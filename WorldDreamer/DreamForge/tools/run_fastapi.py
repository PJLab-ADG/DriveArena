from fastapi import FastAPI, Request, BackgroundTasks
import base64
from PIL import Image
import io
import copy
import os
# from peft import AutoPeftModelForCausalLM
import uvicorn
from argparse import ArgumentParser
from omegaconf import OmegaConf
from hydra.utils import to_absolute_path
import sys
import hydra
from hydra import initialize, compose
from omegaconf import OmegaConf
sys.path.append(".")  # noqa
from dreamforge.misc.test_utils import (
    build_pipe, run_one_batch, run_one_batch_map, update_progress_bar_config, collate_fn_single, ListSetWrapper, partial
)

from mmcv.parallel.data_container import DataContainer
from mmdet3d.core.bbox.structures import LiDARInstance3DBoxes, Box3DMode
from dreamforge.dataset.map_utils import visualize_bev_hdmap, project_map_to_image, project_box_to_image

import torchvision
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from mmdet.datasets.pipelines import to_tensor
import torch
from fastapi.responses import StreamingResponse  
from io import BytesIO  
from dreamforge.runner.utils import concat_6_views
import numpy as np
from typing import Dict
from queue import Queue  
import json
from pydantic import BaseModel
import requests
from mmdet3d.datasets import build_dataset
import mmcv
import time
from nuscenes.eval.common.utils import Quaternion, quaternion_yaw


class ApiSetWrapper(torch.utils.data.DataLoader):
    def __init__(self, param) -> None:
        self.dataset = [param]
        self.data_template = torch.load('data/data_template.pth')
        self.object_classes = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

    def get_can_bus(self, frame):
        ego2global = np.array(frame['metas']['ego_pos'])
        translation = ego2global[:3, 3]
        rotation = Quaternion(matrix=ego2global[:3, :3].astype(np.float64))
        can_bus = np.zeros(9)
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        return can_bus

    def __getitem__(self, idx):
        """This is called by `__getitem__`
            index is the index of the clip_infos
        """
        frames = self.dataset[idx]
        if None in frames:
            return None
        examples = []
        for i, frame in enumerate(frames):
            example = self.process_single(frame)
            # process can bus information
            can_bus = self.get_can_bus(frame)
            if i == 0:
                prev_pos = copy.deepcopy(can_bus[:3])
                prev_angle = copy.deepcopy(can_bus[-1])
                can_bus[:3] = 0
                can_bus[-1] = 0
            else:
                tmp_pos = copy.deepcopy(can_bus[:3])
                tmp_angle = copy.deepcopy(can_bus[-1])
                can_bus[:3] -= prev_pos
                can_bus[-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)
            example['can_bus'] = DataContainer(to_tensor(can_bus), cpu_only=False)
            examples.append(example)

        # fake ref for collate_fn
        examples = [{'img': DataContainer(torch.zeros(6, 3, 224, 400), cpu_only=False), 'can_bus': DataContainer(torch.zeros(9), cpu_only=False)}]*2 + examples

        return examples

    def process_single(self, data):
        mmdet3d_format = {}

        # from data template
        lidar2camera = self.data_template['lidar2camera']
        camera_intrinsics = self.data_template['camera_intrinsics']
        camera2ego = self.data_template['camera2ego']
        mmdet3d_format['lidar2camera'] = DataContainer(lidar2camera)
        mmdet3d_format['camera_intrinsics'] = DataContainer(camera_intrinsics)

        # recompute
        camera2lidar = torch.eye(4, dtype=lidar2camera.dtype)
        camera2lidar = torch.stack([camera2lidar] * len(lidar2camera))
        camera2lidar[:, :3, :3] = lidar2camera[:, :3, :3].transpose(1, 2)
        camera2lidar[:, :3, 3:] = torch.bmm(-camera2lidar[:, :3, :3], lidar2camera[:, :3, 3:])
        lidar2image = torch.bmm(camera_intrinsics, lidar2camera)
        mmdet3d_format['camera2lidar'] = DataContainer(camera2lidar)
        mmdet3d_format['lidar2image'] = DataContainer(lidar2image)

        # from data
        mmdet3d_format['img'] = DataContainer(torch.zeros(6, 3, 224, 400))
        mmdet3d_format['gt_labels_3d'] = DataContainer(torch.tensor(data['gt_labels_3d']))
        mmdet3d_format['metas'] = DataContainer(data['metas'])
        mmdet3d_format['img_aug_matrix'] = DataContainer(self.data_template['img_aug_matrix'])

        # special class
        if torch.tensor(data['gt_bboxes_3d']).size(0) == 0:
            gt_bboxes_3d = torch.zeros(0, 9)   # or all, either can work
        else:
            gt_bboxes_3d = torch.tensor(data['gt_bboxes_3d'])   # or all, either can work
        
        mmdet3d_format['gt_bboxes_3d'] = DataContainer(LiDARInstance3DBoxes(
                gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1],
                origin=(0.5, 0.5, 0)).convert_to(Box3DMode.LIDAR))

        gt_vecs_label = data['gt_vecs_label']
        gt_lines_instance = data['gt_lines_instance']
        drivable_mask = torch.tensor(data['drivable_mask'])
        bev_map = visualize_bev_hdmap(gt_lines_instance, gt_vecs_label, [200, 200], vis_format='polyline_pts', drivable_mask=drivable_mask)
        bev_map = bev_map.transpose(2, 0, 1)
        mmdet3d_format['bev_hdmap'] = DataContainer(torch.tensor(bev_map), cpu_only=False)

        layout_canvas = []
        for i in range(len(lidar2camera)):
            map_canvas = project_map_to_image(gt_lines_instance, gt_vecs_label, camera_intrinsics[i], camera2ego[i], drivable_mask=drivable_mask)
            gt_bboxes= LiDARInstance3DBoxes(gt_bboxes_3d, 
                                            box_dim=gt_bboxes_3d.shape[-1],
                                            origin=(0.5, 0.5, 0)).convert_to(Box3DMode.LIDAR)
            box_canvas = project_box_to_image(gt_bboxes, torch.tensor(data['gt_labels_3d']), lidar2image[i], object_classes=self.object_classes)
            layout_canvas.append(np.concatenate([map_canvas, box_canvas], axis=-1))

        layout_canvas = np.stack(layout_canvas, axis=0)
        layout_canvas = np.transpose(layout_canvas, (0, 3, 1, 2))    # 6, N_channel, H, W
        mmdet3d_format['layout_canvas'] = DataContainer(torch.from_numpy(layout_canvas), cpu_only=False)

        return mmdet3d_format

    def __len__(self):
        return len(self.dataset)


def load_data(cfg, pipe, param=None, scene_idx='0'):
    if param is not None:
        val_dataset = ApiSetWrapper(param)
    else:
        #### datasets ####
        val_dataset = build_dataset(
            OmegaConf.to_container(cfg.dataset.data.val, resolve=True)
        )
        scene_key_frame = val_dataset.scene_key_frame[scene_idx]
        #### dataloader ####
        if cfg.runner.validation_index != "all":
            val_dataset = ListSetWrapper(val_dataset, scene_key_frame)

    return val_dataset


def load_pipe(resume_from_checkpoint, config_name="test_config_single", device='cuda'):
    try:
        initialize(version_base=None, config_path="../configs")
    except:
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        initialize(version_base=None, config_path="../configs")
    output_dir = to_absolute_path(resume_from_checkpoint)
    original_overrides = OmegaConf.load(
        os.path.join(output_dir, "hydra/overrides.yaml"))
    overrides = original_overrides
    cfg = compose(config_name, overrides=overrides)
    cfg.resume_from_checkpoint = resume_from_checkpoint

    #### model ####
    assert cfg.resume_from_checkpoint is not None, "Please set model to load"
    pipe, weight_dtype = build_pipe(cfg, device)

    update_progress_bar_config(pipe, leave=False)

    return pipe, cfg, weight_dtype


def _get_args():
    parser = ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11000)
    parser.add_argument("--agent_port", type=int, default=11001)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--only_keyframe", action='store_true', default=True)
    parser.add_argument("--save_flag", type=bool, default=True)
    parser.add_argument("--save_dir", type=str, default="./tmp")
    parser.add_argument("--is_overlap_condition", type=bool, default=True)
    parser.add_argument("--is_refinement", type=bool, default=True)
    parser.add_argument("--refinement_freq", type=int, default=10)
    parser.add_argument("--model_single", type=str, default='./pretrained/dreamforge-s')
    parser.add_argument("--model", type=str, default='./pretrained/dreamforge-t')

    args = parser.parse_args()
    return args


app = FastAPI()
args = _get_args()
os.makedirs(args.save_dir, exist_ok=True)

### global variants
scene_idx = '0'
batch_index = 0
timestamp = 0.
image_queue = Queue() 
prev_pos = None
prev_angle=None 
ref_images = []
ref_can_bus = []
overlap_images = []
generator = None
agent_command = 2
overlap_length = 1 if args.debug else 2
###
    
pipe_single, cfg_single, _ = load_pipe(args.model_single, config_name="test_config_single")
pipe, cfg, weight_dtype = load_pipe(args.model, config_name="test_config")
val_dataset = load_data(cfg, pipe)

transparent_bg = False
target_map_size = 400

canbus_info = mmcv.load('./data/canbus_infos_trainval.pkl')
info_dict = {}
for sample in canbus_info:
    info_dict[sample['token']] = {'can_bus': sample['can_bus'], 'command': sample['command']}


def _get_can_bus_info(sample_token):
    can_bus = info_dict[sample_token]['can_bus']
    command = info_dict[sample_token]['command']

    ego_pose = np.eye(4)
    ego_pose[:3 ,3] = can_bus[:3]
    ego_pose[:3, :3] = Quaternion(can_bus[3:7]).rotation_matrix

    accel = can_bus[7:10]
    rotation_rate = can_bus[10:13]
    vel = can_bus[13:16]

    return ego_pose, accel, rotation_rate, vel, command


def format_image(image_list):
    image_list = [concat_6_views(framei[0], oneline=True) for framei in image_list]
    formatted_images = []
    for image in image_list:
        formatted_images.append(np.asarray(image))

    # formatted_images = np.stack(formatted_images)
    # 0-255 np -> 0-1 tensor -> grid -> 0-255 pil -> np
    formatted_images = torchvision.utils.make_grid(
        [torchvision.transforms.functional.to_tensor(im) for im in formatted_images], nrow=1)
    formatted_images = to_pil_image(formatted_images)

    return formatted_images


class Data(BaseModel):
    task_id: str
    data: Dict


class ImageNormalize: ## Important !!! check the mean and std which should be consistent with the dataset.
    def __init__(self, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        self.mean = mean
        self.std = std
        self.compose = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.compose(img)


def generate_ref_with_single_pipe(val_input, weight_dtype):
    keys = ['meta_data', 'captions', "pixel_values", "camera_param", 'kwargs', "bev_hdmap", "layout_canvas"]
    val_input_single = {}
    for key in keys:
        if key == 'meta_data':
            meta_data_keys = ['metas', 'lidar2image', 'gt_bboxes_3d', 'gt_bboxes_3d']
            val_input_single[key] = {meta_key: val_input[key][meta_key][:1] for meta_key in meta_data_keys}
        elif key == 'kwargs':
            val_input_single[key] = {'bboxes_3d_data': {k:v[:1] for k,v in val_input[key]['bboxes_3d_data'].items()} if val_input[key]['bboxes_3d_data'] is not None else None}
        else:
            val_input_single[key] = val_input[key][:1]
    cfg_single.runner.validation_times = 1
    cfg_single.show_box = False
    gen_imgs_list = run_one_batch(cfg_single, pipe_single, val_input_single, weight_dtype)[3]
    
    ref_image = torch.stack([ImageNormalize()(x) for x in gen_imgs_list[0][0]])
    ref_images = torch.stack([ref_image, ref_image.clone()])
    val_input['ref_values'] = ref_images
    
    return val_input


def refining_ref_with_single_pipe(start_idx, ref_images, val_input, weight_dtype):
    keys = ['meta_data', 'captions', "pixel_values", "camera_param", 'kwargs', "bev_hdmap", "layout_canvas"]
    val_input_single = {}
    for key in keys:
        if key == 'meta_data':
            meta_data_keys = ['metas', 'lidar2image', 'gt_bboxes_3d', 'gt_bboxes_3d']
            val_input_single[key] = {meta_key: val_input[key][meta_key][start_idx:start_idx+2] for meta_key in meta_data_keys}
        elif key == 'kwargs':
            val_input_single[key] = {'bboxes_3d_data': {k:v[start_idx:start_idx+2] for k,v in val_input[key]['bboxes_3d_data'].items()} if val_input[key]['bboxes_3d_data'] is not None else None}
        else:
            val_input_single[key] = val_input[key][start_idx:start_idx+2]
    val_input['conditional_values'] = torch.stack(ref_images)

    cfg_single.runner.validation_times = 1
    cfg_single.show_box = False
    gen_imgs_list = run_one_batch(cfg_single, pipe_single, val_input_single, weight_dtype)[3]
    ref_image = [torch.stack([ImageNormalize()(x) for x in gen_imgs_list[idx][0]]) for idx in range(2)]

    return ref_image


@app.post("/dreamer-api/")
async def process(request: Request, background_tasks: BackgroundTasks):
    global val_dataset, batch_index, generator, ref_images, ref_can_bus, overlap_images, prev_pos, prev_angle, overlap_length, agent_command, timestamp
    param = await request.json()
    if isinstance(param, str):
        param = json.loads(param)
    
    if not args.debug:
        val_dataset = load_data(cfg, pipe, param=param)
        key_param = param[-1]
        agent_command = key_param['agent_command']

    if args.debug:
        print(f'DEBUG: batch_index: {batch_index}; len(ref_images): {len(ref_images)}; len(ref_can_bus): {len(ref_can_bus)}; len(overlap_images): {len(overlap_images)}; overlap_length: {overlap_length}')

    raw_data = val_dataset[batch_index] if args.debug else val_dataset[0]  # cannot index loader
    val_input = collate_fn_single(
        raw_data, cfg.dataset.template, is_train=False,
        bbox_mode=cfg.model.bbox_mode,
        bbox_view_shared=cfg.model.bbox_view_shared,
        ref_length=cfg.model.ref_length,
    )
    batch_index += 1
        
    if len(ref_images) > 0:
        val_input['ref_values'] = torch.stack(ref_images)
        val_input['can_bus'][0][:3] = prev_pos
        val_input['can_bus'][0][-1] = prev_angle 
        ref_can_bus[0][:3] = 0
        ref_can_bus[0][-1] = 0
        val_input['ref_can_bus'] = torch.stack(ref_can_bus)
        if args.is_overlap_condition:
            val_input['overlap_values'] = torch.stack(overlap_images)
    else:
        val_input = generate_ref_with_single_pipe(val_input, weight_dtype)

    return_tuples = run_one_batch_map(cfg, pipe, val_input, weight_dtype,
                                transparent_bg=transparent_bg, generator=generator,
                                map_size=target_map_size)
    generator, return_tuples = return_tuples[-1], return_tuples[:-1]

    ref_idxs = [cfg.model.video_length-overlap_length-2, cfg.model.video_length-overlap_length-1]
    ref_images = [torch.stack([ImageNormalize()(x) for x in return_tuples[3][idx][0]]) for idx in ref_idxs]

    if args.is_refinement and batch_index % args.refinement_freq == 0:
        assert ref_idxs[0] == ref_idxs[1] - 1
        ref_images = refining_ref_with_single_pipe(ref_idxs[0], ref_images, val_input, weight_dtype)

    if args.is_overlap_condition:
        overlap_images = [torch.stack([ImageNormalize()(x) for x in return_tuples[3][idx][0]]) for idx in range(cfg.model.video_length-overlap_length, cfg.model.video_length)]
    ref_can_bus = [val_input['can_bus'][idx] for idx in ref_idxs]

    prev_pos, prev_angle = val_input['ref_can_bus'][0][:3], val_input['ref_can_bus'][0][-1]
    for idx in range(ref_idxs[-1]+1, cfg.model.video_length-overlap_length+1):
        prev_pos += val_input['can_bus'][idx][:3]
        prev_angle += val_input['can_bus'][idx][-1]
    
    if args.save_flag:
        # combined_image = format_image(return_tuples[4])  # 7, 1, 6
        image_list = [concat_6_views(framei[0], oneline=True) for framei in return_tuples[4]]
        os.makedirs(os.path.join(args.save_dir, f'diffusion_{str(int(timestamp*2)).zfill(3)}'), exist_ok=True)
        for i, x in enumerate(image_list):
            x.save(os.path.join(args.save_dir, f'diffusion_{str(int(timestamp*2)).zfill(3)}/{i}.jpg'))

    gen_imgs = return_tuples[3][-1][0] # the last frame is keyframe
    ## To TrafficManager
    if not args.only_keyframe:
        combined_image = format_image(return_tuples[4])  # 7, 1, 6
    else:
        combined_image = Image.new("RGB", (gen_imgs[0].width, sum(img.height for img in gen_imgs)))  
        y_offset = 0  
        for img in gen_imgs:  
            combined_image.paste(img, (0, y_offset))  
            y_offset += img.height
    img_byte_array = BytesIO()  
    combined_image.save(img_byte_array, format="PNG")  
    img_byte_array.seek(0)  
    
    ## To Agent
    if not args.debug:
        image_queue.put({'param': key_param, 'img': gen_imgs})
    else:
        sample_token = val_input['meta_data']['metas'][-1].data['token']
        ego_pos, accel, rotation_rate, vel, agent_command = _get_can_bus_info(sample_token)
        param = {
            'metas': {'ego_pos': ego_pos.tolist(), 'accel': accel.tolist(), 'rotation_rate': rotation_rate.tolist(), 'vel': vel.tolist()}
        }
        image_queue.put({'param': param, 'img': gen_imgs})
    background_tasks.add_task(send2agent)

    return StreamingResponse(io.BytesIO(img_byte_array.read()), media_type="image/png")  


async def send2agent():
    global timestamp, agent_command
    print("current timestamp:", timestamp)
    if args.debug:
        print(f'DEBUG: agent_command: {agent_command}')
    if image_queue.qsize() >= 1:
        cur_data = image_queue.get() 
        gen_imgs = cur_data['img']
        combined_image = Image.new("RGB", (gen_imgs[0].width, sum(img.height for img in gen_imgs)))  
        y_offset = 0  
        for img in gen_imgs:  
            combined_image.paste(img, (0, y_offset))  
            y_offset += img.height
        img_byte_array = io.BytesIO()  
        combined_image.save(img_byte_array, format="PNG")  
        img_byte_array = img_byte_array.getvalue()
        image_base64 = base64.b64encode(img_byte_array).decode('utf-8')
        ego_pose = cur_data['param']['metas']['ego_pos']
        accel = cur_data['param']['metas']['accel']
        rotation_rate = cur_data['param']['metas']['rotation_rate']
        vel = cur_data['param']['metas']['vel']
        send_data = {
            'timestamp':timestamp, 
            'img_byte_array': image_base64, 
            'ego_pose':ego_pose, 
            'command': agent_command, 
            'accel': accel, 
            'rotation_rate': rotation_rate, 
            'vel': vel}
        response = requests.post("http://localhost:{}/driver-api".format(args.agent_port), json=send_data)
        timestamp = timestamp + 0.5


@app.get("/dreamer-clean/")
async def clean_history():
    global batch_index, generator, ref_images, ref_can_bus, overlap_images, prev_pos, prev_angle, agent_command, timestamp
    print('[world dreamer] cleaned out past frames')
    batch_index = 0
    generator = None
    prev_pos = None
    prev_angle = None
    ref_images = []
    ref_can_bus = []
    overlap_images = []
    timestamp = 0.
    agent_command = 2
    while image_queue.qsize() > 0:  
        image_queue.get()  


if __name__ == '__main__':
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")