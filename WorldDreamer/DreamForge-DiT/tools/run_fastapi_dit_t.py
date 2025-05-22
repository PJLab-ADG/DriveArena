from fastapi import FastAPI, Request, BackgroundTasks
import base64
from PIL import Image
import io
import copy
import os
import mmcv
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import sys
import numpy as np
import torch
import uvicorn
from argparse import ArgumentParser
import importlib.util
from fastapi.responses import StreamingResponse
import requests
import json
import queue
from einops import rearrange, repeat
from typing import Tuple, List, Dict, Any
from mmcv.parallel import DataContainer as DC
from mmcv.parallel.data_container import DataContainer
from mmdet.datasets.pipelines import to_tensor
from nuscenes.eval.common.utils import Quaternion
import logging
import debugpy

sys.path.append(".")  # noqa

import torchvision
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image


from dreamforgedit.registry import MODELS, SCHEDULERS, build_module
from dreamforgedit.utils.config_utils import parse_configs, merge_dataset_cfg
from dreamforgedit.utils.misc import to_torch_dtype, move_to
from dreamforgedit.datasets.map_utils.map_tools import visualize_bev_hdmap, project_box_to_image, project_lines_on_view
from dreamforgedit.mmdet_plugin.core.bbox import LiDARInstance3DBoxes
from dreamforgedit.mmdet_plugin.core.bbox.structures.box_3d_mode import Box3DMode
from dreamforgedit.utils.dataset_utils import concat_6_views
from dreamforgedit.datasets.utils import trans_boxes_to_views
# Add DreamForgeDiT inference utils
import os
import copy
import logging
from collections import OrderedDict
from functools import partial

import torch
import torch.distributed as dist
from colossalai.cluster import DistCoordinator, ProcessGroupMesh
from colossalai.booster.plugin import LowLevelZeroPlugin

from dreamforgedit.acceleration.parallel_states import set_data_parallel_group, set_sequence_parallel_group, get_data_parallel_group
from dreamforgedit.acceleration.plugin import ZeroSeqParallelPlugin
from dreamforgedit.datasets import save_sample
from dreamforgedit.acceleration.communications import gather_tensors


from dreamforgedit.utils.misc import get_logger, collate_bboxes_to_maxlen, move_to, add_box_latent, warn_once
from dreamforgedit.utils.inference_utils import add_null_condition, concat_6_views_pt, enable_offload
from dreamforgedit.registry import DATASETS, MODELS, SCHEDULERS, build_module
from nuscenes.eval.common.utils import Quaternion, quaternion_yaw

META_KEY_LIST = [
    "gt_bboxes_3d",
    "gt_labels_3d",
    "camera_intrinsics",
    "camera2ego",
    "lidar2ego",
    "lidar2camera",
    "camera2lidar",
    "lidar2image",
    "img_aug_matrix",
    "metas",
]


logger = logging.getLogger(__name__)

parser = ArgumentParser()
parser.add_argument("--config_single", type=str, default="./configs/dreamforge/inference/1x448x800_stdit3_CogVAE_noTemp_xCE_wSST.py", help="single-frame config of DiT path")
parser.add_argument("--config", type=str, default="./configs/dreamforge/inference/r17x448x800_stdit3_CogVAE_boxTDS_wCT_xCE_wSST.py", help="multi-frame config of DiT path")
parser.add_argument("--host", type=str, default="0.0.0.0", help="")
parser.add_argument("--port", type=int, default=11000, help="")
parser.add_argument("--debug", action="store_true", help="")
parser.add_argument("--agent_port", type=int, default=11001, help="")
parser.add_argument("--single_model_path", type=str, default='./pretrained/DreamForgeDiT-s/ema.pt', help="")
parser.add_argument("--model_path", type=str, default= './pretrained/DreamForgeDiT-t/ema.pt', help="")
parser.add_argument("--ref_refine_freq", type=int, default=5)
parser.add_argument("--is_overlap_condition", action="store_true", help="")
parser.add_argument("--ref_image_path", type=str, default=None, help="")
parser.add_argument("--cfg_options", nargs="+", default=[])
args = parser.parse_args()


app = FastAPI()


pipe = None
batch_index = 0
prev_pos = None
prev_angle = None
ref_samples = None
timestamp = 0.
agent_command = 2
image_queue = queue.Queue()
device = None
weight_dtype = None

def load_pipe(cfg_path, model_path=None):
    global weight_dtype, device
    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = to_torch_dtype("bf16")
    
    # use parse_configs to load config
    cfg_dict = parse_configs(cfg_path)
    
    # if model_path, set from_pretrained
    if model_path:
        cfg_dict["model"]["from_pretrained"] = model_path
    
    # build text encoder
    text_encoder = build_module(cfg_dict["text_encoder"], MODELS, device=device)
    # build vae
    vae = build_module(cfg_dict["vae"], MODELS).to(device, weight_dtype).eval()
    
    # if vae_tiling, set tiling param
    if "vae_tiling" in cfg_dict:
        TILING_PARAM = {
            "1": {"decoder": True},
            "2": {"encoder": True, "decoder": True}
        }
        vae.module.enable_tiling(**TILING_PARAM[str(cfg_dict["vae_tiling"])])
        print(f"VAE tiling: {TILING_PARAM[str(cfg_dict['vae_tiling'])]}")
    
    # build model
    model = build_module(
        cfg_dict["model"],
        MODELS,
        input_size=(None, None, None),
        in_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length,
        enable_sequence_parallelism=False,
    ).to(device, weight_dtype).eval()
    
    # set y_embedder
    text_encoder.y_embedder = model.y_embedder
    
    # build scheduler
    scheduler_cfg = cfg_dict.get("val", {}).get("scheduler", cfg_dict["scheduler"])
    scheduler = build_module(scheduler_cfg, SCHEDULERS)
    
    # get mv_order_map
    mv_order_map = cfg_dict.get("mv_order_map", None)
    t_order_map = cfg_dict.get("t_order_map", None)
    
    # build pipe
    pipe = {
        'model': model,
        'vae': vae,
        'text_encoder': text_encoder,
        'scheduler': scheduler,
        'cfg': cfg_dict,
        'mv_order_map': mv_order_map,
        't_order_map': t_order_map
    }
    
    print("build pipe success")
    return pipe



def ensure_canvas(coords, canvas_size: Tuple[int, int]):
    """Box with any point in range of canvas should be kept.

    Args:
        coords (_type_): _description_
        canvas_size (Tuple[int, int]): _description_

    Returns:
        np.array: mask on first axis.
    """
    (h, w) = canvas_size
    c_mask = np.any(coords[..., 2] > 0, axis=1)
    w_mask = np.any(np.logical_and(
        coords[..., 0] > 0, coords[..., 0] < w), axis=1)
    h_mask = np.any(np.logical_and(
        coords[..., 1] > 0, coords[..., 1] < h), axis=1)
    c_mask = np.logical_and(c_mask, np.logical_and(w_mask, h_mask))
    return c_mask


def ensure_positive_z(coords):
    c_mask = np.any(coords[..., 2] > 0, axis=1)
    return c_mask

def format_image(image_list):
    image_list = [concat_6_views(framei[0], oneline=True) for framei in image_list]
    formatted_images = []
    for image in image_list:
        formatted_images.append(np.asarray(image))

    formatted_images = torchvision.utils.make_grid(
        [torchvision.transforms.functional.to_tensor(im) for im in formatted_images], nrow=1)
    formatted_images = to_pil_image(formatted_images)

    return formatted_images

class ApiSetWrapper:
    def __init__(self, param) -> None:
        self.param = param
        self.dataset = [param]
        

        try:
            self.data_template = torch.load('data/data_template.pth')
            print("load data template success")
        except Exception as e:
            print(f"load data template failed: {e}")

            self.data_template = self._create_default_template()
            
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
        
    def __len__(self):
        return len(self.param)
    
    def __getitem__(self, idx):
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
        mmdet3d_format['img'] = DataContainer(torch.zeros(6, 3, 224 , 400))
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
        bev_map = visualize_bev_hdmap(gt_lines_instance, gt_vecs_label, [200, 200], drivable_mask=None)
        bev_map = bev_map.transpose(2, 0, 1)
        mmdet3d_format['bev_hdmap'] = DataContainer(torch.tensor(bev_map), cpu_only=False)

        layout_canvas = []
        layout_canvas = []
        image_size = (448, 800)
        for i in range(len(lidar2image)):
            # TODO: Should we consider img_aug_matrix?
            map_canvas = project_lines_on_view(gt_lines_instance, gt_vecs_label, camera_intrinsics[i], camera2ego[i], image_size=image_size)
            gt_bboxes= LiDARInstance3DBoxes(gt_bboxes_3d, 
                                            box_dim=gt_bboxes_3d.shape[-1],
                                            origin=(0.5, 0.5, 0)).convert_to(Box3DMode.LIDAR)
            box_canvas = project_box_to_image(gt_bboxes, torch.tensor(data['gt_labels_3d']), lidar2image[i], object_classes=self.object_classes, image_size=image_size)
            layout_canvas.append(np.concatenate([map_canvas, box_canvas], axis=-1))

        layout_canvas = np.stack(layout_canvas, axis=0)
        layout_canvas = np.transpose(layout_canvas, (0, 3, 1, 2))    # 6, N_channel, H, W
        mmdet3d_format['layout_canvas'] = DataContainer(torch.from_numpy(layout_canvas), cpu_only=False)
        
        # 8. get pixel values shape
        mmdet3d_format['pixel_values_shape'] = torch.tensor(data.get('pixel_shape', [1, 16, 6, 3, 448, 800]))
        mmdet3d_format['captions'] = [data.get('prompt', 'A driving scene image')]
        mmdet3d_format['height'] = data.get('height', 448)
        mmdet3d_format['width'] = data.get('width', 800)
        mmdet3d_format['num_frames'] = data.get('num_frames', 16)
        mmdet3d_format['fps'] = data.get('fps', 12)
        
        mmdet3d_format['metas'] = DataContainer(data['metas'])

        mmdet3d_format['pixel_values'] = DataContainer(torch.zeros(6, 3, 448, 800), cpu_only=False)

        camera_param = torch.stack([torch.cat([
            camera_intrinsics.data[:, :3, :3],  # 3x3 is enough
            lidar2camera.data[:, :3],  # only first 3 rows meaningful
        ], dim=-1)], dim=0)
        mmdet3d_format['camera_param'] = DataContainer(camera_param, cpu_only=False)
        frame_emb = torch.eye(4)
        mmdet3d_format['relative_pose'] = DataContainer(frame_emb, cpu_only=False)
        
        return mmdet3d_format
    
def stack_tensors_in_dicts(
        dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """stack any Tensor in list of dicts. If holder is provided, dicts will be
    stacked ahead of holder tensor. Make sure no dict is changed in place.

    Args:
        dicts (List[Dict[str, Any]]): dicts to stack, without the desired dim.

    Raises:
        TypeError: if the datatype for values are not Tensor or dict.

    Returns:
        Dict[str, Any]: stacked dict.
    """
    out_dict = {}
    out_dict['bboxes_3d_data'] = {}
    out_dict['meta_data'] = {}
    for key in ["pixel_values", "bev_hdmap", "layout_canvas", "camera_param", "frame_emb"]:
        out_dict[key] = torch.stack([ret_dict[key] for ret_dict in dicts])
    if dicts[0]['bboxes_3d_data'] is not None:
        for key in dicts[0]['bboxes_3d_data'].keys():
            out_dict['bboxes_3d_data'][key] = torch.stack([ret_dict['bboxes_3d_data'][key] for ret_dict in dicts])
    else:
        b, t, nc= out_dict['pixel_values'].shape[:3]
        out_dict['bboxes_3d_data'] = {}
        out_dict['bboxes_3d_data']['bboxes'] = torch.zeros(b, t, nc, 1, 8, 3)
        out_dict['bboxes_3d_data']['classes'] = -torch.ones(b, t, nc, 1)
        out_dict['bboxes_3d_data']['masks'] = torch.ones(b, t, nc, 1)

    for key in ['fps', 'num_frames', 'height', 'width']:
        out_dict[key] = to_tensor([ret_dict[key] for ret_dict in dicts][0])

    # caption
    out_dict['captions'] = [[ret_dict['captions'][0] for ret_dict in dicts]] # for each clip only one caption, T is the first dim

        # other meta data
    for key in dicts[0]['meta_data'].keys():
        out_dict['meta_data'][key] = [ret_dict['meta_data'][key] for ret_dict in dicts]

    return out_dict

def collate_fn(
    examples: Tuple[dict, ...],
    template: str,
    bbox_mode: str = None,
    return_raw_data = False,
    is_train: bool = False,
    with_temporal_dim: bool = False,
    is_clip: bool = False,
    drop_ori_imgs: bool = False,
):
    """
    We need to handle:
    1. make multi-view images (img) into tensor -> [N, 6, 3, H, W]
    2. make bev_hdmap into tensor, 
        -> [N, 3, 200, 200]
    3. make layout_canvas into tensor
        -> [N, 6, 13 = 3 map + 10 obj, H, W]
    4. extract camera parameters (camera_intrinsics, camera2lidar)
        camera2lidar: A @ v_camera = v_lidar
        -> [N, 6, 3, 7]
    We keep other meta data as original.
    """
    if return_raw_data:
        return example

    # multi-view images
    pixel_values = torch.stack([example["pixel_values"].data for example in examples])
    pixel_values = pixel_values.to(
        memory_format=torch.contiguous_format).float()

    # layout_canvas
    layout_canvas = torch.stack(
        [example["layout_canvas"].data for example in examples])
    layout_canvas = layout_canvas.to(
        memory_format=torch.contiguous_format).float()

    # bev_hdmap
    bev_hdmap = torch.stack(
        [example["bev_hdmap"].data for example in examples])
    bev_hdmap = bev_hdmap.to(
        memory_format=torch.contiguous_format).float()

    # camera param
    camera_param = torch.stack([torch.cat([
        example["camera_intrinsics"].data[:, :3, :3],  # 3x3 is enough
        example["lidar2camera"].data[:, :3],  # only first 3 rows meaningful
    ], dim=-1) for example in examples], dim=0)

    camera_int = torch.stack([
        example["camera_intrinsics"].data[:, :3, :3]  # 3x3 is enought
        for example in examples], dim=0)
    camera_ext = torch.stack([
        example["lidar2camera"].data for example in examples], dim=0)
    # aug is either eye or has values
    camera_aug = torch.stack([
        example["img_aug_matrix"].data for example in examples], dim=0)

    ret_dict = {
        "pixel_values": pixel_values,
        "bev_hdmap": bev_hdmap,
        "layout_canvas": layout_canvas,
        "camera_param": camera_param,
        "camera_param_raw": {
            "int": camera_int,
            "ext": camera_ext,
            "aug": camera_aug,
        },
    }

    # placeholder: frame embedding, fps.
    ret_dict['frame_emb'] = torch.stack([example["relative_pose"].data for example in examples])

    if not is_clip:
        for key in ['fps', 'num_frames', 'height', 'width']:
            ret_dict[key] = to_tensor([example[key] for example in examples])
    else:
        for key in ['fps', 'num_frames', 'height', 'width']:
            ret_dict[key] = examples[0][key]

    # bboxes_3d, convert to tensor
    # here we consider:
    # 1. do we need to filter bboxes for each view? use `view_shared`
    # 2. padding for one batch of data if need (with zero), and output mask.
    # 3. what is the expected output format? dict of kwargs to bbox embedder
    # TODO: should we change to frame's coordinate?
    canvas_size = pixel_values.shape[-2:]
    if bbox_mode is not None:
        # NOTE: both can be None
        bboxes_3d_input, _ = _preprocess_bbox(
            bbox_mode, canvas_size, examples)
        # if bboxes_3d_input is not None:
        #     bboxes_3d_input["cam_params"] = camera_param
        ret_dict["bboxes_3d_data"] = bboxes_3d_input

    # captions: one real caption with one null caption
    captions = []
    for example in examples:
        caption = template.format(**example["metas"].data)
        captions.append(caption)
    ret_dict["captions"] = captions  # list of str

    # other meta data
    meta_list_dict = dict()
    for key in META_KEY_LIST:
        try:
            meta_list = [example[key] for example in examples]
            meta_list_dict[key] = meta_list
        except KeyError:
            continue
    ret_dict['meta_data'] = meta_list_dict

    if with_temporal_dim:
        for key in ["pixel_values", "bev_hdmap", "layout_canvas", "camera_param", "frame_emb"]:
            ret_dict[key] = ret_dict[key].unsqueeze(1)
        if 'bboxes_3d_data' in ret_dict and ret_dict['bboxes_3d_data'] is not None:
            for k, v in ret_dict['bboxes_3d_data'].items():
                ret_dict['bboxes_3d_data'][k] = v.unsqueeze(1)
        else:
            b, t, nc = ret_dict['pixel_values'].shape[:3]
            ret_dict['bboxes_3d_data'] = {}
            ret_dict['bboxes_3d_data']['bboxes'] = torch.zeros(b, t, nc, 1, 8, 3)
            ret_dict['bboxes_3d_data']['classes'] = -torch.ones(b, t, nc, 1)
            ret_dict['bboxes_3d_data']['masks'] = torch.ones(b, t, nc, 1)

        ret_dict['captions'] = [ret_dict['captions']] # T is the first dim

    if drop_ori_imgs:
        ret_dict['pixel_values_shape'] = torch.IntTensor(list(ret_dict['pixel_values'].shape))
        ret_dict.pop('pixel_values')

    return ret_dict


def collate_fn_t(
    examples: Tuple[dict, ...],
    template: str,
    bbox_mode: str = None,
    return_raw_data = False,
    is_train: bool = False,
    is_clip: bool = True,
    drop_ori_imgs: bool = False,
):
    """
    We need to handle:
    1. make multi-view images (img) into tensor -> [N, 6, 3, H, W]
    2. make bev_hdmap into tensor, 
        -> [N, 3, 200, 200]
    3. make layout_canvas into tensor
        -> [N, 6, 13 = 3 map + 10 obj, H, W]
    4. extract camera parameters (camera_intrinsics, camera2lidar)
        camera2lidar: A @ v_camera = v_lidar
        -> [N, 6, 3, 7]
    We keep other meta data as original.
    """
    if return_raw_data:
        return examples

    ret_dicts = []
    bbox_maxlen = 0
    for example_ti in examples:
        ret_dict = collate_fn(
            example_ti, template=template, bbox_mode=bbox_mode, return_raw_data=return_raw_data, is_train=is_train, is_clip=is_clip)
        if ret_dict['bboxes_3d_data'] is not None:
            bb_shape = ret_dict['bboxes_3d_data']['bboxes'].shape
            bbox_maxlen = max(bbox_maxlen, bb_shape[2])
        ret_dicts.append(ret_dict)

    if bbox_maxlen != 0:
        for ret_dict in ret_dicts:
            bboxes_3d_data = ret_dict['bboxes_3d_data']
            # if it is None while others not, we replace it will all padding.
            bboxes_3d_data = {} if bboxes_3d_data is None else bboxes_3d_data
            new_data = pad_bboxes_to_maxlen(
                bb_shape, bbox_maxlen, **bboxes_3d_data)
            ret_dict['bboxes_3d_data'].update(new_data)

    ret_dicts = stack_tensors_in_dicts(ret_dicts) 

    if drop_ori_imgs:
        ret_dicts['pixel_values_shape'] = torch.IntTensor(list(ret_dicts['pixel_values'].shape))
        ret_dicts.pop('pixel_values')

    return ret_dicts


def _transform_all(examples, matrix_key, proj):
    """project all bbox to views, return 2d coordinates.

    Args:
        examples (List): collate_fn input.

    Returns:
        2-d list: List[List[np.array]] for B, N_cam. Can be [].
    """
    gt_bboxes_3d: List[LiDARInstance3DBoxes] = [
        example["gt_bboxes_3d"].data for example in examples]
    # lidar2image (np.array): lidar to image view transformation
    trans_matrix = np.stack([example[matrix_key].data.numpy()
                            for example in examples], axis=0)
    # img_aug_matrix (np.array): augmentation matrix
    img_aug_matrix = np.stack([example['img_aug_matrix'].data.numpy()
                               for example in examples], axis=0)
    B, N_cam = trans_matrix.shape[:2]

    bboxes_coord = []
    # for each keyframe set
    for idx in range(B):
        # if zero, add empty list
        if len(gt_bboxes_3d[idx]) == 0:
            # keep N_cam dim for convenient
            bboxes_coord.append([None for _ in range(N_cam)])
            continue

        coords_list = trans_boxes_to_views(
            gt_bboxes_3d[idx], trans_matrix[idx], img_aug_matrix[idx], proj)
        bboxes_coord.append(coords_list)
    return bboxes_coord

def _preprocess_bbox(bbox_mode, canvas_size, examples, use_3d_filter=True):
    """Pre-processing for bbox
    .. code-block:: none

                                       up z
                        front x           ^
                             /            |
                            /             |
              (x1, y0, z1) + -----------  + (x1, y1, z1)
                          /|            / |
                         / |           /  |
           (x0, y0, z1) + ----------- +   + (x1, y1, z0)
                        |  /      .   |  /
                        | / origin    | /
        left y<-------- + ----------- + (x0, y1, z0)
            (x0, y0, z0)

    Args:
        bbox_mode (str): type of bbox raw data.
            cxyz -> x1y1z1, x1y0z1, x1y1z0, x0y1z1;
            all-xyz -> all 8 corners xyz;
            owhr -> center, l, w, h, z-orientation.
        canvas_size (2-tuple): H, W of input images
        examples: collate_fn input
    Return:
        in form of dict:
            bboxes (Tensor): B, N_cam, max_len, ...
            classes (LongTensor): B, N_cam, max_len
            masks: 1 for data, 0 for padding
    """
    # init data
    bboxes = []
    classes = []
    max_len = 0
    gt_bboxes_3d: List[LiDARInstance3DBoxes] = [
        example["gt_bboxes_3d"].data for example in examples]
    gt_labels_3d: List[torch.Tensor] = [
        example["gt_labels_3d"].data for example in examples]
    
    # params
    B = len(gt_bboxes_3d)
    N_out = len(examples[0]['lidar2image'].data.numpy())

    bboxes_coord = None
    if not use_3d_filter:
        bboxes_coord = _transform_all(examples, 'lidar2image', True)
    else:
        bboxes_coord_3d = _transform_all(examples, 'lidar2camera', False)

    # set value for boxes
    for bi in range(B):
        bboxes_kf = gt_bboxes_3d[bi]
        classes_kf = gt_labels_3d[bi]

        # if zero, add zero length tensor (for padding).
        if len(bboxes_kf) == 0:
            bboxes.append([None] * N_out)
            classes.append([None] * N_out)
            continue

        # filtered by 2d projection.
        index_list = []  # each view has a mask
        if use_3d_filter:
            coords_list = bboxes_coord_3d[bi]
            filter_func = ensure_positive_z
        else:
            # filter bbox according to 2d projection on image canvas
            coords_list = bboxes_coord[bi]
            # judge coord by cancas_size
            filter_func = partial(ensure_canvas, canvas_size=canvas_size)
        # we do not need to handle None since we already filter for len=0
        for coords in coords_list:
            c_mask = filter_func(coords)
            index_list.append(c_mask)
            max_len = max(max_len, c_mask.sum())
        
        # == mask all done here ==

        # == bboxes & classes, same across the whole batch ==
        if bbox_mode == 'cxyz':
            # x1y1z1, x1y0z1, x1y1z0, x0y1z1
            bboxes_pt = bboxes_kf.corners[:, [6, 5, 7, 2]]
        elif bbox_mode == 'all-xyz':
            bboxes_pt = bboxes_kf.corners  # n x 8 x 3
        elif bbox_mode == 'owhr':
            raise NotImplementedError("Not sure how to do this.")
        else:
            raise NotImplementedError(f"Wrong mode {bbox_mode}")
        bboxes.append([bboxes_pt[ind] for ind in index_list])
        classes.append([classes_kf[ind] for ind in index_list])
        bbox_shape = bboxes_pt.shape[1:]

     # there is no (visible) boxes in this batch
    if max_len == 0:
        return None, None

    # pad and construct mask
    # `bbox_shape` should be set correctly
    ret_dict = pad_bboxes_to_maxlen(
        [B, N_out, max_len, *bbox_shape], max_len, bboxes, classes)
    
    return ret_dict, bboxes_coord


def pad_bboxes_to_maxlen(
        bbox_shape, max_len, bboxes=None, classes=None, masks=None, **kwargs):
    B, N_out = bbox_shape[:2]
    ret_bboxes = torch.zeros(B, N_out, max_len, *bbox_shape[3:])
    # we set unknown to -1. since we have mask, it does not matter.
    ret_classes = -torch.ones(B, N_out, max_len, dtype=torch.long)
    ret_masks = torch.zeros(B, N_out, max_len, dtype=torch.bool)
    if bboxes is not None:
        for _b in range(B):
            _bboxes = bboxes[_b]
            _classes = classes[_b]
            for _n in range(N_out):
                if _bboxes[_n] is None:
                    continue  # empty for this view
                this_box_num = len(_bboxes[_n])
                ret_bboxes[_b, _n, :this_box_num] = _bboxes[_n]
                ret_classes[_b, _n, :this_box_num] = _classes[_n]
                if masks is not None:
                    ret_masks[_b, _n, :this_box_num] = masks[_b, _n]
                else:
                    ret_masks[_b, _n, :this_box_num] = True

    # assemble as input format
    ret_dict = {
        "bboxes": ret_bboxes,
        "classes": ret_classes,
        "masks": ret_masks
    }
    return ret_dict


# load data
def load_data(cfg, pipe, param=None):
    if param is not None:
        val_dataset = ApiSetWrapper(param)
    else:
        # use merge_dataset_cfg to load dataset
        dataset, val_dataset = merge_dataset_cfg(
            cfg, cfg["data_cfg_name"], cfg.get("dataset_cfg_overrides", []),
            cfg["num_frames"])
    return val_dataset

# move_to function
def custom_move_to(obj, device, dtype=None, filter=lambda x: True):
    if torch.is_tensor(obj):
        if filter(obj):
            if dtype is None:
                dtype = obj.dtype
            return obj.to(device, dtype)
        else:
            return obj
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = custom_move_to(v, device, dtype, filter)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(custom_move_to(v, device, dtype, filter))
        return res
    elif isinstance(obj, DC):
        # move DataContainer
        if obj._cpu_only:
            return obj
        else:
            return DC(custom_move_to(obj._data, device, dtype, filter), 
                     cpu_only=obj._cpu_only, stack=obj._stack)
    elif obj is None:
        return obj
    else:
        # move other object
        return obj

# generate images
def generate_images(val_input, weight_dtype, ref_input=None):
    global ref_samples
    dtype = to_torch_dtype(pipe['cfg'].get("dtype", "bf16"))
    verbose = pipe['cfg'].get("verbose", 1)
    batch = copy.deepcopy(val_input)
    B, T, NC = batch["pixel_values"].shape[:3]
    # breakpoint()
    latent_size = pipe['vae'].get_latent_size((T, *batch["pixel_values"].shape[-2:]))
    if ref_input is not None:
        ref_samples = ref_input.to(device, dtype)

    # == prepare batch prompts ==
    y = batch.pop("captions")[0]  # B, just take first frame

    batch_prompts = y
    neg_prompts = None
    maps = batch.pop("bev_hdmap").to(device, dtype)  # B, T, C, H, W
    layouts = batch.pop("layout_canvas").to(device, dtype)
    bbox = batch.pop("bboxes_3d_data")
    # B, T, NC, 3, 7
    cams = batch.pop("camera_param").to(device, dtype)
    cams = rearrange(cams, "B T NC ... -> (B NC) T 1 ...")  # BxNC, T, 1, 3, 7
    rel_pos = batch.pop("frame_emb").to(device, dtype)
    rel_pos = repeat(rel_pos, "B T ... -> (B NC) T 1 ...", NC=NC)  # BxNC, T, 1, 4, 4

    # == model input format ==
    model_args = {}
    model_args["maps"] = maps
    model_args["layouts"] = layouts
    model_args["bbox"] = bbox
    model_args["cams"] = cams
    model_args["rel_pos"] = rel_pos
    model_args["fps"] = torch.tensor([12.], device=device, dtype=dtype)
    model_args["height"] = torch.tensor([448.], device=device, dtype=dtype)
    model_args["width"] = torch.tensor([800.], device=device, dtype=dtype)
    if batch_index == 0 and T == 1:
        model_args["num_frames"] = torch.tensor([1.], device=device, dtype=dtype)
    else:
        model_args["num_frames"] = torch.tensor([17.], device=device, dtype=dtype)
    model_args = move_to(model_args, device=device, dtype=dtype)
    model_args["mv_order_map"] = pipe['cfg'].get("mv_order_map", None)
    model_args["t_order_map"] = pipe['cfg'].get("t_order_map", None)
    model_args["img_metas"] = batch.pop("meta_data")

    logging.info('start gather fps ...')
    # _fpss = gather_tensors(model_args['fps'], pg=get_data_parallel_group())
    logging.info('end gather fps ...')

    # == sampling ==
    torch.manual_seed(1024)  # NOTE: not sure how to handle loop, just change here.
    z = torch.randn(len(batch_prompts), pipe['vae'].out_channels * NC, *latent_size, device=device, dtype=dtype)

    
    # == sample box ==
    if bbox is not None:
        # null set values to all zeros, this should be safe
        if T == 1:
            bbox = add_box_latent(bbox, B, NC, T, pipe_single['model'].sample_box_latent)
        else:
            bbox = add_box_latent(bbox, B, NC, T, pipe['model'].sample_box_latent)
        # overwrite!
        new_bbox = {}
        for k, v in bbox.items():
            new_bbox[k] = rearrange(v, "B T NC ... -> (B NC) T ...")  # BxNC, T, len, 3, 7
        model_args["bbox"] = move_to(new_bbox, device=device, dtype=dtype)

    # == add null condition ==
    if pipe['cfg']['scheduler']['type'] == "dpm-solver" and pipe['cfg']['scheduler']['cfg_scale'] == 1.0 or (
        pipe['cfg']['scheduler']['type'] in ["rflow-slice",]
    ):
        _model_args = copy.deepcopy(model_args)
    else:
        _model_args = add_null_condition(
            copy.deepcopy(model_args),
            pipe['model'].camera_embedder.uncond_cam.to(device),
            pipe['model'].frame_embedder.uncond_cam.to(device),
            prepend=(pipe['cfg']['scheduler']['type'] == "dpm-solver"),
        )

    # == inference ==
    masks = torch.full((1, z.shape[2]), True, dtype=torch.bool, device=device)

    if T == 1:
        masks[0, :1] = True
        z = z
    elif ref_samples is not None and batch_index == 1:
        masks[0, :1] = False
        x_encoded = rearrange(pipe['vae'].encode(ref_samples[:, :, :1]), "(B NC) C T ... -> B (C NC) T ...", NC=NC)
        z[0, :, :1] = x_encoded
    else:
        masks[0, :3] = False
        x_encoded = rearrange(pipe['vae'].encode(ref_samples[:, :, -9:]), "(B NC) C T ... -> B (C NC) T ...", NC=NC)
        z[0, :, :3] = x_encoded
    if batch_index == 0 and T == 1:
        samples = pipe_single['scheduler'].sample(
            pipe_single['model'],
            pipe_single['text_encoder'],
            z=z,
            prompts=batch_prompts,
            neg_prompts=neg_prompts,
            device=device,
            additional_args=_model_args,
            progress=verbose >= 1,
            mask=masks,
        )
    else:
        samples = pipe['scheduler'].sample(
            pipe['model'],
            pipe['text_encoder'],
            z=z,
            prompts=batch_prompts,
            neg_prompts=neg_prompts,
            device=device,
            additional_args=_model_args,
            progress=verbose >= 1,
            mask=masks,
        )
    
    torch.cuda.empty_cache()

    samples = rearrange(samples, "B (C NC) T ... -> (B NC) C T ...", NC=NC)

    samples = pipe['vae'].decode(samples.to(dtype), num_frames=_model_args["num_frames"])
    ref_samples = copy.deepcopy(samples)


    samples = rearrange(samples, "(B NC) C T ... -> B NC C T ...", NC=NC)
    
    return samples, ref_samples

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


@app.post("/dreamer-api/")
async def dreamer_api(request: Request, background_tasks: BackgroundTasks):
    global pipe, batch_index, ref_samples, timestamp, agent_command, device, weight_dtype

    param = await request.json()
    if isinstance(param, str):
        param = json.loads(param)
    
    if batch_index == 0:
        param_to_use = param[0]
        agent_command = param_to_use.get('agent_command', 2)
        val_dataset = load_data(pipe_single['cfg'], pipe_single, param_to_use)
        # key_param = param[-1]
        # agent_command = key_param.get('agent_command', 2)
        # val_dataset = load_data(pipe['cfg'], pipe, param)
    else:
        key_param = param[-1]  
        agent_command = key_param.get('agent_command', 2)
        val_dataset = load_data(pipe['cfg'], pipe, param)
    
    only_keyframe = True
    
    if not pipe:
        return {'status': 'error', 'message': 'construct pipe failed'}
    
    bbox_mode = pipe['cfg'].get('bbox_mode', 'all-xyz')  
    with_temporal_dim = pipe['cfg'].get('with_temporal_dim', True) 
    
    if 'img_collate_param_train' in pipe['cfg']:
        collate_params = pipe['cfg']['img_collate_param_train']
        bbox_mode = collate_params.get('bbox_mode', bbox_mode)
        with_temporal_dim = collate_params.get('with_temporal_dim', with_temporal_dim)
    
    template = pipe['cfg'].get('template', 'A driving scene image at {location}. {description}.')
    
    if batch_index == 0:
        param_to_use = param[0]
        agent_command = param_to_use.get('agent_command', 2)
        val_dataset = load_data(pipe_single['cfg'], pipe_single, param_to_use)
        val_input = collate_fn(
            [val_dataset.process_single(val_dataset.param)],
            template=template, 
            bbox_mode=bbox_mode,
            with_temporal_dim=with_temporal_dim,
            is_train=False,  
            drop_ori_imgs=False
        )
        images, ref_samples = generate_images(val_input, weight_dtype, ref_samples)
        # images = images[:, :, :, :-1]

        key_param = param[-1]  
        agent_command = key_param.get('agent_command', 2)
        val_dataset = load_data(pipe['cfg'], pipe, param)
        raw_data = val_dataset  
        val_input = collate_fn_t(
            raw_data,
            template=template, 
            bbox_mode=bbox_mode,
            is_train=False,
            is_clip=False,
            drop_ori_imgs=False
        )
    else:
        raw_data = val_dataset  
        val_input = collate_fn_t(
            raw_data,
            template=template, 
            bbox_mode=bbox_mode,
            is_train=False,
            is_clip=False,
            drop_ori_imgs=False
        )
    batch_index += 1  
    
    images, ref_samples = generate_images(val_input, weight_dtype, ref_samples)
    images = images[:, :, :, :1]
    
    

    if only_keyframe:
        views = []
        for v in range(images.shape[1]):
            img = images[0, v, :, -1].float().permute(1, 2, 0).cpu().numpy()
            img = ((img * 0.5 + 0.5) * 255).astype(np.uint8)
            views.append(Image.fromarray(img))

        total_width = views[0].width  # 800
        total_height = sum(img.height for img in views)  # 6 * 448 = 2688
        concat_img = Image.new('RGB', (total_width, total_height))
        
        y_offset = 0
        for img in views:
            concat_img.paste(img, (0, y_offset))
            y_offset += img.height
    else:
        all_images = []
        for t in range(images.shape[3]):
            views = []
            for v in range(images.shape[1]):
                img = images[0, v, :, t].float().permute(1, 2, 0).cpu().numpy()
                img = ((img * 0.5 + 0.5) * 255).astype(np.uint8)
                views.append(Image.fromarray(img))
            
            all_images.append(concat_6_views(tuple(views), oneline=False))
        
        total_width = sum(img.width for img in all_images)
        total_height = all_images[0].height
        
        concat_img = Image.new('RGB', (total_width, total_height))
        
        x_offset = 0
        for img in all_images:
            concat_img.paste(img, (x_offset, 0))
            x_offset += img.width
    

    img_byte_array = io.BytesIO()
    concat_img.save(img_byte_array, format='PNG')
    img_byte_array.seek(0)

    ## To Agent
    if not args.debug:
        image_queue.put({'param': param, 'img': concat_img})
    else:
        sample_token = val_input['meta_data']['metas'][-1].data['token']
        ego_pos, accel, rotation_rate, vel, agent_command = _get_can_bus_info(sample_token)
        param = {
            'metas': {'ego_pos': ego_pos.tolist(), 'accel': accel.tolist(), 'rotation_rate': rotation_rate.tolist(), 'vel': vel.tolist()}
        }
        image_queue.put({'param': param, 'img': concat_img})
    
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
        gen_imgs = [gen_imgs]
        combined_image = Image.new("RGB", (gen_imgs[0].width, sum(img.height for img in gen_imgs)))  
        y_offset = 0  
        for img in gen_imgs:  
            combined_image.paste(img, (0, y_offset))  
            y_offset += img.height
        combined_image = combined_image.resize((400, 1344), Image.LANCZOS)
        
        img_byte_array = io.BytesIO()
        combined_image.save(img_byte_array, format="PNG")   
        img_byte_array = img_byte_array.getvalue()
        image_base64 = base64.b64encode(img_byte_array).decode('utf-8')
        ego_pose = cur_data['param'][0]['metas']['ego_pos']
        accel = cur_data['param'][0]['metas']['accel']
        rotation_rate = cur_data['param'][0]['metas']['rotation_rate']
        vel = cur_data['param'][0]['metas']['vel']
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

def parse_configs(config_path, overrides=None):
    if overrides is None:
        overrides = []
    
    abs_config_path = os.path.abspath(config_path)
    
    if not os.path.exists(abs_config_path):
        print(f"error: config file {abs_config_path} not found")
        return None

    try:
        spec = importlib.util.spec_from_file_location("config_module", abs_config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        

        config_dict = {k: v for k, v in config_module.__dict__.items() 
                      if not k.startswith('__') and not callable(v)}
        

        for override in overrides:
            if '=' in override:
                key, value = override.split('=', 1)

                try:
                    value = eval(value)  
                except:
                    pass  
                

                keys = key.split('.')
                d = config_dict
                for k in keys[:-1]:
                    if k not in d:
                        d[k] = {}
                    d = d[k]
                d[keys[-1]] = value
        

        print("load config success")
        return config_dict
    except Exception as e:
        print(f"load config failed: {e}")
        return None

@app.on_event("startup")
async def startup_event():
    global pipe, device, weight_dtype
    if pipe['cfg'].get("cpu_offload", False):
        pipe['text_encoder'].t5.model.to("cpu")
        pipe['model'].to("cpu")
        pipe['vae'].to("cpu")
        pipe['text_encoder'].t5.model, pipe['model'], pipe['vae'], pipe['last_hook'] = enable_offload(
            pipe['text_encoder'].t5.model, pipe['model'], pipe['vae'], device)

@app.get("/dreamer-clean/")
async def clean_history():
    global batch_index, ref_samples, ref_can_bus, overlap_images, prev_pos, prev_angle, agent_command, timestamp
    print('[world dreamer] cleaned out past frames')
    batch_index = 0
    prev_pos = None
    prev_angle = None
    ref_samples = None
    ref_can_bus = []
    overlap_images = []
    timestamp = 0.
    agent_command = 2
    while image_queue.qsize() > 0:  
        image_queue.get() 

if __name__ == '__main__':
    # start debug server, wait for connection
    torch.set_grad_enabled(False)
    # try:
    #     debugpy.listen(("0.0.0.0", 5666))
    #     print("waiting for debugger connection...")
    #     debugpy.wait_for_client()
    #     print("Debugger attached!")
    # except Exception as e:
    #     print(f"Failed to start debugpy: {e}")
    
    pipe_single = load_pipe(args.config_single, args.single_model_path)
    pipe = load_pipe(args.config, args.model_path)
    
    # start server
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

