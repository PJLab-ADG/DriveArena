import cv2
import sklearn
import os
import os.path as osp
import io
import numpy as np
import copy
import sys
import warnings
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from fastapi import FastAPI, Request, status
from fastapi.responses import Response, JSONResponse
import uvicorn
from fastapi import FastAPI, BackgroundTasks
import base64
from PIL import Image

import mmcv
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.uniad.apis.test import custom_multi_gpu_test, custom_single_gpu_test
from mmdet.datasets import replace_ImageToTensor

from CAMixerSR.codes.basicsr.archs.CAMixerSR_arch import CAMixerSR
from pyquaternion import Quaternion
from queue import Queue
from tools.analysis_tools.visualize.mini_run import Visualizer


app = FastAPI()

class SequenceDataset(Dataset):
    def __init__(self, data) -> None:
        self.dataset = data
        self.dataset = [self.dataset]
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        return data

    def __len__(self):
        return len(self.dataset)
    

def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--config', default='projects/configs/stage2_e2e/base_e2e.py', help='test config file path')
    parser.add_argument('--checkpoint', default='ckpts/uniad_base_e2e.pth', help='checkpoint file')
    parser.add_argument('--out', default='output/md_results_single_data.pkl', help='output result file in pickle format')
    parser.add_argument('--data_path', default='data_temp/data_template.pth', help='data template path')
    parser.add_argument('--out_dir', default='output/demo', help='output dir')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11001)
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    
    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def split_image(img, target_height):
    """
    将一个大图片按照指定的高度切分成多个小图片

    参数:
    - img: PIL.Image.Image对象，即原始大图片
    - target_height: int，目标小图片的高度

    返回值:
    - 一个列表，包含切分后的所有PIL.Image.Image对象
    """
    img_width, img_height = img.size  # 获取原始图片的尺寸

    # 计算可以切分成多少个指定高度的小图片
    num_images = img_height // target_height

    # 初始化小图片列表
    gen_imgs = []

    for i in range(num_images):
        # 计算每个小图片的上下边界
        top = i * target_height
        bottom = (i + 1) * target_height

        # 切分图片
        img_cropped = img.crop((0, top, img_width, bottom))
        
        # 将切分后的小图片添加到列表
        gen_imgs.append(img_cropped)

    return gen_imgs


def prepare_SR_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CAMixerSR(scale=4)
    model.load_state_dict(torch.load('ckpts/CAMixerSRx4_DF.pth')['params_ema'], strict=True)
    model.eval()
    model = model.to(device)
    return model


@app.post("/driver-api/")
async def process(request: Request, background_tasks: BackgroundTasks):
    param = await request.json()
    base64_image = param.get('img_byte_array', None)
    timestamp = param.get('timestamp', None)
    command = param.get('command', 2) # 0: Right 1:Left 2:Forward
    accel = torch.tensor(param.get('accel', [0, 0, 9.8]))
    rotation_rate = torch.tensor(param.get('rotation_rate', [0, 0, 0]))
    vel = torch.tensor(param.get('vel', [0, 0, 0]))
    # breakpoint()
    ego_pose = torch.tensor(param.get('ego_pose', None))
    base64_image_bytes = base64.b64decode(base64_image)
    img = Image.open(io.BytesIO(base64_image_bytes)).convert('RGB')
    gen_imgs = split_image(img, 224)
    # 图像替换
    uniad_img_vis_list = []
    uniad_img_list = []
    for cami in range(6):
        img_i = gen_imgs[cami]
        img_i = np.array(img_i) / 255    # RGB
        img_i = torch.from_numpy(np.transpose(img_i, (2, 0, 1))).float()
        img_i = img_i.unsqueeze(0).cuda()
        # upsample imgs
        with torch.no_grad():
            out_img_i = SR_model(img_i)
            out_img_i = out_img_i.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            out_img_i = np.transpose(out_img_i, (1, 2, 0))    # RGB
            out_img_i = (out_img_i * 255.0).round().astype(np.uint8)
        uniad_img_vis_list.append(out_img_i)
        # normalize
        norm_img = mmcv.imnormalize(out_img_i, np.array([103.530, 116.280, 123.675]), np.array([1.0, 1.0, 1.0]), False)
        # pad img to UniAD shape, which is [928, 1600, 3]
        padded_img = mmcv.impad(norm_img, shape=img_pad_size, pad_val=0)
        uniad_img_list.append(padded_img) # todo output
    
    # process uniad data
    uniad_img = np.stack(uniad_img_list)    # [6, 928, 1600, 3]
    # diffusion order is ['front left', 'front', 'front_right', 'back right', 'back', 'back left']
    # uniad order is ['front', 'front right', 'front left', 'back', 'back left', 'back right']
    uniad_img = uniad_img[[1,2,0,4,5,3], ...]
    uniad_img_vis = np.stack(uniad_img_vis_list)
    uniad_img = torch.from_numpy(uniad_img).cuda()
    uniad_img = uniad_img.permute(0, 3, 1, 2)[None,...]
    curr_uniad_data = copy.deepcopy(uniad_data_template)
    curr_uniad_data['img'][0] = uniad_img
    # ego_pose
    trans = ego_pose[:3 ,3]
    rot = ego_pose[:3, :3]
    rot = Quaternion(matrix=rot.numpy().astype(np.float64))
    patch_angle = quaternion_yaw(rot) / np.pi * 180
    # patch_angle = torch.arctan(rot[1,0] / rot[0,0]) / np.pi * 180
    if patch_angle < 0:
        patch_angle += 360
    curr_uniad_data['img_metas'][0][0]['can_bus'][:3] = trans
    curr_uniad_data['img_metas'][0][0]['can_bus'][3:7] = rot
    curr_uniad_data['img_metas'][0][0]['can_bus'][7:10] = accel
    curr_uniad_data['img_metas'][0][0]['can_bus'][10:13] = rotation_rate
    curr_uniad_data['img_metas'][0][0]['can_bus'][13:16] = vel
    curr_uniad_data['img_metas'][0][0]['can_bus'][-2] = patch_angle / 180 * np.pi
    curr_uniad_data['img_metas'][0][0]['can_bus'][-1] = patch_angle
    print('can_bus', curr_uniad_data['img_metas'][0][0]['can_bus'])
    # TODO: can_bus need to be modified!!
    curr_uniad_data['timestamp'][0] = torch.Tensor([timestamp]).to(torch.float64).cuda()
    curr_uniad_data["command"][0] = torch.Tensor([command]).to(torch.int64).cuda()
    dataset = SequenceDataset(curr_uniad_data)
    # inference
    outputs = custom_single_gpu_test(model, dataset, args.tmpdir, args.gpu_collect, eval_occ=False, eval_planning=False)
    outputs["bev_pred_img"] = get_bev_img(outputs["bbox_results"][0])
    output_queue.put(outputs)
    # breakpoint()

    background_tasks.add_task(save_visualize_img, outputs["bbox_results"][0], timestamp, uniad_img_vis)
    
    return 1
    
    # pass

# TODO 1
def get_bev_img(outputs):
    render_cfg = dict(
        with_occ_map=False,
        with_map=True,
        with_planning=True,
        with_pred_box=True,
        with_pred_traj=True,
        show_command=True,
        show_sdc_car=True,
        show_legend=True,
        show_sdc_traj=False
    )
    viser = Visualizer(**render_cfg)
    prediction_dict = viser._parse_predictions(outputs)
    viser.bev_render.reset_canvas(dx=1, dy=1)
    viser.bev_render.set_plot_cfg()

    viser.bev_render.render_pred_box_data(
            prediction_dict['predicted_agent_list'])
    viser.bev_render.render_pred_traj(
            prediction_dict['predicted_agent_list'])
    viser.bev_render.render_pred_map_data(
            prediction_dict['predicted_map_seg'])
    viser.bev_render.render_pred_box_data(
        [prediction_dict['predicted_planning']])
    viser.bev_render.render_planning_data(
        prediction_dict['predicted_planning'], show_command=viser.show_command)
    viser.bev_render.render_sdc_car()
    viser.bev_render.render_legend()
    
    # convert img to byteIO
    img_buffer = io.BytesIO()
    viser.bev_render.save_fig(img_buffer)
    img_buffer.seek(0)
    
    # convert byteIO to base64
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    
    return img_base64


# TODO 2
async def save_visualize_img(outputs, timestamp, visual_imgs=None):
    #visiualize
    render_cfg = dict(
        with_occ_map=False,
        with_map=True,
        with_planning=True,
        with_pred_box=True,
        with_pred_traj=True,
        show_command=True,
        show_sdc_car=True,
        show_legend=True,
        show_sdc_traj=False
    )
    CAM_NAMES = [
        'CAM_FRONT_LEFT',
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT',
        'CAM_BACK',
        'CAM_BACK_RIGHT',
    ]

    viser = Visualizer(**render_cfg)
    prediction_dict = viser._parse_predictions(outputs)
    metas = mmcv.load('data_temp/nus_vis_pose_info.pkl')
    # breakpoint()

    image_dict = {} # TODO: surrounding images
    if visual_imgs is not None:
        visual_imgs = np.array(visual_imgs, dtype=np.float32)
        visual_imgs /= 255.
        temp=np.zeros((6,4,1600,3))
        visual_imgs = np.concatenate([visual_imgs,temp],1)
        image_dict['CAM_FRONT_LEFT'] = visual_imgs[0]
        image_dict['CAM_FRONT'] = visual_imgs[1]
        image_dict['CAM_FRONT_RIGHT'] = visual_imgs[2]
        image_dict['CAM_BACK_RIGHT'] = visual_imgs[3]
        image_dict['CAM_BACK'] = visual_imgs[4]
        image_dict['CAM_BACK_LEFT'] = visual_imgs[5]
    else:
        for cam in CAM_NAMES:
            image_dict[cam] = np.zeros((900, 1600, 3)) # TODO: replace the images
    sample_info = {}
    sample_info['images'] = {}
    sample_info['metas'] = metas
    sample_info['images'] = image_dict
    '''
    sample_info:
        - 'images': 
            'CAM_FRONT': np.array
        - 'metas': 
            'lidar_cs_record'
            'CAM_FRONT':
                'cs_record'
                'imsize'
                'cam_intrinsic'
    }
    '''

    out_folder = args.out_dir
    os.makedirs(out_folder, exist_ok=True)
    project_to_cam = True
    viser.visualize_bev(prediction_dict, os.path.join(out_folder, str(int(timestamp*2)).zfill(3)))

    if project_to_cam:
        viser.visualize_cam(prediction_dict, sample_info, os.path.join(out_folder, str(int(timestamp*2)).zfill(3)))
        viser.combine(os.path.join(out_folder, str(int(timestamp*2)).zfill(3))) 


# 返回一帧
@app.get("/driver-get/")
async def get_output():
    # 删除直到只有一帧
   if output_queue.qsize() >= 1:
       output_data = output_queue.get()
       output_data["bbox_results"][0]["planning_traj"] = output_data["bbox_results"][0]["planning_traj"].cpu().numpy().tolist()
    #    breakpoint()
       output_data["bbox_results"][0].pop('pts_bbox')
       return output_data
   else:
       return Response(status_code=status.HTTP_204_NO_CONTENT)

   
# 清空队列
@app.get("/driver-clean/")
async def clean_history():
    # 删除直到只有一帧
    print('[uniad] cleaned out past frames')
    while output_queue.qsize() > 0:  
        output_queue.get()  
    model.test_track_instances = None


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    img_pad_size = [928, 1600]
    SR_model = prepare_SR_model()
    uniad_data_template = torch.load(args.data_path)
    uniad_data_template["command"][0][0] = 2 # 0: Right 1:Left 2:Forward 
    output_queue = Queue()

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']


    model = model.cuda()
    # 加载完模型后，等着访问即可调用
    
    # 开启api
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
