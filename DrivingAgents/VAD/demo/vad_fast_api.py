# import cv2
import sklearn
import os
import math
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
from typing import List, Dict
from fastapi import FastAPI, BackgroundTasks
import base64
from PIL import Image
from scipy.spatial.transform import Rotation as R
import mmcv
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
import sys
import os
from mmdet3d.apis import single_gpu_test
from nuscenes.eval.common.data_classes import EvalBoxes,EvalBox

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.VAD.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor

from CAMixerSR.codes.basicsr.archs.CAMixerSR_arch import CAMixerSR
from pyquaternion import Quaternion
from queue import Queue
from tools.analysis_tools.visualize.mini_run import Visualizer
from matplotlib.collections import LineCollection

from projects.mmdet3d_plugin.core.bbox.structures.nuscenes_box import CustomNuscenesBox, CustomDetectionBox, color_map
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
app = FastAPI()
ego_width, ego_length = 1.85, 4.084

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
    # parser.add_argument('--config', default='projects/configs/VAD/VADv2_config_voca4096.py', help='test config file path')
    parser.add_argument('--config', default='projects/configs/VAD/VAD_base_stage_2.py', help='test config file path')
    parser.add_argument('--checkpoint', default='ckpts/VAD_base.pth', help='checkpoint file')
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


def quart_to_rpy(qua):
    x, y, z, w = qua
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - x * z))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return roll, pitch, yaw

def boxes_to_sensor(boxes: List[EvalBox], l2g_r_mat, l2g_t):
    """
    Map boxes from global coordinates to the vehicle's sensor coordinate system.
    :param boxes: The boxes in global coordinates.
    :param pose_record: The pose record of the vehicle at the current timestamp.
    :param cs_record: The calibrated sensor record of the sensor.
    :return: The transformed boxes.
    """
    boxes_out = []
    for box in boxes:
        # Create Box instance.
        # box = CustomNuscenesBox(
        #     box.translation, box.size, Quaternion(box.rotation), box.fut_trajs, name=box.detection_name
        # )
        r = R.from_matrix(l2g_r_mat)
        q_xyzw = r.as_quat()[0]
        q_wxyz = Quaternion([q_xyzw[3],q_xyzw[0],q_xyzw[1],q_xyzw[2]])
        box.translate(-np.array(l2g_t)[0])
        box.rotate(q_wxyz.inverse)

        boxes_out.append(box)

    return boxes_out

@app.post("/driver-api/")
async def process(request: Request, background_tasks: BackgroundTasks):
    param = await request.json()
    base64_image = param.get('img_byte_array', None)
    timestamp = param.get('timestamp', None)
    command = param.get('command', 2) # 0: Right 1:Left 2:Forward
    if command == 0:
        command = torch.Tensor([1, 0, 0])  # Turn Right
    elif command == 1:
        command = torch.Tensor([0, 1, 0])  # Turn Left
    elif command == 2:
        command = torch.Tensor([0, 0, 1])  # Go Straight
        
    accel = torch.tensor(param.get('accel', [0, 0, 9.8]))
    rotation_rate = torch.tensor(param.get('rotation_rate', [0, 0, 0]))
    vel = torch.tensor(param.get('vel', [0, 0, 0]))
    # gt_bboxes_3d = torch.tensor(param.get('gt_bboxes_3d', [0, 0, 0]))
    gt_labels_3d = torch.tensor(param.get('gt_labels_3d', [0, 0, 0]))
    # breakpoint()
    ego_pose = torch.tensor(param.get('ego_pose', None))
    base64_image_bytes = base64.b64decode(base64_image)
    img = Image.open(io.BytesIO(base64_image_bytes)).convert('RGB')
    img.save('output_image.png', format='PNG')
    gen_imgs = split_image(img, 224)
    # 图像替换
    vad_img_vis_list = []
    vad_img_list = []
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
        vad_img_vis_list.append(out_img_i)
        # normalize
        norm_img = mmcv.imnormalize(out_img_i, np.array([103.530, 116.280, 123.675]), np.array([1.0, 1.0, 1.0]), False)
        # pad img to VAD shape, which is [896, 1600, 3]
        padded_img = mmcv.impad(norm_img, shape=img_pad_size, pad_val=0)
        vad_img_list.append(padded_img) # todo output
    
    # process uniad data
    vad_img = np.stack(vad_img_list)    # [6, 896, 1600, 3]
    # diffusion order is ['front left', 'front', 'front_right', 'back right', 'back', 'back left']
    # uniad order is ['front', 'front right', 'front left', 'back', 'back left', 'back right']
    vad_img = vad_img[[1,2,0,4,5,3], ...]
    vad_img_vis = np.stack(vad_img_vis_list)
    vad_img = torch.from_numpy(vad_img).cuda()
    vad_img = vad_img.permute(0, 3, 1, 2)[None,...]
    curr_uniad_data = copy.deepcopy(vad_data_template)
    curr_uniad_data['img'][0] = vad_img
    # ego_pose
    trans = ego_pose[:3 ,3]
    rot = ego_pose[:3, :3]
    rot = Quaternion(matrix=rot.numpy().astype(np.float64))
    _,_,yaw = quart_to_rpy(rot)
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
    curr_uniad_data['img_metas'][0][0]['img_nrom_cfg'] = {'mean': np.array([103.53 , 116.28 , 123.675]), 'std': np.array([1., 1., 1.]), 'to_rgb': False}

    # TODO: can_bus need to be modified!!
    curr_uniad_data['timestamp'][0] = torch.Tensor([timestamp]).to(torch.float64).cuda()
    # curr_uniad_data["ego_fut_cmd"][0] = torch.Tensor(command).to(torch.int64).cuda()

    if 'gt_bboxes_3d' not in curr_uniad_data:
        curr_uniad_data['gt_bboxes_3d'] = []
    if 'gt_labels_3d' not in curr_uniad_data:
        curr_uniad_data['gt_labels_3d'] = []
    if "ego_fut_cmd" not in curr_uniad_data:
        curr_uniad_data['ego_fut_cmd'] = [[]]
    if 'ego_his_trajs' not in curr_uniad_data:
        curr_uniad_data['ego_his_trajs'] = torch.Tensor([[-0.00076206, -0.02482096],
                                                        [-0.00076206, -0.02482096]])
    if 'ego_fut_trajs' not in curr_uniad_data:
        curr_uniad_data['ego_fut_trajs'] = torch.Tensor([[-0.0007222844, -0.000527013]])
    
    if 'gt_attr_labels' not in curr_uniad_data:
        curr_uniad_data['gt_attr_labels'] = [[]]

    # curr_uniad_data['gt_bboxes_3d'] = gt_bboxes_3d.to(torch.float64).cuda()
    curr_uniad_data['gt_labels_3d'] = gt_labels_3d.to(torch.long).cuda()
    curr_uniad_data["ego_fut_cmd"][0] = torch.Tensor(command).to(torch.int64).cuda()

    delta_x = curr_uniad_data['ego_his_trajs'][-1, 0] + curr_uniad_data['ego_fut_trajs'][0, 0]
    delta_y = curr_uniad_data['ego_his_trajs'][-1, 1] + curr_uniad_data['ego_fut_trajs'][0, 1]
    v0 = np.sqrt(delta_x**2 + delta_y**2)
    Kappa = 0
    
    ### ego lcf feat (vx, vy, ax, ay, w, length, width, vel, steer), w: yaw角速度
    ego_lcf_feat = np.zeros(9)
    ego_lcf_feat[:2] = np.array(vel[:2])
    ego_lcf_feat[2:4] = accel[:2]
    ego_lcf_feat[4] = rotation_rate[-1] #can_bus[12]
    ego_lcf_feat[5:7] = np.array([ego_length, ego_width])
    ego_lcf_feat[7] = v0
    ego_lcf_feat[8] = Kappa

    if 'ego_lcf_feat' not in curr_uniad_data:
        curr_uniad_data['ego_lcf_feat'] = torch.Tensor(ego_lcf_feat)
    print('Command: ', curr_uniad_data["ego_fut_cmd"][0])
    dataset = SequenceDataset(curr_uniad_data)
    # inference
    outputs = single_gpu_test(model, dataset, args.show, args.show_dir)
    outputs[0]["bev_pred_img"] = visualize_sample(pts_bbox=outputs[0]['pts_bbox'], command=command,savepath='.', l2g_r_mat=curr_uniad_data['l2g_r_mat'].cpu(), l2g_t=curr_uniad_data['l2g_t'].cpu()) #, gt_bbox=curr_uniad_data['gt_bboxes_3d'].cpu()
    output_queue.put(outputs)

    return 1
    

def visualize_sample(pts_bbox,
                    #  gt_bbox,
                     command,
                     l2g_r_mat,
                     l2g_t,
                     nsweeps: int = 1,
                     conf_th: float = 0.45,
                     pc_range: list = [-30.0, -30.0, -4.0, 30.0, 30.0, 4.0],
                     verbose: bool = True,
                     savepath: str = None,
                     traj_use_perstep_offset: bool = True,
                     data_root='data/nuscenes/',
                     map_pc_range: list = [-15.0, -30.0, -4.0, 15.0, 30.0, 4.0],
                     padding_value=-10000,
                     map_classes=['divider', 'ped_crossing', 'boundary'],
                     map_fixed_ptsnum_per_line=20,
                     gt_format=['fixed_num_pts'],
                     colors_plt = ['cornflowerblue', 'royalblue', 'slategrey'],
                     pred_data = None) -> None:
    """
    Visualizes a sample from BEV with annotations and detection results.
    :param nusc: NuScenes object.
    :param sample_token: The nuScenes sample token.
    :param gt_boxes: Ground truth boxes grouped by sample.
    :param pred_boxes: Prediction grouped by sample.
    :param nsweeps: Number of sweeps used for lidar visualization.
    :param conf_th: The confidence threshold used to filter negatives.
    :param eval_range: Range in meters beyond which boxes are ignored.
    :param verbose: Whether to print to stdout.
    :param savepath: If given, saves the the rendering here instead of displaying.
    """
    gt_boxes = EvalBoxes()
    gt_boxes_list = []

    pred_boxes = EvalBoxes()
    pred_boxes_list = []
    boxes_est = pts_bbox['boxes_3d']
    scores_est = pts_bbox['scores_3d']
    labels_est = pts_bbox['labels_3d']
    trajs_est = pts_bbox['trajs_3d']
    map_pts_3d = pts_bbox['map_pts_3d']
    map_labels_3d = pts_bbox['map_labels_3d']
    map_scores_3d = pts_bbox['map_scores_3d']

    # need to transform the boxes to CustomNuscenesBox to use the render functions
    for box_est, score_est, label_est, traj_est in zip(boxes_est, scores_est,labels_est,trajs_est):
        center = box_est[:3]
        # print('center: ', center)
        size = box_est[3:6]
        yaw = - box_est[6] - 1.5708
        # print('size: ', size)
        # print('yaw: ', box_est[6])
        orientation = Quaternion([np.cos(yaw/2),0,0,np.sin(yaw/2)])
        box = CustomNuscenesBox(center=center, size=size, orientation=orientation, fut_trajs=traj_est, label=label_est, score=score_est)
        pred_boxes_list.append(box)

    # for bbox in gt_bbox:
    #     center = bbox[:3]
    #     size = bbox[3:6]
    #     yaw = - bbox[6] - 1.5708
    #     orientation = Quaternion([np.cos(yaw/2),0,0,np.sin(yaw/2)])
    #     gt_box = CustomNuscenesBox(center=center, size=size, orientation=orientation, fut_trajs=None, label=0, score=1.0)
    #     gt_boxes_list.append(gt_box)

    pred_boxes.add_boxes(sample_token="", boxes=pred_boxes_list)
    # gt_boxes.add_boxes(sample_token="", boxes=gt_boxes_list)

    boxes_est_global = pred_boxes[""]
    # boxes_gt_global = gt_boxes[""]
    # boxes_est = boxes_to_sensor(boxes_est_global, l2g_r_mat, l2g_t)
    # Init axes.
    fig, axes = plt.subplots(1, 1, figsize=(4, 4))
    plt.xlim(xmin=-30, xmax=30)
    plt.ylim(ymin=-30, ymax=30)

    # Show Pred Map
    for pred_pts_3d, pred_label_3d, pred_confidence in zip(map_pts_3d, map_labels_3d, map_scores_3d):
        if pred_confidence < 0.6:
            continue
        pts_x = np.array([pt[0] for pt in pred_pts_3d])
        pts_y = np.array([pt[1] for pt in pred_pts_3d])
        axes.plot(pts_x, pts_y, color=colors_plt[pred_label_3d],linewidth=1,alpha=0.8,zorder=-1)
        axes.scatter(pts_x, pts_y, color=colors_plt[pred_label_3d],s=1,alpha=0.8,zorder=-1)  

    # for i, box in enumerate(boxes_gt_global):
    #     if abs(box.center[0]) > 15 or abs(box.center[1]) > 30:
    #         continue
    #     print('gt box: ', box.center)
    #     print('gt size: ', box.wlh)
    #     print('gt rotation: ', box.orientation.rotation_matrix)
    #     box.render(axes, view=np.eye(4), colors=('blue', 'blue', 'blue'), linewidth=1, box_idx=None)
        

    # Show Pred boxes.
    for i, box in enumerate(boxes_est_global):
        # if box.name in ignore_list:
        #     continue
        # Show only predictions with a high score.
        assert not np.isnan(box.score), 'Error: Box score cannot be NaN!'
        
        if box.score < conf_th or abs(box.center[0]) > 15 or abs(box.center[1]) > 30:
            continue
        print('box: ', box.center)
        print('size: ', box.wlh)
        print('rotation: ', box.orientation.rotation_matrix)

        
        # if box.score < 0.5:
        #     box.render(axes, view=np.eye(4), colors=('blue', 'blue', 'blue'), linewidth=1, box_idx=None)
        # else:
        box.render(axes, view=np.eye(4), colors=('tomato', 'tomato', 'tomato'), linewidth=1, box_idx=None)
        # if box.name in ['pedestrian']:
        #     continue
        if traj_use_perstep_offset:
            mode_idx = [0, 1, 2, 3, 4, 5]
            box.render_fut_trajs_grad_color(axes, linewidth=1, mode_idx=mode_idx, fut_ts=6, cmap='autumn')
        else:
            box.render_fut_trajs_coords(axes, color='tomato', linewidth=1)

    # Plot ego vehicle
    axes.plot([-0.9, -0.9], [-2, 2], color='mediumseagreen', linewidth=1, alpha=0.8)
    axes.plot([-0.9, 0.9], [2, 2], color='mediumseagreen', linewidth=1, alpha=0.8)
    axes.plot([0.9, 0.9], [2, -2], color='mediumseagreen', linewidth=1, alpha=0.8)
    axes.plot([0.9, -0.9], [-2, -2], color='mediumseagreen', linewidth=1, alpha=0.8)
    axes.plot([0.0, 0.0], [0.0, 2], color='mediumseagreen', linewidth=1, alpha=0.8)

    # Show Planning.
    plan_cmd = pts_bbox['ego_fut_cmd']
    plan_cmd_idx = torch.nonzero(plan_cmd)[0, 0]
    plan_traj= pts_bbox['ego_fut_preds'][plan_cmd_idx]
    # breakpoint()
    plan_traj[abs(plan_traj) < 0.01] = 0.0
    plan_traj = plan_traj.cumsum(axis=0)
    cmd_list = ['Turn Right', 'Turn Left', 'Go Straight']
    plan_cmd_str = cmd_list[plan_cmd_idx]

    plan_traj = np.concatenate((np.zeros((2, plan_traj.shape[1])), plan_traj), axis=0)
    # Extract x and y coordinates of the trajectory points
    x_points = plan_traj[:, 0]
    y_points = plan_traj[:, 1]
    print('planj: ', plan_traj)
    # Plot the trajectory points
    axes.scatter(x_points, y_points, color='red', s=10, zorder=5)
    plan_traj = np.stack((plan_traj[:-1], plan_traj[1:]), axis=1)
    plan_vecs = None
    for i in range(plan_traj.shape[0]):
        plan_vec_i = plan_traj[i]
        x_linspace = np.linspace(plan_vec_i[0, 0], plan_vec_i[1, 0], 51)
        y_linspace = np.linspace(plan_vec_i[0, 1], plan_vec_i[1, 1], 51)
        xy = np.stack((x_linspace, y_linspace), axis=1)
        xy = np.stack((xy[:-1], xy[1:]), axis=1)
        if plan_vecs is None:
            plan_vecs = xy
        else:
            plan_vecs = np.concatenate((plan_vecs, xy), axis=0)

    cmap = 'winter'
    y = np.sin(np.linspace(1/2*np.pi, 3/2*np.pi, 301))
    colors = color_map(y[:-1], cmap)
    line_segments = LineCollection(plan_vecs, colors=colors, linewidths=1, linestyles='solid', cmap=cmap)
    axes.add_collection(line_segments)

    axes.axes.xaxis.set_ticks([])
    axes.axes.yaxis.set_ticks([])
    axes.axis('off')
    axes.text(
        0.05,                # X-coordinate (5% from the left)
        0.95,                # Y-coordinate (95% from the bottom)
        plan_cmd_str,        # The text string to display
        transform=axes.transAxes,
        fontsize=14,         # Adjust font size as needed
        color='black',       # Text color
        verticalalignment='top',
        #bbox=dict(facecolor='black', alpha=0.5, pad=5)  # Optional: Add a semi-transparent background box
    )
    fig.set_tight_layout(True)
    fig.canvas.draw()
    # plt.savefig(savepath+'/bev_pred.png', bbox_inches='tight', dpi=200)
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=200)
    img_buffer.seek(0)
    file_bytes = np.asarray(bytearray(img_buffer.read()), dtype=np.uint8)
    
    # img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # # Apply the text overlay using cv2.putText
    # cv2.putText(img, plan_cmd_str, (20, 770), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA)

    # # Encode the modified image back to PNG format
    # is_success, buffer = cv2.imencode(".png", img)
    # if not is_success:
    #     # Handle the error
    #     raise ValueError("Could not encode image")

    # Convert the buffer to a base64 string
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    plt.close()

    return img_base64
    

# 返回一帧
@app.get("/driver-get/")
async def get_output():
    # 删除直到只有一帧
   if output_queue.qsize() >= 1:
        output_data = output_queue.get()
       
        cmd = output_data[0]["pts_bbox"]['ego_fut_cmd']
        if (cmd == torch.Tensor([1, 0, 0])).all():
            cmd_idx = 0  # Turn Right
        elif (cmd == torch.Tensor([0, 1, 0])).all():
            cmd_idx = 1  # Turn Left
        elif (cmd == torch.Tensor([0, 0, 1])).all():
            cmd_idx = 2  # Go Straight
        plan_traj = output_data[0]["pts_bbox"]["ego_fut_preds"][cmd_idx]#.cpu().numpy() #.tolist()
        plan_traj[abs(plan_traj) < 0.01] = 0.0
        plan_traj = plan_traj.cumsum(axis=0)
        # plan_traj = np.concatenate((np.zeros((2, plan_traj.shape[1])), plan_traj), axis=0)
        # plan_traj = np.stack((plan_traj[:-1], plan_traj[1:]), axis=1)

        # output_data[0]["planning_traj"] = plan_traj.cpu().numpy().tolist()
        output_data[0]["planning_traj"] = plan_traj.cpu().numpy().tolist() #output_data[0]["pts_bbox"]["ego_fut_preds"][cmd_idx].cpu().numpy().tolist()
        print(output_data[0]["planning_traj"])

        output_data[0].pop('pts_bbox')
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
    img_pad_size = [896, 1600]
    SR_model = prepare_SR_model()
    vad_data_template = torch.load(args.data_path)
    vad_data_template["command"][0][0] = 2 # 0: Right 1:Left 2:Forward 

    # 需要改
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
