import cv2
import torch
import argparse
import os
import sys
import glob
import numpy as np
import mmcv
import matplotlib
import matplotlib.pyplot as plt

from PIL import Image
sys.path.append('/cpfs01/user/yangxuemeng/code/UniAD/')
from tools.analysis_tools.visualize.utils import AgentPredictionData
from tools.analysis_tools.visualize.render.bev_render import BEVRender
from tools.analysis_tools.visualize.render.cam_render import CustomCameraRender


CAM_NAMES = [
    'CAM_FRONT_LEFT',
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
]
Nuplan_cam_names = [
    'CAM_L0',
    'CAM_F0',
    'CAM_R0',
    'CAM_R1',
    'CAM_B0',
    'CAM_L1',

]


class Visualizer:
    """
    BaseRender class
    """

    def __init__(
            self,
            with_occ_map=False,
            with_map=False,
            with_planning=False,
            with_pred_box=True,
            with_pred_traj=False,
            show_command=False,
            show_sdc_car=False,
            show_sdc_traj=False,
            show_legend=False):
        self.with_occ_map = with_occ_map
        self.with_map = with_map
        self.with_planning = with_planning
        self.show_command = show_command
        self.show_sdc_car = show_sdc_car
        self.show_sdc_traj = show_sdc_traj
        self.show_legend = show_legend
        self.with_pred_traj = with_pred_traj
        self.with_pred_box = with_pred_box
        self.veh_id_list = [0, 1, 2, 3, 4, 6, 7]

        self.bev_render = BEVRender()
        self.cam_render = CustomCameraRender()

    def _parse_predictions(self, outputs):
        if self.show_sdc_traj:
            outputs['pts_bbox']['boxes_3d'].tensor = torch.cat(
                [outputs['pts_bbox']['boxes_3d'].tensor, outputs['pts_bbox']['sdc_boxes_3d'].tensor], dim=0)
            outputs['pts_bbox']['scores_3d'] = torch.cat(
                [outputs['pts_bbox']['scores_3d'], outputs['pts_bbox']['sdc_scores_3d']], dim=0)
            outputs['pts_bbox']['labels_3d'] = torch.cat([outputs['pts_bbox']['labels_3d'], torch.zeros(
                (1,), device=outputs['pts_bbox']['labels_3d'].device)], dim=0)
        # detection
        bboxes = outputs['pts_bbox']['boxes_3d']
        scores = outputs['pts_bbox']['scores_3d']
        labels = outputs['pts_bbox']['labels_3d']
        
        track_scores = scores.cpu().detach().numpy()
        track_labels = labels.cpu().detach().numpy()
        track_boxes = bboxes.tensor.cpu().detach().numpy()
        
        track_centers = bboxes.gravity_center.cpu().detach().numpy()
        track_dims = bboxes.dims.cpu().detach().numpy()
        track_yaw = bboxes.yaw.cpu().detach().numpy()

        if 'track_ids' in outputs:
            track_ids = outputs['track_ids'].cpu().detach().numpy()
        else:
            track_ids = None

        # speed
        track_velocity = bboxes.tensor.cpu().detach().numpy()[:, -2:]
            
        # trajectories
        # trajs = outputs[f'traj'].numpy()
        # traj_scores = outputs[f'traj_scores'].numpy()

        trajs = outputs['pts_bbox'][f'trajs_3d'].numpy()
        traj_scores = outputs['pts_bbox'][f'scores_3d'].numpy()

        predicted_agent_list = []

        # occflow
        if self.with_occ_map:
            if 'topk_query_ins_segs' in outputs['occ']:
                occ_map = outputs['occ']['topk_query_ins_segs'][0].cpu(
                ).numpy()
            else:
                occ_map = np.zeros((1, 5, 200, 200))
        else:
            occ_map = None

        occ_idx = 0
        for i in range(track_scores.shape[0]):
            if track_scores[i] < 0.25:
                continue
            if occ_map is not None and track_labels[i] in self.veh_id_list:
                occ_map_cur = occ_map[occ_idx, :, ::-1]
                occ_idx += 1
            else:
                occ_map_cur = None
            if track_ids is not None:
                if i < len(track_ids):
                    track_id = track_ids[i]
                else:
                    track_id = 0
            else:
                track_id = None
            # if track_labels[i] not in [0, 1, 2, 3, 4, 6, 7]:
            #     continue
            predicted_agent_list.append(
                AgentPredictionData(
                    track_scores[i],
                    track_labels[i],
                    track_centers[i],
                    track_dims[i],
                    track_yaw[i],
                    track_velocity[i],
                    trajs[i],
                    traj_scores[i],
                    pred_track_id=track_id,
                    pred_occ_map=occ_map_cur,
                    past_pred_traj=None
                )
            )

        if self.with_map:
            map_thres = 0.7
            breakpoint()

            score_list = outputs['pts_bbox']['map_scores_3d'].cpu().numpy().transpose([
                1, 2, 0])
            predicted_map_seg = outputs['pts_bbox']['lane_score'].cpu().numpy().transpose([
                1, 2, 0])  # H, W, C
            predicted_map_seg[..., -1] = score_list[..., -1]
            predicted_map_seg = (predicted_map_seg > map_thres) * 1.0
            predicted_map_seg = predicted_map_seg[::-1, :, :]
        else:
            predicted_map_seg = None

        if self.with_planning:
            # detection
            bboxes =outputs['pts_bbox']['boxes_3d']
            scores = outputs['pts_bbox']['scores_3d']
            labels = 0

            track_scores = scores.cpu().detach().numpy()
            track_labels = labels
            track_boxes = bboxes.tensor.cpu().detach().numpy()

            track_centers = bboxes.gravity_center.cpu().detach().numpy()
            track_dims = bboxes.dims.cpu().detach().numpy()
            track_yaw = bboxes.yaw.cpu().detach().numpy()
            track_velocity = bboxes.tensor.cpu().detach().numpy()[:, -2:]

            if self.show_command:
                command = outputs['command'][0].cpu().detach().numpy()
            else:
                command = None
            planning_agent = AgentPredictionData(
                track_scores[0],
                track_labels,
                track_centers[0],
                track_dims[0],
                track_yaw[0],
                track_velocity[0],
                outputs['planning_traj'][0].cpu().detach().numpy(),
                1,
                pred_track_id=-1,
                pred_occ_map=None,
                past_pred_traj=None,
                is_sdc=True,
                command=command,
            )
            predicted_agent_list.append(planning_agent)
        else:
            planning_agent = None

        return dict(predicted_agent_list=predicted_agent_list,
                    predicted_map_seg=predicted_map_seg,
                    predicted_planning=planning_agent)

    def visualize_bev(self, prediction_dict, out_filename, t=None):
        self.bev_render.reset_canvas(dx=1, dy=1)
        self.bev_render.set_plot_cfg()

        if self.with_pred_box:
            self.bev_render.render_pred_box_data(
                prediction_dict['predicted_agent_list'])
        if self.with_pred_traj:
            self.bev_render.render_pred_traj(
                prediction_dict['predicted_agent_list'])
        if self.with_map:
            self.bev_render.render_pred_map_data(
                prediction_dict['predicted_map_seg'])
        if self.with_occ_map:
            self.bev_render.render_occ_map_data(
                prediction_dict['predicted_agent_list'])
        if self.with_planning:
            self.bev_render.render_pred_box_data(
                [prediction_dict['predicted_planning']])
            self.bev_render.render_planning_data(
                prediction_dict['predicted_planning'], show_command=self.show_command)
        if self.show_sdc_car:
            self.bev_render.render_sdc_car()
        if self.show_legend:
            self.bev_render.render_legend()
        self.bev_render.save_fig(out_filename + '.jpg')

    def visualize_cam(self, prediction_dict, sample_info, out_filename):
        self.cam_render.reset_canvas(dx=2, dy=3, tight_layout=True)
        self.cam_render.render_image_data(sample_info)
        self.cam_render.render_pred_track_bbox(
            prediction_dict['predicted_agent_list'], sample_info)
        self.cam_render.render_pred_traj(
            prediction_dict['predicted_agent_list'], sample_info, render_sdc=self.with_planning)
        self.cam_render.save_fig(out_filename + '_cam.jpg')

    def combine(self, out_filename):
        # pass
        bev_image = cv2.imread(out_filename + '.jpg')
        cam_image = cv2.imread(out_filename + '_cam.jpg')
        merge_image = cv2.hconcat([cam_image, bev_image])
        cv2.imwrite(out_filename + '.jpg', merge_image)
        os.remove(out_filename + '_cam.jpg')

    def to_video(self, folder_path, out_path, fps=4, downsample=1):
        imgs_path = glob.glob(os.path.join(folder_path, '*.jpg'))
        imgs_path = sorted(imgs_path)
        img_array = []
        for img_path in imgs_path:
            img = cv2.imread(img_path)
            height, width, channel = img.shape
            img = cv2.resize(img, (width//downsample, height //
                             downsample), interpolation=cv2.INTER_AREA)
            height, width, channel = img.shape
            size = (width, height)
            img_array.append(img)
        out = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()


def main(args):
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

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder) 

    metas = mmcv.load('/cpfs01/user/yangxuemeng/code/UniAD/data/nup_vis_pose_info.pkl')
    # if os.path.isfile('output/tmp.pkl'):
    #     predictions = mmcv.load('output/tmp.pkl')
    # else:
    #     predictions = mmcv.load(args.predroot)['bbox_results'] 
    #     mmcv.dump(predictions[:10], 'output/tmp.pkl')
    # predictions = mmcv.load(args.predroot)['bbox_results']
    predictions = mmcv.load(args.predroot)['bbox_results']
    for i, outputs in enumerate(predictions):
        prediction_dict = viser._parse_predictions(outputs)

        image_dict = {} # TODO: surrounding images
        # for cam in CAM_NAMES:
            # image_dict[cam] = np.zeros((900, 1600, 3)) # TODO: replace the images

        for idx, cam in enumerate(CAM_NAMES):
            tmp_img = cv2.imread(f'/cpfs01/user/yangxuemeng/code/UniAD/output/demo_mini/nup_boston/{i*10}_{Nuplan_cam_names[idx]}.jpg')
            tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
            tmp_img = cv2.resize(tmp_img, (1600, 900))
            image_dict[cam] = tmp_img


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

        viser.visualize_bev(prediction_dict, os.path.join(args.out_folder, str(i).zfill(3)))

        if args.project_to_cam:
            viser.visualize_cam(prediction_dict, sample_info, os.path.join(args.out_folder, str(i).zfill(3)))
            viser.combine(os.path.join(args.out_folder, str(i).zfill(3)))

    viser.to_video(args.out_folder, args.demo_video, fps=4, downsample=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predroot', default='/cpfs01/user/yangxuemeng/code/UniAD/output/results_nup_boston_data.pkl', help='Path to results.pkl')
    parser.add_argument('--out_folder', default='/cpfs01/user/yangxuemeng/code/UniAD/output/demo_mini/nup_boston', help='Output folder path')
    parser.add_argument('--demo_video', default='mini_val_final.avi', help='Demo video name')
    parser.add_argument('--project_to_cam', default=True, help='Project to cam (default: True)')
    args = parser.parse_args()
    main(args)
