# nuScenes dev-kit.
# Code written by Sergi Adipraja Widjaja, 2019.
# + Map mask by Kiwoo Shin, 2019.
# + Methods operating on NuScenesMap and NuScenes by Holger Caesar, 2019.

import json
import os
import os.path as osp
import copy
import random
from typing import Dict, List, Tuple, Optional, Union

import cv2
import descartes
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Arrow
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from pyquaternion import Quaternion
from shapely import affinity, ops
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString, Point, box
from collections.abc import Sequence

import mmcv
from nuscenes.map_expansion.arcline_path_utils import discretize_lane, ArcLinePath
from nuscenes.map_expansion.bitmap import BitMap
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from TrafficManager.utils.map_api import NuScenesMap, NuScenesMapExplorer
# Recommended style to use as the plots will show grids.
plt.style.use('seaborn-whitegrid')

# Define a map geometry type for polygons and lines.
Geometry = Union[Polygon, LineString]

locations = ['singapore-onenorth', 'singapore-hollandvillage',
             'singapore-queenstown', 'boston-seaport']
colors_plt = ['orange', 'b', 'g']


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


def visualize_bev_hdmap(gt_bboxes_3d, gt_labels_3d, canvas_size, bound=[-50.0, 50.0], vis_format='polyline_pts'):
    canvas = np.zeros((3, *canvas_size, 3), dtype=np.uint8)
    gt_lines_instance = gt_bboxes_3d.instance_list
    for gt_line_instance, gt_label_3d in zip(gt_lines_instance, gt_labels_3d):
        pts = np.array(list(gt_line_instance.coords))
        for p in pts:
            pp = ((p - bound[0]) / (bound[1] - bound[0])
                  * canvas_size[0]).astype(int)
            cv2.circle(canvas[int(gt_label_3d)], tuple(pp), 1, (1, 0, 0), -1)
            # import pdb; pdb.set_trace()

        for i in range(len(pts)-1):
            pp1 = ((pts[i] - bound[0]) / (bound[1] - bound[0])
                   * canvas_size[0]).astype(int)
            pp2 = ((pts[i+1] - bound[0]) / (bound[1] - bound[0])
                   * canvas_size[0]).astype(int)
            cv2.line(canvas[int(gt_label_3d)], tuple(
                pp1), tuple(pp2), (1, 0, 0), 1)

    canvas = canvas[..., 0]
    canvas = np.transpose(canvas, (2, 1, 0))
    # canvas = canvas[::-1, :, :]

    # cv2.imwrite('./GT_polyline_pts_MAP.png', canvas)
    return canvas


def project_map_to_image(gt_bboxes_3d, gt_labels_3d, intrinsic, extrinsic, image=None):
    z = 0
    if image is not None:
        canvas = image
    else:
        canvas = np.zeros((3, 900, 1600, 3), dtype=np.uint8)
    gt_lines_instance = gt_bboxes_3d.instance_list
    for gt_line_instance, gt_label_3d in zip(gt_lines_instance, gt_labels_3d):
        pts = torch.Tensor(list(gt_line_instance.coords))
        pts = pts[:, [1, 0]]
        pts[:, 1] = -pts[:, 1]
        dummy_pts = torch.cat([pts, torch.ones((pts.shape[0], 1))*z], dim=-1)
        # dummy_pts = torch.cat([dummy_pts, torch.ones((pts.shape[0], 1))], dim=-1)
        points_in_cam_cor = torch.matmul(
            extrinsic[:3, :3].T, (dummy_pts.T - extrinsic[:3, 3].reshape(3, -1)))
        points_in_cam_cor = points_in_cam_cor[:, points_in_cam_cor[2, :] > 0]
        if points_in_cam_cor.shape[1] > 1:
            points_on_image_cor = intrinsic[:3, :3] @ points_in_cam_cor
            points_on_image_cor = points_on_image_cor / \
                (points_on_image_cor[-1, :].reshape(1, -1))
            points_on_image_cor = points_on_image_cor[:2, :].T
            points_on_image_cor = points_on_image_cor.int().numpy()
        else:
            points_on_image_cor = []

        if image is not None:
            for p in points_on_image_cor:
                cv2.circle(canvas, tuple(p), 4, (255, 0, 0), -1)
            for i in range(len(points_on_image_cor)-1):
                cv2.line(canvas, tuple(points_on_image_cor[i]), tuple(
                    points_on_image_cor[i+1]), (255, 0, 0), 4)
        else:
            # for p in points_on_image_cor:
            #     cv2.circle(canvas[int(gt_label_3d)], tuple(p), 40, (150,0,0), -1)
            for i in range(len(points_on_image_cor)-1):
                cv2.line(canvas[int(gt_label_3d)], tuple(points_on_image_cor[i]), tuple(
                    points_on_image_cor[i+1]), (1, 0, 0), 4)
    if image is not None:
        cv2.imwrite('./project.png', canvas)
    else:
        canvas = canvas[..., 0]
        canvas = np.transpose(canvas, (1, 2, 0))
        canvas = canvas[::4, ::4, :][1:, ...]
        # cv2.imwrite('./project.png', canvas)

    return canvas


class LiDARInstanceLines(object):
    """Line instance in LIDAR coordinates

    """
    def __init__(self,
                 instance_line_list,
                 sample_dist=1,
                 num_samples=250,
                 padding=False,
                 fixed_num=-1,
                 padding_value=-10000,
                 patch_size=None):
        assert isinstance(instance_line_list, list)
        assert patch_size is not None
        if len(instance_line_list) != 0:
            assert isinstance(instance_line_list[0], LineString)
        self.patch_size = patch_size
        self.max_x = self.patch_size[1] / 2
        self.max_y = self.patch_size[0] / 2
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.fixed_num = fixed_num
        self.padding_value = padding_value

        self.instance_list = instance_line_list

    @property
    def start_end_points(self):
        """
        return torch.Tensor([N,4]), in xstart, ystart, xend, yend form
        """
        assert len(self.instance_list) != 0
        instance_se_points_list = []
        for instance in self.instance_list:
            se_points = []
            se_points.extend(instance.coords[0])
            se_points.extend(instance.coords[-1])
            instance_se_points_list.append(se_points)
        instance_se_points_array = np.array(instance_se_points_list)
        instance_se_points_tensor = to_tensor(instance_se_points_array)
        instance_se_points_tensor = instance_se_points_tensor.to(
            dtype=torch.float32)
        instance_se_points_tensor[:, 0] = torch.clamp(
            instance_se_points_tensor[:, 0], min=-self.max_x, max=self.max_x)
        instance_se_points_tensor[:, 1] = torch.clamp(
            instance_se_points_tensor[:, 1], min=-self.max_y, max=self.max_y)
        instance_se_points_tensor[:, 2] = torch.clamp(
            instance_se_points_tensor[:, 2], min=-self.max_x, max=self.max_x)
        instance_se_points_tensor[:, 3] = torch.clamp(
            instance_se_points_tensor[:, 3], min=-self.max_y, max=self.max_y)
        return instance_se_points_tensor

    @property
    def bbox(self):
        """
        return torch.Tensor([N,4]), in xmin, ymin, xmax, ymax form
        """
        assert len(self.instance_list) != 0
        instance_bbox_list = []
        for instance in self.instance_list:
            # bounds is bbox: [xmin, ymin, xmax, ymax]
            instance_bbox_list.append(instance.bounds)
        instance_bbox_array = np.array(instance_bbox_list)
        instance_bbox_tensor = to_tensor(instance_bbox_array)
        instance_bbox_tensor = instance_bbox_tensor.to(
            dtype=torch.float32)
        instance_bbox_tensor[:, 0] = torch.clamp(
            instance_bbox_tensor[:, 0], min=-self.max_x, max=self.max_x)
        instance_bbox_tensor[:, 1] = torch.clamp(
            instance_bbox_tensor[:, 1], min=-self.max_y, max=self.max_y)
        instance_bbox_tensor[:, 2] = torch.clamp(
            instance_bbox_tensor[:, 2], min=-self.max_x, max=self.max_x)
        instance_bbox_tensor[:, 3] = torch.clamp(
            instance_bbox_tensor[:, 3], min=-self.max_y, max=self.max_y)
        return instance_bbox_tensor

    @property
    def fixed_num_sampled_points(self):
        """
        return torch.Tensor([N,fixed_num,2]), in xmin, ymin, xmax, ymax form
            N means the num of instances
        """
        assert len(self.instance_list) != 0
        instance_points_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            sampled_points = np.array([list(instance.interpolate(
                distance).coords) for distance in distances]).reshape(-1, 2)
            instance_points_list.append(sampled_points)
        instance_points_array = np.array(instance_points_list)
        instance_points_tensor = to_tensor(instance_points_array)
        instance_points_tensor = instance_points_tensor.to(
            dtype=torch.float32)
        instance_points_tensor[:, :, 0] = torch.clamp(
            instance_points_tensor[:, :, 0], min=-self.max_x, max=self.max_x)
        instance_points_tensor[:, :, 1] = torch.clamp(
            instance_points_tensor[:, :, 1], min=-self.max_y, max=self.max_y)
        return instance_points_tensor

    @property
    def fixed_num_sampled_points_ambiguity(self):
        """
        return torch.Tensor([N,fixed_num,2]), in xmin, ymin, xmax, ymax form
            N means the num of instances
        """
        assert len(self.instance_list) != 0
        instance_points_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            sampled_points = np.array([list(instance.interpolate(
                distance).coords) for distance in distances]).reshape(-1, 2)
            instance_points_list.append(sampled_points)
        instance_points_array = np.array(instance_points_list)
        instance_points_tensor = to_tensor(instance_points_array)
        instance_points_tensor = instance_points_tensor.to(
            dtype=torch.float32)
        instance_points_tensor[:, :, 0] = torch.clamp(
            instance_points_tensor[:, :, 0], min=-self.max_x, max=self.max_x)
        instance_points_tensor[:, :, 1] = torch.clamp(
            instance_points_tensor[:, :, 1], min=-self.max_y, max=self.max_y)
        instance_points_tensor = instance_points_tensor.unsqueeze(1)
        return instance_points_tensor

    @property
    def fixed_num_sampled_points_torch(self):
        """
        return torch.Tensor([N,fixed_num,2]), in xmin, ymin, xmax, ymax form
            N means the num of instances
        """
        assert len(self.instance_list) != 0
        instance_points_list = []
        for instance in self.instance_list:
            # distances = np.linspace(0, instance.length, self.fixed_num)
            # sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
            poly_pts = to_tensor(np.array(list(instance.coords)))
            poly_pts = poly_pts.unsqueeze(0).permute(0, 2, 1)
            sampled_pts = torch.nn.functional.interpolate(
                poly_pts, size=(self.fixed_num), mode='linear', align_corners=True)
            sampled_pts = sampled_pts.permute(0, 2, 1).squeeze(0)
            instance_points_list.append(sampled_pts)
        # instance_points_array = np.array(instance_points_list)
        # instance_points_tensor = to_tensor(instance_points_array)
        instance_points_tensor = torch.stack(instance_points_list, dim=0)
        instance_points_tensor = instance_points_tensor.to(
            dtype=torch.float32)
        instance_points_tensor[:, :, 0] = torch.clamp(
            instance_points_tensor[:, :, 0], min=-self.max_x, max=self.max_x)
        instance_points_tensor[:, :, 1] = torch.clamp(
            instance_points_tensor[:, :, 1], min=-self.max_y, max=self.max_y)
        return instance_points_tensor

    @property
    def shift_fixed_num_sampled_points(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        fixed_num_sampled_points = self.fixed_num_sampled_points
        instances_list = []
        is_poly = False
        # is_line = False
        # import pdb;pdb.set_trace()
        for fixed_num_pts in fixed_num_sampled_points:
            # [fixed_num, 2]
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            fixed_num = fixed_num_pts.shape[0]
            shift_pts_list = []
            if is_poly:
                # import pdb;pdb.set_trace()
                for shift_right_i in range(fixed_num):
                    shift_pts_list.append(fixed_num_pts.roll(shift_right_i, 0))
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list, dim=0)

            shift_pts[:, :, 0] = torch.clamp(
                shift_pts[:, :, 0], min=-self.max_x, max=self.max_x)
            shift_pts[:, :, 1] = torch.clamp(
                shift_pts[:, :, 1], min=-self.max_y, max=self.max_y)

            if not is_poly:
                padding = torch.full(
                    [fixed_num-shift_pts.shape[0], fixed_num, 2], self.padding_value)
                shift_pts = torch.cat([shift_pts, padding], dim=0)
                # padding = np.zeros((self.num_samples - len(sampled_points), 2))
                # sampled_points = np.concatenate([sampled_points, padding], axis=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(
            dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v1(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        fixed_num_sampled_points = self.fixed_num_sampled_points
        instances_list = []
        is_poly = False
        # is_line = False
        # import pdb;pdb.set_trace()
        for fixed_num_pts in fixed_num_sampled_points:
            # [fixed_num, 2]
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            pts_num = fixed_num_pts.shape[0]
            shift_num = pts_num - 1
            if is_poly:
                pts_to_shift = fixed_num_pts[:-1, :]
            shift_pts_list = []
            if is_poly:
                for shift_right_i in range(shift_num):
                    shift_pts_list.append(pts_to_shift.roll(shift_right_i, 0))
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list, dim=0)

            if is_poly:
                _, _, num_coords = shift_pts.shape
                tmp_shift_pts = shift_pts.new_zeros(
                    (shift_num, pts_num, num_coords))
                tmp_shift_pts[:, :-1, :] = shift_pts
                tmp_shift_pts[:, -1, :] = shift_pts[:, 0, :]
                shift_pts = tmp_shift_pts

            shift_pts[:, :, 0] = torch.clamp(
                shift_pts[:, :, 0], min=-self.max_x, max=self.max_x)
            shift_pts[:, :, 1] = torch.clamp(
                shift_pts[:, :, 1], min=-self.max_y, max=self.max_y)

            if not is_poly:
                padding = torch.full(
                    [shift_num-shift_pts.shape[0], pts_num, 2], self.padding_value)
                shift_pts = torch.cat([shift_pts, padding], dim=0)
                # padding = np.zeros((self.num_samples - len(sampled_points), 2))
                # sampled_points = np.concatenate([sampled_points, padding], axis=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(
            dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v2(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        assert len(self.instance_list) != 0
        instances_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            poly_pts = np.array(list(instance.coords))
            start_pts = poly_pts[0]
            end_pts = poly_pts[-1]
            is_poly = np.equal(start_pts, end_pts)
            is_poly = is_poly.all()
            shift_pts_list = []
            pts_num, coords_num = poly_pts.shape
            shift_num = pts_num - 1
            final_shift_num = self.fixed_num - 1
            if is_poly:
                pts_to_shift = poly_pts[:-1, :]
                for shift_right_i in range(shift_num):
                    shift_pts = np.roll(pts_to_shift, shift_right_i, axis=0)
                    pts_to_concat = shift_pts[0]
                    pts_to_concat = np.expand_dims(pts_to_concat, axis=0)
                    shift_pts = np.concatenate(
                        (shift_pts, pts_to_concat), axis=0)
                    shift_instance = LineString(shift_pts)
                    shift_sampled_points = np.array([list(shift_instance.interpolate(
                        distance).coords) for distance in distances]).reshape(-1, 2)
                    shift_pts_list.append(shift_sampled_points)
                # import pdb;pdb.set_trace()
            else:
                sampled_points = np.array([list(instance.interpolate(
                    distance).coords) for distance in distances]).reshape(-1, 2)
                flip_sampled_points = np.flip(sampled_points, axis=0)
                shift_pts_list.append(sampled_points)
                shift_pts_list.append(flip_sampled_points)

            multi_shifts_pts = np.stack(shift_pts_list, axis=0)
            shifts_num, _, _ = multi_shifts_pts.shape

            if shifts_num > final_shift_num:
                index = np.random.choice(
                    multi_shifts_pts.shape[0], final_shift_num, replace=False)
                multi_shifts_pts = multi_shifts_pts[index]

            multi_shifts_pts_tensor = to_tensor(multi_shifts_pts)
            multi_shifts_pts_tensor = multi_shifts_pts_tensor.to(
                dtype=torch.float32)

            multi_shifts_pts_tensor[:, :, 0] = torch.clamp(
                multi_shifts_pts_tensor[:, :, 0], min=-self.max_x, max=self.max_x)
            multi_shifts_pts_tensor[:, :, 1] = torch.clamp(
                multi_shifts_pts_tensor[:, :, 1], min=-self.max_y, max=self.max_y)
            # if not is_poly:
            if multi_shifts_pts_tensor.shape[0] < final_shift_num:
                padding = torch.full(
                    [final_shift_num-multi_shifts_pts_tensor.shape[0], self.fixed_num, 2], self.padding_value)
                multi_shifts_pts_tensor = torch.cat(
                    [multi_shifts_pts_tensor, padding], dim=0)
            instances_list.append(multi_shifts_pts_tensor)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(
            dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v3(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        assert len(self.instance_list) != 0
        instances_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            poly_pts = np.array(list(instance.coords))
            start_pts = poly_pts[0]
            end_pts = poly_pts[-1]
            is_poly = np.equal(start_pts, end_pts)
            is_poly = is_poly.all()
            shift_pts_list = []
            pts_num, coords_num = poly_pts.shape
            shift_num = pts_num - 1
            final_shift_num = self.fixed_num - 1
            if is_poly:
                pts_to_shift = poly_pts[:-1, :]
                for shift_right_i in range(shift_num):
                    shift_pts = np.roll(pts_to_shift, shift_right_i, axis=0)
                    pts_to_concat = shift_pts[0]
                    pts_to_concat = np.expand_dims(pts_to_concat, axis=0)
                    shift_pts = np.concatenate(
                        (shift_pts, pts_to_concat), axis=0)
                    shift_instance = LineString(shift_pts)
                    shift_sampled_points = np.array([list(shift_instance.interpolate(
                        distance).coords) for distance in distances]).reshape(-1, 2)
                    shift_pts_list.append(shift_sampled_points)
                flip_pts_to_shift = np.flip(pts_to_shift, axis=0)
                for shift_right_i in range(shift_num):
                    shift_pts = np.roll(flip_pts_to_shift,
                                        shift_right_i, axis=0)
                    pts_to_concat = shift_pts[0]
                    pts_to_concat = np.expand_dims(pts_to_concat, axis=0)
                    shift_pts = np.concatenate(
                        (shift_pts, pts_to_concat), axis=0)
                    shift_instance = LineString(shift_pts)
                    shift_sampled_points = np.array([list(shift_instance.interpolate(
                        distance).coords) for distance in distances]).reshape(-1, 2)
                    shift_pts_list.append(shift_sampled_points)
            else:
                sampled_points = np.array([list(instance.interpolate(
                    distance).coords) for distance in distances]).reshape(-1, 2)
                flip_sampled_points = np.flip(sampled_points, axis=0)
                shift_pts_list.append(sampled_points)
                shift_pts_list.append(flip_sampled_points)

            multi_shifts_pts = np.stack(shift_pts_list, axis=0)
            shifts_num, _, _ = multi_shifts_pts.shape
            if shifts_num > 2*final_shift_num:
                index = np.random.choice(
                    shift_num, final_shift_num, replace=False)
                flip0_shifts_pts = multi_shifts_pts[index]
                flip1_shifts_pts = multi_shifts_pts[index+shift_num]
                multi_shifts_pts = np.concatenate(
                    (flip0_shifts_pts, flip1_shifts_pts), axis=0)

            multi_shifts_pts_tensor = to_tensor(multi_shifts_pts)
            multi_shifts_pts_tensor = multi_shifts_pts_tensor.to(
                dtype=torch.float32)

            multi_shifts_pts_tensor[:, :, 0] = torch.clamp(
                multi_shifts_pts_tensor[:, :, 0], min=-self.max_x, max=self.max_x)
            multi_shifts_pts_tensor[:, :, 1] = torch.clamp(
                multi_shifts_pts_tensor[:, :, 1], min=-self.max_y, max=self.max_y)
            if multi_shifts_pts_tensor.shape[0] < 2*final_shift_num:
                padding = torch.full(
                    [final_shift_num*2-multi_shifts_pts_tensor.shape[0], self.fixed_num, 2], self.padding_value)
                multi_shifts_pts_tensor = torch.cat(
                    [multi_shifts_pts_tensor, padding], dim=0)
            instances_list.append(multi_shifts_pts_tensor)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(
            dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v4(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        fixed_num_sampled_points = self.fixed_num_sampled_points
        instances_list = []
        is_poly = False
        for fixed_num_pts in fixed_num_sampled_points:
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            pts_num = fixed_num_pts.shape[0]
            shift_num = pts_num - 1
            shift_pts_list = []
            if is_poly:
                pts_to_shift = fixed_num_pts[:-1, :]
                for shift_right_i in range(shift_num):
                    shift_pts_list.append(pts_to_shift.roll(shift_right_i, 0))
                flip_pts_to_shift = pts_to_shift.flip(0)
                for shift_right_i in range(shift_num):
                    shift_pts_list.append(
                        flip_pts_to_shift.roll(shift_right_i, 0))
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list, dim=0)

            if is_poly:
                _, _, num_coords = shift_pts.shape
                tmp_shift_pts = shift_pts.new_zeros(
                    (shift_num*2, pts_num, num_coords))
                tmp_shift_pts[:, :-1, :] = shift_pts
                tmp_shift_pts[:, -1, :] = shift_pts[:, 0, :]
                shift_pts = tmp_shift_pts

            shift_pts[:, :, 0] = torch.clamp(
                shift_pts[:, :, 0], min=-self.max_x, max=self.max_x)
            shift_pts[:, :, 1] = torch.clamp(
                shift_pts[:, :, 1], min=-self.max_y, max=self.max_y)

            if not is_poly:
                padding = torch.full(
                    [shift_num*2-shift_pts.shape[0], pts_num, 2], self.padding_value)
                shift_pts = torch.cat([shift_pts, padding], dim=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(
            dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_torch(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        fixed_num_sampled_points = self.fixed_num_sampled_points_torch
        instances_list = []
        is_poly = False

        for fixed_num_pts in fixed_num_sampled_points:
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            fixed_num = fixed_num_pts.shape[0]
            shift_pts_list = []
            if is_poly:
                for shift_right_i in range(fixed_num):
                    shift_pts_list.append(fixed_num_pts.roll(shift_right_i, 0))
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list, dim=0)

            shift_pts[:, :, 0] = torch.clamp(
                shift_pts[:, :, 0], min=-self.max_x, max=self.max_x)
            shift_pts[:, :, 1] = torch.clamp(
                shift_pts[:, :, 1], min=-self.max_y, max=self.max_y)

            if not is_poly:
                padding = torch.full(
                    [fixed_num-shift_pts.shape[0], fixed_num, 2], self.padding_value)
                shift_pts = torch.cat([shift_pts, padding], dim=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(
            dtype=torch.float32)
        return instances_tensor


class VectorizedLocalMap(object):
    CLASS2LABEL = {
        'road_divider': 0,
        'lane_divider': 0,
        'ped_crossing': 1,
        'contours': 2,
        'others': -1
    }

    def __init__(self,
                 dataroot,
                 map_name,
                 patch_size,
                 map_classes=['divider', 'ped_crossing', 'boundary'],
                 line_classes=['road_divider', 'lane_divider'],
                 ped_crossing_classes=['ped_crossing'],
                 contour_classes=['road_segment', 'lane'],
                 sample_dist=1,
                 num_samples=250,
                 padding=False,
                 fixed_ptsnum_per_line=-1,
                 padding_value=-10000,):
        '''
        Args:
            fixed_ptsnum_per_line = -1 : no fixed num
        '''
        super().__init__()
        self.data_root = dataroot
        self.MAPS = [map_name]
        self.vec_classes = map_classes
        self.line_classes = line_classes
        self.ped_crossing_classes = ped_crossing_classes
        self.polygon_classes = contour_classes
        self.nusc_maps = {}
        self.map_explorer = {}
        for loc in self.MAPS:
            self.nusc_maps[loc] = NuScenesMap(
                dataroot=self.data_root, map_name=loc)
            self.map_explorer[loc] = NuScenesMapExplorer(self.nusc_maps[loc])

        self.patch_size = patch_size
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.fixed_num = fixed_ptsnum_per_line
        self.padding_value = padding_value

    def gen_vectorized_samples(self, location, lidar2global_translation, lidar2global_rotation):
        '''
        use lidar2global to get gt map layers
        '''

        map_pose = lidar2global_translation[:2]
        rotation = Quaternion(axis=[0, 0, 1], angle=lidar2global_rotation)

        patch_box = (map_pose[0], map_pose[1],
                     self.patch_size[0], self.patch_size[1])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        vectors = []
        for vec_class in self.vec_classes:
            if vec_class == 'divider':
                line_geom = self.get_map_geom(
                    patch_box, patch_angle, self.line_classes, location)
                line_instances_dict = self.line_geoms_to_instances(line_geom)
                for line_type, instances in line_instances_dict.items():
                    for instance in instances:
                        vectors.append(
                            (instance, self.CLASS2LABEL.get(line_type, -1)))
            elif vec_class == 'ped_crossing':
                ped_geom = self.get_map_geom(
                    patch_box, patch_angle, self.ped_crossing_classes, location)
                ped_instance_list = self.ped_poly_geoms_to_instances(ped_geom)
                for instance in ped_instance_list:
                    vectors.append(
                        (instance, self.CLASS2LABEL.get('ped_crossing', -1)))
            elif vec_class == 'boundary':
                polygon_geom = self.get_map_geom(
                    patch_box, patch_angle, self.polygon_classes, location)
                poly_bound_list = self.poly_geoms_to_instances(polygon_geom)
                for contour in poly_bound_list:
                    vectors.append(
                        (contour, self.CLASS2LABEL.get('contours', -1)))
            else:
                raise ValueError(f'WRONG vec_class: {vec_class}')

        filtered_vectors = []
        gt_pts_loc_3d = []
        gt_pts_num_3d = []
        gt_labels = []
        gt_instance = []
        for instance, type in vectors:
            if type != -1:
                gt_instance.append(instance)
                gt_labels.append(type)
        gt_instance = LiDARInstanceLines(gt_instance, self.sample_dist,
                                         self.num_samples, self.padding, self.fixed_num, self.padding_value, patch_size=self.patch_size)

        anns_results = dict(
            gt_vecs_pts_loc=gt_instance,
            gt_vecs_label=gt_labels,

        )
        return anns_results

    def get_map_geom(self, patch_box, patch_angle, layer_names, location):
        map_geom = []
        for layer_name in layer_names:
            if layer_name in self.line_classes:
                geoms = self.get_divider_line(
                    patch_box, patch_angle, layer_name, location)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.polygon_classes:
                geoms = self.get_contour_line(
                    patch_box, patch_angle, layer_name, location)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.ped_crossing_classes:
                geoms = self.get_ped_crossing_line(
                    patch_box, patch_angle, location)
                map_geom.append((layer_name, geoms))
        return map_geom

    def _one_type_line_geom_to_vectors(self, line_geom):
        line_vectors = []

        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for single_line in line.geoms:
                        line_vectors.append(
                            self.sample_pts_from_line(single_line))
                elif line.geom_type == 'LineString':
                    line_vectors.append(self.sample_pts_from_line(line))
                else:
                    raise NotImplementedError
        return line_vectors

    def _one_type_line_geom_to_instances(self, line_geom):
        line_instances = []

        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for single_line in line.geoms:
                        line_instances.append(single_line)
                elif line.geom_type == 'LineString':
                    line_instances.append(line)
                else:
                    raise NotImplementedError
        return line_instances

    def poly_geoms_to_vectors(self, polygon_geom):
        roads = polygon_geom[0][1]
        lanes = polygon_geom[1][1]
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        union_segments = ops.unary_union([union_roads, union_lanes])
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_vectors(results)

    def ped_poly_geoms_to_instances(self, ped_geom):
        ped = ped_geom[0][1]
        union_segments = ops.unary_union(ped)
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x - 0.2, -max_y - 0.2, max_x + 0.2, max_y + 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_instances(results)

    def poly_geoms_to_instances(self, polygon_geom):
        roads = polygon_geom[0][1]
        lanes = polygon_geom[1][1]
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        union_segments = ops.unary_union([union_roads, union_lanes])
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_instances(results)

    def line_geoms_to_vectors(self, line_geom):
        line_vectors_dict = dict()
        for line_type, a_type_of_lines in line_geom:
            one_type_vectors = self._one_type_line_geom_to_vectors(
                a_type_of_lines)
            line_vectors_dict[line_type] = one_type_vectors

        return line_vectors_dict

    def line_geoms_to_instances(self, line_geom):
        line_instances_dict = dict()
        for line_type, a_type_of_lines in line_geom:
            one_type_instances = self._one_type_line_geom_to_instances(
                a_type_of_lines)
            line_instances_dict[line_type] = one_type_instances

        return line_instances_dict

    def ped_geoms_to_vectors(self, ped_geom):
        ped_geom = ped_geom[0][1]
        union_ped = ops.unary_union(ped_geom)
        if union_ped.geom_type != 'MultiPolygon':
            union_ped = MultiPolygon([union_ped])

        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        results = []
        for ped_poly in union_ped:
            # rect = ped_poly.minimum_rotated_rectangle
            ext = ped_poly.exterior
            if not ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            results.append(lines)

        return self._one_type_line_geom_to_vectors(results)

    def get_contour_line(self, patch_box, patch_angle, layer_name, location):
        if layer_name not in self.map_explorer[location].map_api.non_geometric_polygon_layers:
            raise ValueError('{} is not a polygonal layer'.format(layer_name))

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer[location].get_patch_coord(
            patch_box, patch_angle)

        records = getattr(self.map_explorer[location].map_api, layer_name)

        polygon_list = []
        if layer_name == 'drivable_area':
            for record in records:
                polygons = [self.map_explorer[location].map_api.extract_polygon(
                    polygon_token) for polygon_token in record['polygon_tokens']]

                for polygon in polygons:
                    new_polygon = polygon.intersection(patch)
                    if not new_polygon.is_empty:
                        new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                      origin=(patch_x, patch_y), use_radians=False)
                        new_polygon = affinity.affine_transform(new_polygon,
                                                                [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                        if new_polygon.geom_type is 'Polygon':
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)

        else:
            for record in records:
                polygon = self.map_explorer[location].map_api.extract_polygon(
                    record['polygon_token'])

                if polygon.is_valid:
                    new_polygon = polygon.intersection(patch)
                    if not new_polygon.is_empty:
                        new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                      origin=(patch_x, patch_y), use_radians=False)
                        new_polygon = affinity.affine_transform(new_polygon,
                                                                [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                        if new_polygon.geom_type is 'Polygon':
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)

        return polygon_list

    def get_divider_line(self, patch_box, patch_angle, layer_name, location):
        if layer_name not in self.map_explorer[location].map_api.non_geometric_line_layers:
            raise ValueError("{} is not a line layer".format(layer_name))

        if layer_name is 'traffic_light':
            return None

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer[location].get_patch_coord(
            patch_box, patch_angle)

        line_list = []
        records = getattr(self.map_explorer[location].map_api, layer_name)
        for record in records:
            line = self.map_explorer[location].map_api.extract_line(
                record['line_token'])
            if line.is_empty:  # Skip lines without nodes.
                continue

            new_line = line.intersection(patch)
            if not new_line.is_empty:
                new_line = affinity.rotate(
                    new_line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                new_line = affinity.affine_transform(new_line,
                                                     [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                line_list.append(new_line)

        return line_list

    def get_ped_crossing_line(self, patch_box, patch_angle, location):
        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer[location].get_patch_coord(
            patch_box, patch_angle)
        polygon_list = []
        records = getattr(self.map_explorer[location].map_api, 'ped_crossing')
        # records = getattr(self.nusc_maps[location], 'ped_crossing')
        for record in records:
            polygon = self.map_explorer[location].map_api.extract_polygon(
                record['polygon_token'])
            if polygon.is_valid:
                new_polygon = polygon.intersection(patch)
                if not new_polygon.is_empty:
                    new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                  origin=(patch_x, patch_y), use_radians=False)
                    new_polygon = affinity.affine_transform(new_polygon,
                                                            [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                    if new_polygon.geom_type is 'Polygon':
                        new_polygon = MultiPolygon([new_polygon])
                    polygon_list.append(new_polygon)

        return polygon_list

    def sample_pts_from_line(self, line):
        if self.fixed_num < 0:
            distances = np.arange(0, line.length, self.sample_dist)
            sampled_points = np.array([list(line.interpolate(
                distance).coords) for distance in distances]).reshape(-1, 2)
        else:
            # fixed number of points, so distance is line.length / self.fixed_num
            distances = np.linspace(0, line.length, self.fixed_num)
            sampled_points = np.array([list(line.interpolate(
                distance).coords) for distance in distances]).reshape(-1, 2)

        num_valid = len(sampled_points)

        if not self.padding or self.fixed_num > 0:
            return sampled_points, num_valid

        # fixed distance sampling need padding!
        num_valid = len(sampled_points)

        if self.fixed_num < 0:
            if num_valid < self.num_samples:
                padding = np.zeros((self.num_samples - len(sampled_points), 2))
                sampled_points = np.concatenate(
                    [sampled_points, padding], axis=0)
            else:
                sampled_points = sampled_points[:self.num_samples, :]
                num_valid = self.num_samples

        return sampled_points, num_valid
