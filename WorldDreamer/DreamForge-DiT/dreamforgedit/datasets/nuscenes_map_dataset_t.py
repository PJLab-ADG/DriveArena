import logging
import mmcv
import numpy as np
import torch
from typing import Tuple, List, Dict, Any
import random
from functools import partial

from mmcv.parallel import DataContainer as DC
from ..registry import DATASETS

from mmdet.datasets.pipelines import to_tensor
from nuscenes.eval.common.utils import Quaternion

from ..mmdet_plugin.core.bbox import LiDARInstance3DBoxes
from ..mmdet_plugin.datasets import NuScenesDataset

from .map_utils import LiDARInstanceLines, VectorizedLocalMap, project_lines_on_bev, project_lines_on_view, project_boxes_on_view
from .utils import trans_boxes_to_views
from .nuscenes_map_dataset import collate_fn, pad_bboxes_to_maxlen


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


def obtain_next2top(first, current, epsilon=1e-6, v2=True):
    l2e_r = first["lidar2ego_rotation"]
    l2e_t = first["lidar2ego_translation"]
    e2g_r = first["ego2global_rotation"]
    e2g_t = first["ego2global_translation"]
    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = Quaternion(e2g_r).rotation_matrix

    l2e_r_s = current["lidar2ego_rotation"]
    l2e_t_s = current["lidar2ego_translation"]
    e2g_r_s = current["ego2global_rotation"]
    e2g_t_s = current["ego2global_translation"]

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T -= (
        e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        + l2e_t @ np.linalg.inv(l2e_r_mat).T
    )
    next2lidar_rotation = R.T  # points @ R.T + T
    next2lidar_translation = T
    if v2:
        # inverse, point trans from lidar to next
        _R = np.concatenate([next2lidar_rotation.T, np.array(
            [[0.,] * 3], dtype=T.dtype)], axis=0)
        _T = -next2lidar_rotation.T @ next2lidar_translation
        _T = np.concatenate(
            [_T[..., np.newaxis], np.array([[1.]], dtype=T.dtype)], axis=0)
        # shape like:
        # | R T |
        # | 0 1 |
        # A @ point lidar -> point next
        next2lidar = np.concatenate([_R, _T], axis=1)
    else:
        _R = np.concatenate(
            [next2lidar_rotation, np.array([[0.,]] * 3, dtype=T.dtype)], axis=1)
        _T = np.concatenate(
            [next2lidar_translation, np.array([1.], dtype=T.dtype)], axis=0)
        # shape like:
        # | R 0 |
        # | T 1 |.T
        next2lidar = np.concatenate(
            [_R, _T[np.newaxis, ...]], axis=0,
        ).T  # A @ [points, 1].T
    if epsilon is not None:
        next2lidar[np.abs(next2lidar) < epsilon] = 0.
    return next2lidar


@DATASETS.register_module()
class NuScenesMapDatasetT(NuScenesDataset):
    def __init__(
        self,
        ann_file,
        pipeline=None,
        dataset_root=None,
        object_classes=None,
        map_classes=['divider', 'ped_crossing','boundary'],
        load_interval=1,
        with_velocity=True,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        next2topv2=True,
        eval_version="detection_cvpr_2019",
        use_valid_flag=False,
        force_all_boxes=False,
        video_length=None,
        ref_length=None,
        candidate_length=None,
        start_on_keyframe=False,
        fixed_ptsnum_per_line=-1,
        padding_value=-10000,
        img_collate_param={},
        xbound = [-50.0, 50.0, 0.5],
        ybound = [-50.0, 50.0, 0.5],
        fps = 12,
        # is_train=False,
    ) -> None:
        
        self.video_length = video_length
        self.ref_length = ref_length
        self.candidate_length = candidate_length
        self.start_on_keyframe = start_on_keyframe
        self.fps = fps
        self.padding_value = padding_value
        self.fixed_num = fixed_ptsnum_per_line
        self.next2topv2 = next2topv2
        super().__init__(
            ann_file, pipeline, dataset_root, object_classes, map_classes,
            load_interval, with_velocity, modality, box_type_3d,
            filter_empty_gt, test_mode, eval_version, use_valid_flag,
            force_all_boxes)
        self.object_classes = object_classes
        self.map_classes = map_classes
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.bound = (xbound, ybound)
        self.vector_map = VectorizedLocalMap(dataset_root, 
                            patch_size=self.patch_size, map_classes=map_classes, 
                            fixed_ptsnum_per_line=fixed_ptsnum_per_line,
                            padding_value=padding_value)
        self.img_collate_param = img_collate_param
        
    
    def build_clips(self, data_infos, scene_tokens, is_train=True):
        """Since the order in self.data_infos may change on loading, we
        calculate the index for clips after loading.

        Args:
            data_infos (list of dict): loaded data_infos
            scene_tokens (2-dim list of str): 2-dim list for tokens to each
            scene 

        Returns:
            2-dim list of int: int is the index in self.data_infos
        """
        self.token_data_dict = {
            item['token']: idx for idx, item in enumerate(data_infos)}
        all_clips = []
        self.scene_clips = {}
        self.valid_clips = {}

        for si, scene in enumerate(scene_tokens):
            self.scene_clips[si] = []
            for start in range(0, len(scene) - self.video_length + 1):
                if self.start_on_keyframe and ";" in scene[start]:
                    continue  # this is not a keyframe
                if self.start_on_keyframe and len(scene[start]) >= 33:
                    continue  # this is not a keyframe
                clip = [self.token_data_dict[token]
                        for token in scene[start: start + self.video_length]]
                # Here we need to check ref_idx in the start of the clip is equal to the ref_length
                if start == 0:
                    ref_idx = [0] * self.ref_length
                else:
                    candidate_range = list(range(max(start - self.candidate_length, 0), start))
                    if len(candidate_range) < self.ref_length:
                        ref_idx = sorted(random.choices(candidate_range, k=self.ref_length))
                    else:
                        ref_idx = sorted(random.sample(candidate_range, self.ref_length))
                ref = [self.token_data_dict[scene[idx]] for idx in ref_idx]
                clip = ref + clip
                all_clips.append(clip)

        for si, scene in enumerate(scene_tokens):
            self.scene_clips[si] = []
            for start in range(0, len(scene) - self.video_length + 1):
                if self.start_on_keyframe and ";" in scene[start]:
                    continue  # this is not a keyframe
                if self.start_on_keyframe and len(scene[start]) >= 33:
                    continue  # this is not a keyframe
                clip = [self.token_data_dict[token]
                        for token in scene[start: start + self.video_length]]
                # Here we need to check ref_idx in the start of the clip is equal to the ref_length]
                if not self.start_on_keyframe:
                    ref_idx = [0] * max(self.ref_length-start, 0) + list(range(max(start-self.ref_length, 0), start))
                else:
                    ref_idx = [0] * self.ref_length if start == 0 else [0] * max(self.ref_length-start, 0) + list(range(max(start-self.ref_length, 0), start))

                if start >= self.ref_length and (start-self.ref_length) % (self.video_length -1) == 0: # TODO: there is any need to repeat the first frame for ${ref_length} times for the first clip
                    ref = [self.token_data_dict[scene[idx]] for idx in ref_idx]
                    valid_clip = ref + clip
                    # valid_clip = [valid_clip[0]] + valid_clip[:-1]
                    self.scene_clips[si].append(valid_clip)
                    # break
                ref = [self.token_data_dict[scene[idx]] for idx in ref_idx]
                clip = ref + clip
        # For 16-clip validation
        ref_length = 0
        video_length = 17
        for si, scene in enumerate(scene_tokens):
            self.valid_clips[si] = []
            for start in range(0, len(scene) - video_length + 1):
                if self.start_on_keyframe and ";" in scene[start]:
                    continue  # this is not a keyframe
                if self.start_on_keyframe and len(scene[start]) >= 33:
                    continue  # this is not a keyframe
                clip = [self.token_data_dict[token]
                        for token in scene[start: start + video_length]]
                # Here we need to check ref_idx in the start of the clip is equal to the ref_length]
                if not self.start_on_keyframe:
                    ref_idx = [0] * max(ref_length-start, 0) + list(range(max(start-ref_length, 0), start))
                else:
                    ref_idx = [0] * ref_length if start == 0 else [0] * max(ref_length-start, 0) + list(range(max(start-ref_length, 0), start))

                if start >= self.ref_length and (start-self.ref_length) % (video_length -1) == 0: # TODO: there is any need to repeat the first frame for ${ref_length} times for the first clip
                    ref = [self.token_data_dict[scene[idx]] for idx in ref_idx]
                    valid_clip = ref + clip
                    valid_clip = [valid_clip[0]] + valid_clip[:-1]
                    self.valid_clips[si].append(valid_clip)
                    break
                ref = [self.token_data_dict[scene[idx]] for idx in ref_idx]
                clip = ref + clip
        
        


        logging.info(f"[{self.__class__.__name__}] Got {len(scene_tokens)} "
                     f"continuous scenes. Cut into {self.video_length + self.ref_length}-clip ({self.ref_length}/{self.video_length}), "
                     f"which has {len(all_clips)} in total.")
        return all_clips
    
    def load_annotations(self, ann_file, is_train=True):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        # breakpoint()
        data_infos = data_infos[:: self.load_interval]
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]
        self.clip_infos = self.build_clips(data_infos, data['scene_tokens'],is_train=is_train)
        return data_infos
    
    def __len__(self):
        return len(self.clip_infos)
    
    def get_ann_info(self, index):
        anns_results, mask = super().get_ann_info(index)
        info = self.data_infos[index]
        if "gt_box_ids" not in info:
            return anns_results, mask

        gt_bboxes_3d = anns_results['gt_bboxes_3d'].tensor

        # add token
        gt_bboxes_token = [info["gt_box_ids"][i] for i in np.where(mask)[0]]
        token_idxes = torch.arange(
            len(gt_bboxes_token),
            dtype=gt_bboxes_3d.dtype)
        gt_bboxes_3d = torch.cat([
            gt_bboxes_3d, token_idxes.unsqueeze(-1)], dim=-1)

        # rebuild boxes
        # important! put origin on the box centre
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0.5),
            tokens=gt_bboxes_token,
        )
        anns_results['gt_bboxes_3d'] = gt_bboxes_3d
        return anns_results, mask
    
    def load_clip(self, clip):
        frames = []
        first_info = self.data_infos[clip[0]]
        for frame in clip:
            frame_info = super().get_data_info(frame)
            info = self.data_infos[frame]
            next2top = obtain_next2top(first_info, info, v2=self.next2topv2)
            frame_info['relative_pose'] = next2top
            frames.append(frame_info)
        return frames

    def get_data_info(self, index):
        """We should sample from clip_infos
        """
        clip = self.clip_infos[int(index)]
        frames = self.load_clip(clip)
        return frames

    def vectormap_pipeline(self, example, input_dict):
        '''
        `example` type: <class 'dict'>
            keys: 'img_metas', 'gt_bboxes_3d', 'gt_labels_3d', 'img';
                  all keys type is 'DataContainer';
                  'img_metas' cpu_only=True, type is dict, others are false;
                  'gt_labels_3d' shape torch.size([num_samples]), stack=False,
                                padding_value=0, cpu_only=False
                  'gt_bboxes_3d': stack=False, cpu_only=True
        '''
        
        lidar2ego = input_dict['lidar2ego']
        ego2global = input_dict['ego2global']
        lidar2global = ego2global @ lidar2ego

        lidar2global_translation = list(lidar2global[:3,3])
        lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)
        location = input_dict['location']
        anns_results = self.vector_map.gen_vectorized_samples(location, lidar2global_translation, lidar2global_rotation)
        '''
        anns_results, type: dict
            'gt_vecs_pts_loc': list[num_vecs], vec with num_points*2 coordinates
            'gt_vecs_pts_num': list[num_vecs], vec with num_points
            'gt_vecs_label': list[num_vecs], vec with cls index
        '''
        gt_vecs_label = to_tensor(anns_results['gt_vecs_label'])
        if isinstance(anns_results['gt_vecs_pts_loc'], LiDARInstanceLines):
            gt_vecs_pts_loc = anns_results['gt_vecs_pts_loc']
            gt_lines_instance = gt_vecs_pts_loc.instance_list
            gt_map_pts = []
            for i in range(len(gt_lines_instance)):
                pts = np.array(list(gt_lines_instance[i].coords))
                gt_map_pts.append(pts)
        example['gt_vecs_label'] = DC(gt_vecs_label, cpu_only=False)
        example['gt_vecs_pts_loc'] = DC(gt_map_pts, cpu_only=True)

        bev_hdmap = project_lines_on_bev(example['gt_vecs_pts_loc'].data, example['gt_vecs_label'].data, num_classes=len(self.map_classes), xbound=self.bound[0], ybound=self.bound[1])
        bev_hdmap = bev_hdmap.transpose(2, 0, 1)    # 3, 200, 200
        example['bev_hdmap'] = DC(to_tensor(bev_hdmap.copy()), cpu_only=False)
        
        return example

    def project_on_views(self, example):
        image_size = example['img'].data.shape[-2:]
        lidar2image = example['lidar2image'].data
        camera2ego = example['camera2ego'].data
        camera_intrinsics = example['camera_intrinsics'].data

        layout_canvas = []
        for i in range(len(lidar2image)):
            # TODO: Should we consider img_aug_matrix?
            map_canvas = project_lines_on_view(example['gt_vecs_pts_loc'].data, example['gt_vecs_label'].data, camera_intrinsics[i], camera2ego[i], image_size=image_size)
            box_canvas = project_boxes_on_view(example['gt_bboxes_3d'].data, example['gt_labels_3d'].data, lidar2image[i], object_classes=self.object_classes, image_size=image_size)
            layout_canvas.append(np.concatenate([map_canvas, box_canvas], axis=-1))

        layout_canvas = np.stack(layout_canvas, axis=0)
        layout_canvas = np.transpose(layout_canvas, (0, 3, 1, 2))    # 6, 3+10, H, W
        example['layout_canvas'] = DC(to_tensor(layout_canvas), cpu_only=False)

        return example

    def prepare_train_data(self, index):
        """This is called by `__getitem__`
            index is the index of the clip_infos
        """
        frames = self.get_data_info(index)
        if None in frames:
            return None
        examples = []
        for frame in frames:
            self.pre_pipeline(frame)
            example = self.pipeline(frame)
            example = self.vectormap_pipeline(example, frame)
            example = self.project_on_views(example)

            example['height'] = example['img'].data.shape[-2]
            example['width'] = example['img'].data.shape[-1]
            example['fps'] = self.fps
            example['num_frames'] = self.video_length + self.ref_length

            if self.filter_empty_gt and frame['is_key_frame'] and (
                example is None or ~(example["gt_labels_3d"]._data != -1).any()
            ):
                return None
            examples.append(example)
 
        return examples
    

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
        out_dict[key] = to_tensor([ret_dict[key] for ret_dict in dicts])

    # caption
    out_dict['captions'] = [[ret_dict['captions'][0] for ret_dict in dicts]] # for each clip only one caption, T is the first dim

        # other meta data
    for key in dicts[0]['meta_data'].keys():
        out_dict['meta_data'][key] = [ret_dict['meta_data'][key] for ret_dict in dicts]

    return out_dict
