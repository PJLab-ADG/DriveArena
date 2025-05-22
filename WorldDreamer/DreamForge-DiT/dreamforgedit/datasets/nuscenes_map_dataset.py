import logging
import mmcv
import numpy as np
import torch
import os
from typing import Tuple, List
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


@DATASETS.register_module()
class NuScenesMapDataset(NuScenesDataset):
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
        eval_version="detection_cvpr_2019",
        use_valid_flag=False,
        force_all_boxes=False,
        fixed_ptsnum_per_line=-1,
        padding_value=-10000,
        img_collate_param={},
        xbound = [-50.0, 50.0, 0.5],
        ybound = [-50.0, 50.0, 0.5],
        fps = 2
    ) -> None:
        self.use_2Hz = True if '12Hz' in ann_file and fps == 2 else False

        super().__init__(
            ann_file, pipeline, dataset_root, object_classes, map_classes,
            load_interval, with_velocity, modality, box_type_3d,
            filter_empty_gt, test_mode, eval_version, use_valid_flag,
            force_all_boxes)
        
        self.fps = fps
        self.padding_value = padding_value
        self.fixed_num = fixed_ptsnum_per_line
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
        
    def extracting_keyframes(self, data_infos, scene_tokens):
        self.token_data_dict = {
            item['token']: idx for idx, item in enumerate(data_infos)}
        keyframes = []
        for scene in scene_tokens:
            for x in scene:
                if ";" not in x and len(x)<33:
                    keyframes.append(self.token_data_dict[x])

        logging.info(f"[{self.__class__.__name__}] Got {len(scene_tokens)} "
                     f"continuous scenes. Extract {len(keyframes)} keyframes in 2Hz.")
        return keyframes

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        data_infos = data_infos[:: self.load_interval]
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]
        if self.use_2Hz:
            self.keyframes = self.extracting_keyframes(data_infos, data['scene_tokens'])
        return data_infos
    
    def __len__(self):
        return len(self.keyframes)
    
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
    
    def get_data_info(self, index):
        if self.use_2Hz:
            index = self.keyframes[index]
        frame = super().get_data_info(index)
        # add placeholder
        # for single-frame version, it's always eye matrix.
        frame['relative_pose'] = np.eye(4)

        return frame

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
        frame = self.get_data_info(index)
        if frame == None:
            return None
        
        self.pre_pipeline(frame)
        example = self.pipeline(frame)
        example = self.vectormap_pipeline(example, frame)
        example = self.project_on_views(example)

        example['height'] = example['img'].data.shape[-2]
        example['width'] = example['img'].data.shape[-1]
        example['fps'] = self.fps # placeholder
        example['num_frames'] = 1 # placeholder

        if self.filter_empty_gt and frame['is_key_frame'] and (
            example is None or ~(example["gt_labels_3d"]._data != -1).any()
        ):
             return None
 
        return example
    

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
    pixel_values = torch.stack([example["img"].data for example in examples])
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
        for k, v in ret_dict['bboxes_3d_data'].items():
            ret_dict['bboxes_3d_data'][k] = v.unsqueeze(1)

        ret_dict['captions'] = [ret_dict['captions']] # T is the first dim

    if drop_ori_imgs:
        ret_dict['pixel_values_shape'] = torch.IntTensor(list(ret_dict['pixel_values'].shape))
        ret_dict.pop('pixel_values')


    return ret_dict


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


def random_0_to_1(mask: np.array, num):
    assert mask.ndim == 1
    inds = np.where(mask == 0)[0].tolist()
    random.shuffle(inds)
    mask = np.copy(mask)
    mask[inds[:num]] = 1
    return mask


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
