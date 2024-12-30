import logging
import numpy as np
import random
import copy
import mmcv
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from mmdet.datasets.pipelines import to_tensor
from nuscenes.eval.common.utils import Quaternion, quaternion_yaw

from .map_utils import LiDARInstanceLines, VectorizedLocalMap, visualize_bev_hdmap, project_map_to_image, project_box_to_image


@DATASETS.register_module()
class NuScenesMapDataset(NuScenesDataset):
    def __init__(
        self,
        ann_file,
        pipeline=None,
        dataset_root=None,
        object_classes=None,
        map_classes=None,
        load_interval=1,
        with_velocity=True,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        eval_version="detection_cvpr_2019",
        use_valid_flag=False,
        force_all_boxes=False,
        video_length=None,
        ref_length=None,
        candidate_length=None,
        start_on_keyframe=True,
        fixed_ptsnum_per_line=-1,
        padding_value=-10000,
        temporal=True,
        use_2Hz=False
    ) -> None:
        self.video_length = video_length
        self.ref_length = ref_length
        self.candidate_length = candidate_length
        self.start_on_keyframe = start_on_keyframe
        self.temporal = temporal
        self.use_2Hz = use_2Hz
        super().__init__(
            ann_file, pipeline, dataset_root, object_classes, map_classes,
            load_interval, with_velocity, modality, box_type_3d,
            filter_empty_gt, test_mode, eval_version, use_valid_flag,
            force_all_boxes)
        self.padding_value = padding_value
        self.fixed_num = fixed_ptsnum_per_line
        self.object_classes = object_classes
        self.map_classes = map_classes
        xbound = [-50.0, 50.0, 0.5]
        ybound = [-50.0, 50.0, 0.5]
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.vector_map = VectorizedLocalMap(dataset_root, 
                            patch_size=self.patch_size, map_classes=['divider', 'ped_crossing','boundary'], 
                            fixed_ptsnum_per_line=fixed_ptsnum_per_line,
                            padding_value=padding_value)


    def __len__(self):
        return len(self.clip_infos)

    def build_2Hz_clips(self, data_infos, scene_tokens):
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
        assert self.temporal
        self.scene_key_frame = {}
        num_clip = 0
        for si, scene in enumerate(scene_tokens):
            scene_2Hz = []
            for x in scene:
                if ";" not in x and len(x)<33:
                    scene_2Hz.append(x)
            scene = scene_2Hz

            for start in range(0, len(scene), self.video_length):
                clip = [self.token_data_dict[token]
                        for token in scene[start: min(start + self.video_length, len(scene))]]
                if len(clip) < self.video_length:
                    clip += [clip[-1]]*(self.video_length-len(clip))

                ref_idx = [0, 0] if start == 0 else sorted(random.sample(range(max(start-self.candidate_length, 0), start), self.ref_length))
                ref = [self.token_data_dict[scene[idx]] for idx in ref_idx]
                clip = ref + clip

                if f'{si}' not in self.scene_key_frame:
                    self.scene_key_frame[f'{si}'] = []
                self.scene_key_frame[f'{si}'].append(num_clip)

                num_clip += 1

                all_clips.append(clip)
        logging.info(f"[{self.__class__.__name__}] Got {len(scene_tokens)} "
                     f"continuous scenes. Cut into {self.video_length + self.ref_length}-clip ({self.ref_length}/{self.video_length}), "
                     f"which has {len(all_clips)} in total.")
        return all_clips

    def build_clips(self, data_infos, scene_tokens):
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
        if not self.temporal:
            self.ref_length = 0
            self.video_length = 1
        self.scene_key_frame = {}
        num_clip = 0
        for si, scene in enumerate(scene_tokens):
            for start in range(0, len(scene) - self.video_length + 1):
                if self.start_on_keyframe and ";" in scene[start]:
                    continue  # this is not a keyframe
                if self.start_on_keyframe and len(scene[start]) >= 33:
                    continue  # this is not a keyframe
                clip = [self.token_data_dict[token]
                        for token in scene[start: start + self.video_length]]

                if self.temporal:
                    ref_idx = [0, 0] if start == 0 else sorted(random.sample(range(max(start-self.candidate_length, 0), start), self.ref_length))
                    ref = [self.token_data_dict[scene[idx]] for idx in ref_idx]
                    clip = ref + clip

                if not (";" in scene[start + self.video_length-1] or len(scene[start + self.video_length-1]) >= 33):
                    if f'{si}' not in self.scene_key_frame:
                        self.scene_key_frame[f'{si}'] = []
                    self.scene_key_frame[f'{si}'].append(num_clip)

                num_clip += 1

                all_clips.append(clip)
        logging.info(f"[{self.__class__.__name__}] Got {len(scene_tokens)} "
                     f"continuous scenes. Cut into {self.video_length + self.ref_length}-clip ({self.ref_length}/{self.video_length}), "
                     f"which has {len(all_clips)} in total.")
        return all_clips

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
        self.clip_infos = self.build_clips(data_infos, data['scene_tokens']) if not self.use_2Hz else self.build_2Hz_clips(data_infos, data['scene_tokens'])
        return data_infos

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

        drivable_mask = example['gt_masks_bev'][0, ...] + example['gt_masks_bev'][-1, ...]    # 200*200
        drivable_mask = drivable_mask.astype(bool)
        bev_map = visualize_bev_hdmap(example['gt_vecs_pts_loc'].data, example['gt_vecs_label'].data, self.canvas_size, vis_format='polyline_pts', drivable_mask=drivable_mask)

        bev_map = bev_map.transpose(2, 0, 1)    # N_channel, H, W

        example['bev_hdmap'] = DC(to_tensor(bev_map.copy()), cpu_only=False)
        
        return example

    def project_bev2img(self, example):
        image_size = example['img'].data.shape[-2:]
        lidar2image = example['lidar2image'].data
        camera2ego = example['camera2ego'].data
        camera_intrinsics = example['camera_intrinsics'].data
        camera_channels = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"]

        drivable_mask = example['gt_masks_bev'][0, ...] + example['gt_masks_bev'][-1, ...]    # 200*200
        drivable_mask = drivable_mask.astype(bool)
        layout_canvas = []
        for i in range(len(lidar2image)):
            map_canvas = project_map_to_image(example['gt_vecs_pts_loc'].data, example['gt_vecs_label'].data, camera_intrinsics[i], camera2ego[i], drivable_mask=drivable_mask, image_size=image_size)
            box_canvas = project_box_to_image(example['gt_bboxes_3d'].data, example['gt_labels_3d'].data, lidar2image[i], object_classes=self.object_classes, image_size=image_size)
            layout_canvas.append(np.concatenate([map_canvas, box_canvas], axis=-1))

        layout_canvas = np.stack(layout_canvas, axis=0)
        layout_canvas = np.transpose(layout_canvas, (0, 3, 1, 2))    # 6, N_channel, H, W
        example['layout_canvas'] = DC(to_tensor(layout_canvas), cpu_only=False)
        return example
    
    def get_can_bus(self, frame):
        ego2global = frame['ego2global']
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
    
    def get_data_info(self, index):
        """We should sample from clip_infos
        """
        clip = self.clip_infos[index]
        frames = []
        for frame in clip:
            frame_info = super().get_data_info(frame)
            frames.append(frame_info)

        return frames

    def prepare_train_data(self, index):
        """This is called by `__getitem__`
            index is the index of the clip_infos
        """
        frames = self.get_data_info(index)
        if None in frames:
            return None
        examples = []
        for i, frame in enumerate(frames):
            self.pre_pipeline(frame)
            example = self.pipeline(frame)
            example = self.vectormap_pipeline(example, frame)
            example = self.project_bev2img(example)

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
            example['can_bus'] = DC(to_tensor(can_bus), cpu_only=False)


            if self.filter_empty_gt and frame['is_key_frame'] and (
                example is None or ~(example["gt_labels_3d"]._data != -1).any()
            ):
                return None
            examples.append(example)

        if not self.temporal:
            return examples[0]
 
        return examples

