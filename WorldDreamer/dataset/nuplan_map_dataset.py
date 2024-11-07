import logging
import random

import cv2
import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmdet3d.datasets import NuPlanDataset
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import to_tensor

from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB
from nuplan.database.maps_db.map_api import NuPlanMapWrapper
from nuscenes.eval.common.utils import Quaternion

from .map_utils import (
    LiDARInstanceLines,
    VectorizedLocalMap,
    project_box_to_image,
    project_map_to_image_nuplan,
    visualize_bev_hdmap,
)

rotation_z_neg90 = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
MAP_LOCATIONS = {
    "sg-one-north",
    "us-ma-boston",
    "us-nv-las-vegas-strip",
    "us-pa-pittsburgh-hazelwood",
}


@DATASETS.register_module()
class NuPlanMapDataset(NuPlanDataset):
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
        map_bound=None,
        video_length=None,
        start_on_keyframe=True,
        start_on_firstframe=False,
        fixed_ptsnum_per_line=-1,
        padding_value=-10000,
        fps=12
    ) -> None:
        self.video_length = video_length
        self.start_on_keyframe = start_on_keyframe
        self.start_on_firstframe = start_on_firstframe
        self.fps = fps
        super().__init__(
            ann_file,
            pipeline,
            dataset_root,
            object_classes,
            map_classes,
            load_interval,
            with_velocity,
            modality,
            box_type_3d,
            filter_empty_gt,
            test_mode,
            eval_version,
            use_valid_flag,
            force_all_boxes,
        )
        self.padding_value = padding_value
        self.fixed_num = fixed_ptsnum_per_line
        self.object_classes = object_classes
        self.map_classes = map_classes
        xbound = map_bound['x']
        ybound = map_bound['y']
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)

        self.mapdb = GPKGMapsDB(
            map_version="nuplan-maps-v1.0", map_root=dataset_root + "/maps"
        )
        self.maps = {}
        for location in MAP_LOCATIONS:
            map_api = NuPlanMapWrapper(self.mapdb, map_name=location)
            self.maps[location] = map_api

        self.vector_map = VectorizedLocalMap(
            dataset_root,
            patch_size=self.patch_size,
            map_classes=["divider", "ped_crossing", "boundary"],
            fixed_ptsnum_per_line=fixed_ptsnum_per_line,
            padding_value=padding_value,
            nuplan_map_api = self.maps
        )
        

        self.nuplan = True

    def __len__(self):
        return len(self.clip_infos)

    def build_clips(self, data_infos, scene_tokens, start_on_keyframe=False):
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
            item["token"]: idx for idx, item in enumerate(data_infos)
        }
        all_clips = []
        for scene in scene_tokens:
            for start in range(len(scene) - self.video_length + 1):
                if self.start_on_keyframe and ";" in scene[start]:
                    continue  # this is not a keyframe
                if self.start_on_keyframe and len(scene[start]) >= 33:
                    continue  # this is not a keyframe
                clip = [
                    self.token_data_dict[token]
                    for token in scene[start : start + self.video_length]
                ]
                all_clips.append(clip)
                if self.start_on_firstframe:
                    break
        logging.info(
            f"[{self.__class__.__name__}] Got {len(scene_tokens)} "
            f"continuous scenes. Cut into {self.video_length}-clip, "
            f"which has {len(all_clips)} in total."
        )
        return all_clips

    def build_clips_2hz(self, data_infos, scene_tokens):
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
            item["token"]: idx for idx, item in enumerate(data_infos)
        }
        all_clips = []

        for scene in scene_tokens:
            keyframes_list = []
            # breakpoint()

            for idx in range(len(scene)):
                if len(scene[idx]) == 16:  # 32
                    keyframes_list.append(scene[idx])
            # breakpoint()
            data_infos[self.token_data_dict[keyframes_list[0]]]["is_first_frame"] = True
            # breakpoint()
            all_clips.append(
                [
                    self.token_data_dict[keyframes_list[0]],
                    self.token_data_dict[keyframes_list[0]],
                ]
            )
            # print('keyframe id:', len(all_clips)-1)
            for idx in range(len(keyframes_list) - 1):
                clip = [
                    self.token_data_dict[keyframes_list[idx]],
                    self.token_data_dict[keyframes_list[idx + 1]],
                ]
                all_clips.append(clip)
        logging.info(
            f"[{self.__class__.__name__}] Got {len(scene_tokens)} "
            f"continuous scenes. Cut into 2-clip, only keyframes, "
            f"which has {len(all_clips)} in total."
        )
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
        if self.fps == 2 and self.test_mode:
            self.clip_infos = self.build_clips_2hz(data_infos, data["scene_tokens"])
        else:
            self.clip_infos = self.build_clips(data_infos, data["scene_tokens"])
        return data_infos

    def vectormap_pipeline(self, example, input_dict):
        """
        Process vector map data for input example, using transformation matrices and
        generating annotations.
        """

        lidar2ego = input_dict["lidar2ego"]
        ego2global = input_dict["ego2global"]
        lidar2global = ego2global @ lidar2ego
        lidar2global = rotation_z_neg90 @ lidar2global
        lidar2global_translation = list(lidar2global[:3, 3])
        lidar2global_rotation = Quaternion(matrix=lidar2global)

        anns_results = self.vector_map.gen_vectorized_samples_nuplan(
            self.maps[input_dict["location"]],
            lidar2global_translation,
            lidar2global_rotation,
        )
        gt_vecs_label = to_tensor(anns_results["gt_vecs_label"])
        if isinstance(anns_results["gt_vecs_pts_loc"], LiDARInstanceLines):
            gt_vecs_pts_loc = anns_results["gt_vecs_pts_loc"]
            gt_lines_instance = gt_vecs_pts_loc.instance_list
            gt_map_pts = [np.array(list(line.coords)) for line in gt_lines_instance]
        example["gt_vecs_label"] = DC(gt_vecs_label, cpu_only=False)
        example["gt_vecs_pts_loc"] = DC(gt_map_pts, cpu_only=True)

        # Visualizing BEV map
        drivable_mask = (
            example["gt_masks_bev"][0, ...] + example["gt_masks_bev"][-1, ...]
        ).astype(bool)

        bev_map = visualize_bev_hdmap(
            example["gt_vecs_pts_loc"].data,
            example["gt_vecs_label"].data,
            self.canvas_size,
            # vis_format="polyline_pts",
            drivable_mask=drivable_mask,
            nuplan=True,
        )
        bev_map = bev_map.transpose(2, 0, 1)
        example["bev_hdmap"] = DC(to_tensor(bev_map.copy()), cpu_only=False)

        return example

    def project_bev2img(self, example):
        lidar2image = example["lidar2image"].data
        img_file_list = example["metas"].data["filename"]
        camera2ego = example["camera2ego"].data
        camera_intrinsics = example["camera_intrinsics"].data

        rotation_z_neg90 = torch.Tensor(
            [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        camera2ego = rotation_z_neg90 @ camera2ego

        drivable_mask = (
            example["gt_masks_bev"][0, ...] + example["gt_masks_bev"][-1, ...]
        )  # 200*200
        drivable_mask = drivable_mask.astype(bool)
        layout_canvas = []
        for i in range(len(lidar2image)):
            image = cv2.imread(img_file_list[i])
            map_canvas = project_map_to_image_nuplan(
                example["gt_vecs_pts_loc"].data,
                example["gt_vecs_label"].data,
                camera_intrinsics[i],
                camera2ego[i],
                image=image,
                drivable_mask=drivable_mask,
            )

            box_canvas = project_box_to_image(
                example["gt_bboxes_3d"].data,
                example["gt_labels_3d"].data,
                lidar2image[i],
                object_classes=self.object_classes,
            )

            layout_canvas.append(np.concatenate([map_canvas, box_canvas], axis=-1))

        layout_canvas = np.stack(layout_canvas, axis=0)
        layout_canvas = np.transpose(layout_canvas, (0, 3, 1, 2))  # 6, N_channel, H, W
        example["layout_canvas"] = DC(to_tensor(layout_canvas), cpu_only=False)
        return example

    def process_ref_image(self, example, random_index):
        random_sample_dict = self.get_data_info(random_index)
        self.pre_pipeline(random_sample_dict)
        random_sample_dict = self.pipeline(random_sample_dict)
        example["ref_images"] = random_sample_dict["img"]
        relative_pose = torch.matmul(
            torch.inverse(example["ego2global"].data),
            random_sample_dict["ego2global"].data,
        )

        example["relative_pose"] = DC(relative_pose, cpu_only=False)
        example["ref_bboxes_3d"] = random_sample_dict["gt_bboxes_3d"]
        example["ref_labels_3d"] = random_sample_dict["gt_labels_3d"]
        example["obj_ids"] = example["metas"].data["obj_ids"]
        example["ref_obj_ids"] = random_sample_dict["metas"].data["obj_ids"]

        return example

    def prepare_train_data(self, index):
        """This is called by `__getitem__`
        index is the index of the clip_infos
        """
        clips = self.clip_infos[index]
        curr_index = clips[-1]
        random_index = random.choice(clips)
        input_dict = self.get_data_info(curr_index)
        if input_dict is None:
            return None

        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        example = self.vectormap_pipeline(example, input_dict)
        example['index'] = index
        example = self.project_bev2img(example)

        # Timing process_ref_image step
        example = self.process_ref_image(example, random_index)

        # Check for empty ground truths and filter if necessary
        if (
            self.filter_empty_gt
            and input_dict["is_key_frame"]
            and (example is None or ~(example["gt_labels_3d"]._data != -1).any())
        ):
            return None

        return example

    def prepare_test_data(self, index):
        """This is called by `__getitem__`
        index is the index of the clip_infos
        """
        clips = self.clip_infos[index]
        curr_index = clips[-1]
        random_index = clips[-2]
        # curr_index = 0
        input_dict = self.get_data_info(curr_index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        # example = self.vectormap_pipeline(example, input_dict)
        example = self.vectormap_pipeline(example, input_dict)
        example['index'] = index
        example = self.project_bev2img(example)
        # process the random ref image
        example = self.process_ref_image(example, random_index)

        if (
            self.filter_empty_gt
            and input_dict["is_key_frame"]
            and (example is None or ~(example["gt_labels_3d"]._data != -1).any())
        ):
            return None

        return example
