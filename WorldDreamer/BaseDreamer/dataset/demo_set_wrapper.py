import numpy as np
import torch
import copy
from mmcv.parallel.data_container import DataContainer
from mmdet3d.core.bbox.structures import LiDARInstance3DBoxes, Box3DMode
from .map_utils import visualize_bev_hdmap, project_map_to_image, project_box_to_image


class ApiSetWrapper(torch.utils.data.DataLoader):
    def __init__(self, data) -> None:
        if isinstance(data, str):
            self.dataset = torch.load(data)
        else:
            self.dataset = [data]
        self.data_template = torch.load('data/demo_data/data_template.pth')
        self.object_classes = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        mmdet3d_format = {}

        # from data template
        lidar2camera = self.data_template['lidar2camera']
        camera_intrinsics = self.data_template['camera_intrinsics']
        camera2ego = self.data_template['camera2ego']

        mmdet3d_format['lidar2camera'] = DataContainer(lidar2camera)
        mmdet3d_format['camera_intrinsics'] = DataContainer(camera_intrinsics)
        mmdet3d_format['img_aug_matrix'] = DataContainer(self.data_template['img_aug_matrix'])
        # from data
        mmdet3d_format['img'] = DataContainer(torch.zeros(6,3,1,1))
        mmdet3d_format['gt_labels_3d'] = DataContainer(torch.tensor(data['gt_labels_3d']))
        mmdet3d_format['metas'] = DataContainer(copy.deepcopy(data['metas']))
        mmdet3d_format['metas'].data['ego_pos'] = torch.tensor(mmdet3d_format['metas'].data['ego_pos'])
        mmdet3d_format['relative_pose'] = DataContainer(torch.tensor(data['relative_pose']), cpu_only=False)

        # special class
        if torch.tensor(data['gt_bboxes_3d']).size(0) == 0:
            gt_bboxes_3d = torch.zeros(0, 9)
        else:
            gt_bboxes_3d = torch.tensor(data['gt_bboxes_3d'])

        mmdet3d_format['gt_bboxes_3d'] = DataContainer(LiDARInstance3DBoxes(
                gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1],
                origin=(0.5, 0.5, 0)).convert_to(Box3DMode.LIDAR))
        
        mmdet3d_format['ref_images'] = DataContainer(torch.ones(6,3,224,400), cpu_only=False)
        gt_vecs_label = data['gt_vecs_label']
        gt_lines_instance = data['gt_lines_instance']
        drivable_mask = torch.Tensor(data['drivable_mask'])
        bev_map = visualize_bev_hdmap(gt_lines_instance, gt_vecs_label, [200, 200], drivable_mask=drivable_mask)
        bev_map = bev_map.transpose(2, 0, 1)
        mmdet3d_format['bev_hdmap'] = DataContainer(torch.tensor(bev_map), cpu_only=False)
        mmdet3d_format['gt_masks_bev'] = torch.tensor(bev_map)

        layout_canvas = []
        for i in range(len(lidar2camera)):
            lidar2image = camera_intrinsics[i] @ lidar2camera[i] 
            map_canvas = project_map_to_image(gt_lines_instance, gt_vecs_label, camera_intrinsics[i], camera2ego[i], drivable_mask=drivable_mask)

            gt_bboxes= LiDARInstance3DBoxes(gt_bboxes_3d, 
                                            box_dim=gt_bboxes_3d.shape[-1],
                                            origin=(0.5, 0.5, 0)).convert_to(Box3DMode.LIDAR)
            box_canvas = project_box_to_image(gt_bboxes, torch.tensor(data['gt_labels_3d']), lidar2image, object_classes=self.object_classes)

            layout_canvas.append(np.concatenate([map_canvas, box_canvas], axis=-1))

        layout_canvas = np.stack(layout_canvas, axis=0)
        layout_canvas = np.transpose(layout_canvas, (0, 3, 1, 2))    # 6, N_channel, H, W
        mmdet3d_format['layout_canvas'] = torch.from_numpy(layout_canvas)

        # recompute
        camera2lidar = torch.eye(4, dtype=lidar2camera.dtype)
        camera2lidar = torch.stack([camera2lidar] * len(lidar2camera))
        camera2lidar[:, :3, :3] = lidar2camera[:, :3, :3].transpose(1, 2)
        camera2lidar[:, :3, 3:] = torch.bmm(-camera2lidar[:, :3, :3], lidar2camera[:, :3, 3:])
        mmdet3d_format['camera2lidar'] = DataContainer(camera2lidar)
        mmdet3d_format['lidar2image'] = DataContainer(
            torch.bmm(camera_intrinsics, lidar2camera)
        )

        # fmt: on
        return mmdet3d_format

    def __len__(self):
        return len(self.dataset)
    

class ListSetWrapper(torch.utils.data.DataLoader):
    def __init__(self, dataset, list) -> None:
        self.dataset = dataset
        self.list = list

    def __getitem__(self, idx):
        return self.dataset[self.list[idx]]

    def __len__(self):
        return len(self.list)


class FileSetWrapper(torch.utils.data.DataLoader):
    def __init__(self, file_name) -> None:
        print(file_name)
        self.dataset = torch.load(file_name)
        self.data_template = torch.load('data/demo_data/data_template.pth')

    def __getitem__(self, idx):
        data = self.dataset[idx]
        mmdet3d_format = {}

        # from data template
        lidar2camera = self.data_template['lidar2camera']
        camera_intrinsics = self.data_template['camera_intrinsics']
        mmdet3d_format['lidar2camera'] = DataContainer(lidar2camera)
        mmdet3d_format['camera_intrinsics'] = DataContainer(camera_intrinsics)
        mmdet3d_format['img_aug_matrix'] = DataContainer(self.data_template['img_aug_matrix'])

        # from data
        data['metas']['description'] = 'Narrow road, following bus, nature'
        mmdet3d_format['img'] = DataContainer(data['img'])
        mmdet3d_format['gt_labels_3d'] = DataContainer(data['gt_labels_3d'])
        mmdet3d_format['metas'] = DataContainer(data['metas'])
        mmdet3d_format['layout_canvas'] = DataContainer(data['layout_canvas'], cpu_only=False)
        mmdet3d_format['bev_hdmap'] = DataContainer(data['bev_hdmap'], cpu_only=False)
        mmdet3d_format['gt_masks_bev'] = data['bev_hdmap']
        mmdet3d_format['relative_pose'] = DataContainer(data['relative_pose'], cpu_only=False)
        
        # special class
        gt_bboxes_3d = data['gt_bboxes_3d']
        mmdet3d_format['gt_bboxes_3d'] = DataContainer(LiDARInstance3DBoxes(
                gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1],
                origin=(0.5, 0.5, 0)).convert_to(Box3DMode.LIDAR))
        
        mmdet3d_format['ref_images'] = DataContainer(torch.zeros(6,3,224,400), cpu_only=False)

        # recompute
        camera2lidar = torch.eye(4, dtype=lidar2camera.dtype)
        camera2lidar = torch.stack([camera2lidar] * len(lidar2camera))
        camera2lidar[:, :3, :3] = lidar2camera[:, :3, :3].transpose(1, 2)
        camera2lidar[:, :3, 3:] = torch.bmm(-camera2lidar[:, :3, :3], lidar2camera[:, :3, 3:])
        mmdet3d_format['camera2lidar'] = DataContainer(camera2lidar)
        mmdet3d_format['lidar2image'] = DataContainer(
            torch.bmm(camera_intrinsics, lidar2camera)
        )

        # fmt: on
        return mmdet3d_format

    def __len__(self):
        return len(self.dataset)
