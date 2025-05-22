import copy
import cv2
import numpy as np
import torch


def project_lines_on_bev(
        gt_lines_instance, 
        gt_labels_3d, 
        num_classes=3, 
        xbound=[-50.0, 50.0, 0.5],
        ybound=[-50.0, 50.0, 0.5]
        ):
    """
    Project 3D lines onto a 2D BEV (Bird's Eye View) canvas and draw them on different channels based on their labels.

    Args:
        gt_lines_instance (list): A list of 3D lines, where each line consists of multiple points.
        gt_labels_3d (list): A list of labels (integers) corresponding to each line.
        num_classes (int): Number of classes (channels) in the output canvas.
        xbound (list): BEV x-axis bounds and resolution [min, max, resolution].
        ybound (list): BEV y-axis bounds and resolution [min, max, resolution].

    Returns:
        canvas (np.ndarray): A canvas of shape (H, W, num_classes), where each channel corresponds to a label.
    """
    # Calculate canvas dimensions based on bounds and resolution
    canvas_h = int((ybound[1] - ybound[0]) / ybound[2])
    canvas_w = int((xbound[1] - xbound[0]) / xbound[2])

    # Initialize the canvas
    canvas = np.zeros((num_classes, canvas_h, canvas_w, 3), dtype=np.uint8)

    # Define normalization bounds
    bound = np.array([
        [xbound[0], ybound[0]],  # min bound (x, y)
        [xbound[1], ybound[1]]   # max bound (x, y)
    ])

    # Iterate over each line and its label
    for gt_line_instance, gt_label_3d in zip(gt_lines_instance, gt_labels_3d):
        # Convert 3D points to 2D BEV coordinates (ignore z-axis)
        pts = np.array(gt_line_instance)[:, :2]  # Extract (x, y) coordinates
        pts_canvas = ((pts - bound[0]) / (bound[1] - bound[0]) * np.array([canvas_w, canvas_h])).astype(int)

        # Draw points on the canvas
        for i in range(len(pts_canvas)):
            cv2.circle(canvas[int(gt_label_3d)], tuple(pts_canvas[i]), 1, (1, 0, 0), -1)
            if i > 0:
                cv2.line(canvas[int(gt_label_3d)], tuple(pts_canvas[i-1]), tuple(pts_canvas[i]), (1, 0, 0), 1)

    # Keep only the first color channel
    canvas = canvas[..., 0]
    canvas = np.transpose(canvas, (2, 1, 0))

    return canvas
    
def visualize_bev_hdmap(gt_lines_instance, gt_labels_3d, canvas_size, num_classes=3, bound=[-50.0, 50.0], drivable_mask=None, nuplan=None):
    canvas = np.zeros((num_classes, *canvas_size, 3), dtype=np.uint8)
    for gt_line_instance, gt_label_3d in zip(gt_lines_instance, gt_labels_3d):
        pts = np.array(gt_line_instance)
        for p in pts:
            pp = ((p - bound[0]) / (bound[1] - bound[0]) * canvas_size[0]).astype(int)
            cv2.circle(canvas[int(gt_label_3d)], tuple(pp), 1, (1,0,0), -1)

        for i in range(len(pts)-1):
            pp1 = ((pts[i] - bound[0]) / (bound[1] - bound[0]) * canvas_size[0]).astype(int)
            pp2 = ((pts[i+1] - bound[0]) / (bound[1] - bound[0]) * canvas_size[0]).astype(int)
            cv2.line(canvas[int(gt_label_3d)], tuple(pp1), tuple(pp2), (1,0,0), 1)
    canvas = canvas[..., 0]
    
    if drivable_mask is not None:
        drivable_mask = drivable_mask[None, ...]
        drivable_mask = np.transpose(drivable_mask, (0, 2, 1))
        canvas = np.concatenate([canvas, drivable_mask], 0)
    canvas = np.transpose(canvas, (2, 1, 0))    # H, W, C
    # if nuplan:
    #     tmp = canvas[..., 1:] * 255
    #     cv2.imwrite("proj_map_nuplan.jpg", tmp)
    return canvas

def project_lines_on_view(
    gt_lines_instance, 
    gt_labels_3d, 
    intrinsic, 
    extrinsic,
    num_classes=3,
    image_size=(224, 400)
):
    """
    Process ground truth lines, project them to image coordinates, and draw them on a canvas and optionally on an image.

    Args:
        gt_lines_instance (list): List of 3D line instances.
        gt_labels_3d (list): List of labels corresponding to each line.
        intrinsic (torch.Tensor): Camera intrinsic matrix, shape (3, 3).
        extrinsic (torch.Tensor): Camera extrinsic matrix, shape (4, 4).
        num_classes (int): Number of classes (channels) in the output canvas.
        drivable_mask (np.ndarray): Optional drivable mask to merge with the canvas (default is None).
        image_size (tuple): Size of the output canvas (default is (224, 400)).

    Returns:
        canvas (np.ndarray): Final canvas with drawn lines, shape (H, W, C).
    """
    z = 0
    canvas = np.zeros((num_classes, 900, 1600, 3), dtype=np.uint8)
    for gt_line_instance, gt_label_3d in zip(gt_lines_instance, gt_labels_3d):
        pts = torch.Tensor(gt_line_instance)
        pts = pts[:,[1,0]]
        pts[:,1] = -pts[:, 1]
        dummy_pts = torch.cat([pts, torch.ones((pts.shape[0], 1))*z], dim=-1)
        points_in_cam_cor = torch.matmul(extrinsic[:3, :3].T, (dummy_pts.T - extrinsic[:3, 3].reshape(3, -1)))
        points_in_cam_cor = points_in_cam_cor[:, points_in_cam_cor[2, :] > 0]
        if points_in_cam_cor.shape[1] > 1:
            points_on_image_cor = intrinsic[:3,:3] @ points_in_cam_cor
            points_on_image_cor = points_on_image_cor / (points_on_image_cor[-1, :].reshape(1, -1))
            points_on_image_cor = points_on_image_cor[:2, :].T
            points_on_image_cor = points_on_image_cor.int().numpy()
        else:
            points_on_image_cor = []

        for i in range(len(points_on_image_cor)-1):
            cv2.line(canvas[int(gt_label_3d)], tuple(points_on_image_cor[i]), tuple(points_on_image_cor[i+1]), (1,0,0), 4)

    canvas = canvas[..., 0]
    canvas = np.transpose(canvas, (1, 2, 0))
    canvas = cv2.resize(canvas, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)

    return canvas


def project_boxes_on_view(
    gt_bboxes_3d, 
    gt_labels_3d, 
    transform, 
    object_classes, 
    image_size=[224, 400]
):
    """
    Project 3D bounding boxes onto a 2D image and draw them on a canvas.

    Args:
        gt_bboxes_3d (np.ndarray): 3D bounding boxes, shape (N, 8, 3).
        gt_labels_3d (np.ndarray): Labels for the bounding boxes, shape (N,).
        transform (np.ndarray): Transformation matrix, shape (4, 4).
        object_classes (list): List of object class names.
        image_size (tuple): Size of the output canvas (default is (224, 400)).

    Returns:
        canvas (np.ndarray): Final canvas with drawn bounding boxes, shape (H, W, C).
    """
    canvas = np.zeros((len(object_classes), 900, 1600, 3), dtype=np.uint8)

    if gt_bboxes_3d is not None and len(gt_bboxes_3d) > 0:
        corners = gt_bboxes_3d.corners
        num_bboxes = corners.shape[0]

        # Convert to homogeneous coordinates
        coords = np.concatenate(
            [corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
        )
        transform = copy.deepcopy(transform.numpy()).reshape(4, 4)
        coords = coords @ transform.T
        coords = coords.reshape(-1, 8, 4)

        # Filter bounding boxes in front of the camera (Z > 0)
        indices = np.all(coords[..., 2] > 0, axis=1)
        coords = coords[indices]
        gt_labels_3d = gt_labels_3d[indices]

        # Sort bounding boxes by depth (minimum Z-coordinate)
        indices = np.argsort(-np.min(coords[..., 2], axis=1))
        coords = coords[indices]
        gt_labels_3d = gt_labels_3d[indices]

        # Project to 2D
        coords = coords.reshape(-1, 4)
        coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
        coords[:, 0] /= coords[:, 2]
        coords[:, 1] /= coords[:, 2]
        coords = coords[..., :2].reshape(-1, 8, 2)

        for index in range(coords.shape[0]):
            for pi, (start, end) in enumerate([
                (0, 3),
                (1, 2),
                (3, 2),
                (3, 7),
                (4, 7),
                (2, 6),
                (5, 6),
                (6, 7),
                ######
                (0, 1),
                (0, 4),
                (1, 5),
                (4, 5),
            ]):
                try:
                    cv2.line(
                        canvas[int(gt_labels_3d[index])],
                        coords[index, start].astype(np.int),
                        coords[index, end].astype(np.int),
                        (1, 0, 0) if pi < 8 else (2, 0, 0),
                        4,
                        cv2.LINE_AA,
                    )
                except:
                    pass

    canvas = canvas[..., 0]
    canvas = np.transpose(canvas, (1, 2, 0))
    canvas = cv2.resize(canvas, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)

    return canvas
    
