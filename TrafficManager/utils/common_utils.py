import numpy as np
from shapely.geometry import MultiPoint, box
import copy
import cv2
import torch


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def boxes_to_corners_3d(boxes3d):
    """
        3 -------- 0
       /|         /|
      2 -------- 1 .
      | |        | |
      . 7 -------- 4
      |/         |/
      6 -------- 5
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d = np.array(boxes3d, dtype=np.float32)
    boxes3d = boxes3d.reshape(1, -1) if len(boxes3d.shape) == 1 else boxes3d
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]
    corners3d = corners3d.squeeze()

    return corners3d.numpy() if is_numpy else corners3d


def post_process_coords(corner_coords, imsize=(1600, 900)):
    """
    Get the intersection of the convex hull of the reprojected bbox corners and the image canvas, return None if no
    intersection.
    :param corner_coords: Corner coordinates of reprojected bounding box.
    :param imsize: Size of the image canvas.
    :return: Intersection of the convex hull of the 2D box corners and the image canvas.
    """
    if len(corner_coords) < 3:
        return None
    
    if len(np.unique(corner_coords, axis=0)) < 3 or (corner_coords[:, 0]>0).sum() < 3 or (corner_coords[:, 1]>0).sum() < 3 or (corner_coords[:,0] >= imsize[0]).sum() >=5 or (corner_coords[:,1] >= imsize[1]).sum() >= 5:
        x_max = max(min(corner_coords[:, 0].max(), imsize[0]), 0)
        x_min = max(corner_coords[:, 0].min(), 0)
        y_max = max(min(corner_coords[:, 1].max(), imsize[1]), 0)
        y_min = max(corner_coords[:, 1].min(), 0)
        return int(x_min), int(y_min), int(x_max), int(y_max)
    
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        try:
            intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])
        except:
            import pdb; pdb.set_trace()

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return int(min_x), int(min_y), int(max_x), int(max_y)
    else:
        return None


def project_box_to_image(box, transform):
    canvas = np.zeros((900, 1600, 3), dtype=np.uint8)

    corners = boxes_to_corners_3d(box)

    coords = np.concatenate(
        [corners, np.ones((corners.shape[0], 1))], axis=-1
    )
    transform = copy.deepcopy(transform.numpy()).reshape(4, 4)
    coords = coords @ transform.T

    mask = coords[:, 2] > 0
    coords = coords[mask]

    coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
    coords[:, 0] /= coords[:, 2]
    coords[:, 1] /= coords[:, 2]

    coords = coords[..., :2].astype(np.int32)
    coords_h = post_process_coords(coords)

    if coords_h is None:
        return None
    
    min_x, min_y, max_x, max_y = coords_h
    if max_x-min_x <= 0 or max_y-min_y <= 0:
        return None
    
    # cv2.rectangle(canvas, (min_x, min_y), (max_x, max_y), (0, 255, 0), 5)
    # canvas = cv2.resize(canvas, (400, 224), interpolation=cv2.INTER_NEAREST)
    # cv2.imwrite(f'./results/tmp.png', canvas)

    min_x /= 1600.0
    max_x /= 1600.0
    min_y /= 900.0
    max_y /= 900.0

    return (min_x, max_x, min_y, max_y)