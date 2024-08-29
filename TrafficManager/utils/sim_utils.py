import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch

from TrafficManager.LimSim.utils.trajectory import State, Trajectory

# Add LimSim to sys.path
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "LimSim"))  # noqa
from TrafficManager.utils.map_utils import (
    LiDARInstanceLines,
    VectorizedLocalMap,
    to_tensor,
    visualize_bev_hdmap,
)


def limsim2diffusion(
    vehicles,
    data_template,
    vectorized_map: VectorizedLocalMap,
    map_name,
    agent_command=2,
    last_pose=torch.eye(4),
    drivable_mask=np.ones((200, 200), dtype=np.uint8),
    accel=[0, 0, 9.81],
    rotation_rate=[0, 0, 0],
    vel=[5, 0, 0],
    gen_location="singapore-onenorth",
    gen_prompts="daytime, cloudy, downtown, gray buildings, white cars",
):
    VEH_LENGTH = 4.7
    VEH_WIDTH = 1.6
    VEH_HEIGHT = 1.4

    ego_vehicle = vehicles["egoCar"]
    ego_x, ego_y, ego_yaw = (
        ego_vehicle["xQ"][-1],
        ego_vehicle["yQ"][-1],
        ego_vehicle["yawQ"][-1],
    )
    ego_yaw_deg = ego_vehicle["yawQ"][-1] * 180 / np.pi

    bbox_list = []
    label_list = []

    def transform(pos, origin):
        # pos是要变换的坐标和朝向，origin是新的原点的坐标和朝向
        # 返回变换后的坐标和朝向
        x, y, yaw = pos
        x0, y0, yaw0 = origin
        # 计算相对于新原点的位移和角度
        dx = x - x0
        dy = y - y0
        dtheta = yaw - yaw0
        # 计算新坐标系下的坐标和朝向
        x_new = dx * np.cos(yaw0) + dy * np.sin(yaw0)
        y_new = -dx * np.sin(yaw0) + dy * np.cos(yaw0)
        yaw_new = dtheta
        return x_new, y_new, yaw_new

    def plot_vehicle(pos, color):
        # pos是车辆的中心坐标和朝向，color是车辆的颜色
        # 绘制车辆的矩形
        x, y, yaw = pos
        # 计算车辆的四个顶点的坐标
        x1 = x + VEH_LENGTH / 2 * np.cos(yaw) - VEH_WIDTH / 2 * np.sin(yaw)
        y1 = y + VEH_LENGTH / 2 * np.sin(yaw) + VEH_WIDTH / 2 * np.cos(yaw)
        x2 = x + VEH_LENGTH / 2 * np.cos(yaw) + VEH_WIDTH / 2 * np.sin(yaw)
        y2 = y + VEH_LENGTH / 2 * np.sin(yaw) - VEH_WIDTH / 2 * np.cos(yaw)
        x3 = x - VEH_LENGTH / 2 * np.cos(yaw) + VEH_WIDTH / 2 * np.sin(yaw)
        y3 = y - VEH_LENGTH / 2 * np.sin(yaw) - VEH_WIDTH / 2 * np.cos(yaw)
        x4 = x - VEH_LENGTH / 2 * np.cos(yaw) - VEH_WIDTH / 2 * np.sin(yaw)
        y4 = y - VEH_LENGTH / 2 * np.sin(yaw) + VEH_WIDTH / 2 * np.cos(yaw)
        # 绘制矩形
        plt.fill([x1, x2, x3, x4], [y1, y2, y3, y4], color=color)

    # for sur_veh in vehicles['carInAoI']:
    #     sur_x, sur_y, sur_yaw = sur_veh['xQ'][-1], sur_veh['yQ'][-1], sur_veh['yawQ'][-1]
    #     tran_x, tran_y, tran_yaw = transform((sur_x, sur_y, sur_yaw), (ego_x, ego_y, ego_yaw - np.pi/2))
    #     print(sur_veh['id'], tran_x, tran_y, tran_yaw,  tran_yaw + np.pi/2)
    #     bbox_list.append([tran_x, tran_y, -0.8, VEH_WIDTH, VEH_LENGTH, VEH_HEIGHT, tran_yaw + np.pi/2, 0, 0])

    #     plot_vehicle((tran_x, tran_y, tran_yaw), color='blue')
    #     label_list.append(0) # 0 for vehicle

    # plot_vehicle(transform((ego_x, ego_y, ego_yaw), (ego_x, ego_y, ego_yaw - np.pi/2)), color='red')
    # plt.axis('equal')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Transformed vehicles')
    # # 显示图像
    # plt.show()

    for sur_veh in vehicles["carInAoI"]:
        sur_x, sur_y, sur_yaw = (
            sur_veh["xQ"][-1],
            sur_veh["yQ"][-1],
            sur_veh["yawQ"][-1],
        )
        tran_x, tran_y, tran_yaw = transform(
            (sur_x, sur_y, sur_yaw), (ego_x, ego_y, ego_yaw)
        )
        tran_x, tran_y, tran_yaw = transform(
            (tran_x, tran_y, tran_yaw), (0, 0, -np.pi / 2)
        )
        # print(sur_veh['id'], tran_x, tran_y, tran_yaw,  tran_yaw+np.pi/2)
        bbox_list.append(
            [
                tran_x,
                tran_y,
                -0.8,
                VEH_WIDTH,
                VEH_LENGTH,
                VEH_HEIGHT,
                -(tran_yaw + np.pi / 2),
                0,
                0,
            ]
        )

        # plot_vehicle((tran_x, tran_y, tran_yaw), color='blue')
        label_list.append(0)  # 0 for vehicle

    # tran_x, tran_y, tran_yaw = transform((ego_x, ego_y, ego_yaw), (ego_x, ego_y, ego_yaw))
    # tran_x, tran_y, tran_yaw = transform((tran_x, tran_y, tran_yaw), (0, 0, -np.pi/2))
    # plot_vehicle((tran_x, tran_y, tran_yaw), color='red')
    # plt.axis('equal')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Transformed vehicles')
    # # 显示图像
    # plt.show()

    send_data = {}
    # ------------ meta ------------ #
    send_data["metas"] = data_template["metas"]
    send_data["metas"][
        "location"
    ] = gen_location  #'singapore-onenorth' #'singapore-hollandvillage' #  'boston-seaport' # 'singapore-hollandvillage' for night #map_name #'boston-seaport' #map_name
    send_data["metas"][
        "description"
    ] = gen_prompts  #'daytime, cloudy, downtown, gray buildings, white cars' #'daytime, cloudy, nature, green trees, black cars' # 'night, clear, suburban, streetlights' # 'daytime, rainy, suburban, low buildings, wet surface'  # 'daytime, sunny, downtown, red buildings, trees, black cars' #'night, clear, downtown, streetlights'#'daytime, cloudy, suburban, red buildings, black cars' #'night, clear, suburban, streetlights' # 'daytime, rainy, downtown, red buildings, black cars' #'rain, buildings, parked bicycles, many vehicles'
    send_data["metas"]["ego_pos"] = torch.Tensor(
        [
            [np.cos(ego_yaw), -np.sin(ego_yaw), 0, ego_x],
            [np.sin(ego_yaw), np.cos(ego_yaw), 0, ego_y],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    send_data["metas"]["accel"] = accel
    send_data["metas"]["rotation_rate"] = rotation_rate
    send_data["metas"]["vel"] = vel

    # ------------ bboxes ------------ #
    if len(bbox_list) != 0:
        gt_bboxes_3d = torch.tensor(bbox_list)
        send_data["gt_bboxes_3d"] = gt_bboxes_3d
        send_data["gt_labels_3d"] = torch.tensor(label_list)
    else:
        gt_bboxes_3d = torch.empty(0, 9)
        send_data["gt_bboxes_3d"] = gt_bboxes_3d
        send_data["gt_labels_3d"] = torch.empty(0)

    # ------------ HDMap ------------ #
    anns_results = vectorized_map.gen_vectorized_samples(
        map_name, [ego_x, ego_y], np.deg2rad(ego_yaw_deg - 90)
    )

    gt_vecs_label = to_tensor(anns_results["gt_vecs_label"])
    if isinstance(anns_results["gt_vecs_pts_loc"], LiDARInstanceLines):
        gt_vecs_pts_loc = anns_results["gt_vecs_pts_loc"]
    else:
        gt_vecs_pts_loc = to_tensor(anns_results["gt_vecs_pts_loc"])
        try:
            gt_vecs_pts_loc = gt_vecs_pts_loc.flatten(1).to(dtype=torch.float32)
        except:
            gt_vecs_pts_loc = gt_vecs_pts_loc
    send_data["gt_vecs_label"] = gt_vecs_label
    gt_lines_instance = gt_vecs_pts_loc.instance_list
    gt_map_pts = []
    for i in range(len(gt_lines_instance)):
        pts = np.array(list(gt_lines_instance[i].coords))
        gt_map_pts.append(pts.tolist())
    send_data["gt_lines_instance"] = gt_map_pts

    # ---------------ref pose------------------#
    send_data["relative_pose"] = torch.matmul(
        torch.inverse(send_data["metas"]["ego_pos"]), last_pose
    )

    # ---------------drivable mask- -----------------#
    send_data["drivable_mask"] = drivable_mask

    # ---------------Agent command-----------------#
    send_data["agent_command"] = agent_command

    return send_data


def normalize_angle(angle: float) -> float:
    """Normalize angle to be within the interval [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def transform_to_ego_frame(curr_state: State, target_state: State):
    ego_yaw = curr_state.yaw
    R = np.array(
        [[np.cos(-ego_yaw), -np.sin(-ego_yaw)], [np.sin(-ego_yaw), np.cos(-ego_yaw)]]
    )
    translated = np.array(
        [target_state.x - curr_state.x, target_state.y - curr_state.y]
    )
    rotated = R @ translated
    yaw_adjusted = normalize_angle(target_state.yaw - curr_state.yaw)
    return rotated[0], rotated[1], yaw_adjusted


def interpolate_traj(ego_vehicle, path_points, Ti_path=0.5) -> Trajectory:
    ego_x, ego_y, ego_yaw = (
        ego_vehicle["xQ"][-1],
        ego_vehicle["yQ"][-1],
        ego_vehicle["yawQ"][-1],
    )
    ego_vel, ego_acc = ego_vehicle["speedQ"][-1], ego_vehicle["accelQ"][-1]
    Ti_traj = 0.1

    global_points = [
        (
            ego_x
            + px * math.cos(ego_yaw - math.pi / 2)
            - py * math.sin(ego_yaw - math.pi / 2),
            ego_y
            + px * math.sin(ego_yaw - math.pi / 2)
            + py * math.cos(ego_yaw - math.pi / 2),
        )
        for px, py in path_points
    ]

    states = [State(t=0, x=ego_x, y=ego_y, yaw=ego_yaw, vel=ego_vel, acc=ego_acc)]
    for i in range(1, len(global_points)):
        x1, y1 = global_points[i - 1]
        x2, y2 = global_points[i]
        dx, dy = x2 - x1, y2 - y1
        distance = math.sqrt(dx**2 + dy**2)
        yaw = math.atan2(dy, dx)
        vel = distance / Ti_path

        if i < len(global_points) - 1:
            x3, y3 = global_points[i + 1]
            dx2, dy2 = x3 - x2, y3 - y2
            distance2 = math.sqrt(dx2**2 + dy2**2)
            vel2 = distance2 / Ti_path
            acc = (vel2 - vel) / Ti_path
        else:
            acc = states[-1].acc

        states.append(State(t=i * Ti_path, x=x2, y=y2, yaw=yaw, vel=vel, acc=acc))

    trajectory = Trajectory()
    for i in range(1, len(states)):
        prev_state, curr_state = states[i - 1], states[i]
        t = prev_state.t
        for _ in range(int((curr_state.t - prev_state.t) / Ti_traj)):
            ratio = (t - prev_state.t) / (curr_state.t - prev_state.t)
            trajectory.states.append(
                State(
                    t=t,
                    x=prev_state.x + ratio * (curr_state.x - prev_state.x),
                    y=prev_state.y + ratio * (curr_state.y - prev_state.y),
                    yaw=prev_state.yaw
                    + ratio * normalize_angle(curr_state.yaw - prev_state.yaw),
                    vel=prev_state.vel + ratio * (curr_state.vel - prev_state.vel),
                    acc=prev_state.acc + ratio * (curr_state.acc - prev_state.acc),
                )
            )
            t += Ti_traj

    return trajectory
