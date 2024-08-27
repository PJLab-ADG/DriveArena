import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
from math import cos, pi, sin


class MatplotlibRenderer:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.ax.set_aspect('equal')
        self.view_range = 50  # 视图范围，单位为米

    def get_transformed_points(self, points, ex, ey, ego_yaw):
        # 先平移，再旋转
        transformed = []
        for p in points:
            dx, dy = p[0] - ex, p[1] - ey
            # 旋转角度为 -ego_yaw + pi/2，使得车头朝向与 y 轴正方向对齐
            rx = dx * cos(-ego_yaw + pi/2) - dy * sin(-ego_yaw + pi/2)
            ry = dx * sin(-ego_yaw + pi/2) + dy * cos(-ego_yaw + pi/2)
            transformed.append((rx, ry))
        return transformed

    def is_in_range(self, x, y):
        return abs(x) <= self.view_range and abs(y) <= self.view_range

    def drawLane(self, lrd, ex, ey, ego_yaw, flag: int):
        if flag & 0b10:
            return
        else:
            left_bound_tf = self.get_transformed_points(lrd.left_bound, ex, ey, ego_yaw)
            x, y = zip(*left_bound_tf)
            self.ax.plot(x, y, color='black', linewidth=2, alpha=0.4)

    def drawEdge(self, erd, rgrd, ex, ey, ego_yaw):
        right_bound_tf = None
        left_bound_tf = None

        for lane_index in range(erd.num_lanes):
            lane_id = erd.id + '_' + str(lane_index)
            lrd = rgrd.get_lane_by_id(lane_id)
            flag = 0b00
            if lane_index == 0:
                flag += 1
                right_bound_tf = self.get_transformed_points(lrd.right_bound, ex, ey, ego_yaw)
            if lane_index == erd.num_lanes - 1:
                flag += 2
                left_bound_tf = self.get_transformed_points(lrd.left_bound, ex, ey, ego_yaw)
            self.drawLane(lrd, ex, ey, ego_yaw, flag)

        if right_bound_tf and left_bound_tf:
            polygon = right_bound_tf + left_bound_tf[::-1] + [right_bound_tf[0]]
            x, y = zip(*polygon)
            self.ax.fill(x, y, facecolor='black', edgecolor='black', alpha=0.1, linewidth=2)

    def drawJunctionLane(self, jlrd, ex, ey, ego_yaw):
        if jlrd.center_line:
            center_line_tf = self.get_transformed_points(jlrd.center_line, ex, ey, ego_yaw)
            x, y = zip(*center_line_tf)
            if jlrd.currTlState:
                if jlrd.currTlState == 'r':
                    jlColor = (1, 0.42, 0.51, 0.4)
                elif jlrd.currTlState == 'y':
                    jlColor = (0.98, 0.77, 0.19, 0.4)
                elif jlrd.currTlState in ['g', 'G']:
                    jlColor = (0.15, 0.68, 0.38, 0.2)
            else:
                jlColor = (0, 0, 0, 0.12)
            self.ax.plot(x, y, color=jlColor, linewidth=30)

    def drawRoadgraph(self, rgrd, ex, ey, ego_yaw):
        for erd in rgrd.edges.values():
            self.drawEdge(erd, rgrd, ex, ey, ego_yaw)

        for jlrd in rgrd.junction_lanes.values():
            self.drawJunctionLane(jlrd, ex, ey, ego_yaw)

    def plotVehicle(self, ex: float, ey: float, ego_yaw: float, vtag: str, vrd):
        # 计算相对位置和朝向
        dx, dy = vrd.x - ex, vrd.y - ey
        rx = dx * cos(-ego_yaw + pi/2) - dy * sin(-ego_yaw + pi/2)
        ry = dx * sin(-ego_yaw + pi/2) + dy * cos(-ego_yaw + pi/2)
        
        if not self.is_in_range(rx, ry):
            return  # 如果车辆不在视图范围内，不绘制
        
        relative_yaw = vrd.yaw - ego_yaw + pi/2

        rotateMat = np.array([
            [cos(relative_yaw), -sin(relative_yaw)],
            [sin(relative_yaw), cos(relative_yaw)]
        ])
        vertexes = np.array([
            [vrd.length/2, vrd.width/2],
            [vrd.length/2, -vrd.width/2],
            [-vrd.length/2, -vrd.width/2],
            [-vrd.length/2, vrd.width/2]
        ]).T
        rotVertexes = rotateMat @ vertexes
        x = rx + rotVertexes[0]
        y = ry + rotVertexes[1]

        if vtag == 'ego':
            vcolor = (0.83, 0.33, 0)
        elif vtag == 'AoI':
            vcolor = (0.16, 0.5, 0.73)
        else:
            vcolor = (0.39, 0.43, 0.45)

        self.ax.fill(x, y, color=vcolor, alpha=1.0)
        self.ax.text(rx, ry, vrd.id, color='black', fontsize=10, ha='center', va='center')

    def plotTrajectory(self, ex: float, ey: float, ego_yaw: float, vrd):
        traj_points = list(zip(vrd.trajectoryXQ, vrd.trajectoryYQ))
        transformed_traj = self.get_transformed_points(traj_points, ex, ey, ego_yaw)
        x, y = zip(*transformed_traj)
        self.ax.plot(x, y, color=(0.8, 0.52, 0.95), linewidth=2)

    def drawVehicles(self, VRDDict: Dict[str, List], ex: float, ey: float, ego_yaw: float):
        egoVRD = VRDDict['egoCar'][0]
        if egoVRD.trajectoryXQ:
            self.plotTrajectory(ex, ey, ego_yaw, egoVRD)
        for avrd in VRDDict['carInAoI']:
            self.plotVehicle(ex, ey, ego_yaw, 'AoI', avrd)
            if avrd.trajectoryXQ:
                self.plotTrajectory(ex, ey, ego_yaw, avrd)
        for svrd in VRDDict['outOfAoI']:
            self.plotVehicle(ex, ey, ego_yaw, 'other', svrd)
        self.plotVehicle(ex, ey, ego_yaw, 'ego', egoVRD)

    def render(self, roadgraphRenderData, VRDDict, filename = "output.png"):
        # self.fig, self.ax = plt.subplots(figsize=(12, 12))
        # self.ax.set_aspect('equal')
        self.ax.clear()
        egoVRD = VRDDict['egoCar'][0]
        ex, ey, ego_yaw = egoVRD.x, egoVRD.y, egoVRD.yaw
        self.drawRoadgraph(roadgraphRenderData, ex, ey, ego_yaw)
        self.drawVehicles(VRDDict, ex, ey, ego_yaw)
        
        # 设置坐标轴范围为以ego为中心的100x100正方形
        self.ax.set_xlim(-self.view_range, self.view_range)
        self.ax.set_ylim(-self.view_range, self.view_range)
        
        # 设置坐标轴标签
        # self.ax.set_xlabel("X (meters)")
        # self.ax.set_ylabel("Y (meters)")
        # self.ax.set_title("Ego-Centered View")
        
        # 添加网格线
        self.ax.grid(True, linestyle='--', alpha=0.5)
        
        # # 添加ego车辆位置和方向指示
        # self.ax.plot(0, 0, 'ro', markersize=10)  # ego车辆位置
        # self.ax.arrow(0, 0, 0, 5, head_width=2, head_length=2, fc='r', ec='r')  # ego车辆方向

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        # plt.show()

# 使用示例
# renderer = MatplotlibRenderer()
# renderer.render(roadgraphRenderData, VRDDict)