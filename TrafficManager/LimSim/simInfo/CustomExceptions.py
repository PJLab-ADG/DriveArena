import sqlite3
import time

from simModel.Model import Model
from utils.trajectory import Rectangle, RecCollide
import cv2
import matplotlib.pyplot as plt
import numpy as np

class CollisionException(Exception):
    def __init__(self, ErrorInfo: str) -> None:
        super().__init__(self)
        self.errorinfo = ErrorInfo
    
    def __str__(self) -> str:
        return self.errorinfo

class CollisionChecker:
    def __init__(self):
        pass
    
    def CollisionCheck(self, model: Model) -> bool:
        # vehicle trajectory collision need to be checked in every frame
        for key, value in model.ms.vehINAoI.items():
            if value.id == model.ms.ego.id:
                continue
            recA = Rectangle([model.ms.ego.x, model.ms.ego.y],
                                model.ms.ego.length, model.ms.ego.width, model.ms.ego.yaw)
            recB = Rectangle([value.x, value.y],
                                value.length, value.width, value.yaw)
            rc = RecCollide(recA, recB)
            # if the car collide, stop the simulation
            if rc.isCollide():
                raise CollisionException("Ego car have a collision with vehicle {}".format(key))
        return False

    def check(self, model: Model) -> bool:
        return self.CollisionCheck(model)

class OffRoadException(Exception):
    def __init__(self, ErrorInfo: str) -> None:
        super().__init__(self)
        self.errorinfo = ErrorInfo
    
    def __str__(self) -> str:
        return self.errorinfo

class OffRoadChecker:
    def __init__(self, view_range=50, image_size=200):
        self.view_range = view_range
        self.image_size = image_size
        self.scale = image_size / (2 * view_range)

    def transform_points(self, points, ex, ey, ego_yaw):
        cos_yaw, sin_yaw = np.cos(ego_yaw), np.sin(ego_yaw)
        transformed = []
        for x, y in points:
            dx, dy = x - ex, y - ey
            rx = dx * cos_yaw + dy * sin_yaw
            ry = -dx * sin_yaw + dy * cos_yaw
            transformed.append((rx * self.scale + self.image_size // 2,
                                -ry * self.scale + self.image_size // 2))
        return np.array(transformed, dtype=np.int32)

    def draw_lane(self, img, lrd, ex, ey, ego_yaw, flag):
        if flag & 0b10:
            return
        left_bound = self.transform_points(lrd.left_bound, ex, ey, ego_yaw)
        cv2.polylines(img, [left_bound], False, 255, 1)

    def draw_edge(self, img, erd, rgrd, ex, ey, ego_yaw):
        right_bound, left_bound = None, None
        for lane_index in range(erd.num_lanes):
            lane_id = erd.id + '_' + str(lane_index)
            lrd = rgrd.get_lane_by_id(lane_id)
            flag = 0
            if lane_index == 0:
                flag += 1
                right_bound = self.transform_points(lrd.right_bound, ex, ey, ego_yaw)
            if lane_index == erd.num_lanes - 1:
                flag += 2
                left_bound = self.transform_points(lrd.left_bound, ex, ey, ego_yaw)
            self.draw_lane(img, lrd, ex, ey, ego_yaw, flag)

        if right_bound is not None and left_bound is not None:
            polygon = np.concatenate([right_bound, left_bound[::-1]])
            cv2.fillPoly(img, [polygon], 255)

    def draw_junction_lane(self, img, jlrd, ex, ey, ego_yaw):
        if jlrd.center_line:
            center_line = self.transform_points(jlrd.center_line, ex, ey, ego_yaw)
            cv2.polylines(img, [center_line], False, 255, 6)

    def draw_roadgraph(self, img, rgrd, ex, ey, ego_yaw):
        for erd in rgrd.edges.values():
            self.draw_edge(img, erd, rgrd, ex, ey, ego_yaw)
        for jlrd in rgrd.junction_lanes.values():
            self.draw_junction_lane(img, jlrd, ex, ey, ego_yaw)

    def check(self, model: Model) -> bool:
        roadgraphRenderData, VRDDict = model.renderQueue.get()
        egoVRD = VRDDict['egoCar'][0]
        ex, ey, ego_yaw = egoVRD.x, egoVRD.y, egoVRD.yaw

        img = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        self.draw_roadgraph(img, roadgraphRenderData, ex, ey, ego_yaw)

        car_length = int(5 * self.scale)
        car_width = int(2 * self.scale)
        center = self.image_size // 2
        car_area = img[center - car_width//2:center + car_width//2,
                    center - car_length//2:center + car_length//2]
        
        white_percentage = np.mean(car_area == 0) * 100 # white is not road block
        # print(f"[New] white percentage: {white_percentage:.2f}%")

        # plt.figure(figsize=(10, 10))
        # plt.imshow(img, cmap='binary')
        # car_rect = plt.Rectangle((center - car_length//2, center - car_width//2),
        #                         car_length, car_width,
        #                         linewidth=1, edgecolor='r', facecolor='none')
        # plt.gca().add_patch(car_rect)
        # plt.axis('off')
        # plt.tight_layout(pad=0)

        if white_percentage > 90:
            raise OffRoadException(f"Ego car has {white_percentage:.2f}% body off road")
        return False 


class LaneChangeException(Exception):
    def __init__(self) -> None:
        super().__init__(self)
        self.errorinfo = "you need to change lane, but your lane change is not successful"
    
    def __str__(self) -> str:
        return self.errorinfo  
    
def record_result(model: Model, start_time: float, result: bool, reason: str = "", error: Exception = None) -> None:
    conn = sqlite3.connect(model.dataBase)
    cur = conn.cursor()
    # add result data
    cur.execute(
        """INSERT INTO resultINFO (
            egoID, result, total_score, complete_percentage, drive_score, use_time, fail_reason
            ) VALUES (?,?,?,?,?,?,?);""",
        (
            model.ms.ego.id, result, 0, 0, 0, time.time() - start_time, reason
        )
    )
    conn.commit()
    conn.close()
    return 

    
class BrainDeadlockException(Exception):
    def __init__(self) -> None:
        super().__init__(self)
        self.errorinfo = "Your reasoning and decision-making result is in deadlock."

    def __str__(self) -> str:
        return self.errorinfo
    
class TimeOutException(Exception):
    def __init__(self) -> None:
        super().__init__(self)
        self.errorinfo = "You failed to complete the route within 100 seconds, exceeding the allotted time."

    def __str__(self) -> str:
        return self.errorinfo
    

class NoPathFoundError(Exception):
    def __init__(self, errorInfo: str) -> None:
        super().__init__(self)
        self.errorinfo = errorInfo

    def __str__(self) -> str:
        return self.errorinfo
