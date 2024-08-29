import base64
import io
import json
import math
import os
import sys
import time
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import cv2
import dearpygui.dearpygui as dpg
import numpy as np
import requests
import torch
import yaml
from matplotlib import pyplot as plt
from PIL import Image

# Add LimSim to sys.path
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "LimSim")
)  # noqa
from TrafficManager.LimSim.simInfo.CustomExceptions import (
    CollisionChecker,
    OffRoadChecker,
)
from TrafficManager.LimSim.simModel.DataQueue import CameraImages
from TrafficManager.LimSim.simModel.Model import Model
from TrafficManager.LimSim.simModel.MPGUI import GUI
from TrafficManager.LimSim.trafficManager.traffic_manager import TrafficManager
from TrafficManager.LimSim.utils.trajectory import State, Trajectory
from TrafficManager.utils.map_utils import VectorizedLocalMap
from TrafficManager.utils.matplot_render import MatplotlibRenderer
from TrafficManager.utils.scorer import Scorer
from TrafficManager.utils.sim_utils import (
    interpolate_traj,
    limsim2diffusion,
    normalize_angle,
    transform_to_ego_frame,
)


class SimulationManager:
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.setup_constants()
        self.setup_paths()
        self.model: Optional[Model] = None
        self.planner: Optional[TrafficManager] = None
        self.vectorized_map: Optional[VectorizedLocalMap] = None
        self.gui: Optional[GUI] = None
        self.renderer: Optional[MatplotlibRenderer] = None
        self.checkers: List = []
        self.scorer: Optional[Scorer] = None
        self.timestamp: float = -0.5
        self.data_template: Optional[torch.Tensor] = None
        self.last_pose: torch.Tensor = torch.eye(4)
        self.accel: List[float] = [0, 0, 9.80]
        self.rotation_rate: List[float] = [0, 0, 0]
        self.vel: List[float] = [0, 0, 0]
        self.agent_command: int = 2
        self.result_path = f"./results/{datetime.now().strftime('%m-%d-%H%M%S')}/"
        os.makedirs(self.result_path, exist_ok=True)

    @staticmethod
    def load_config(config_path: str) -> Dict:
        with open(config_path, "r") as config_file:
            return yaml.safe_load(config_file)

    def setup_constants(self):
        self.DIFFUSION_SERVER = self.config["servers"]["diffusion"]
        self.DRIVER_SERVER = self.config["servers"]["driver"]
        self.STEP_LENGTH = self.config["simulation"]["step_length"]
        self.GUI_DISPLAY = self.config["simulation"]["gui_display"]
        self.MAX_SIM_TIME = self.config["simulation"]["max_sim_time"]
        self.EGO_ID = self.config["simulation"]["ego_id"]
        self.MAP_NAME = self.config["map"]["name"]
        self.IMAGE_SIZE = self.config["image"]["size"]
        self.TARGET_SIZE = tuple(self.config["image"]["target_size"])

    def setup_paths(self):
        data_root = os.path.dirname(os.path.abspath(__file__))
        self.SUMO_CFG_FILE = os.path.join(
            data_root,
            self.config["map"]["sumo_cfg_file"].format(map_name=self.MAP_NAME),
        )
        self.SUMO_NET_FILE = os.path.join(
            data_root,
            self.config["map"]["sumo_net_file"].format(map_name=self.MAP_NAME),
        )
        self.SUMO_ROU_FILE = os.path.join(
            data_root,
            self.config["map"]["sumo_rou_file"].format(map_name=self.MAP_NAME),
        )
        self.DATA_TEMPLATE_PATH = os.path.join(
            data_root, self.config["data"]["template_path"]
        )
        self.NU_SCENES_DATA_ROOT = os.path.join(
            data_root,
            self.config["data"]["nu_scenes_root"].format(map_name=self.MAP_NAME),
        )

    @staticmethod
    def normalize_angle(angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def get_drivable_mask(self, model: Model) -> np.ndarray:
        img = np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=np.uint8)
        roadgraphRenderData, VRDDict = model.renderQueue.get()
        egoVRD = VRDDict["egoCar"][0]
        ex, ey, ego_yaw = egoVRD.x, egoVRD.y, egoVRD.yaw

        OffRoadChecker().draw_roadgraph(img, roadgraphRenderData, ex, ey, ego_yaw)
        return img.astype(bool)

    def initialize_simulation(self):
        # Initialising models, planners, maps etc
        self.model = Model(
            egoID=self.EGO_ID,
            netFile=self.SUMO_NET_FILE,
            rouFile=self.SUMO_ROU_FILE,
            cfgFile=self.SUMO_CFG_FILE,
            dataBase=self.result_path + "limsim.db",
            SUMOGUI=False,
            CARLACosim=False,
        )
        self.model.start()
        self.planner = TrafficManager(
            self.model,
            config_file_path="./TrafficManager/LimSim/trafficManager/config.yaml",
        )

        self.data_template = torch.load(self.DATA_TEMPLATE_PATH)
        self.vectorized_map = VectorizedLocalMap(
            dataroot=self.NU_SCENES_DATA_ROOT,
            map_name=self.MAP_NAME,
            patch_size=[100, 100],
            fixed_ptsnum_per_line=-1,
        )

        self.gui = GUI(self.model)
        if self.GUI_DISPLAY:
            self.gui.start()

        self.renderer = MatplotlibRenderer()
        self.checkers = [OffRoadChecker(), CollisionChecker()]

    def process_frame(self):
        # Single frame processing logic
        if self.scorer is None:
            self.scorer = Scorer(
                self.model,
                map_name=self.MAP_NAME,
                save_file_path=self.result_path + "drive_arena.pkl",
            )
        try:
            for checker in self.checkers:
                checker.check(self.model)
        except Exception as e:
            print(f"WARNING: Checker failed @ timestep {self.model.timeStep}. {e}")
            raise e

        drivable_mask = self.get_drivable_mask(self.model)
        if self.model.timeStep % 5 == 0:
            self.timestamp += 0.5
            if self.timestamp >= self.MAX_SIM_TIME:
                print("Simulation time end.")
                return False

            limsim_trajectories = self.planner.plan(
                self.model.timeStep * 0.1, self.roadgraph, self.vehicles
            )
            if not limsim_trajectories[self.EGO_ID].states:
                return True

            traj_len = min(len(limsim_trajectories[self.EGO_ID].states) - 1, 25)
            local_x, local_y, local_yaw = transform_to_ego_frame(
                limsim_trajectories[self.EGO_ID].states[0],
                limsim_trajectories[self.EGO_ID].states[traj_len],
            )
            self.agent_command = (
                2
                if local_x <= 5.0
                else (1 if local_y > 4.0 else 0 if local_y < -4.0 else 2)
            )
            print(
                "Agent command:",
                traj_len,
                self.agent_command,
                local_x,
                local_y,
                local_yaw,
            )

            diffusion_data = limsim2diffusion(
                self.vehicles,
                self.data_template,
                self.vectorized_map,
                self.MAP_NAME,
                self.agent_command,
                self.last_pose,
                drivable_mask,
                self.accel,
                self.rotation_rate,
                self.vel,
            )
            self.last_pose = diffusion_data["metas"]["ego_pos"]

            self.scorer.record_frame(drivable_mask, is_planning_frame=False)
        else:
            if self.scorer is not None:
                self.scorer.record_frame(drivable_mask, is_planning_frame=False)

        return True

    def run_simulation(self):
        self.initialize_simulation()
        try:
            while not self.model.tpEnd:
                self.model.moveStep()
                self.roadgraph, self.vehicles = self.model.exportSce()
                if self.vehicles is not None and "egoCar" in self.vehicles:
                    if not self.process_frame():
                        break
                self.model.updateVeh()
        finally:
            self.cleanup()

    def cleanup(self):
        # Cleaning up resources, saving scores, etc.
        print("Simulation ends")
        if self.scorer:
            self.scorer.save()
        self.model.destroy()
        self.gui.terminate()
        self.gui.join()


def main():
    sim_manager = SimulationManager("config.yaml")
    sim_manager.run_simulation()


if __name__ == "__main__":
    main()
