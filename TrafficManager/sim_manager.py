import base64
from datetime import datetime
import io
import json
import os
import sys
import time
import math
from typing import Dict, List, Optional, Tuple
import dearpygui.dearpygui as dpg
from matplotlib import pyplot as plt
import requests
import numpy as np
import torch
import cv2
from PIL import Image
from io import BytesIO
import yaml
# Add LimSim to sys.path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "LimSim"))  # noqa
from TrafficManager.utils.sim_utils import limsim2diffusion, normalize_angle, transform_to_ego_frame, interpolate_traj
from TrafficManager.utils.map_utils import VectorizedLocalMap
from LimSim.utils.trajectory import Trajectory, State
from LimSim.trafficManager.traffic_manager import TrafficManager
from LimSim.simModel.MPGUI import GUI
from LimSim.simModel.Model import Model
from LimSim.simModel.DataQueue import CameraImages
from TrafficManager.utils.matplot_render import MatplotlibRenderer
from LimSim.simInfo.CustomExceptions import CollisionChecker, OffRoadChecker
from TrafficManager.utils.scorer import Scorer



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
        self.agent_command: int = 2 # Defined by UniAD  0: Right 1:Left 2:Forward 
        self.result_path = f"./results/{datetime.now().strftime('%m-%d-%H%M%S')}/"
        self.img_save_path = f"{self.result_path}imgs/"
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(self.img_save_path, exist_ok=True)


    @staticmethod
    def load_config(config_path: str) -> Dict:
        with open(config_path, 'r') as config_file:
            return yaml.safe_load(config_file)

    def setup_constants(self):
        self.DIFFUSION_SERVER = self.config['servers']['diffusion']
        self.DRIVER_SERVER = self.config['servers']['driver']
        self.STEP_LENGTH = self.config['simulation']['step_length']
        self.GUI_DISPLAY = self.config['simulation']['gui_display']
        self.MAX_SIM_TIME = self.config['simulation']['max_sim_time']
        self.EGO_ID = self.config['simulation']['ego_id']
        self.MAP_NAME = self.config['map']['name']
        self.IMAGE_SIZE = self.config['image']['size']
        self.TARGET_SIZE = tuple(self.config['image']['target_size'])

    def setup_paths(self):
        data_root = os.path.dirname(os.path.abspath(__file__))
        self.SUMO_CFG_FILE = os.path.join(data_root, self.config['map']['sumo_cfg_file'].format(map_name=self.MAP_NAME))
        self.SUMO_NET_FILE = os.path.join(data_root, self.config['map']['sumo_net_file'].format(map_name=self.MAP_NAME))
        self.SUMO_ROU_FILE = os.path.join(data_root, self.config['map']['sumo_rou_file'].format(map_name=self.MAP_NAME))
        self.DATA_TEMPLATE_PATH = os.path.join(data_root, self.config['data']['template_path'])
        self.NU_SCENES_DATA_ROOT = os.path.join(data_root, self.config['data']['nu_scenes_root'].format(map_name=self.MAP_NAME))

    @staticmethod
    def normalize_angle(angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi


    def send_request_diffusion(self, diffusion_data: Dict) -> Optional[np.ndarray]:
        serialized_data = {
            k: v.numpy().tolist() if isinstance(v, torch.Tensor) else 
               {k2: v2.numpy().tolist() if isinstance(v2, torch.Tensor) else v2 for k2, v2 in v.items()} if isinstance(v, dict) else
               v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in diffusion_data.items()
        }

        try:
            print(f"Sending data to WorldDreamer server...")
            response = requests.post(self.DIFFUSION_SERVER + "dreamer-api/", json=serialized_data)
            if response.status_code == 200 and 'image' in response.headers['Content-Type']:
                image = Image.open(BytesIO(response.content))
                images_array = np.array(np.split(np.array(image), 6, axis=0))
                combined_image = np.vstack((np.hstack(images_array[:3]), np.hstack(images_array[3:])))
                cv2.imwrite(f"{self.img_save_path}diffusion_{str(int(self.timestamp*2)).zfill(3)}.jpg", cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))
                return np.array(np.split(np.array(image), 6, axis=0))
        except requests.exceptions.RequestException as e:
            print(f"Warning: Request failed due to {e}")
        return None


    def get_drivable_mask(self, model:Model) -> np.ndarray:    
        img = np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=np.uint8)
        roadgraphRenderData, VRDDict = model.renderQueue.get()
        egoVRD = VRDDict['egoCar'][0]
        ex, ey, ego_yaw = egoVRD.x, egoVRD.y, egoVRD.yaw

        OffRoadChecker().draw_roadgraph(img, roadgraphRenderData, ex, ey, ego_yaw)
        return img.astype(bool)


    def initialize_simulation(self):        
        # Initialising models, planners, maps etc
        self.model = Model(
            egoID=self.EGO_ID, netFile=self.SUMO_NET_FILE, rouFile=self.SUMO_ROU_FILE,
            cfgFile=self.SUMO_CFG_FILE, dataBase=self.result_path+"limsim.db", SUMOGUI=False,
            CARLACosim=False,
        )
        self.model.start()
        self.planner = TrafficManager(self.model, config_file_path='./TrafficManager/LimSim/trafficManager/config.yaml')

        print(f"Testing connection to WorldDreamer & Driver servers...")
        requests.get(self.DIFFUSION_SERVER + "dreamer-clean/")
        requests.get(self.DRIVER_SERVER + "driver-clean/")

        self.data_template = torch.load(self.DATA_TEMPLATE_PATH)
        self.vectorized_map = VectorizedLocalMap(dataroot=self.NU_SCENES_DATA_ROOT, map_name=self.MAP_NAME, patch_size=[100, 100], fixed_ptsnum_per_line=-1)
        
        self.gui = GUI(self.model)
        if self.GUI_DISPLAY:
            self.gui.start()

        self.renderer = MatplotlibRenderer()
        self.checkers = [OffRoadChecker(), CollisionChecker()]


    def process_frame(self):
        #Single frame processing logic
        if  self.scorer is None:
            self.scorer = Scorer(self.model, map_name=self.MAP_NAME, save_file_path=self.result_path+"drive_arena.pkl")
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

            limsim_trajectories = self.planner.plan(self.model.timeStep * 0.1, self.roadgraph, self.vehicles)
            if not limsim_trajectories[self.EGO_ID].states:
                return True

            traj_len = min(len(limsim_trajectories[self.EGO_ID].states) - 1, 25)
            local_x, local_y, local_yaw = transform_to_ego_frame(limsim_trajectories[self.EGO_ID].states[0], limsim_trajectories[self.EGO_ID].states[traj_len])
            self.agent_command = 2 if local_x <= 5.0 else (1 if local_y > 4.0 else 0 if local_y < -4.0 else 2)
            print("Agent command:", self.agent_command)

            diffusion_data = limsim2diffusion(
                self.vehicles, self.data_template, self.vectorized_map, self.MAP_NAME, self.agent_command, self.last_pose, drivable_mask,
                self.accel, self.rotation_rate, self.vel,
                gen_location = "boston-seaport", 
                gen_prompts ="daytime, cloudy, downtown, red buildings, white cars"
            )
            self.last_pose = diffusion_data['metas']['ego_pos']
            gen_images = self.send_request_diffusion(diffusion_data)

            if gen_images is not None:
                front_left_image, front_image, front_right_image = [Image.fromarray(img).convert('RGBA') for img in gen_images[:3]]
            else:
                raise ValueError("No images generated!")

            new_width, new_height = self.TARGET_SIZE[0], int((self.TARGET_SIZE[0] / front_image.width) * front_image.height)
            resized_images = [img.resize((new_width, new_height), Image.Resampling.LANCZOS) for img in [front_left_image, front_image, front_right_image]]

            ci = CameraImages()
            ci.CAM_FRONT_LEFT, ci.CAM_FRONT, ci.CAM_FRONT_RIGHT = [np.array(img) for img in resized_images]
            print("Current timestamp:", self.timestamp)

            response = requests.get(self.DRIVER_SERVER + "driver-get/")
            while response.status_code != 200 or response.text == "false":
                # print("The Driver Agent not processing done, try again in 1s")
                time.sleep(0.5)
                response = requests.get(self.DRIVER_SERVER + "driver-get/")
                # print("Driver Agent", response.status_code)

            driver_output = json.loads(response.text)
            path_points = driver_output["bbox_results"][0]["planning_traj"][0]
            print("Driver Agent's Path:", path_points)

            #add driver predict BEV 
            pred_bev_base64 = driver_output["bev_pred_img"]
            pred_bev_img = base64.b64decode(pred_bev_base64)
            pred_bev_img = Image.open(io.BytesIO(pred_bev_img))
            # save image
            pred_bev_img = pred_bev_img.convert('RGB')
            pred_bev_img.save(f"{self.img_save_path}agent_{str(int(self.timestamp*2)).zfill(3)}.jpg")
            pred_bev_img = pred_bev_img.convert('RGBA')
            pred_bev_img = pred_bev_img.resize((800, 800), Image.Resampling.LANCZOS)
            ci.PRED_BEV =  np.array(pred_bev_img, dtype=np.float32)

            self.model.imageQueue.put(ci)


            path_points.insert(0, [0.0, 0.0])
            ego_vehicle = self.vehicles['egoCar']
            ego_traj = interpolate_traj(ego_vehicle, path_points)

            if len(limsim_trajectories[self.EGO_ID].states) < 10:
                yaw_rate = 0
            else:
                yaw_rate = limsim_trajectories[self.EGO_ID].states[9].yaw - limsim_trajectories[self.EGO_ID].states[0].yaw
            vx_1, vx_2 = path_points[2][0] - path_points[0][0], path_points[3][0] - path_points[1][0]
            vy_1, vy_2 = path_points[2][1] - path_points[0][1], path_points[3][1] - path_points[1][1]
            ax, ay = (vx_2 - vx_1) / 0.5, (vy_2 - vy_1) / 0.5
            self.accel = [ax, ay, 9.80]
            self.rotation_rate = [0, 0, yaw_rate]
            self.vel = [limsim_trajectories[self.EGO_ID].states[0].vel, 0, 0]
            print("Accel:", self.accel, "\nRotation rate:", self.rotation_rate, "\nVel:", self.vel)

            self.model.putRenderData()
            roadgraphRenderData, VRDDict = self.model.renderQueue.get()
            self.renderer.render(roadgraphRenderData, VRDDict, f'{self.img_save_path}bev_{str(int(self.timestamp*2)).zfill(3)}.png')

            self.scorer.record_frame(drivable_mask, is_planning_frame=True, planned_traj=ego_traj, ref_traj=limsim_trajectories[self.EGO_ID])

            USE_DRIVER_ACTION = False
            limsim_trajectories = {}
            if USE_DRIVER_ACTION and self.timestamp > 8.0:
                limsim_trajectories[self.EGO_ID] = ego_traj
            self.model.setTrajectories(limsim_trajectories)
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
                if self.vehicles and 'egoCar' in self.vehicles:
                    if not self.process_frame():
                        break
                self.model.updateVeh()
        finally:
            self.cleanup()

    def cleanup(self):
        print("Simulation ends")
        if self.scorer:
            self.scorer.save()
        self.model.destroy()
        self.gui.terminate()
        self.gui.join()


def main():
    sim_manager = SimulationManager('config.yaml')
    sim_manager.run_simulation()

if __name__ == '__main__':
    main()
