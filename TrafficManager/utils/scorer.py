import pickle
from LimSim.simModel.Model import Model
import traci
from datetime import datetime
from LimSim.utils.trajectory import Trajectory, State


class Scorer:
    def __init__(self, model: Model, veh_id: str = None, map_name: str = "Unknown", ego_id = None, save_file_path: str = 'test_arena.pkl'):
        if veh_id is None:
            # default to evaulate the ego vehicle
            veh_id = model.ego.id

        self.veh_id = veh_id
        self.model = model
        self.veh_route_list = traci.vehicle.getRoute(veh_id)
        self.total_route_length = self.calculate_total_route_length()
        self.drive_distance = 0.0
        self.pkl_file_path = save_file_path

        self.data = {}  # save to pkl file
        self.data["metas"] = {
            "location": map_name,
            "runtime":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ego_id": ego_id
        }
        self.data["type"] = 'closed-loop'
        self.data["route_length"] = self.total_route_length
        self.data["frames"] = []

    def calculate_total_route_length(self) -> float:
        total_length = 0
        for ego_route_id in self.veh_route_list:
            lanes_ids_set = self.model.nb.getEdge(ego_route_id).lanes
            lanes_ids_list = list(lanes_ids_set)
            lane_length = self.model.nb.getLane(
                lanes_ids_list[0]).spline_length
            total_length += lane_length
        return total_length

    def update_driving_distance(self):
        veh_edge_id = traci.vehicle.getRoadID(self.veh_id)
        edge_distance = traci.vehicle.getLanePosition(self.veh_id)

        if veh_edge_id not in self.veh_route_list:
            if traci.vehicle.getLaneID(self.veh_id) in self.model.nb.junctionLanes.keys():
                # vehicle in junction, not update the driving distance
                pass
            else:
                # vehicle has left the preset route
                print(
                    "[Warning]: Ego vehicle has left the preset route, cannot calculate driving distance.")
                pass
        else:
            drive_distance = 0.0
            for route_id in self.veh_route_list:
                if veh_edge_id == route_id:
                    drive_distance += edge_distance
                    break
                else:
                    lanes_ids_set = self.model.nb.getEdge(route_id).lanes
                    lanes_ids_list = list(lanes_ids_set)
                    lane_length = self.model.nb.getLane(
                        lanes_ids_list[0]).spline_length
                    drive_distance += lane_length
            self.drive_distance = max(drive_distance, self.drive_distance)
            print("Ego vehicle driving distance: ",
                  self.drive_distance, "/", self.total_route_length)

        return

    def record_frame(self, drivable_mask, is_planning_frame: bool = False, planned_traj: Trajectory = None, ref_traj: Trajectory = None):
        frame_data = {}
        self.update_driving_distance()

        frame_data["time_stamp"] = self.model.timeStep / 10  # running in 10Hz

        frame_data["obj_names"] = []
        frame_data["obj_boxes"] = []
        for vid, veh in {**self.model.ms.vehINAoI, **self.model.ms.outOfAoI}.items():
            # supposez=0 and car height= 1.5 m
            frame_data["obj_boxes"].append(
                (veh.x, veh.y, 0, veh.width, veh.length, 1.5, veh.yaw)
            )
            frame_data["obj_names"].append("car")
        frame_data["drivable_mask"] = drivable_mask
        frame_data["is_key_frame"] = is_planning_frame
        frame_data["ego_box"] = (self.model.ego.x, self.model.ego.y, 0,
                                 self.model.ego.width, self.model.ego.length, 1.5, self.model.ego.yaw)
        
        if is_planning_frame:
            min_len = min(len(planned_traj.states), len(ref_traj.states))
            step = 5
            frame_data['planned_traj'] = {
                'timestep': 0.5,
                'traj': [(state.x, state.y, state.yaw) for state in planned_traj.states[4:min_len+1:step]]
            }
            frame_data['ref_traj'] = {
                'timestep': 0.5,
                'traj': [(state.x, state.y, state.yaw) for state in ref_traj.states[3:min_len+1:step]]
            }
            # breakpoint()

        self.data["frames"].append(frame_data)
        # breakpoint()
        return

    def save(self):
        self.data['drive_length'] = self.drive_distance
        datas = [self.data]
        with open(self.pkl_file_path, 'wb') as f:
            pickle.dump(datas, f)
        return
