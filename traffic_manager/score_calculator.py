import csv
import pickle
import sys
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from shapely.geometry import LineString, Point
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import Point
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import threading


# Define boundaries
boundaries = {
    'max_abs_lat_accel': 4.89,  # [m/s^2]
    'max_lon_accel': 2.40,  # [m/s^2]
    'min_lon_accel': -4.05,  # [m/s^2]
    'max_abs_yaw_accel': 1.93,  # [rad/s^2]
    'max_abs_lon_jerk': 8.37,  # [m/s^3],
    'max_abs_yaw_rate': 0.95,  # [rad/s]
}

score_weight = {
    'ttc': 5,
    'c': 2,
    'ep': 5,
}


class ScoreCalculator:
    def __init__(self, data):
        self.data = data

        self.pdms = 0.0
        self.driving_score = None
        pass

    def transform_to_ego_frame(self, traj, ego_box):
        """
        Transform trajectory from global frame to ego-centric frame.

        :param traj: List of tuples (x, y, yaw) in global frame
        :param ego_box: Tuple (x, y, z, w, l, h, yaw) of ego vehicle in global frame
        :return: Numpy array of transformed trajectory
        """
        ego_x, ego_y, _, _, _, _, ego_yaw = ego_box

        # Create rotation matrix
        c, s = np.cos(-ego_yaw), np.sin(-ego_yaw)
        R = np.array([[c, -s], [s, c]])

        # Transform each point
        transformed_traj = []
        for x, y, yaw in traj:
            # Translate
            x_translated, y_translated = x - ego_x, y - ego_y

            # Rotate
            x_rotated, y_rotated = R @ np.array([x_translated, y_translated])

            # Adjust yaw
            yaw_adjusted = yaw - ego_yaw

            transformed_traj.append((x_rotated, y_rotated, yaw_adjusted))

        return np.array(transformed_traj)

    def get_vehicle_corners(self, x, y, yaw, length, width):
        """
        Calculate the corner points of the vehicle given its position, orientation, and dimensions.

        :param x: x-coordinate of the vehicle's center
        :param y: y-coordinate of the vehicle's center
        :param yaw: orientation of the vehicle in radians
        :param length: length of the vehicle
        :param width: width of the vehicle
        :return: numpy array of corner coordinates (4x2)
        """
        c, s = np.cos(yaw), np.sin(yaw)
        front_left = np.array([x + c * length / 2 - s * width / 2,
                               y + s * length / 2 + c * width / 2])
        front_right = np.array([x + c * length / 2 + s * width / 2,
                                y + s * length / 2 - c * width / 2])
        rear_left = np.array([x - c * length / 2 - s * width / 2,
                              y - s * length / 2 + c * width / 2])
        rear_right = np.array([x - c * length / 2 + s * width / 2,
                               y - s * length / 2 - c * width / 2])
        return np.array([front_left, front_right, rear_right, rear_left])

    def plot_trajectory_on_drivable_mask(self, drivable_mask, transformed_traj, vehicle_width, vehicle_length):
        """
        Plot the transformed trajectory and vehicle bounding boxes on the drivable mask.

        :param drivable_mask: 2D numpy array representing the drivable area (200x200)
        :param transformed_traj: Numpy array of transformed trajectory points
        :param vehicle_width: Width of the vehicle in meters
        :param vehicle_length: Length of the vehicle in meters
        """

        plt.figure(figsize=(10, 10))
        plt.imshow(drivable_mask, cmap='gray', extent=[-50, 50, -50, 50])

        # Scale factor (200 pixels represent 100 meters)
        scale_factor = 200 / 100  # pixels per meter

        # Plot trajectory
        x_coords, y_coords, yaws = transformed_traj.T
        plt.plot(x_coords, y_coords, 'r-', linewidth=2)

        # Plot vehicle bounding boxes
        for x, y, yaw in transformed_traj:
            corners = self.get_vehicle_corners(
                x, y, yaw, vehicle_length, vehicle_width)
            plt.gca().add_patch(Polygon(corners, fill=False, edgecolor='blue'))

        # Plot start and end points
        plt.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start')
        plt.plot(x_coords[-1], y_coords[-1], 'bo', markersize=10, label='End')

        plt.title('Trajectory and Vehicle Bounding Boxes on Drivable Mask')
        plt.legend()
        plt.xlabel('x (meters)')
        plt.ylabel('y (meters)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def _calculate_drivable_area_compliance(self, drivable_mask, transformed_traj, vehicle_width, vehicle_length):
        """
        Check if the vehicle bounding box is within the drivable area for each frame.

        :param drivable_mask: 2D numpy array representing the drivable area (200x200)
        :param transformed_traj: Numpy array of transformed trajectory points
        :param vehicle_width: Width of the vehicle in meters
        :param vehicle_length: Length of the vehicle in meters
        :return: List of tuples (traj_ti, drivable_ratio)
        """
        def meter_to_pixel(x, y, mask_size=200, real_size=100):
            """
            Convert coordinates from meters to pixels.

            :param x: x-coordinate in meters
            :param y: y-coordinate in meters
            :param mask_size: size of the mask in pixels
            :param real_size: size of the area in meters
            :return: tuple of (x, y) in pixel coordinates
            """
            scale = mask_size / real_size
            px = int((x + real_size/2) * scale)
            py = int((real_size/2 - y) * scale)  # Flip y-axis
            return px, py

        dac = 1.0
        mask_height, mask_width = drivable_mask.shape
        for traj_ti, (x, y, yaw) in enumerate(transformed_traj):
            # Get vehicle corners in meter coordinates
            corners = self.get_vehicle_corners(
                x, y, yaw, vehicle_length, vehicle_width)

            # Convert corners to pixel coordinates
            corners_px = np.array([meter_to_pixel(cx, cy)
                                  for cx, cy in corners])

            # Create a mask for the vehicle
            vehicle_mask = np.zeros_like(drivable_mask, dtype=bool)
            vehicle_poly = ShapelyPolygon(corners_px)

            # Check each pixel within the bounding box of the vehicle polygon
            min_x, min_y = np.floor(np.min(corners_px, axis=0)).astype(int)
            max_x, max_y = np.ceil(np.max(corners_px, axis=0)).astype(int)

            for i in range(max(0, min_y), min(mask_height, max_y + 1)):
                for j in range(max(0, min_x), min(mask_width, max_x + 1)):
                    if vehicle_poly.contains(Point(j, i)):
                        vehicle_mask[i, j] = True
            # Calculate the ratio of drivable area within the vehicle mask
            total_vehicle_pixels = np.sum(vehicle_mask)
            drivable_vehicle_pixels = np.sum(drivable_mask[vehicle_mask])
            drivable_ratio = drivable_vehicle_pixels / \
                total_vehicle_pixels if total_vehicle_pixels > 0 else 1.0

            if drivable_ratio < 0.2:
                dac = 0.0
                return dac
            elif drivable_ratio < 0.5:
                dac = 0.5

            # print(f"Frame {traj_ti}: Drivable ratio = {drivable_ratio}")

        return dac

    def _calculate_progress(self, planned_traj, ref_taj):
        def calculate_curve_length(points):
            """Calculate the total length of a curve given by a set of points."""
            curve = LineString(points)
            return curve.length

        def project_curve_onto_curve(curve_a, curve_b):
            """Project curve_b onto curve_a and calculate the projected length."""
            projected_points = []
            for point in curve_b.coords:
                projected_point = curve_a.interpolate(
                    curve_a.project(Point(point)))
                projected_points.append(projected_point)
            projected_curve = LineString(projected_points)
            return projected_curve.length

        # Create Shapely LineString objects
        plan_curve = LineString([(x, y) for x, y, _ in planned_traj])
        ref_curve = LineString([(x, y) for x, y, _ in ref_taj])

        # Calculate lengths
        plan_curve_length = calculate_curve_length(plan_curve)
        ref_curve_length = calculate_curve_length(ref_curve)
        projected_length = project_curve_onto_curve(ref_curve, plan_curve)
        # print(f"plan_curve_length: {plan_curve_length}, ref_curve_length: {ref_curve_length}, project plan to ref_length: {projected_length}")

        ep = 0.0
        if max(plan_curve_length, ref_curve_length) < 5.0 or ref_curve_length < 1e-6:
            ep = 1.0
        else:
            ep = projected_length / ref_curve_length
        return ep

    def _calculate_is_comfortable(self, traj, timestep):
        """
        Check if all kinematic parameters of a trajectory are within specified boundaries.

        :param traj: List of tuples (x, y, yaw) representing the trajectory, in ego's local frame
        :param timestep: Time interval between trajectory points in seconds
        :return: 1.0 if all parameters are within boundaries, 0.0 otherwise
        """

        def calculate_trajectory_kinematics(traj, timestep):
            """
            Calculate kinematic parameters for a given trajectory.

            :param traj: List of tuples (x, y, yaw) for each point in the trajectory
            :param timestep: Time interval between each point in the trajectory
            :return: Dictionary containing lists of calculated parameters
            """
            # Convert trajectory to numpy array for easier calculations
            x, y, yaw = zip(*traj)
            x, y, yaw = np.array(x), np.array(y), np.array(yaw)

            # Calculate velocities
            dx = np.diff(x) / timestep
            dy = np.diff(y) / timestep

            # Calculate yaw rate
            dyaw = np.diff(yaw)
            dyaw = np.where(dyaw > np.pi, dyaw - 2*np.pi, dyaw)
            dyaw = np.where(dyaw < -np.pi, dyaw + 2*np.pi, dyaw)
            dyaw = dyaw / timestep
            ddyaw = np.diff(dyaw) / timestep

            # Calculate speed
            speed = np.sqrt(dx**2 + dy**2)

            # Calculate accelerations
            accel = np.diff(speed) / timestep
            jerk = np.diff(accel) / timestep

            # Calculate yaw rate (already calculated as dyaw)
            yaw_rate = dyaw
            # Calculate yaw acceleration
            yaw_accel = ddyaw

            # # Calculate jerk
            # ddx = np.diff(dx) / timestep
            # ddy = np.diff(dy) / timestep
            # dddx = np.diff(ddx) / timestep
            # dddy = np.diff(ddy) / timestep
            # heading = np.arctan2(dy, dx)
            # lon_accel = ddx * np.cos(heading[:-1]) + ddy * np.sin(heading[:-1])
            # lat_accel = -ddx * np.sin(heading[:-1]) + ddy * np.cos(heading[:-1])
            # # Calculate longitudinal jerk
            # lon_jerk = dddx * np.cos(heading[:-2]) + dddy * np.sin(heading[:-2])

            lon_accel = accel
            lat_accel = np.zeros_like(lon_accel)
            lon_jerk = jerk

            # Pad arrays to match the original trajectory length
            yaw_rate = np.pad(yaw_rate, (0, 1), 'edge')
            yaw_accel = np.pad(yaw_accel, (0, 2), 'edge')
            lon_accel = np.pad(lon_accel, (0, 2), 'edge')
            lat_accel = np.pad(lat_accel, (0, 2), 'edge')
            lon_jerk = np.pad(lon_jerk, (0, 3), 'edge')

            return {
                'speed': speed,
                'yaw_rate': yaw_rate,
                'yaw_accel': yaw_accel,
                'lon_accel': lon_accel,
                'lat_accel': lat_accel,
                'lon_jerk': lon_jerk,
            }

        # Calculate kinematic parameters
        if len(traj) < 4:
            return 1.0

        kinematics = calculate_trajectory_kinematics(traj, timestep)

        # Check each parameter against its boundary
        checks = [
            np.all(np.abs(kinematics['lat_accel']) <=
                   boundaries['max_abs_lat_accel']),
            np.all(kinematics['lon_accel'] <= boundaries['max_abs_lat_accel']),
            np.all(kinematics['lon_accel'] >= boundaries['min_lon_accel']),
            np.all(np.abs(kinematics['lon_jerk']) <=
                   boundaries['max_abs_lon_jerk']),
            np.all(np.abs(kinematics['yaw_accel']) <=
                   boundaries['max_abs_yaw_accel']),
            np.all(np.abs(kinematics['yaw_rate']) <=
                   boundaries['max_abs_yaw_rate'])
        ]

        if not all(checks):
            print(traj)
            print(kinematics)
        print(f"comfortable: {all(checks)} ,{checks}")

        # Return 1.0 if all checks pass, 0.0 otherwise
        return 1.0 if all(checks) else 0.0

    def _calculate_no_collision(self, ego_box, planned_traj, obs_lists):
        def create_rectangle(center_x, center_y, width, length, yaw):
            """Create a rectangle polygon."""
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)

            x_offs = [length/2, length/2, -length/2, -length/2]
            y_offs = [width/2, -width/2, -width/2, width/2]

            x_pts = [center_x + x_off*cos_yaw - y_off *
                     sin_yaw for x_off, y_off in zip(x_offs, y_offs)]
            y_pts = [center_y + x_off*sin_yaw + y_off *
                     cos_yaw for x_off, y_off in zip(x_offs, y_offs)]

            return ShapelyPolygon(zip(x_pts, y_pts))

        ego_x, ego_y, _, ego_w, ego_l, ego_h, ego_yaw = ego_box
        for idx, obs_list in enumerate(obs_lists):
            ego_x, ego_y, ego_yaw = planned_traj[idx]  # ego_state= (x,y,yaw)
            ego_poly = create_rectangle(ego_x, ego_y, ego_w, ego_l, ego_yaw)
            for obs in obs_list:
                # obs = (x,y,z,w,l,h,yaw)
                obs_x, obs_y, _, obs_w, obs_l, _, obs_yaw = obs
                obs_poly = create_rectangle(
                    obs_x, obs_y, obs_w, obs_l, obs_yaw)
                if ego_poly.intersects(obs_poly):
                    print(f"collision with obstacle detected! @ timestep{idx}")
                    print(
                        f"ego_poly: {(ego_x, ego_y,ego_yaw,obs_w, obs_l)}, obs_poly: {(obs_x, obs_y, obs_yaw,obs_w, obs_l )}")
                    return 0.0  # Collision detected

        return 1.0

    def _calculate_time_to_collision(self, ego_box, planned_traj, obs_lists, timestep):
        # breakpoint()
        t_list = [0.5, 1]  # ttc time

        for t in t_list:
            # Calculate velocities
            velocities = np.diff(planned_traj[:, :2], axis=0) / timestep

            # Use the velocity of the second point for the first point
            velocities = np.vstack([velocities[0], velocities])

            # Calculate the displacement
            displacement = velocities * t

            # Create the new trajectory
            new_traj = planned_traj.copy()
            new_traj[:, :2] += displacement

            is_collide_score = self._calculate_no_collision(
                ego_box, new_traj, obs_lists)
            if is_collide_score == 0.0:
                print(f" failed to pass ttc collision check, t={t}")
                # breakpoint()
                return 0.0

        return 1.0

    def calculate(self, ):

        print(f"current exp has {len(self.data['frames'])} frames")
        if len(self.data['frames']) == 0:
            return None
        # todo: time_step need modify
        score_list = {}
        for i in range(0, len(self.data['frames']), 1):
            frame = self.data['frames'][i]
            if frame['is_key_frame'] == False:
                continue

            print(f"frame {i} / {len(self.data['frames'])}")
            timestamp = frame['time_stamp']
            planned_last_timestamp = timestamp + \
                len(frame['planned_traj']['traj']) * \
                frame['planned_traj']['timestep']
            ego_x, ego_y, _, ego_w, ego_l, _, ego_yaw = frame['ego_box']
            # frame['planned_traj']['traj']
            if len(frame['planned_traj']['traj'])<2:
                continue
            planned_traj = np.concatenate(
                ([np.array([ego_x, ego_y, ego_yaw])], frame['planned_traj']['traj']), axis=0)
            ref_traj = np.concatenate(
                ([np.array([ego_x, ego_y, ego_yaw])], frame['ref_traj']['traj']), axis=0)
            
            # planned_traj =  frame['planned_traj']['traj']
            # planned_traj= ref_traj
            # ref_traj =frame['ref_traj']['traj']

            # if the car is stopped, there may be error in the yaw of the planned trajectory
            traj_distance =  np.linalg.norm(planned_traj[-1, :2]  - planned_traj[0, :2] )
            if traj_distance<1:
                 planned_traj[:, 2] = planned_traj[0, 2]  # set all yaw to the first yaw

            current_timestamp = timestamp
            current_frame_idx = i
            obs_lists = []
            while current_timestamp <= planned_last_timestamp+1e-5:
                if abs(current_timestamp - self.data['frames'][current_frame_idx]['time_stamp']) < 1e-5:
                    obs_list = []
                    for idx, obj in enumerate(self.data['frames'][current_frame_idx]['obj_boxes']):
                        # obs_list.append(obj)
                        if self.data['frames'][current_frame_idx]['obj_names'][idx] == 'car':
                            obs_list.append(obj)
                    obs_lists.append(obs_list)
                    # obs_lists.append(
                    #     self.data['frames'][current_frame_idx]['obj_boxes'])
                    current_timestamp += frame['planned_traj']['timestep']

                current_frame_idx += 1
                if current_frame_idx >= len(self.data['frames']):
                    break

            # breakpoint()
            # plt.imshow(frame['drivable_mask'].astype(np.uint8))
            # plt.show()
            # transformed_traj = self.transform_to_ego_frame(frame['planned_traj']['traj'], frame['ego_box'])
            transformed_traj = self.transform_to_ego_frame(
                planned_traj, frame['ego_box'])
            # breakpoint()

            score_nc = self._calculate_no_collision(
                frame['ego_box'], planned_traj, obs_lists)
            score_dac = self._calculate_drivable_area_compliance(
                frame['drivable_mask'].astype(np.uint8), transformed_traj, ego_w, ego_l)
            score_ttc = self._calculate_time_to_collision(
                frame['ego_box'], planned_traj, obs_lists, frame['planned_traj']['timestep'])
            score_c = self._calculate_is_comfortable(
                transformed_traj, frame['planned_traj']['timestep'])
            score_ep = self._calculate_progress(
                planned_traj, ref_traj)
            score_pdms = score_nc*score_dac*(score_weight['ttc']*score_ttc+score_weight['c']*score_c+score_weight['ep']*score_ep)/(
                score_weight['ttc']+score_weight['c']+score_weight['ep'])
            score_list[timestamp] = {'nc': score_nc, 'dac': score_dac,
                                     'ttc': score_ttc, 'c': score_c, 'ep': score_ep, 'pdms': score_pdms}
            
            # self.plot_trajectory_on_drivable_mask(
            #     frame['drivable_mask'].astype(np.uint8), transformed_traj, ego_w, ego_l)

        totals = {metric: 0 for metric in next(iter(score_list.values()))}
        for scores in score_list.values():
            for metric, value in scores.items():
                totals[metric] += value

        # avg scores
        num_entries = len(score_list)
        averages = {metric: total / num_entries for metric,
                    total in totals.items()}

        # write to csv
        with open('pdms_data.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(averages.values())

        print(averages)
        mean_score = averages['pdms']
        if self.data['type'] == 'closed-loop':
            route_completion = self.data['drive_length'] / \
                self.data['route_length']
            print(route_completion)
            driving_score = mean_score*route_completion
        else:
            route_completion = None
            driving_score = mean_score
        return mean_score, route_completion, driving_score, averages


def main(data):
    print(f"this pkl file contains {len(data)} experiment records.")
    # print(f"the first item metadata is {data[0]['metas']}.")
    # breakpoint()

    def process_exp_data(exp_data):
        score_calc = ScoreCalculator(exp_data)
        score = score_calc.calculate()
        print(f"The score of experiment is {score}.")
        return score[3]

    def multi_threaded_process(data, max_workers=None):
        all_averages = []

        # 使用线程锁来确保线程安全的追加操作
        lock = threading.Lock()

        def append_result(future):
            result = future.result()
            with lock:
                all_averages.append(result)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_exp_data, exp_data)
                       for exp_data in data]
            for future in futures:
                future.add_done_callback(append_result)

        return all_averages

    # 使用方法
    all_averages = multi_threaded_process(data)

    # 收集所有循环的值
    collected_values = defaultdict(list)
    for averages in all_averages:
        for key, value in averages.items():
            collected_values[key].append(value)

    # 计算每个指标的平均值和标准差
    results = {}
    for key, values in collected_values.items():
        avg = np.mean(values)
        std = np.std(values)
        results[key] = f"{avg:.4f}±{std:.4f}"

    # 输出结果
    print("=============================Results=============================")
    for key, value in results.items():
        print(f"'{key}': {value}")
    return


if __name__ == "__main__":
    # Check if a file name is provided as a command-line argument
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
    else:
        # Default file name if no argument is provided
        file_name = "./results/08-23-150733/"+"drive_arena.pkl"
        # file_name = "gen_pdms_data.pkl"
    
    # Print the file name being used
    print(f"Using file: {file_name}")

    # Open the file and load the data
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    
    # Call the main function with the loaded data
    main(data)