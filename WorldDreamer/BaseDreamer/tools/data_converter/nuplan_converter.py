import base64
import os
import pickle
import threading
import time

from datetime import datetime
from functools import reduce
from os import path as osp

import mmcv
import numpy as np
import requests
import yaml
from nuplan.database.nuplan_db_orm.frame import Frame
from nuplan.database.nuplan_db_orm.nuplandb_wrapper import NuPlanDBWrapper
from nuplan.database.nuplan_db_orm.rendering_utils import lidar_pc_closest_image
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

id2local = {
    0: "generic_object",
    1: "car",
    2: "pedestrian",
    3: "bicycle",
    4: "traffic_cone",
    5: "barrier",
    6: "czone_sign",
}

global2local = {
    "generic_object": "generic_object",
    "vehicle": "car",
    "pedestrian": "ped",
    "bicycle": "bike",
    "traffic_cone": "traffic_cone",
    "barrier": "barrier",
    "czone_sign": "czone_sign",
}


camera_types = [
    "CAM_B0",
    "CAM_F0",
    "CAM_L0",
    "CAM_L1",
    "CAM_R0",
    "CAM_R1",
]


frames_t = 6
use_break = eval(os.environ.get("USE_BREAK", "False"))


def load_and_combine_pkl_files(directory, test=False):
    """Load multiple .pkl files and combine 'train.pkl' and 'val.pkl' files into separate summaries."""
    if test:
        data_types = ['test']
    else:
        data_types = ['train', 'val']

    data_summary = {}

    for data_type in data_types:
        files = [f for f in os.listdir(directory) if f.endswith(f"{data_type}.pkl")]
        infos = []
        scene_tokens = []
        metadata = None
        
        with tqdm(total=len(files), desc=f"Processing {data_type} files") as pbar:
            for filename in files:
                filepath = os.path.join(directory, filename)
                try:
                    with open(filepath, "rb") as file:
                        data = pickle.load(file)
                        infos.extend(data["infos"])
                        scene_tokens.extend(data["scene_tokens"])
                        if metadata is None:
                            metadata = data["metadata"]
                except Exception as e:
                    print(f"Failed to process {filename}: {e}")
                pbar.update(1)
        
        if files: 
            data_summary[data_type] = {
                "infos": infos,
                "scene_tokens": scene_tokens,
                "metadata": metadata
            }

    if test:
        return {"test": data_summary.get("test", {})}
    else:
        return {k: v for k, v in data_summary.items() if k in ['train', 'val']}


def save_combined_data(data, output_filepath):
    """Save the combined data to a .pkl file."""
    with open(output_filepath, "wb") as file:
        pickle.dump(data, file)


def create_nuplan_infos(
    root_path,
    info_prefix,
    split_yaml,
    version="v1.1-mini",
    map_version="nuplan-maps-v1.0",
    out_path=None,
):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
        max_sweeps (int): Max number of sweeps.
            Default: 10
    """

    out_path = out_path if out_path else root_path
    tmp_out_path = out_path + "/tmp/"
    available_vers = [
        "v1.1-trainval",
        "v1.1-test",
        "v1.1-mini",
        "dreamer-trainval",
    ]

    # load the split distribution from yaml
    with open(split_yaml, "r") as file:
        data = yaml.safe_load(file)

    log_splits = data.get("log_splits", {})

    assert version in available_vers
    if version in "v1.1-trainval":
        train_scenes = log_splits["train"]
        val_scenes = log_splits["val"]
    elif version == "v1.1-test":
        train_scenes = log_splits["test"]
        val_scenes = []
    elif version == "v1.1-mini":
        train_scenes = log_splits["mini_train"]
        val_scenes = log_splits["mini_val"]
    elif version == "dreamer-trainval":
        train_scenes = log_splits["dreamer_train"]
        val_scenes = log_splits["dreamer_val"]
    else:
        raise ValueError("unknown")
    
    test = "test" in version
    if test:
        print("test scene: {}".format(len(train_scenes)))
    else:
        print(
            "train scene: {}, val scene: {}".format(len(train_scenes), len(val_scenes))
        )
    
    _fill_trainval_infos(
        root_path,
        train_scenes,
        val_scenes,
        version=version,
        test=test,
        map_version="nuplan-maps-v1.0",
        out_path=tmp_out_path,
    )

    # Load and combine the PKL files
    combined_data = load_and_combine_pkl_files(tmp_out_path, test)

    # Save the combined data
    if test:
        save_combined_data(combined_data["test"], out_path + "/nuplan_infos_test.pkl")
    else:
        save_combined_data(combined_data["train"], out_path + "/nuplan_infos_train.pkl")
        save_combined_data(combined_data["val"], out_path + "/nuplan_infos_val.pkl")

    print(f"Combined pkl files saved to: {out_path}")


def isTimeFormat(data: str, format: str) -> bool:
    try:
        time.strptime(data, format)
        return True
    except ValueError:
        return False


def quaternion_multiply(q1, q2):
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def _sample_data_transform(
    nuplandb,
    lidar_pc_rec,
    description, 
):
    lidar_pc_token = lidar_pc_rec.token
    lidar_rec = nuplandb.lidar[0]
    pose_rec = lidar_pc_rec.ego_pose

    location = nuplandb.log.map_version

    logfile = nuplandb.log.logfile
    datetime_str = logfile.split("_")[0]

    datetime_obj = datetime.strptime(datetime_str, "%Y.%m.%d.%H.%M.%S")
    timeofday = datetime_obj.strftime("%Y-%m-%d-%H-%M-%S")

    lidar_path = (
        nuplandb.data_root + "nuplan-v1.1/sensor_blobs/" + lidar_pc_rec.filename
    )
    lidar_path = str(lidar_path)

    rotation_z_pos90 = np.array(
        [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )

    e2g = np.eye(4)
    e2g = pose_rec.trans_matrix
    e2g = rotation_z_pos90 @ e2g

    e2g_t = e2g[:3, 3]
    e2g_r = e2g[:3, :3]
    e2g_q = Quaternion(quaternion_from_rotation_matrix(e2g_r))
    e2g_q = np.array([e2g_q.w, e2g_q.x, e2g_q.y, e2g_q.z])

    info = {
        "lidar_path": lidar_path,
        "token": lidar_pc_token,
        "sweeps": [],
        "cams": dict(),
        "lidar2ego_translation": lidar_rec.translation_np,
        "lidar2ego_rotation": lidar_rec.rotation,
        "ego2global_translation": e2g_t, 
        "ego2global_rotation": e2g_q,
        "timestamp": lidar_pc_rec.timestamp,
        "location": location,
        "description": description,
        "timeofday": timeofday,
        "is_key_frame": True,
    }

    l2e_r = info["lidar2ego_rotation"]
    l2e_t = info["lidar2ego_translation"]
    e2g_r = info["ego2global_rotation"]
    e2g_t = info["ego2global_translation"]
    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = Quaternion(e2g_r).rotation_matrix

    # obtain 8 image's information per frame
    image_sample_list = lidar_pc_closest_image(
        lidar_pc=lidar_pc_rec, camera_channels=camera_types
    )

    for image_rec in image_sample_list:
        cam_rec = image_rec.camera
        cam = cam_rec.channel
        cam_intrinsics = cam_rec.intrinsic_np
        cam_info = obtain_sensor2top(nuplandb, image_rec, lidar_pc_rec, cam)
        cam_info.update(camera_intrinsics=cam_intrinsics)
        info["cams"].update({cam: cam_info})
    info["sweeps"] = []

    annotations = lidar_pc_rec.lidar_boxes
    velocity = []
    for ann in annotations:
        vel = ann.velocity[:2]
        velocity.append(vel)

    velocity = np.array(velocity)

    # boxes in lidar coordinate frame
    boxes = lidar_pc_rec.boxes(frame=Frame.SENSOR)  # list of Box3D objects
    locs = np.array([b.center for b in boxes]).reshape(-1, 3)
    dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
    rots = np.array([b.yaw for b in boxes]).reshape(-1, 1)

    #To minimize the differences in sensor setup between the NuScenes and NuPlan 
    # datasets, we applied a 90-degree rotation around the Z-axis to the LiDAR/ego 
    # axis in the NuPlan dataset during the information generation process.
    theta = np.pi / 2  # 90 degrees
    rotation_matrix_z = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )

    # Rotate locations
    locs_rotated = np.dot(locs, rotation_matrix_z.T)

    for i in range(len(boxes)):
        velo = np.array([*velocity[i], 0.0])
        velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
        velo = np.dot(velo, rotation_matrix_z)
        velocity[i] = velo[:2]

    labels = [b.label for b in boxes]
    for i in range(len(labels)):
        if labels[i] in id2local:
            labels[i] = id2local[labels[i]]
    labels = np.array(labels)


    gt_boxes = np.concatenate([locs_rotated, dims, -rots - np.pi], axis=1)
    assert len(gt_boxes) == len(annotations), f"{len(gt_boxes)}, {len(annotations)}"
    info["gt_boxes"] = np.array(gt_boxes)
    info["gt_names"] = np.array(labels)
    info["gt_velocity"] = np.array(velocity.reshape(-1, 2))
    info["num_lidar_pts"] = np.ones(len(annotations), dtype=bool)
    info["num_radar_pts"] = np.ones(len(annotations), dtype=bool)
    info["valid_flag"] = np.ones(len(annotations), dtype=bool)
    info["obj_ids"] = np.array([a.track_token for a in annotations])
    return info


def find_dictionary_by_token(dict_list, search_token):
    for dictionary in dict_list:
        if dictionary["token"] == search_token:
            return dictionary
    return None


def split_list(data, num_splits):
    """Split the list into nearly equal parts."""
    k, m = divmod(len(data), num_splits)
    return (
        data[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(num_splits)
    )


def process_scenes(chunk, results, index, nuplandb, progress=None):
    local_train_nuplan_infos = []
    local_train_scene_tokens = []

    if progress is None:
        progress = tqdm(total=len(chunk), desc="Processing scenes", leave=False)

    for scene in chunk:
        scene_token = scene.token
        _scene_tokens = []
        infos = []

        lidar_pcs = nuplandb.lidar_pc.select_many(scene_token=scene_token)[::2]
        idx = int(len(lidar_pcs) / 2)
        image_path = (
            nuplandb.data_root
            + "nuplan-v1.1/sensor_blobs/"
            + lidar_pc_closest_image(lidar_pcs[idx], ["CAM_F0"])[0].filename_jpg
        )

        for lidar_pc in lidar_pcs:
            lidar_pc_token = lidar_pc.token
            _scene_tokens.append(lidar_pc_token)
            info = _sample_data_transform(nuplandb, lidar_pc, description='') 
            infos.append(info)
        local_train_scene_tokens.append(_scene_tokens)
        local_train_nuplan_infos.extend(infos)
        progress.update(1)
    results[index] = (local_train_scene_tokens, local_train_nuplan_infos)


def is_log_name_in_dict(log_name, log_names):
    """Check if the log name is present in any of the lists in the dictionary."""
    for scenes in log_names:
        if log_name == scenes:
            return True
    return False


def _fill_trainval_infos(
    root_path,
    train_scenes,
    val_scenes,
    version="v1.1-mini",
    test=False,
    map_version="nuplan-maps-v1.0",
    out_path=None,
):
    """Generate the train/val infos from the raw data.

    Args:
        nuplandb_wrapper (:obj:`NuplanDBWrapper`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool): Whether use the test mode. In the test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """

    ver = version.split("-")[-1]
    nuplandb_wrapper = NuPlanDBWrapper(
        data_root=root_path,
        map_root=f"{root_path}/maps",
        db_files=f"{root_path}/nuplan-v1.1/splits/{ver}",
        map_version=map_version,
    )

    out_path = out_path if out_path else root_path

    log_dbs = nuplandb_wrapper.log_dbs
    num_cores = 64
    results = {} 
    threads = []
    for log_db in log_dbs:

        scenes = log_db.scene
        log_name = log_db.log_name

        if not is_log_name_in_dict(log_name, train_scenes) and not is_log_name_in_dict(
            log_name, val_scenes
        ):
            print(f"skip log_db {log_name}")
            continue
        elif is_log_name_in_dict(log_name, train_scenes) and test is False:
            file_path = os.path.join(out_path, f"{log_name}_infos_train.pkl")
            val = False
        elif is_log_name_in_dict(log_name, train_scenes) and test is True:
            file_path = os.path.join(out_path, f"{log_name}_infos_test.pkl")
            val = False
        elif is_log_name_in_dict(log_name, val_scenes):
            file_path = os.path.join(out_path, f"{log_name}_infos_val.pkl")
            val = True

        if os.path.exists(file_path):
            print(f"Skipping {log_name}, already processed.")
            continue

        # Create chunks of scenes to distribute to threads
        parts = list(split_list(scenes, num_cores))
        with tqdm(
            total=len(log_db.scene),
            desc=f"Processing scenes for log file {log_name}",
        ) as pbar:
            for i, part in enumerate(parts):
                thread = threading.Thread(
                    target=process_scenes, args=(part, results, i, log_db, pbar)
                )
                threads.append(thread)
                thread.start()
            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            # Collect results
            train_nuplan_infos = []
            train_scene_tokens = []
            val_nuplan_infos = []
            val_scene_tokens = []

            for i in sorted(results.keys()):
                local_scene_tokens, local_infos = results[i]
                if val:
                    val_nuplan_infos.extend(local_infos)
                    val_scene_tokens.extend(local_scene_tokens)
                else:
                    train_scene_tokens.extend(local_scene_tokens)
                    train_nuplan_infos.extend(local_infos)

        if val:
            metadata = dict(version=version)
            data = dict(
                infos=val_nuplan_infos,
                metadata=metadata,
                scene_tokens=val_scene_tokens,
            )
        else:
            metadata = dict(version=version)
            data = dict(
                infos=train_nuplan_infos,
                metadata=metadata,
                scene_tokens=train_scene_tokens,
            )

        converted_data = convert_to_primitive(data)
        mmcv.dump(converted_data, file_path)

        print(f"pkl for {log_name} succesfully generated")


def convert_to_primitive(data):
    if isinstance(data, dict):
        return {key: convert_to_primitive(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_primitive(item) for item in data]
    elif "Quaternion" in str(type(data)): 
        return [
            data.w,
            data.x,
            data.y,
            data.z,
        ]
    return data


def quaternion_from_rotation_matrix(R):
    if not np.allclose(np.dot(R, R.T), np.eye(3), atol=1e-6) or not np.allclose(
        np.linalg.det(R), 1, atol=1e-6
    ):
        raise ValueError("The input matrix must be a valid rotation matrix")

    q = np.zeros(4)

    q[0] = 0.5 * np.sqrt(max(1 + R[0, 0] + R[1, 1] + R[2, 2], 0))  # w
    q[1] = 0.5 * np.sqrt(max(1 + R[0, 0] - R[1, 1] - R[2, 2], 0))  # x
    q[2] = 0.5 * np.sqrt(max(1 - R[0, 0] + R[1, 1] - R[2, 2], 0))  # y
    q[3] = 0.5 * np.sqrt(max(1 - R[0, 0] - R[1, 1] + R[2, 2], 0))  # z

    q[1] *= np.sign(q[1] * (R[2, 1] - R[1, 2]))  # x
    q[2] *= np.sign(q[2] * (R[0, 2] - R[2, 0]))  # y
    q[3] *= np.sign(q[3] * (R[1, 0] - R[0, 1]))  # z

    return q


def obtain_sensor2top(nuplandb, sd_rec, lidar_pc_rec, sensor_type="lidar"):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sd_rec (LidarPC/Image object): Sample data object.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    if "CAM" in sensor_type:
        cs_record = sd_rec.camera
        data_path = (
            nuplandb.data_root + "nuplan-v1.1/sensor_blobs/" + sd_rec.filename_jpg
        )
    elif sensor_type == "MergedPointCloud":
        cs_record = nuplandb.lidar
        data_path = nuplandb.data_root + "nuplan-v1.1/sensor_blobs/" + sd_rec.filename

    pose_record = sd_rec.ego_pose
    if os.getcwd() in data_path:
        data_path = data_path.split(f"{os.getcwd()}/")[-1]  # relative path

    rotation_z_pos90 = np.array(
        [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )

    e2g = np.eye(4)
    e2g = pose_record.trans_matrix
    e2g = rotation_z_pos90 @ e2g

    e2g_t = e2g[:3, 3]
    e2g_r = e2g[:3, :3]
    e2g_q = Quaternion(quaternion_from_rotation_matrix(e2g_r))

    s2e = np.eye(4)
    s2e = cs_record.trans_matrix
    s2e = rotation_z_pos90 @ s2e

    s2e_t = s2e[:3, 3]
    s2e_r = s2e[:3, :3]
    s2e_q = Quaternion(quaternion_from_rotation_matrix(s2e_r))

    sweep = {
        "data_path": data_path,
        "type": sensor_type,
        "sample_data_token": sd_rec.token,
        "sensor2ego_translation": s2e_t, 
        "sensor2ego_rotation": s2e_q,
        "ego2global_translation": e2g_t, 
        "ego2global_rotation": e2g_q,
        "timestamp": sd_rec.timestamp,
    }

    transform = reduce(
        np.dot,
        [
            lidar_pc_rec.lidar.trans_matrix_inv,
            lidar_pc_rec.ego_pose.trans_matrix_inv,
            pose_record.trans_matrix,
            cs_record.trans_matrix,
        ],
    )
    transform = rotation_z_pos90 @ transform
    sweep["sensor2lidar_rotation"] = transform[:3, :3]
    sweep["sensor2lidar_translation"] = np.array(transform[:3, 3])
    return sweep
