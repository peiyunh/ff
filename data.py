# data.py
# data loader for nuscenes
import os
import torch
import numpy as np
import warnings
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.eval.common.utils import quaternion_yaw

import sampler as trajectory_sampler

def locate_message(utimes, utime):
    i = np.searchsorted(utimes, utime)
    if i > 0 and utime - utimes[i-1] < utimes[i] - utime:
        i -= 1
    return i

def CollateFn(batch):
    examples = {
        "scene_tokens": [],
        "sample_data_tokens": [],
        "input_points": [],
        "sampled_trajectories": [],
        "drive_commands": [],
        "output_origins": [],
        "output_points": [],
        "gt_trajectories": [],
        "obj_boxes": [],
        "obj_shadows": [],
        "fvf_maps": [],
    }

    max_n_input_points = max([len(example["input_points"]) for example in batch])
    max_n_output_points = max([len(example["output_points"]) for example in batch])

    examples = {
        "scene_tokens": [example["scene_token"] for example in batch],
        "sample_data_tokens": [example["sample_data_token"] for example in batch],
        "input_points": torch.stack([
            torch.nn.functional.pad(
                example["input_points"], (0, 0, 0, max_n_input_points - len(example["input_points"])),
                mode="constant", value=-1,
            ) for example in batch
        ]),
        "sampled_trajectories_fine": torch.stack([example["sampled_trajectories_fine"] for example in batch]),
        "sampled_trajectories": torch.stack([example["sampled_trajectories"] for example in batch]),
        "drive_commands": torch.stack([example["drive_command"] for example in batch]),
        "output_origins": torch.stack([example["output_origin"] for example in batch]),
        "output_points": torch.stack([
            torch.nn.functional.pad(
                example["output_points"], (0, 0, 0, max_n_output_points - len(example["output_points"])),
                mode="constant", value=-1,
            ) for example in batch
        ]),
        "gt_trajectories": torch.stack([example["gt_trajectory"] for example in batch]),
        "obj_boxes": torch.stack([example["obj_boxes"] for example in batch]),
        "obj_shadows": torch.stack([example["obj_shadows"] for example in batch]),
        "fvf_maps": torch.stack([example["fvf_maps"] for example in batch])
    }

    examples["scene_tokens"] = [example["scene_token"] for example in batch]

    return examples


class MyLidarPointCloud(LidarPointCloud):
    def remove_ego(self):
        ego_mask = np.logical_and(
            np.logical_and(-0.8 <= self.points[0], self.points[0] <= 0.8),
            np.logical_and(-1.5 <= self.points[1], self.points[1] <= 2.5)
        )
        self.points = self.points[:, np.logical_not(ego_mask)]


class nuScenesDataset(Dataset):
    N_SWEEPS_PER_SAMPLE = 10
    SAMPLE_INTERVAL = 0.5  # second

    def __init__(self, nusc, nusc_split, kwargs, seed=0):
        super(nuScenesDataset, self).__init__()

        # set seed for split
        np.random.seed(seed)

        self.nusc = nusc
        self.nusc_root = self.nusc.dataroot
        self.nusc_can = NuScenesCanBus(dataroot=self.nusc_root)
        self.nusc_split = nusc_split

        # number of input samples
        self.n_input = kwargs["n_input"]

        # number of sampled trajectories
        self.n_samples = kwargs["n_samples"]

        # number of output samples
        self.n_output = kwargs["n_output"]
        assert(self.n_output == 7)

        #
        self.train_on_all_sweeps = kwargs["train_on_all_sweeps"]

        # scene-0419 does not have vehicle monitor data
        blacklist = [419] + self.nusc_can.can_blacklist

        # NOTE: use the official split (minus the ones in the blacklist)
        if "scene_token" in kwargs and kwargs["scene_token"] != "":
            scene = self.nusc.get("scene", kwargs["scene_token"])
            scenes = [scene]
        else:
            scene_splits = create_splits_scenes(verbose=False)
            scene_names = scene_splits[self.nusc_split]
            scenes = []
            for scene in self.nusc.scene:
                scene_name = scene["name"]
                scene_no = int(scene_name[-4:])
                if (scene_name in scene_names) and (scene_no not in blacklist):
                    scenes.append(scene)

        # list all sample data
        self.valid_index = []
        self.flip_flags = []
        self.scene_tokens = []
        self.sample_data_tokens = []
        for scene in scenes:
            scene_token = scene["token"]
            # location
            log = self.nusc.get("log", scene["log_token"])
            # flip x axis if in left-hand traffic (singapore)
            flip_flag = True if log["location"].startswith("singapore") else False
            # record the token of every key frame
            start_index = len(self.sample_data_tokens)
            first_sample = self.nusc.get("sample", scene["first_sample_token"])
            sample_data_token = first_sample["data"]["LIDAR_TOP"]
            while sample_data_token != "":
                sample_data = self.nusc.get("sample_data", sample_data_token)
                if (self.nusc_split == "train" and self.train_on_all_sweeps) or (sample_data["is_key_frame"]):
                    self.flip_flags.append(flip_flag)
                    self.scene_tokens.append(scene_token)
                    self.sample_data_tokens.append(sample_data_token)
                sample_data_token = sample_data["next"]
            end_index = len(self.sample_data_tokens)
            # NOTE: make sure we have enough number of sweeps for input and output
            if self.nusc_split == "train" and self.train_on_all_sweeps:
                valid_start_index = start_index + self.n_input - 1
                valid_end_index = end_index - (self.n_output - 1) * self.N_SWEEPS_PER_SAMPLE
            else:
                # NEW: acknowledge the fact and skip the first sample
                n_input_samples = self.n_input // self.N_SWEEPS_PER_SAMPLE
                valid_start_index = start_index + n_input_samples
                valid_end_index = end_index - self.n_output + 1
            self.valid_index += list(range(valid_start_index, valid_end_index))
        self._n_examples = len(self.valid_index)
        print(f"{self.nusc_split}: {self._n_examples} valid samples over {len(scenes)} scenes")

    def __len__(self):
        return self._n_examples

    def get_global_pose(self, sd_token, inverse=False):
        sd = self.nusc.get("sample_data", sd_token)
        sd_ep = self.nusc.get("ego_pose", sd["ego_pose_token"])
        sd_cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        if inverse is False:
            global_from_ego = transform_matrix(sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=False)
            ego_from_sensor = transform_matrix(sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=False)
            pose = global_from_ego.dot(ego_from_sensor)
        else:
            sensor_from_ego = transform_matrix(sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=True)
            ego_from_global = transform_matrix(sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=True)
            pose = sensor_from_ego.dot(ego_from_global)
        return pose

    def load_ground_segmentation(self, sample_data_token):
        path = f"{self.nusc.dataroot}/grndseg/{self.nusc.version}/{sample_data_token}_grndseg.bin"
        labels = np.fromfile(path, np.uint8)
        return labels

    def load_future_visible_freespace(self, sample_data_token):
        path = f"{self.nusc.dataroot}/fvfmaps/{self.nusc.version}/{sample_data_token}.bin"
        if os.path.exists(path):
            fvf_maps = np.fromfile(path, dtype=np.int8)
            fvf_maps = fvf_maps.reshape((7, 704, 400))
        else:
            fvf_maps = np.zeros((7, 704, 400), dtype=np.int8)
            warnings.warn(f"Cannot find fvf_maps at {path}")
        return fvf_maps

    def load_object_boxes(self, sample_data_token):
        path = f"{self.nusc.dataroot}/obj_boxes/{self.nusc.version}/{sample_data_token}.bin"
        if os.path.exists(path):
            obj_boxes = np.fromfile(path, dtype=bool)
            obj_boxes = obj_boxes.reshape((7, 704, 400))
        else:
            obj_boxes = np.zeros((7, 704, 400))
        return obj_boxes

    def load_object_shadows(self, sample_data_token):
        path = f"{self.nusc.dataroot}/obj_shadows/{self.nusc.version}/{sample_data_token}.bin"
        if os.path.exists(path):
            obj_shadows = np.fromfile(path, dtype=bool)
            obj_shadows = obj_shadows.reshape((7, 704, 400))
        else:
            obj_shadows = np.zeros((7, 704, 400))
        return obj_shadows

    def __getitem__(self, idx):
        ref_index = self.valid_index[idx]

        ref_sd_token = self.sample_data_tokens[ref_index]
        ref_scene_token = self.scene_tokens[ref_index]
        flip_flag = self.flip_flags[ref_index]

        # reference coordinate frame
        ref_from_global = self.get_global_pose(ref_sd_token, inverse=True)

        # NOTE: input
        input_sds = []
        sd_token = ref_sd_token
        while sd_token != "" and len(input_sds) < self.n_input:
            curr_sd = self.nusc.get("sample_data", sd_token)
            input_sds.append(curr_sd)
            sd_token = curr_sd["prev"]

        # call out when we have less than the desired number of input sweeps
        # if len(input_sds) < self.n_input:
        #     warnings.warn(f"The number of input sweeps {len(input_sds)} is less than {self.n_input}.", RuntimeWarning)

        # get input sweep frames
        input_points_list = []
        for i, curr_sd in enumerate(input_sds):
            # load the current lidar sweep
            curr_lidar_pc = MyLidarPointCloud.from_file(f"{self.nusc_root}/{curr_sd['filename']}")

            # remove ego returns
            curr_lidar_pc.remove_ego()

            # transform from the current lidar frame to global and then to the reference lidar frame
            global_from_curr = self.get_global_pose(curr_sd["token"], inverse=False)
            ref_from_curr = ref_from_global.dot(global_from_curr)
            curr_lidar_pc.transform(ref_from_curr)

            # NOTE: check if we are in Singapore (if so flip x)
            if flip_flag:
                curr_lidar_pc.points[0] *= -1

            #
            points = np.asarray(curr_lidar_pc.points[:3].T)
            tindex = np.full((len(points), 1), i)
            points = np.concatenate((points, tindex), axis=1)

            #
            input_points_list.append(points.astype(np.float32))

        # NOTE: output
        # get output sample frames and ground truth trajectory
        output_origin_list = []
        output_points_list = []
        gt_trajectory = np.zeros((self.n_output, 3), np.float64)
        for i in range(self.n_output):
            if self.nusc_split == "train" and self.train_on_all_sweeps:
                index = ref_index + i * self.N_SWEEPS_PER_SAMPLE
            else:
                index = ref_index + i

            # if this exists a valid target
            if index < len(self.scene_tokens) and self.scene_tokens[index] == ref_scene_token:
                curr_sd_token = self.sample_data_tokens[index]
                curr_sd = self.nusc.get("sample_data", curr_sd_token)

                # load the current lidar sweep
                curr_lidar_pc = LidarPointCloud.from_file(f"{self.nusc_root}/{curr_sd['filename']}")

                # transform from the current lidar frame to global and then to the reference lidar frame
                global_from_curr = self.get_global_pose(curr_sd_token, inverse=False)
                ref_from_curr = ref_from_global.dot(global_from_curr)
                curr_lidar_pc.transform(ref_from_curr)

                #
                theta = quaternion_yaw(Quaternion(matrix=ref_from_curr))

                # NOTE: check if we are in Singapore (if so flip x)
                if flip_flag:
                    ref_from_curr[0, 3] *= -1
                    curr_lidar_pc.points[0] *= -1
                    theta *= -1

                origin = np.array(ref_from_curr[:3, 3])
                points = np.array(curr_lidar_pc.points[:3].T)
                gt_trajectory[i, :] = [origin[0], origin[1], theta]

                tindex = np.full(len(points), i)

                labels = self.load_ground_segmentation(curr_sd_token)
                assert(len(labels) == len(points))
                mask = np.logical_and(labels >= 1, labels <= 30)

                points = np.concatenate((points, tindex[:, None], labels[:, None]), axis=1)
                points = points[mask, :]

            else:  # filler
                raise RuntimeError(f"The {i}-th output frame is not available")
                origin = np.array([0.0, 0.0, 0.0])
                points = np.full((0, 5), -1)

            # origin
            output_origin_list.append(origin.astype(np.float32))

            # points
            output_points_list.append(points.astype(np.float32))

        # NOTE: trajectory sampling
        ref_scene = self.nusc.get("scene", ref_scene_token)

        # NOTE: rely on pose and steeranglefeedback data instead of vehicle_monitor
        vm_msgs = self.nusc_can.get_messages(ref_scene["name"], "vehicle_monitor")
        vm_uts = [msg["utime"] for msg in vm_msgs]
        pose_msgs = self.nusc_can.get_messages(ref_scene["name"], "pose")
        pose_uts = [msg["utime"] for msg in pose_msgs]
        steer_msgs = self.nusc_can.get_messages(ref_scene["name"], "steeranglefeedback")
        steer_uts = [msg["utime"] for msg in steer_msgs]

        # locate the closest message by universal timestamp
        ref_sd = self.nusc.get("sample_data", ref_sd_token)
        ref_utime = ref_sd["timestamp"]
        vm_index = locate_message(vm_uts, ref_utime)
        vm_data = vm_msgs[vm_index]
        pose_index = locate_message(pose_uts, ref_utime)
        pose_data = pose_msgs[pose_index]
        steer_index = locate_message(steer_uts, ref_utime)
        steer_data = steer_msgs[steer_index]

        # initial speed
        # v0 = vm_data["vehicle_speed"] / 3.6  # km/h to m/s
        v0 = pose_data["vel"][0]  # [0] means longitudinal velocity

        # curvature (positive: turn left)
        # steering = np.deg2rad(vm_data["steering"])
        steering = steer_data["value"]
        if flip_flag:
            steering *= -1
        Kappa = 2 * steering / 2.588

        #
        left_signal = vm_data["left_signal"]
        right_signal = vm_data["right_signal"]
        if flip_flag:
            left_signal, right_signal = right_signal, left_signal
        drive_command = [left_signal, right_signal]

        # initial state
        T0 = np.array([0.0, 1.0])  # define front
        N0 = np.array([1.0, 0.0]) if Kappa <= 0 else np.array([-1.0, 0.0])  # define side

        #
        # tt = np.arange(self.n_output) * self.SAMPLE_INTERVAL
        # tt = np.arange(0, self.n_output + self.SAMPLE_INTERVAL, self.SAMPLE_INTERVAL)
        t_start = 0  # second
        t_end = (self.n_output-1) * self.SAMPLE_INTERVAL  # second
        t_interval = self.SAMPLE_INTERVAL / 10
        tt = np.arange(t_start, t_end + t_interval, t_interval)
        sampled_trajectories_fine = trajectory_sampler.sample(v0, Kappa, T0, N0, tt, self.n_samples)
        sampled_trajectories = sampled_trajectories_fine[:, ::10]

        #
        obj_boxes = self.load_object_boxes(ref_sd_token)
        obj_shadows = self.load_object_shadows(ref_sd_token)

        #
        fvf_maps = self.load_future_visible_freespace(ref_sd_token)

        #
        example = {
            "scene_token": ref_scene_token,
            "sample_data_token": ref_sd_token,
            "input_points": torch.from_numpy(np.concatenate(input_points_list)),
            "sampled_trajectories_fine": torch.from_numpy(sampled_trajectories_fine),
            "sampled_trajectories": torch.from_numpy(sampled_trajectories),
            "drive_command": torch.tensor(drive_command),
            "output_origin": torch.from_numpy(np.stack(output_origin_list)),
            "output_points": torch.from_numpy(np.concatenate(output_points_list)),
            "gt_trajectory": torch.from_numpy(gt_trajectory),
            "obj_boxes": torch.from_numpy(obj_boxes),
            "obj_shadows": torch.from_numpy(obj_shadows),
            "fvf_maps": torch.from_numpy(fvf_maps),
        }
        return example
