# rasterize.py
# rasterize annotated bounding box to BEV maps
# this is required to identify collisions
import os
import argparse
import numpy as np
from tqdm import tqdm
from functools import reduce

from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.common.utils import boxes_to_sensor
from nuscenes.utils.geometry_utils import transform_matrix

from skimage.draw import polygon

parser = argparse.ArgumentParser()
parser.add_argument("--nusc-root", type=str, default="/data/nuscenes")
parser.add_argument("--nusc-version", type=str, default="v1.0-trainval")
args = parser.parse_args()

nusc_version = args.nusc_version
nusc_root = args.nusc_root

nusc = NuScenes(nusc_version, nusc_root)

obj_box_dir = f"{nusc_root}/obj_boxes/{nusc_version}"
if not os.path.exists(obj_box_dir):
    os.makedirs(obj_box_dir)

obj_shadow_dir = f"{nusc_root}/obj_shadows/{nusc_version}"
if not os.path.exists(obj_shadow_dir):
    os.makedirs(obj_shadow_dir)

def traverse(nusc, name, pointer, length, token):
    tokens = []
    entry = nusc.get(name, token)
    while entry[pointer] != "" and len(tokens) < length:
        tokens.append(entry[pointer])
        entry = nusc.get(name, entry[pointer])
    return tokens

def draw_object_shadow(boxes, x0, y0, xlim, ylim):

    ymin, ymax, ydelta = ylim
    xmin, xmax, xdelta = xlim

    H = int(np.ceil((ymax - ymin) / ydelta))
    W = int(np.ceil((xmax - xmin) / xdelta))

    y0 = (y0 - ymin) / (ymax - ymin) * H
    x0 = (x0 - xmin) / (xmax - xmin) * W

    # NOTE: setting corner at (0,0) may lead to the ray not intersecting
    # vertices = [(0, 0), (H, 0), (H, W), (0, W)]
    # NOTE: we extend the ray such that it always intersect with something
    vertices = [(-1, -1), (H, -1), (H, W), (-1, W)]

    # initialize line segments
    segments = []
    for i in range(len(vertices)):
        src = vertices[i]
        dst = vertices[i + 1] if i < len(vertices) - 1 else vertices[0]
        segments.append((src, dst))

    # break boxes down to vertices and line segments
    for box in boxes:
        # get corner coordinates in top-down 2d view
        X, Y = box.bottom_corners()[:2, :]

        # discretize
        Y = (Y - ymin) / (ymax - ymin) * H
        X = (X - xmin) / (xmax - xmin) * W

        for i in range(len(Y)):
            src = (Y[i], X[i])
            vertices.append(src)
            dst = (Y[i + 1], X[i + 1]) if i < len(Y) - 1 else (Y[0], X[0])
            segments.append((src, dst))

    # the angle of all rays
    thetas = np.array([np.arctan2(y - y0, x - x0) for (y, x) in vertices])

    # augmented angles
    augmented_thetas = []
    for theta in thetas:
        augmented_thetas.extend([theta - 0.00001, theta, theta + 0.00001])

    # sort augmented thetas (pi to -pi)
    order = np.argsort(augmented_thetas)[::-1]
    augmented_thetas = [augmented_thetas[idx] for idx in order]

    #
    intersections = []
    for theta in augmented_thetas:
        r_px, r_py, r_dx, r_dy = x0, y0, np.cos(theta), np.sin(theta)
        r_mag = np.sqrt(r_dx**2 + r_dy**2)

        closest_intersection = None
        closest_T1 = 10000000.0
        for (src, dst) in segments:
            s_px, s_py, s_dx, s_dy = src[1], src[0], (dst[1] - src[1]), (dst[0] - src[0])
            s_mag = np.sqrt(s_dx**2 + s_dy**2)

            # test if ray and line segment are parallel to each other
            if r_dx / r_mag == s_dx / s_mag and r_dy / r_mag == s_dy / s_mag:
                continue

            # solve the intersection equation
            T2 = (r_dx * (s_py - r_py) + r_dy * (r_px - s_px)) / (s_dx * r_dy - s_dy * r_dx)
            T1 = (s_px + s_dx * T2 - r_px) / r_dx

            # intersect behind the sensor
            if T1 < 0:
                continue

            # intersect outside the line segment
            if T2 < 0 or T2 > 1:
                continue

            # derive the coordinate of the intersection
            x, y = r_px + r_dx * T1, r_py + r_dy * T1
            if closest_intersection is None or T1 < closest_T1:
                closest_intersection = (y, x)
                closest_T1 = T1
        # this should not really happen
        if closest_intersection is not None:
            intersections.append(closest_intersection)

    # default to True
    object_shadow = np.ones((H, W), bool)
    for i in range(len(intersections)):
        y1, x1 = intersections[i]
        y2, x2 = intersections[i + 1] if i < len(intersections) - 1 else intersections[0]
        rr, cc = polygon([y0, y1, y2], [x0, x1, x2])
        I = np.logical_and(
            np.logical_and(rr >= 0, rr < H),
            np.logical_and(cc >= 0, cc < W),
        )
        object_shadow[rr[I], cc[I]] = False

    return object_shadow

def draw_obj_boxes(boxes, xlim, ylim):
    ymin, ymax, ydelta = ylim
    xmin, xmax, xdelta = xlim

    H = int(np.ceil((ymax - ymin) / ydelta))
    W = int(np.ceil((xmax - xmin) / xdelta))

    object_mask = np.zeros((H, W), dtype=bool)

    for box in boxes:

        xx, yy = box.bottom_corners()[:2, :]

        yi = np.round((yy - ymin) / (ymax - ymin) * H).astype(int)
        xi = np.round((xx - xmin) / (xmax - xmin) * W).astype(int)

        rr, cc = polygon(yi, xi)

        I = np.logical_and(
            np.logical_and(rr >= 0, rr < H),
            np.logical_and(cc >= 0, cc < W),
        )
        object_mask[rr[I], cc[I]] = True

    return object_mask


# enumerate every sample
xlim = [-40.0, 40.0, 0.2]
ylim = [-70.4, 70.4, 0.2]
n_next = 6

# for ref_sample in tqdm(nusc.sample):
def rasterize(ref_sample):

    obj_box_list = []
    obj_shadow_list = []

    ref_sample_token = ref_sample["token"]

    ref_lidar_data = nusc.get("sample_data", ref_sample["data"]["LIDAR_TOP"])
    ref_lidar_calib = nusc.get("calibrated_sensor", ref_lidar_data["calibrated_sensor_token"])
    ref_lidar_pose = nusc.get("ego_pose", ref_lidar_data["ego_pose_token"])

    ref_from_car = transform_matrix(ref_lidar_calib["translation"],
                                    Quaternion(ref_lidar_calib["rotation"]), inverse=True)
    car_from_global = transform_matrix(ref_lidar_pose["translation"],
                                       Quaternion(ref_lidar_pose["rotation"]), inverse=True)

    next_sample_tokens = traverse(nusc, "sample", "next", n_next, ref_sample_token)
    for curr_sample_token in ([ref_sample_token] + next_sample_tokens):
        curr_sample = nusc.get("sample", curr_sample_token)

        curr_sample_boxes = []

        for sample_annotation_token in curr_sample["anns"]:

            sample_annotation = nusc.get("sample_annotation", sample_annotation_token)

            detection_name = category_to_detection_name(sample_annotation["category_name"])
            if detection_name is None:  # there are certain categories we will ignore
                continue

            # print(sample_annotation["category_name"], detection_name)
            curr_sample_boxes.append(DetectionBox(
                sample_token=curr_sample_token,
                translation=sample_annotation["translation"],
                size=sample_annotation["size"],
                rotation=sample_annotation["rotation"],
                velocity=nusc.box_velocity(sample_annotation["token"])[:2],
                num_pts=sample_annotation["num_lidar_pts"] + sample_annotation["num_radar_pts"],
                detection_name=detection_name,
            ))

        # NOTE transform boxes to the *reference* frame
        curr_sample_boxes = boxes_to_sensor(curr_sample_boxes, ref_lidar_pose, ref_lidar_calib)

        # NOTE object box binary masks
        curr_obj_box = draw_obj_boxes(curr_sample_boxes, xlim, ylim)
        obj_box_list.append(curr_obj_box)

        curr_sample_data = nusc.get("sample_data", curr_sample["data"]["LIDAR_TOP"])
        curr_lidar_pose = nusc.get("ego_pose", curr_sample_data["ego_pose_token"])
        curr_lidar_calib = nusc.get("calibrated_sensor", curr_sample_data["calibrated_sensor_token"])

        global_from_car = transform_matrix(curr_lidar_pose["translation"],
                                           Quaternion(curr_lidar_pose["rotation"]), inverse=False)
        car_from_curr = transform_matrix(curr_lidar_calib["translation"],
                                         Quaternion(curr_lidar_calib["rotation"]), inverse=False)

        ref_from_curr = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_curr])
        _x0, _y0, _z0 = ref_from_curr[:3, 3]

        curr_obj_shadow = draw_object_shadow(curr_sample_boxes, _x0, _y0, xlim, ylim)
        obj_shadow_list.append(curr_obj_shadow)

    obj_boxes = np.array(obj_box_list)
    obj_shadows = np.array(obj_shadow_list)

    ref_scene = nusc.get("scene", ref_sample["scene_token"])
    ref_log = nusc.get("log", ref_scene["log_token"])
    flip_flag = True if ref_log["location"].startswith("singapore") else False

    if flip_flag:
        obj_boxes = obj_boxes[:, :, ::-1]
        obj_shadows = obj_shadows[:, :, ::-1]

    ref_lidar_token = ref_lidar_data["token"]

    obj_box_path = f"{obj_box_dir}/{ref_lidar_token}.bin"
    if not os.path.exists(obj_box_path):
        obj_boxes.tofile(obj_box_path)

    obj_shadow_path = f"{obj_shadow_dir}/{ref_lidar_token}.bin"
    if not os.path.exists(obj_shadow_path):
        obj_shadows.tofile(obj_shadow_path)


if __name__ == "__main__":
    from multiprocessing import Pool
    with Pool(16) as p:
        results = list(tqdm(p.imap(rasterize, nusc.sample), total=len(nusc.sample)))
