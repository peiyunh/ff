# preprocess.py
# process LIDAR sweeps to identify ground returns
# this is required to identify freespace
import os
import argparse
import numpy as np
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

from lib.grndseg import segmentation

#
def process(sd):

    # read lidar points
    pc = LidarPointCloud.from_file(f"{nusc_root}/{sd['filename']}")

    #
    pts = np.array(pc.points[:3].T)

    # we follow nuscenes's labeling protocol
    # c.f. nuscenes lidarseg
    # 24: driveable surface
    # 30: static.others
    # 31: ego

    # initialize everything to 30 (static others)
    lbls = np.full(len(pts), 30, dtype=np.uint8)

    # identify ego mask based on the car's physical dimensions
    ego_mask = np.logical_and(
        np.logical_and(-0.8 <= pc.points[0], pc.points[0] <= 0.8),
        np.logical_and(-1.5 <= pc.points[1], pc.points[1] <= 2.5)
    )
    lbls[ego_mask] = 31

    # run ground segmentation code
    index = np.flatnonzero(np.logical_not(ego_mask))
    label = segmentation.segment(pts[index])

    #
    grnd_index = np.flatnonzero(label)
    lbls[index[grnd_index]] = 24

    # visualize to double check
    if False:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        colors = np.zeros_like(pts)
        colors[lbls == 24, :] = [1, 0, 0]
        colors[lbls == 30, :] = [0, 1, 0]
        colors[lbls == 31, :] = [0, 0, 1]
        pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([pcd])

    #
    res_file = os.path.join(res_dir, f"{sd['token']}_grndseg.bin")
    lbls.tofile(res_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--nusc-root", type=str, default="/data/nuscenes")
    parser.add_argument("--nusc-version", type=str, default="v1.0-trainval")
    parser.add_argument("--num-workers", type=int, default=16)
    args = parser.parse_args()

    # nusc_version = "v1.0-mini"
    nusc_version = args.nusc_version
    nusc_root = args.nusc_root

    nusc = NuScenes(nusc_version, nusc_root)

    res_dir = f"{nusc_root}/grndseg/{nusc_version}"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    #
    sds = [sd for sd in nusc.sample_data if sd["channel"] == "LIDAR_TOP"]
    print("total number of sample data:", len(sds))

    from multiprocessing import Pool
    with Pool(args.num_workers) as p:
        results = list(tqdm(p.imap(process, sds), total=len(sds)))
