# test.py
# main testing script
import os
import json
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from nuscenes.nuscenes import NuScenes

from data import nuScenesDataset, CollateFn

import matplotlib.pyplot as plt
import numpy as np

from skimage.draw import polygon

def make_data_loader(cfg, args):
    if "train_on_all_sweeps" not in cfg:
        train_on_all_sweeps = False
    else:
        train_on_all_sweeps = cfg["train_on_all_sweeps"]
    dataset_kwargs = {
        "n_input": cfg["n_input"],
        "n_samples": args.n_samples,
        "n_output": cfg["n_output"],
        "train_on_all_sweeps": train_on_all_sweeps
    }
    data_loader_kwargs = {
        "pin_memory": False,  # NOTE
        "shuffle": True,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers
    }
    nusc = NuScenes(cfg["nusc_version"], cfg["nusc_root"])
    data_loader = DataLoader(nuScenesDataset(nusc, args.test_split, dataset_kwargs),
                             collate_fn=CollateFn, **data_loader_kwargs)
    return data_loader

def mkdir_if_not_exists(d):
    if not os.path.exists(d):
        print(f"creating directory {d}")
        os.makedirs(d)

def evaluate_box_coll(obj_boxes, trajectory, pc_range):
    xmin, ymin, _, xmax, ymax, _ = pc_range
    T, H, W = obj_boxes.shape
    collisions = np.full(T, False)
    for t in range(T):
        x, y, theta = trajectory[t]
        corners = np.array([
            (-0.8, -1.5, 1),  # back left corner
            (0.8, -1.5, 1),   # back right corner
            (0.8, 2.5, 1),    # front right corner
            (-0.8, 2.5, 1),   # front left corner
        ])
        tf = np.array([
            [np.cos(theta), -np.sin(theta), x],
            [np.sin(theta), np.cos(theta), y],
            [0, 0, 1],
        ])
        xx, yy = tf.dot(corners.T)[:2]

        yi = np.round((yy - ymin) / (ymax - ymin) * H).astype(int)
        xi = np.round((xx - xmin) / (xmax - xmin) * W).astype(int)
        rr, cc = polygon(yi, xi)
        I = np.logical_and(
            np.logical_and(rr >= 0, rr < H),
            np.logical_and(cc >= 0, cc < W),
        )
        collisions[t] = np.any(obj_boxes[t, rr[I], cc[I]])
    return collisions

def voxelize_point_cloud(points):
    valid = (points[:, -1] == 0)
    x, y, z, t = points[valid].T
    x = ((x + 40.0) / 0.2).astype(int)
    y = ((y + 70.4) / 0.2).astype(int)
    mask = np.logical_and(
        np.logical_and(0 <= x, x < 400),
        np.logical_and(0 <= y, y < 704)
    )
    voxel_map = np.zeros((704, 400), dtype=bool)
    voxel_map[y[mask], x[mask]] = True
    return voxel_map

def make_cost_fig(cost_maps):
    cost_imgs = np.ones_like(cost_maps)
    T = len(cost_maps)
    for t in range(T):
        cost_map = cost_maps[t]
        cost_min, cost_max = cost_map.min(), cost_map.max()
        cost_img = (cost_map - cost_min) / (cost_max - cost_min)
        cost_imgs[t] = cost_img
    return cost_imgs

def test(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_count = torch.cuda.device_count()
    if args.batch_size % device_count != 0:
        raise RuntimeError(f"Batch size ({args.batch_size}) cannot be divided by device count ({device_count})")

    model_dir = args.model_dir
    with open(f"{model_dir}/config.json", 'r') as f:
        cfg = json.load(f)

    # dataset
    data_loader = make_data_loader(cfg, args)

    # instantiate a model and a renderer
    _n_input, _n_output = cfg["n_input"], cfg["n_output"]
    _pc_range, _voxel_size = cfg["pc_range"], cfg["voxel_size"]

    model_type = cfg["model_type"]
    if model_type == "vanilla":
        from model import VanillaNeuralMotionPlanner
        model = VanillaNeuralMotionPlanner(_n_input, _n_output, _pc_range, _voxel_size)
    elif model_type == "vf_guided":
        from model import VFGuidedNeuralMotionPlanner
        model = VFGuidedNeuralMotionPlanner(_n_input, _n_output, _pc_range, _voxel_size)
    elif model_type == "obj_guided":
        from model import ObjGuidedNeuralMotionPlanner
        model = ObjGuidedNeuralMotionPlanner(_n_input, _n_output, _pc_range, _voxel_size)
    elif model_type == "obj_shadow_guided":
        from model import ObjShadowGuidedNeuralMotionPlanner
        model = ObjShadowGuidedNeuralMotionPlanner(_n_input, _n_output, _pc_range, _voxel_size)
    else:
        raise NotImplementedError(f"{model_type} not implemented yet.")

    model = model.to(device)

    # resume
    ckpt_path = f"{args.model_dir}/ckpts/model_epoch_{args.test_epoch}.pth"
    checkpoint = torch.load(ckpt_path, map_location=device)
    # NOTE: ignore renderer's parameters
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    # data parallel
    model = nn.DataParallel(model)
    model.eval()

    # output
    vis_dir = os.path.join(model_dir, "visuals", f"{args.test_split}_epoch_{args.test_epoch}")
    mkdir_if_not_exists(vis_dir)

    #
    counts = np.zeros(cfg["n_output"], dtype=int)
    l2_dist_sum = np.zeros(cfg["n_output"], dtype=float)
    obj_coll_sum = np.zeros(cfg["n_output"], dtype=int)
    obj_box_coll_sum = np.zeros(cfg["n_output"], dtype=int)

    #
    obj_box_dir = f"{cfg['nusc_root']}/obj_boxes/{cfg['nusc_version']}"

    #
    np.set_printoptions(suppress=True, precision=2)
    num_batch = len(data_loader)
    for i, batch in enumerate(data_loader):
        sample_data_tokens = batch["sample_data_tokens"]
        bs = len(sample_data_tokens)
        if bs < device_count:
            print(f"Dropping the last batch of size {bs}")
            continue

        with torch.set_grad_enabled(False):
            results = model(batch, "test")

        best_plans = results["best_plans"].detach().cpu().numpy()

        sampled_plans = batch["sampled_trajectories"].detach().cpu().numpy()
        gt_plans = batch["gt_trajectories"].detach().cpu().numpy()

        plot_on = args.plot_on and (i % args.plot_every == 0)
        cache_on = args.cache_on and (i % args.cache_every == 0)

        if (cache_on or plot_on) and "cost" in results:
            costs = results["cost"].detach().cpu().numpy()
        else:
            costs = None

        for j, sample_data_token in enumerate(sample_data_tokens):
            # visualization:
            # - highlight the low cost regions (sub-zero)
            # - distinguish cost maps from different timestamps
            if plot_on:
                # tt = [2, 4, 6]
                tt = list(range(_n_output))
                if costs is not None:
                    cost = np.concatenate(costs[j, tt], axis=-1)
                    plt.imsave(f"{vis_dir}/{sample_data_token}.png", cost[::-1])

            # rasterized collision ground truth
            obj_box_path = f"{obj_box_dir}/{sample_data_token}.bin"
            obj_boxes = np.fromfile(obj_box_path, dtype=bool).reshape((-1, 704, 400))

            # T tells us how many future frames we have expert data for
            T = len(obj_boxes)
            counts[:T] += 1

            # skip when gt plan is flawed (because of the limits of BEV planning)
            gt_plan = gt_plans[j]
            gt_box_coll = evaluate_box_coll(obj_boxes, gt_plan, _pc_range)

            # compute L2 distance
            # best_plan = best_plans[j, 0]
            output_plan = sampled_plans[j, best_plans[j, 0]]
            l2_dist = np.sqrt(((output_plan[:, :2] - gt_plan[:, :2])**2).sum(axis=-1))
            l2_dist_sum[:T] += l2_dist[:T]

            # test ego-vehicle point against annotated object boxes
            ti = np.arange(T)
            yi = ((output_plan[:T, 1] - _pc_range[1]) / _voxel_size).astype(int)
            xi = ((output_plan[:T, 0] - _pc_range[0]) / _voxel_size).astype(int)
            # when the best plan is outside the boundary
            m1 = np.logical_and(
                np.logical_and(_pc_range[1] <= output_plan[:T, 1], output_plan[:T, 1] < _pc_range[4]),
                np.logical_and(_pc_range[0] <= output_plan[:T, 0], output_plan[:T, 0] < _pc_range[3])
            )
            # exclude cases where even the expert trajectory collides (box)
            # obviously the expert did not crash
            # it is only so because we are considering bird's-eye view
            # e.g. when a person is stepping out of the ego vehicle
            m1 = np.logical_and(m1, np.logical_not(gt_box_coll[ti]))
            obj_coll_sum[ti[m1]] += obj_boxes[ti[m1], yi[m1], xi[m1]].astype(int)

            # test ego-vehicle box against annotated object boxes
            # exclude cases where the expert trajectory collides (box)
            m2 = np.logical_not(gt_box_coll[ti])
            box_coll = evaluate_box_coll(obj_boxes, output_plan, _pc_range)
            obj_box_coll_sum[ti[m2]] += (box_coll[ti[m2]]).astype(int)

        print(f"{args.test_split} Epoch-{args.test_epoch},",
              f"Batch: {i+1}/{num_batch},",
              f"L2: {l2_dist_sum / counts},",
              f"Pt: {obj_coll_sum / counts * 100},",
              f"Box: {obj_box_coll_sum / counts * 100}")

    res_dir = f"{model_dir}/results"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    res_file = f"{res_dir}/{args.test_split}_epoch_{args.test_epoch}.txt"
    with open(res_file, "w") as f:
        f.write(f"Split: {args.test_split}\n")
        f.write(f"Epoch: {args.test_epoch}\n")
        f.write(f"Counts: {counts}\n")
        f.write(f"L2 distances: {l2_dist_sum / counts}\n")
        f.write(f"Point collision rates: {obj_coll_sum / counts * 100}\n")
        f.write(f"Box collision rates: {obj_box_coll_sum / counts * 100}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--test-split", type=str, required=True)
    parser.add_argument("--test-epoch", type=int, default=5)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=36)
    parser.add_argument("--cache-on", action="store_true")
    parser.add_argument("--cache-every", type=int, default=1)
    parser.add_argument("--plot-on", action="store_true")
    parser.add_argument("--plot-every", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=18)

    args = parser.parse_args()

    np.random.seed(0)
    torch.random.manual_seed(0)

    test(args)
