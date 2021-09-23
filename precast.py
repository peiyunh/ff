# precast.py
# raycasting and cache all future visible freespace
# Usage:
#   python -W ignore precast.py
# Note:
#   Run this script twice.
#   Once for the training split and once for the validation split.

import os
import torch
import numpy as np
from nuscenes.nuscenes import NuScenes
from data import nuScenesDataset, CollateFn
from torch.utils.data import DataLoader
from torch.utils.cpp_extension import load
raycaster = load("raycaster", sources=[
    "lib/raycast/raycaster.cpp", "lib/raycast/raycaster.cu"], verbose=True)

torch.random.manual_seed(0)
np.random.seed(0)

# NOTE: mini for debugging
# nusc = NuScenes("v1.0-mini", "/data/nuscenes")
nusc = NuScenes("v1.0-trainval", "/data/nuscenes")

dataset_kwargs = {"n_input": 20, "n_samples": 100, "n_output": 7}

# NOTE: run this script once for train and once for val
dataset = nuScenesDataset(nusc, "train", dataset_kwargs)
# dataset = nuScenesDataset(nusc, "val", dataset_kwargs)

data_loader_kwargs = {"pin_memory": False, "shuffle": True, "batch_size": 32, "num_workers": 4}
data_loader = DataLoader(dataset, collate_fn=CollateFn, **data_loader_kwargs)

pc_range = [-40.0, -70.4, -2.0, 40.0, 70.4, 3.4]
voxel_size = 0.2
output_grid = [7, 704, 400]

device = torch.device("cuda:0")

offset = torch.nn.parameter.Parameter(
    torch.Tensor(pc_range[:3])[None, None, :], requires_grad=False).to(device)
scaler = torch.nn.parameter.Parameter(
    torch.Tensor([voxel_size]*3)[None, None, :], requires_grad=False).to(device)

cache_dir = f"{nusc.dataroot}/fvfmaps/{nusc.version}"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

for i, batch in enumerate(data_loader):
    print(i, len(data_loader))
    output_origins = batch["output_origins"].to(device)
    output_points = batch["output_points"].to(device)

    output_origins[:, :, :3] = (output_origins[:, :, :3] - offset) / scaler
    output_points[:, :, :3] = (output_points[:, :, :3] - offset) / scaler

    # what we would like
    freespace = raycaster.raycast(output_origins, output_points, output_grid)
    freespace = freespace.detach().cpu().numpy().astype(np.int8)

    #
    sd_tokens = batch["sample_data_tokens"]

    #
    for j, sd_token in enumerate(sd_tokens):
        # fvf: future visible freespace
        path = f"{cache_dir}/{sd_token}.bin"
        if not os.path.exists(path):
            if dataset.nusc_split == "val":
                raise RuntimeError("This is unexpected!")
            freespace[j].tofile(path)
