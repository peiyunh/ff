# model.py
# neural net model implementations
import abc

import torch
import torch.nn as nn
import torch.nn.functional as F

# JIT
from torch.utils.cpp_extension import load
voxelizer = load("voxelizer", sources=[
    "lib/voxelize/voxelizer.cpp", "lib/voxelize/voxelizer.cu"], verbose=True)

def conv3x3(in_channels, out_channels, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                     padding=1, bias=bias)

def deconv3x3(in_channels, out_channels, stride):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride,
                              padding=1, output_padding=1, bias=False)

def maxpool2x2(stride):
    return nn.MaxPool2d(kernel_size=2, stride=stride, padding=0)

def relu(inplace=True):
    return nn.ReLU(inplace=inplace)

def bn(num_features):
    return nn.BatchNorm2d(num_features=num_features)

class ConvBlock(nn.Module):
    def __init__(self, num_layer, in_channels, out_channels, max_pool=False):
        super(ConvBlock, self).__init__()

        layers = []
        for i in range(num_layer):
            _in_channels = in_channels if i == 0 else out_channels
            layers.append(conv3x3(_in_channels, out_channels))
            layers.append(bn(out_channels))
            layers.append(relu())

        if max_pool:
            layers.append(maxpool2x2(stride=2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_layers, num_filters):
        super(Encoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = num_filters[4]

        # Block 1-4
        _in_channels = self.in_channels
        self.block1 = ConvBlock(num_layers[0], _in_channels, num_filters[0], max_pool=True)
        self.block2 = ConvBlock(num_layers[1], num_filters[0], num_filters[1], max_pool=True)
        self.block3 = ConvBlock(num_layers[2], num_filters[1], num_filters[2], max_pool=True)
        self.block4 = ConvBlock(num_layers[3], num_filters[2], num_filters[3])

        # Block 5
        _in_channels = sum(num_filters[0:4])
        self.block5 = ConvBlock(num_layers[4], _in_channels, num_filters[4])

    def forward(self, x):
        N, C, H, W = x.shape

        # the first 4 blocks
        c1 = self.block1(x)
        c2 = self.block2(c1)
        c3 = self.block3(c2)
        c4 = self.block4(c3)

        # upsample and concat
        _H, _W = H//4, W//4
        c1_interp = F.interpolate(c1, size=(_H, _W), mode="bilinear", align_corners=True)
        c2_interp = F.interpolate(c2, size=(_H, _W), mode="bilinear", align_corners=True)
        c3_interp = F.interpolate(c3, size=(_H, _W), mode="bilinear", align_corners=True)
        c4_interp = F.interpolate(c4, size=(_H, _W), mode="bilinear", align_corners=True)

        #
        c4_aggr = torch.cat((c1_interp, c2_interp, c3_interp, c4_interp), dim=1)
        c5 = self.block5(c4_aggr)

        return c5


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            deconv3x3(in_channels, 128, stride=2),
            bn(128),
            relu(),
            conv3x3(128, 128),
            bn(128),
            relu(),
            deconv3x3(128, 64, stride=2),
            bn(64),
            relu(),
            conv3x3(64, 64),
            bn(64),
            relu(),
            conv3x3(64, out_channels, bias=True)
        )

    def forward(self, x):
        return self.block(x)


class BaseNeuralMotionPlanner(nn.Module):
    MAX_COST = 1000.0

    def __init__(self, n_input, n_output, pc_range, voxel_size):

        super(BaseNeuralMotionPlanner, self).__init__()

        self.n_input = n_input
        self.n_output = n_output

        self.n_height = int((pc_range[5] - pc_range[2]) / voxel_size)
        self.n_length = int((pc_range[4] - pc_range[1]) / voxel_size)
        self.n_width = int((pc_range[3] - pc_range[0]) / voxel_size)

        self.input_grid = [self.n_input, self.n_height, self.n_length, self.n_width]
        self.output_grid = [self.n_output, self.n_length, self.n_width]

        self.pc_range = pc_range
        self.voxel_size = voxel_size

        self.offset = torch.nn.parameter.Parameter(
            torch.Tensor(self.pc_range[:3])[None, None, :], requires_grad=False)
        self.scaler = torch.nn.parameter.Parameter(
            torch.Tensor([self.voxel_size]*3)[None, None, :], requires_grad=False)

        self._in_channels = self.n_input * self.n_height
        self.encoder = Encoder(self._in_channels, [2, 2, 3, 6, 5], [32, 64, 128, 256, 256])

        # NOTE: initialize the linear predictor (no bias) over history
        self._out_channels = self.n_output
        self.imitation_decoder = Decoder(self.encoder.out_channels, self._out_channels)

    def _compute_L2(self, batch):
        # 1000 sampled trajectories vs 1 gt trajectory
        st, gt = batch["sampled_trajectories"], batch["gt_trajectories"]

        # L2 distance
        return torch.sqrt(((st[:, :, :, :2] - gt[:, None, :, :2]) ** 2).sum(dim=-1))

    def _normalize(self, points):
        points[:, :, :3] = (points[:, :, :3] - self.offset) / self.scaler

    def _discretize(self, trajectories):
        # input: N x n_samples x n_output x 3 (x, y, theta)
        # output: N x n_samples x n_output (yi, xi)
        if trajectories.ndim == 3:  # ground truth trajectories
            trajectories = trajectories[:, None, :, :]

        #
        N, M, T, _ = trajectories.shape

        #
        xx, yy = trajectories[:, :, :, 0], trajectories[:, :, :, 1]

        # discretize
        yi = ((yy - self.pc_range[1]) / self.voxel_size).long()
        yi = torch.clamp(yi, min=0, max=self.n_length-1)

        xi = ((xx - self.pc_range[0]) / self.voxel_size).long()
        xi = torch.clamp(xi, min=0, max=self.n_width-1)

        #
        return yi, xi

    def prepare_input(self, batch):
        # extract data
        input_points = batch["input_points"]

        # convert metric coordinates to grid coordinates
        self._normalize(input_points)

        # voxelize input LiDAR sweeps
        input_tensor = voxelizer.voxelize(input_points, self.input_grid)
        input_tensor = input_tensor.reshape((-1, self._in_channels, self.n_length, self.n_width))

        return input_tensor

    def compute_cost_maps(self, feat):
        return

    def clamp_cost_maps(self, C):
        return torch.clamp(C, min=-self.MAX_COST, max=self.MAX_COST)

    def evaluate_samples(self, batch, C):
        # parse input
        sampled_trajectories = batch["sampled_trajectories"]

        # batch size
        N = len(sampled_trajectories)
        ii = torch.arange(N)
        ti = torch.arange(self.n_output)

        # discretize trajectories
        Syi, Sxi = self._discretize(sampled_trajectories)

        # indexing
        CS = C[ii[:, None, None], ti[None, None, :], Syi, Sxi]

        #
        return CS

    def evaluate_expert(self, batch, C):
        # parse input
        gt_trajectories = batch["gt_trajectories"]

        # batch size
        N = len(gt_trajectories)
        ii = torch.arange(N)
        ti = torch.arange(self.n_output)

        # discretize trajectories
        Gyi, Gxi = self._discretize(gt_trajectories)

        # indexing
        CG = C[ii[:, None, None], ti[None, None, :], Gyi, Gxi]

        #
        return CG

    @abc.abstractmethod
    def compute_cost_margins(self, batch):
        return

    def select_best_plans(self, batch, CS, k=1):
        # select top 5 (lowest) cost trajectories
        CC, KK = torch.topk(CS.sum(-1), k, dim=-1, largest=False)

        #
        return KK

    def forward(self, batch, mode="train"):
        results = {}

        # voxelize input lidar sweeps
        _input = self.prepare_input(batch)

        # extract backbone feature maps
        feat = self.encoder(_input)

        # compute cost maps (model-specific)
        C = self.compute_cost_maps(feat)

        # clamp cost
        C = self.clamp_cost_maps(C)

        # evaluate the cost of every sampled trajectory
        CS = self.evaluate_samples(batch, C)

        if mode == "train":
            # evaluate the cost of the expert trajectory
            CG = self.evaluate_expert(batch, C)

            # compute cost margins (model-specific)
            CM = self.compute_cost_margins(batch)

            # construct the max-margin loss
            L, _ = ((F.relu(CG - CS + CM)).sum(dim=-1)).max(dim=-1)

            # return the margin loss
            results["margin_loss"] = L

        else:
            results["cost"] = C
            results["best_plans"] = self.select_best_plans(batch, CS, 5)

        return results


class VanillaNeuralMotionPlanner(BaseNeuralMotionPlanner):
    def __init__(self, n_input, n_output, pc_range, voxel_size):
        super(VanillaNeuralMotionPlanner, self).__init__(n_input, n_output, pc_range, voxel_size)

    def compute_cost_margins(self, batch):
        return self._compute_L2(batch)

    def compute_cost_maps(self, feat):
        return self.imitation_decoder(feat)

    def forward(self, batch, mode):
        results = super(VanillaNeuralMotionPlanner, self).forward(batch, mode)
        if mode == "train":
            results["loss"] = results["margin_loss"]
        return results


class VFGuidedNeuralMotionPlanner(BaseNeuralMotionPlanner):
    NVF_COST_FACTOR = 200.0

    def __init__(self, n_input, n_output, pc_range, voxel_size):
        super(VFGuidedNeuralMotionPlanner, self).__init__(n_input, n_output, pc_range, voxel_size)

    def compute_cost_margins(self, batch):
        # incorporate visible freespace as part of the cost margin
        freespace = batch["fvf_maps"]

        # discretize sampled trajectories
        sampled_trajectories = batch["sampled_trajectories"]
        N = len(sampled_trajectories)
        ii = torch.arange(N)
        ti = torch.arange(self.n_output)
        Syi, Sxi = self._discretize(sampled_trajectories)

        # index observed future visible freespace with sampled trajectories
        # observed freespace is marked as -1
        label = freespace[ii[:, None, None], ti[None, None, :], Syi, Sxi]

        #
        nvf_cost = self.NVF_COST_FACTOR * (label != -1).float()

        #
        l2_cost = self._compute_L2(batch)

        return (nvf_cost + l2_cost)

    def compute_cost_maps(self, feat):
        return self.imitation_decoder(feat)

    def forward(self, batch, mode):
        results = super(VFGuidedNeuralMotionPlanner, self).forward(batch, mode)
        if mode == "train":
            results["loss"] = results["margin_loss"]
        return results


class ObjGuidedNeuralMotionPlanner(BaseNeuralMotionPlanner):
    OBJ_COST_FACTOR = 200.0

    def __init__(self, n_input, n_output, pc_range, voxel_size):
        super(ObjGuidedNeuralMotionPlanner, self).__init__(n_input, n_output, pc_range, voxel_size)

    def compute_cost_margins(self, batch):
        # incorporate visible freespace as part of the cost margin
        obj_maps = batch["obj_boxes"]

        # discretize sampled trajectories
        sampled_trajectories = batch["sampled_trajectories"]
        N = len(sampled_trajectories)
        ii = torch.arange(N)
        ti = torch.arange(self.n_output)
        Syi, Sxi = self._discretize(sampled_trajectories)

        # index observed future visible freespace with sampled trajectories
        # observed freespace is marked as -1
        label = obj_maps[ii[:, None, None], ti[None, None, :], Syi, Sxi]

        #
        obj_cost = self.OBJ_COST_FACTOR * (label == 1).float()

        #
        l2_cost = self._compute_L2(batch)

        return (obj_cost + l2_cost)

    def compute_cost_maps(self, feat):
        return self.imitation_decoder(feat)

    def forward(self, batch, mode):
        results = super(ObjGuidedNeuralMotionPlanner, self).forward(batch, mode)
        if mode == "train":
            results["loss"] = results["margin_loss"]
        return results


class ObjShadowGuidedNeuralMotionPlanner(BaseNeuralMotionPlanner):
    OBJ_COST_FACTOR = 200.0

    def __init__(self, n_input, n_output, pc_range, voxel_size):
        super(ObjShadowGuidedNeuralMotionPlanner, self).__init__(n_input, n_output, pc_range, voxel_size)

    def compute_cost_margins(self, batch):
        # incorporate visible freespace as part of the cost margin
        # obj_maps = batch["obj_maps"]
        obj_maps = batch["obj_shadows"]

        # discretize sampled trajectories
        sampled_trajectories = batch["sampled_trajectories"]
        N = len(sampled_trajectories)
        ii = torch.arange(N)
        ti = torch.arange(self.n_output)
        Syi, Sxi = self._discretize(sampled_trajectories)

        # index observed future visible freespace with sampled trajectories
        # observed freespace is marked as -1
        label = obj_maps[ii[:, None, None], ti[None, None, :], Syi, Sxi]

        #
        obj_cost = self.OBJ_COST_FACTOR * (label == 1).float()

        #
        l2_cost = self._compute_L2(batch)

        return (obj_cost + l2_cost)

    def compute_cost_maps(self, feat):
        return self.imitation_decoder(feat)

    def forward(self, batch, mode):
        results = super(ObjShadowGuidedNeuralMotionPlanner, self).forward(batch, mode)
        if mode == "train":
            results["loss"] = results["margin_loss"]
        return results
