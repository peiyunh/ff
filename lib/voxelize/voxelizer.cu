#include <torch/extension.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <iostream>

template <typename scalar_t>
__global__ void voxelize_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> points,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> occupancy) {

    // batch index
    const auto n = blockIdx.y;

    // ray index
    const auto c = blockIdx.x * blockDim.x + threadIdx.x;

    // num of rays
    const auto & M = points.size(1);
    const auto & T = occupancy.size(1);
    const auto & H = occupancy.size(2);
    const auto & L = occupancy.size(3);
    const auto & W = occupancy.size(4);

    // we allocated more threads than num_rays
    if (c < M) {
        // ray end point
        const int vt = int(points[n][c][3]);
        const int vz = int(points[n][c][2]);
        const int vy = int(points[n][c][1]);
        const int vx = int(points[n][c][0]);

        if (0 <= vt && vt < T && 
            0 <= vz && vz < H && 
            0 <= vy && vy < L && 
            0 <= vx && vx < W) {
            occupancy[n][vt][vz][vy][vx] = 1;
        }
    }
}

torch::Tensor voxelize_cuda(
    torch::Tensor points,
    const std::vector<int> grid) {

    const auto N = points.size(0); // batch size
    const auto M = points.size(1); // num of rays

    const auto T = grid[0];  // t
    const auto H = grid[1];  // z 
    const auto L = grid[2];  // y
    const auto W = grid[3];  // x

    const auto dtype = points.dtype();
    const auto device = points.device();
    const auto options = torch::TensorOptions().dtype(dtype).device(device).requires_grad(false);

    // initialize to 0 (unknown)
    auto occupancy = torch::zeros({N, T, H, L, W}, options);

    const int threads = 1024;
    const dim3 blocks((M + threads - 1) / threads, N);

    // voxelize point clouds
    AT_DISPATCH_FLOATING_TYPES(points.type(), "voxelize_cuda", ([&] {
                voxelize_cuda_kernel<scalar_t><<<blocks, threads>>>(
                    points.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                    occupancy.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());
            }));

    // synchronize
    cudaDeviceSynchronize();

    //
    return occupancy;
}