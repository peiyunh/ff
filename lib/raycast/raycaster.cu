// TODO: SEE FIXME 
#include <torch/extension.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <iostream>

template <typename scalar_t>
__global__ void init_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> points,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> stopper) {

    // batch index
    const auto n = blockIdx.y;

    // ray index
    const auto c = blockIdx.x * blockDim.x + threadIdx.x;

    // num of rays
    const auto M = points.size(1);
    const auto T = stopper.size(1);

    // we allocated more threads than num_rays
    if (c < M) {
        // ray end point
        const int t = points[n][c][3];

        // invalid points 
        // if t < 0, it is a padded point
        if (t < 0) return;

        //
        const auto vysize = stopper.size(2);
        const auto vxsize = stopper.size(3);

        // end point
        const int vx = int(points[n][c][0]);
        const int vy = int(points[n][c][1]);

        //
        if (0 <= vx && vx < vxsize && 0 <= vy && vy < vysize) {
            const int label = int(points[n][c][4]);
            // set non-drivable-surface returns as stoppers
            if ((1 <= label && label <= 23) || (28 <= label && label <= 30)) {
                stopper[n][t][vy][vx] = 1;
            }
        }
    }
}

template <typename scalar_t>
__global__ void raycast_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> origins,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> points,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> stopper,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> occupancy) {

    // batch index
    const auto n = blockIdx.y;

    // ray index
    const auto c = blockIdx.x * blockDim.x + threadIdx.x;

    // num of rays
    const auto M = points.size(1);
    const auto T = occupancy.size(1);

    // we allocated more threads than num_rays
    if (c < M) {
        // ray end point
        const auto t = points[n][c][3];

        // if t or label < 0, it is a padded point
        if (t < 0) return;

        // invalid points
        assert(t < T);

        // grid shape
        const int vysize = occupancy.size(2);
        const int vxsize = occupancy.size(3);

        // origin
        const double xo = origins[n][t][0];
        const double yo = origins[n][t][1];

        // end point
        const double xe = points[n][c][0];
        const double ye = points[n][c][1];

        // where raycasting starts
        const int vxo = int(xo);
        const int vyo = int(yo);

        // where raycasting ends
        const int vxe = int(xe);
        const int vye = int(ye);

        // NOTE: the ray starts at the origin regardless
        int vx = vxo;
        int vy = vyo;

        // origin to end
        const double rx = xe - xo;
        const double ry = ye - yo;
        double gt_d = sqrt(rx * rx + ry * ry);

        // directional vector
        const double dx = rx / gt_d;
        const double dy = ry / gt_d;

        // In which direction the voxel ids are incremented.
        const int stepX = (dx >= 0) ? 1 : -1;
        const int stepY = (dy >= 0) ? 1 : -1;

        // Distance along the ray to the next voxel border from the current position (tMaxX, tMaxY, tMaxZ).
        const double next_voxel_boundary_x = vx + (stepX < 0 ? 0 : 1);
        const double next_voxel_boundary_y = vy + (stepY < 0 ? 0 : 1);

        // tMaxX, tMaxY, tMaxZ -- distance until next intersection with voxel-border
        // the value of t at which the ray crosses the first vertical voxel boundary
        double tMaxX = (dx!=0) ? (next_voxel_boundary_x - xo)/dx : DBL_MAX; //
        double tMaxY = (dy!=0) ? (next_voxel_boundary_y - yo)/dy : DBL_MAX; //

        // tDeltaX, tDeltaY, tDeltaZ --
        // how far along the ray we must move for the horizontal component to equal the width of a voxel
        // the direction in which we traverse the grid
        // can only be FLT_MAX if we never go in that direction
        const double tDeltaX = (dx!=0) ? stepX/dx : DBL_MAX;
        const double tDeltaY = (dy!=0) ? stepY/dy : DBL_MAX;

        // the ray travels a maximum distance (80m / 0.2m = 400 voxels)
        int step = 0;
        double last_d = 0.0;

        // voxel traversal raycasting
        bool was_inside = false;
        while (true) {
            bool inside = (0 <= vx && vx < vxsize) && (0 <= vy && vy < vysize);
            if (inside) {
                was_inside = true;
                const auto stop = stopper[n][t][vy][vx];
                if (stop == 1) {
                    occupancy[n][t][vy][vx] = 1;
                    break;
                } else if (occupancy[n][t][vy][vx] == 0) {
                    occupancy[n][t][vy][vx] = -1;
                }
            } else if (was_inside){
                break;
            } else if (last_d > gt_d) {
                break;
            }
            // move along the ray by one cell
            double _d = 0.0;
            if (tMaxX < tMaxY) {
                _d = tMaxX;
                vx += stepX;
                tMaxX += tDeltaX;
            } else {
                _d = tMaxY;
                vy += stepY;
                tMaxY += tDeltaY;
            }
            last_d = _d;
            step ++;
        }
    }
}

torch::Tensor raycast_cuda(
    torch::Tensor origins,
    torch::Tensor points,
    const std::vector<int> grid) {

    const auto N = points.size(0); // batch size
    const auto M = points.size(1); // num of rays

    const auto T = grid[0];  // time
    const auto L = grid[1];  // y
    const auto W = grid[2];  // x

    const auto dtype = points.dtype();
    const auto device = points.device();
    const auto options = torch::TensorOptions().dtype(dtype).device(device).requires_grad(false);

    // initialize to 0 (unknown)
    auto stopper = torch::zeros({N, T, L, W}, options);
    auto occupancy = torch::zeros({N, T, L, W}, options);

    const int threads = 1024;
    const dim3 blocks((M + threads - 1) / threads, N);

    // set stopper
    AT_DISPATCH_FLOATING_TYPES(points.type(), "init_cuda", ([&] {
                init_cuda_kernel<scalar_t><<<blocks, threads>>>(
                    points.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                    stopper.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
            }));

    // synchronize
    cudaDeviceSynchronize();

    // raycast
    AT_DISPATCH_FLOATING_TYPES(points.type(), "raycast_cuda", ([&] {
                raycast_cuda_kernel<scalar_t><<<blocks, threads>>>(
                    origins.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                    points.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                    stopper.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                    occupancy.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
            }));

    // synchronize
    cudaDeviceSynchronize();

    return occupancy;
}