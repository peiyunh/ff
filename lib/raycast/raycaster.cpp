#include <torch/extension.h>
#include <string>
#include <vector>

/*
 * CUDA forward declarations
 */
torch::Tensor raycast_cuda(
    torch::Tensor origins,
    torch::Tensor points,
    const std::vector<int> grid);

/*
 * C++ interface
 */
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
torch::Tensor raycast(
    torch::Tensor origins,
    torch::Tensor points,
    const std::vector<int> grid) {
    CHECK_INPUT(origins);
    CHECK_INPUT(points);
    return raycast_cuda(origins, points, grid);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("raycast", &raycast, "Raycast");
}