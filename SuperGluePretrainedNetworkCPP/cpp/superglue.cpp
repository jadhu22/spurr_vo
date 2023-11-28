#include <torch/script.h>
#include <torch/torch.h>

#include <chrono>
#include <experimental/filesystem>
#include <iostream>
#include <utility>

#include "io.h"
#include "viz.h"

using namespace torch;
using namespace torch::indexing;
namespace fs = std::experimental::filesystem;

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> unpack_result(const IValue &result) {
  auto dict = result.toGenericDict();
  return {dict.at("keypoints").toTensorVector()[0], //
          dict.at("scores").toTensorVector()[0],    //
          dict.at("descriptors").toTensorVector()[0]};
}

torch::Dict<std::string, Tensor> toTensorDict(const torch::IValue &value) {
  return c10::impl::toTypedDict<std::string, Tensor>(value.toGenericDict());
}

int main(int argc, const char *argv[]) {
  if (argc <= 3) {
    std::cerr << "Usage:" << std::endl;
    std::cerr << argv[0] << " <image0> <image1> <downscaled_width>" << std::endl;
    return 1;
  }

  torch::manual_seed(1);
  torch::autograd::GradMode::set_enabled(false);

  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Training on GPU." << std::endl;
    device = torch::Device(torch::kCUDA);
  }

  int target_width = std::stoi(argv[3]);
  Tensor image0 = read_image(std::string(argv[1]), target_width).to(device);
  Tensor image1 = read_image(std::string(argv[2]), target_width).to(device);

  // Look for the TorchScript module files in the executable directory
  auto executable_dir = (fs::path(argv[0])).parent_path();
  auto module_path = executable_dir / "SuperPoint.zip";
  if (!fs::exists(module_path)) {
    std::cerr << "Could not find the TorchScript module file " << module_path << std::endl;
    return 1;
  }
  torch::jit::script::Module superpoint = torch::jit::load(module_path);
  superpoint.eval();
  superpoint.to(device);

  module_path = executable_dir / "SuperGlue.zip";
  if (!fs::exists(module_path)) {
    std::cerr << "Could not find the TorchScript module file " << module_path << std::endl;
    return 1;
  }
  torch::jit::script::Module superglue = torch::jit::load(module_path);
  superglue.eval();
  superglue.to(device);

  int N = 50;
  using namespace std::chrono;
  auto t0 = high_resolution_clock::now();
  Tensor keypoints0, scores0, descriptors0;
  Tensor keypoints1, scores1, descriptors1;
  torch::Dict<std::string, Tensor> pred;
  for (int i = 0; i < N; ++i) {
    std::tie(keypoints0, scores0, descriptors0) = unpack_result(superpoint.forward({image0}));
    std::tie(keypoints1, scores1, descriptors1) = unpack_result(superpoint.forward({image1}));

    torch::Dict<std::string, Tensor> input;
    input.insert("image0", image0);
    input.insert("image1", image1);
    input.insert("keypoints0", keypoints0.unsqueeze(0));
    input.insert("keypoints1", keypoints1.unsqueeze(0));
    input.insert("scores0", scores0.unsqueeze(0));
    input.insert("scores1", scores1.unsqueeze(0));
    input.insert("descriptors0", descriptors0.unsqueeze(0));
    input.insert("descriptors1", descriptors1.unsqueeze(0));
    pred = toTensorDict(superglue.forward({input}));
  }
  double period = duration_cast<duration<double>>(high_resolution_clock::now() - t0).count() / N;
  std::cout << period * 1e3 << " ms, FPS: " << 1 / period << std::endl;

  auto matches = pred.at("matches0")[0];
  auto valid = at::nonzero(matches > -1).squeeze();
  auto mkpts0 = keypoints0.index_select(0, valid);
  auto mkpts1 = keypoints1.index_select(0, matches.index_select(0, valid));
  auto confidence = pred.at("matching_scores0")[0].index_select(0, valid);

  std::cout << "Image #0 keypoints: " << keypoints0.size(0) << std::endl;
  std::cout << "Image #1 keypoints: " << keypoints1.size(0) << std::endl;
  std::cout << "Valid match count: " << valid.size(0) << std::endl;

  cv::Mat plot =
      make_matching_plot_fast(image0, image1, keypoints0, keypoints1, mkpts0, mkpts1, confidence);
  cv::imwrite("matches.png", plot);
  std::cout << "Done! Created matches.png for visualization." << std::endl;
}
