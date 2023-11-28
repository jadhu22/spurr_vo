#pragma once

#include <torch/torch.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <string>

torch::Tensor read_image(const std::string &path, int target_width) {
  cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
  int target_height = std::lround((float)target_width / image.cols * image.rows);
  image.convertTo(image, CV_32F, 1.0f / 255.0f);
  cv::resize(image, image, {target_width, target_height});

  torch::Tensor tensor = torch::from_blob(image.data, {1, 1, image.rows, image.cols},
                                          torch::TensorOptions().dtype(torch::kFloat32));
  return tensor.clone();
}

cv::Mat tensor2mat(torch::Tensor tensor) {
  tensor = tensor.to(torch::kCPU).contiguous();
  cv::Mat mat(tensor.size(-2), tensor.size(-1), CV_32F);
  std::memcpy((void *)mat.data, tensor.data_ptr(), sizeof(float) * tensor.numel());
  return mat;
}