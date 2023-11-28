#pragma once

#include <torch/torch.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "io.h"

cv::Mat draw_keypoints(const torch::Tensor &img, const torch::Tensor &keypoints) {
  cv::Mat out = tensor2mat(img);
  out.convertTo(out, CV_8U, 255.0f);
  cv::cvtColor(out, out, cv::COLOR_GRAY2BGR);
  for (int i = 0; i < keypoints.size(0); ++i) {
    auto kp = keypoints[i];
    cv::Point p(std::roundl(kp[0].item<float>()), std::roundl(kp[1].item<float>()));
    cv::circle(out, p, 2, {0, 0, 255}, -1, cv::LINE_AA);
  }
  return out;
}

cv::Mat make_matching_plot_fast(const torch::Tensor &image0, const torch::Tensor &image1,
                                const torch::Tensor &kpts0, const torch::Tensor &kpts1,
                                const torch::Tensor &mkpts0, const torch::Tensor &mkpts1,
                                const torch::Tensor &confidence, bool show_keypoints = true,
                                int margin = 10) {
  cv::Mat imgmat0 = tensor2mat(image0);
  imgmat0.convertTo(imgmat0, CV_8U, 255.0f);
  cv::Mat imgmat1 = tensor2mat(image1);
  imgmat1.convertTo(imgmat1, CV_8U, 255.0f);

  if (show_keypoints) {
    const cv::Scalar white(255, 255, 255);
    const cv::Scalar black(0, 0, 0);
    for (int i = 0; i < kpts0.size(0); ++i) {
      auto kp = kpts0[i];
      cv::Point pt(std::lround(kp[0].item<float>()), std::lround(kp[1].item<float>()));
      cv::circle(imgmat0, pt, 2, black, -1, cv::LINE_AA);
      cv::circle(imgmat0, pt, 1, white, -1, cv::LINE_AA);
    }
    for (int i = 0; i < kpts1.size(0); ++i) {
      auto kp = kpts1[i];
      cv::Point pt(std::lround(kp[0].item<float>()), std::lround(kp[1].item<float>()));
      cv::circle(imgmat1, pt, 2, black, -1, cv::LINE_AA);
      cv::circle(imgmat1, pt, 1, white, -1, cv::LINE_AA);
    }
  }

  int H0 = imgmat0.rows, W0 = imgmat0.cols;
  int H1 = imgmat1.rows, W1 = imgmat1.cols;
  int H = std::max(H0, H1), W = W0 + W1 + margin;

  cv::Mat out = 255 * cv::Mat::ones(H, W, CV_8U);
  imgmat0.copyTo(out.rowRange(0, H0).colRange(0, W0));
  imgmat1.copyTo(out.rowRange(0, H1).colRange(W0 + margin, W));
  cv::cvtColor(out, out, cv::COLOR_GRAY2BGR);

  // Apply colormap to confidences
  cv::Mat conf_mat = tensor2mat(confidence.unsqueeze(0));
  conf_mat.convertTo(conf_mat, CV_8U, 255.0f);
  cv::Mat colors;
  cv::applyColorMap(conf_mat, colors, cv::COLORMAP_JET);

  int n = std::min(mkpts0.size(0), mkpts1.size(0));
  for (int i = 0; i < n; ++i) {
    auto kp0 = mkpts0[i];
    auto kp1 = mkpts1[i];
    cv::Point pt0(std::lround(kp0[0].item<float>()), std::lround(kp0[1].item<float>()));
    cv::Point pt1(std::lround(kp1[0].item<float>()), std::lround(kp1[1].item<float>()));
    auto c = colors.at<cv::Vec3b>({i, 0});
    cv::line(out, pt0, {pt1.x + margin + W0, pt1.y}, c, 1, cv::LINE_AA);
    // display line end-points as circles
    cv::circle(out, pt0, 2, c, -1, cv::LINE_AA);
    cv::circle(out, {pt1.x + margin + W0, pt1.y}, 2, c, -1, cv::LINE_AA);
  }

  return out;
}
