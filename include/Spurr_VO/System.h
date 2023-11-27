#ifndef SYSTEM_H
#define SYSTEM_H

#include<string>
#include<thread>
#include<opencv2/core/core.hpp>
#include <unistd.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <string>
#include <chrono>
#include <experimental/filesystem>
#include <iostream>
#include <utility>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <mutex>
#include "Tracking.h"
#include "Map.h"
#include "LocalMapping.h"

using namespace std;
namespace fs = std::experimental::filesystem;
using namespace torch;
using namespace torch::indexing;
using namespace std::chrono;
namespace SPURR_VO
{
static torch::Device device(torch::kCUDA);
class Map;
class Tracking;
class LocalMapping;

class System
{
public:

    System(std::string sf, std::string sp, std::string sg);

    cv::Mat TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp);
    torch::Tensor process_image(cv::Mat image, int target_width);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> unpack_result(const IValue &result);
    torch::Dict<std::string, Tensor> toTensorDict(const torch::IValue &value);

    void ActivateLocalizationMode();

    void DeactivateLocalizationMode();

    bool MapChanged();

    void Reset();

    void Shutdown();

    void SaveTrajectoryTUM(const string &filename);

    void SaveKeyFrameTrajectoryTUM(const string &filename);

    void SaveTrajectoryKITTI(const string &filename);

    int GetTrackingState();
    std::vector<MapPoint*> GetTrackedMapPoints();
    std::vector<cv::KeyPoint> GetTrackedKeyPointsUn();

private:

    cv::Mat imGray;
    Tensor keypoints0, scores0, descriptors0;
    Tensor keypoints1, scores1, descriptors1;
    std::string spPath, sgPath, strSettingsFile;

    torch::Dict<std::string, Tensor> pred;
    torch::jit::script::Module superpoint, superglue;

    Map* mpMap;

    Tracking* mpTracker;

    LocalMapping* mpLocalMapper;

    std::thread* mptLocalMapping;
    std::mutex mMutexReset;
    bool mbReset;

    std::mutex mMutexMode;
    bool mbActivateLocalizationMode;
    bool mbDeactivateLocalizationMode;

    std::vector<MapPoint*> mTrackedMapPoints;
    std::vector<cv::KeyPoint> mTrackedKeyPointsUn;
    std::mutex mMutexState;
    int mTrackingState;

};

}

#endif // SYSTEM_H
