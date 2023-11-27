#include "System.h"
#include "Converter.h"
#include <thread>
#include <iomanip>

namespace SPURR_VO
{

System::System(std::string sf, std::string sp, std::string sg):strSettingsFile(sf), spPath(sp), sgPath(sg), mbReset(false),mbActivateLocalizationMode(false),
        mbDeactivateLocalizationMode(false)
{
    cout << "Input sensor was set to: ";

    if (!fs::exists(spPath)) {
      std::cerr << "Could not find the TorchScript module file " << spPath << std::endl;
      return;
    }
    superpoint = torch::jit::load(spPath);
    superpoint.eval();
    superpoint.to(device);

    if (!fs::exists(sgPath)) {
      std::cerr << "Could not find the TorchScript module file " << sgPath << std::endl;
      return;
    }
    superglue = torch::jit::load(sgPath);
    superglue.eval();
    superglue.to(device);


    mpMap = new Map();

    mpTracker = new Tracking(this, strSettingsFile, superpoint, superglue, mpMap);

    mpLocalMapper = new LocalMapping(mpMap);
    mpTracker->SetLocalMapper(mpLocalMapper);

    mpLocalMapper->SetTracker(mpTracker);

}

cv::Mat System::TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp)
{
      unique_lock<mutex> lock(mMutexMode);

    if(im.channels()==3)
    {
      if(mpTracker->mbRGB)
          cvtColor(im,imGray,cv::COLOR_RGB2GRAY);
      else
          cvtColor(im,imGray,cv::COLOR_BGR2GRAY);
    }
    else if(im.channels()==4)
    {
      if(mpTracker->mbRGB)
          cvtColor(im,imGray,cv::COLOR_RGBA2GRAY);
      else
          cvtColor(im,imGray,cv::COLOR_BGRA2GRAY);
    }

    // Check reset
    {
    unique_lock<mutex> lock(mMutexReset);
    if(mbReset)
    {
        std::cout<<"Resetting"<<std::endl;
        mpTracker->Reset();
        mbReset = false;
    }
    }
    std::cout<<"System.CPP"<<std::endl;
    torch::Tensor imgT = process_image(imGray, imGray.cols).to(device);
    cv::Mat Tcw = mpTracker->GrabImageRGBD(imGray, imgT,depthmap,timestamp);
    std::cout<<"ASDF__"<<std::endl;

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
    std::cout<<"ASDF"<<std::endl;
    return Tcw;
}

torch::Tensor System::process_image(cv::Mat image, int target_width) {
  int target_height = std::lround((float)target_width / image.cols * image.rows);
  image.convertTo(image, CV_32F, 1.0f / 255.0f);
  cv::resize(image, image, {target_width, target_height});

  torch::Tensor tensor = torch::from_blob(image.data, {1, 1, image.rows, image.cols},
                                          torch::TensorOptions().dtype(torch::kFloat32));
  return tensor.clone();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> System::unpack_result(const IValue &result)
{
  auto dict = result.toGenericDict();
  return {dict.at("keypoints").toTensorVector()[0], //
          dict.at("scores").toTensorVector()[0],    //
          dict.at("descriptors").toTensorVector()[0]};
}

torch::Dict<std::string, Tensor> System::toTensorDict(const torch::IValue &value)
{
  return c10::impl::toTypedDict<std::string, Tensor>(value.toGenericDict());
}

void System::ActivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbActivateLocalizationMode = true;
}

void System::DeactivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbDeactivateLocalizationMode = true;
}

bool System::MapChanged()
{
    static int n=0;
    int curn = mpMap->GetLastBigChangeIdx();
    if(n<curn)
    {
        n=curn;
        return true;
    }
    else
        return false;
}

void System::Reset()
{
    unique_lock<mutex> lock(mMutexReset);
    mbReset = true;
}

void System::Shutdown()
{
    mpLocalMapper->RequestFinish();
    // mpLoopCloser->RequestFinish();
    // if(mpViewer)
    // {
    //     mpViewer->RequestFinish();
    //     while(!mpViewer->isFinished())
    //         usleep(5000);
    // }

    // Wait until all thread have effectively stopped
    while(!mpLocalMapper->isFinished())
    {
        usleep(5000);
    }

}

int System::GetTrackingState()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackingState;
}

vector<MapPoint*> System::GetTrackedMapPoints()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedMapPoints;
}

vector<cv::KeyPoint> System::GetTrackedKeyPointsUn()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedKeyPointsUn;
}

}
