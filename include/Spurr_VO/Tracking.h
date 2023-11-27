#ifndef TRACKING_H
#define TRACKING_H

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include"Map.h"
#include"LocalMapping.h"
#include"Frame.h"
#include "System.h"

#include <mutex>
using namespace std;
using namespace torch;
using namespace torch::indexing;
using namespace std::chrono;
namespace SPURR_VO
{

class Map;
class LocalMapping;
class System;

class Tracking
{

public:
    Tracking(System* pSys, const string &strSettingPath, torch::jit::script::Module sp, torch::jit::script::Module sg, Map *pMap);

    // Preprocess the input and call Track(). Extract features and performs stereo matching.
    cv::Mat GrabImageRGBD(const cv::Mat &imRGB, const torch::Tensor &imRGBT,const cv::Mat &imD, const double &timestamp);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> unpack_result(const IValue &result);
    torch::Dict<std::string, Tensor> toTensorDict(const torch::IValue &value);
    torch::jit::script::Module superpoint;
    torch::jit::script::Module superglue;

    int SearchByProjection(Frame &F, const vector<MapPoint*> &vpMapPoints, const float th);
    int SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th);
    int SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, const set<MapPoint*> &sAlreadyFound, const float th , const int dist);
    void ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);
    void SetLocalMapper(LocalMapping* pLocalMapper);


public:

    // Tracking states
    enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
        LOST=3
    };

    eTrackingState mState;
    eTrackingState mLastProcessedState;
    //
    // // Current Frame
    Frame mCurrentFrame;
    cv::Mat mImGray;

    // //Color order (true RGB, false BGR, ignored if grayscale)
    bool mbRGB;
    //
    // // Initialization Variables (Monocular)
    std::vector<int> mvIniLastMatches;
    std::vector<int> mvIniMatches;
    std::vector<cv::Point2f> mvbPrevMatched;
    std::vector<cv::Point3f> mvIniP3D;
    Frame mInitialFrame;
    //
    // // Lists used to recover the full camera trajectory at the end of the execution.
    // // Basically we store the reference keyframe for each frame and its relative transformation
    list<cv::Mat> mlRelativeFramePoses;
    list<KeyFrame*> mlpReferences;
    list<double> mlFrameTimes;
    list<bool> mlbLost;
    //
    // // True if local mapping is deactivated and we are performing only localization
    // bool mbOnlyTracking;
    //
    void Reset();
    Tensor keypoints0, scores0, descriptors0;
    Tensor keypoints1, scores1, descriptors1;
    torch::Dict<std::string, Tensor> pred;
protected:

    // Main tracking function. It is independent of the input sensor.
    void Track();
    //
    // // Map initialization for stereo and RGB-D
    void Initialization();
    //
    void CheckReplacedInLastFrame();
    bool TrackReferenceKeyFrame();
    void UpdateLastFrame();
    bool TrackWithMotionModel();
    //
    bool Relocalization();
    //
    void UpdateLocalMap();
    void UpdateLocalPoints();
    void UpdateLocalKeyFrames();
    //
    bool TrackLocalMap();
    void SearchLocalPoints();
    //
    bool NeedNewKeyFrame();
    void CreateNewKeyFrame();

    bool mbVO;
    //
    // //Other Thread Pointers
    LocalMapping* mpLocalMapper;

    KeyFrame* mpReferenceKF;
    std::vector<KeyFrame*> mvpLocalKeyFrames;
    std::vector<MapPoint*> mvpLocalMapPoints;
    //
    // // System
    System* mpSystem;
    Map* mpMap;
    //
    // //Calibration matrix
    cv::Mat mK;
    cv::Mat mDistCoef;
    float mbf;
    //
    // //New KeyFrame rules (according to fps)
    int mMinFrames;
    int mMaxFrames;
    //
    float mThDepth;
    //
    float mDepthMapFactor;
    //
    int mnMatchesInliers;
    //
    KeyFrame* mpLastKeyFrame;
    Frame mLastFrame;
    unsigned int mnLastKeyFrameId;
    unsigned int mnLastRelocFrameId;
    //
    // //Motion Model
    cv::Mat mVelocity;
    //

    //
    list<MapPoint*> mlpTemporalPoints;
};

}

#endif // TRACKING_H
