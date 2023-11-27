#include "Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"Converter.h"
#include"Map.h"

#include"Optimizer.h"
#include"PnPSolver.h"

#include<iostream>

#include<mutex>

const int TH_HIGH = 100;
const int TH_LOW = 50;
const int HISTO_LENGTH = 30;

using namespace std;

namespace SPURR_VO
{

Tracking::Tracking(System *pSys, const string &strSettingPath, torch::jit::script::Module sp, torch::jit::script::Module sg, Map *pMap):
    mState(NO_IMAGES_YET), superpoint(sp), superglue(sg), mpMap(pMap), mpSystem(pSys), mnLastRelocFrameId(0)
{
    // Load camera parameters from settings file

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;


    mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
    cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;

    mDepthMapFactor = fSettings["DepthMapFactor"];
    if(fabs(mDepthMapFactor)<1e-5)
        mDepthMapFactor=1;
    else
        mDepthMapFactor = 1.0f/mDepthMapFactor;
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB, const torch::Tensor &imRGBT,const cv::Mat &imD, const double &timestamp)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    // if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
    //     imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    std::tie(keypoints0, scores0, descriptors0) = mpSystem->unpack_result(superpoint.forward({imRGBT}));

    mCurrentFrame = Frame(mImGray,imRGBT,keypoints0,scores0,descriptors0,imDepth,timestamp,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}

void Tracking::Track()
{
  std::cout<<"tracking: "<<mState<<"  ,  "<<mCurrentFrame.N<<std::endl;

    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState=mState;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    if(mState==NOT_INITIALIZED)
    {
        Initialization();

        // mpFrameDrawer->Update(this);

        if(mState!=OK)
            return;
    }
    else
    {
        // System is initialized. Track Frame.
        bool bOK;

        // Local Mapping is activated. This is the normal behaviour, unless
        // you explicitly activate the "only tracking" mode.

        if(mState==OK)
        {
            // Local Mapping might have changed some MapPoints tracked in last frame
            CheckReplacedInLastFrame();

            if(mVelocity.empty() || mCurrentFrame.mnId<mnLastRelocFrameId+2)
            {
                bOK = TrackReferenceKeyFrame();
            }
            else
            {
              std::cout<<"Motion Model!!"<<std::endl;
                bOK = TrackWithMotionModel();
                if(!bOK)
                    bOK = TrackReferenceKeyFrame();
            }
        }
        else
        {
            bOK = Relocalization();
            // if(!bOK)
            // {
            //   cout << "Track lost soon after initialisation, reseting..." << endl;
            //   mpSystem->Reset();
            //   return;
            // }
        }

        mCurrentFrame.mpReferenceKF = mpReferenceKF;
        std::cout<<"STATE:: "<<mState<<std::endl;
        std::cout<<"BOK:: "<<bOK<<std::endl;

        if(bOK)
            bOK = TrackLocalMap();
        std::cout<<"Track LOCAL MAP BOK:: "<<bOK<<std::endl;

        if(bOK)
            mState = OK;
        else
            mState=LOST;

        // Update drawer
        // mpFrameDrawer->Update(this);

        // If tracking were good, check if we insert a keyframe
        if(bOK)
        {
            // Update motion model
            if(!mLastFrame.mTcw.empty())
            {
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                mVelocity = mCurrentFrame.mTcw*LastTwc;
            }
            else
                mVelocity = cv::Mat();

            // std::cout<<"VELOCITY:: "<<mVelocity<<std::endl;
            // mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean VO matches
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }

            // Delete temporal MapPoints
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

            // Check if we need to insert a new keyframe
            if(NeedNewKeyFrame())
                CreateNewKeyFrame();

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        if(mState==LOST)
        {
          if(mpMap->KeyFramesInMap()<=5)
          {
              cout << "Track lost soon after initialisation, reseting..." << endl;
              mpSystem->Reset();
              return;
          }
          // // if(mvpLocalKeyFrames.size()<=3)
          // // {
          //   cout << "Track lost soon after initialisation, reseting...  "<<mvpLocalKeyFrames.size() << endl;
          //   mpSystem->Reset();
          //   return;
          // // }
        }

        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame);
    }

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if(!mCurrentFrame.mTcw.empty())
    {
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState==LOST);
    }
    else
    {
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }

}


void Tracking::Initialization()
{
    if(mCurrentFrame.N>50)
    {
        // Set Frame pose to the origin
        mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

        // Create KeyFrame
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpMap);
        // // Insert KeyFrame in the map
        mpMap->AddKeyFrame(pKFini);
        //
        // // Create MapPoints and asscoiate to KeyFrame
        for(int i=0; i<mCurrentFrame.N;i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap);
                pNewMP->AddObservation(pKFini,i);
                pKFini->AddMapPoint(pNewMP,i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);

                mCurrentFrame.mvpMapPoints[i]=pNewMP;
            }
        }

        cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpMap->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        // mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        mState=OK;
    }
}

void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}


bool Tracking::TrackReferenceKeyFrame()
{
    vector<MapPoint*> vpMapPointMatches;
    const vector<MapPoint*> vpMapPointsKF = mpReferenceKF->GetMapPointMatches();
    vpMapPointMatches = vector<MapPoint*>(mCurrentFrame.N,static_cast<MapPoint*>(NULL));
    std::cout<<"vpMAPPointsKF:: "<<vpMapPointsKF.size()<<std::endl;

    torch::Dict<std::string, Tensor> input;
    input.insert("image0", mpReferenceKF->imgTnsr);
    input.insert("image1", mCurrentFrame.imgTnsr);
    input.insert("keypoints0", mpReferenceKF->keypoints.unsqueeze(0));
    input.insert("keypoints1", mCurrentFrame.keypoints.unsqueeze(0));
    input.insert("scores0", mpReferenceKF->scores.unsqueeze(0));
    input.insert("scores1", mCurrentFrame.scores.unsqueeze(0));
    input.insert("descriptors0", mpReferenceKF->descriptors.unsqueeze(0));
    input.insert("descriptors1", mCurrentFrame.descriptors.unsqueeze(0));
    pred = toTensorDict(superglue.forward({input}));

    auto matches = pred.at("matches0")[0];
    auto valid = at::nonzero(matches > -1).squeeze();
    auto mkpts0 = mpReferenceKF->keypoints.index_select(0, valid);
    auto mkpts1 = mCurrentFrame.keypoints.index_select(0, matches.index_select(0, valid));
    auto confidence = pred.at("matching_scores0")[0].index_select(0, valid);
    std::cout << "Image #0 keypoints: " << mpReferenceKF->keypoints.size(0) << std::endl;
    std::cout << "Image #1 keypoints: " << mCurrentFrame.keypoints.size(0) << std::endl;
    // std::cout << "Valid match count: " << valid.size(0) << std::endl;
    std::cout<<"TOTAL:: "<<matches.size(0)<<"  ,  "<<confidence.size(0)<<"  ,  "<<mkpts0.size(0)<<"  ,  "<<mkpts1.size(0)<<std::endl;

    int nmatches = confidence.size(0);

    if(nmatches<15)
        return false;
    int k=0;
    for(int i=0; i<matches.size(0); i++)
    {
      if(matches[i].item<float>()!=-1)
      {
        MapPoint* pMP = vpMapPointsKF[i];
        if(pMP)
        {
          if(confidence[k].item().to<double>()>0.9)
          {
            auto kp0 = mpReferenceKF->keypoints[i];
            auto mp0 = mkpts0[k];
            auto kp1 = mCurrentFrame.keypoints[matches[i].item<int>()];
            auto mp1 = mkpts1[k];
            // std::cout<<std::lround(kp0[0].item<float>())<<"  ,  "<<std::lround(kp0[1].item<float>())<<std::endl;
            // std::cout<<std::lround(mp0[0].item<float>())<<"  ,  "<<std::lround(mp0[1].item<float>())<<std::endl;
            // std::cout<<std::lround(kp1[0].item<float>())<<"  ,  "<<std::lround(kp1[1].item<float>())<<std::endl;
            // std::cout<<std::lround(mp1[0].item<float>())<<"  ,  "<<std::lround(mp1[1].item<float>())<<std::endl;
            // std::cout<<"@@"<<std::endl;
            vpMapPointMatches[matches[i].item<int>()]=pMP;
          }
        }
        k++;
      }
    }
    std::cout<<"K:: "<<k<<std::endl;
    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.mTcw);

    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }
    std::cout<<"tKRF:: "<<nmatchesMap<<std::endl;
    return nmatchesMap>=10;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Tracking::unpack_result(const IValue &result)
{
  auto dict = result.toGenericDict();
  return {dict.at("keypoints").toTensorVector()[0], //
          dict.at("scores").toTensorVector()[0],    //
          dict.at("descriptors").toTensorVector()[0]};
}

torch::Dict<std::string, Tensor> Tracking::toTensorDict(const torch::IValue &value)
{
  return c10::impl::toTypedDict<std::string, Tensor>(value.toGenericDict());
}
//
void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();

    mLastFrame.SetPose(Tlr*pRef->GetPose());

    if(mnLastKeyFrameId==mLastFrame.mnId)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);
    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
        {
            bCreateNew = true;
        }

        if(bCreateNew)
        {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);

            MapPoint* pNewMP = new MapPoint(x3D,mpMap,&mLastFrame,i);

            mLastFrame.mvpMapPoints[i]=pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;
    }
}

bool Tracking::TrackWithMotionModel()
{
    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    UpdateLastFrame();

    mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);

    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    // Project points seen in previous frame
    int th=15;
    int nmatches = SearchByProjection(mCurrentFrame,mLastFrame,th);
    std::cout<<"Motion Model NMATCHES_1:: "<<nmatches<<std::endl;
    // If few matches, uses a wider window search
    if(nmatches<20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        nmatches = SearchByProjection(mCurrentFrame,mLastFrame,2*th);
    }
    std::cout<<"Motion Model NMATCHES_2:: "<<nmatches<<std::endl;
    if(nmatches<20)
        return false;

    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }
    std::cout<<"Motion Model NMATCHES_2 nMatchesMap:: "<<nmatchesMap<<std::endl;
    return nmatchesMap>=10;
}

bool Tracking::TrackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.

    UpdateLocalMap();

    SearchLocalPoints();

    // Optimize Pose
    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();

                if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                    mnMatchesInliers++;
            }
        }
    }
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
        return false;
    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    std::cout<<"$$$$  mnMATCHES::  "<<mnMatchesInliers<<std::endl;
    if(mnMatchesInliers<30)
        return false;
    else
        return true;
}

bool Tracking::NeedNewKeyFrame()
{
       // If Local Mapping is freezed by a Loop Closure do not insert keyframes
       if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
          return false;

       const int nKFs = mpMap->KeyFramesInMap();
       std::cout<<"nKFs:: "<<nKFs<<std::endl;
       // Do not insert keyframes if not enough frames have passed from last relocalisation
       if(nKFs>mMaxFrames)
           return false;

       // Tracked MapPoints in the reference keyframe
       int nMinObs = 3;
       if(nKFs<=2)
           nMinObs=2;
       int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

       // Local Mapping accept keyframes?
       bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

       // Check how many "close" points are being tracked and how many could be potentially created.
       int nNonTrackedClose = 0;
       int nTrackedClose= 0;

       for(int i =0; i<mCurrentFrame.N; i++)
       {
           if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
           {
               if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                   nTrackedClose++;
               else
                   nNonTrackedClose++;
           }
       }

       bool bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

       // Thresholds
       float thRefRatio = 0.75f;
       if(nKFs<2)
           thRefRatio = 0.4f;

       // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
       const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
       // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
       const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
       // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
       const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15);

       if((c1a||c1b)&&c2)
       {
           // If the mapping accepts keyframes, insert keyframe.
           // Otherwise send a signal to interrupt BA
           if(bLocalMappingIdle)
           {
               return true;
           }
           else
           {
               mpLocalMapper->InterruptBA();

               if(mpLocalMapper->KeyframesInQueue()<3)
                   return true;
               else
                   return false;
           }
       }
       else
           return false;
}

void Tracking::CreateNewKeyFrame()
{
    if(!mpLocalMapper->SetNotStop(true))
        return;

    std::cout<<std::endl<<"Creating New KeyFrame"<<std::endl<<std::endl;
    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap);

    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    mCurrentFrame.UpdatePoseMatrices();

    // We sort points by the measured depth by the stereo/RGBD sensor.
    // We create all those MapPoints whose depth < mThDepth.
    // If there are less than 100 close points we create the 100 closest.
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mCurrentFrame.N);
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        float z = mCurrentFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(!vDepthIdx.empty())
    {
        sort(vDepthIdx.begin(),vDepthIdx.end());

        int nPoints = 0;
        for(size_t j=0; j<vDepthIdx.size();j++)
        {
            int i = vDepthIdx[j].second;

            bool bCreateNew = false;

            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP)
                bCreateNew = true;
            else if(pMP->Observations()<1)
            {
                bCreateNew = true;
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
            }

            if(bCreateNew)
            {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);
                pNewMP->AddObservation(pKF,i);
                pKF->AddMapPoint(pNewMP,i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);

                mCurrentFrame.mvpMapPoints[i]=pNewMP;
                nPoints++;
            }
            else
            {
                nPoints++;
            }

            if(vDepthIdx[j].first>mThDepth && nPoints>100)
                break;
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
    std::cout<<std::endl<<"Created New KeyFrame"<<std::endl<<std::endl;
}

void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch=0;

    // Project points in frame and check its visibility
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;
        // std::cout<<"AXCD"<<std::endl;
        // Project (this fills MapPoint variables for matching)
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }
    // std::cout<<"123456   "<<nToMatch<<std::endl;
    if(nToMatch>0)
    {
        int th = 5;
        // If the camera has been relocalised recently, perform a coarser search

        SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
    }
    // std::cout<<"789010"<<std::endl;
}

int Tracking::SearchByProjection(Frame &F, const vector<MapPoint*> &vpMapPoints, const float th)
{
  int nmatches=0;

  const bool bFactor = th!=1.0;

  for(size_t iMP=0; iMP<vpMapPoints.size(); iMP++)
  {
    MapPoint* pMP = vpMapPoints[iMP];
    if(!pMP->mbTrackInView)
        continue;

    if(pMP->isBad())
        continue;

    const int &nPredictedLevel = pMP->mnTrackScaleLevel;

    float r=0;
    if(pMP->mTrackViewCos>0.998)
        r = 2.5;
    else
        r = 4.0;

    if(bFactor)
        r*=th;

    // std::cout<<pMP->mTrackProjX<<"  ,  "<<pMP->mTrackProjY<<"  ,  "<<r<<std::endl;
    const vector<size_t> vIndices = F.GetFeaturesInArea(pMP->mTrackProjX,pMP->mTrackProjY,r);
    // std::cout<<"ZSERT:: "<<vIndices.size()<<"  ,  "<<pMP->des.size(0)<<std::endl;

    if(vIndices.empty())
        continue;

    cv::Mat pMP_mat(256, 1, CV_32FC1, pMP->des.data<float>());

    int bestDist=256;
    int bestIdx =-1;
    for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
    {
        const size_t idx = *vit;

        if(F.mvpMapPoints[idx])
            if(F.mvpMapPoints[idx]->Observations()>0)
                continue;
        if(F.mvuRight[idx]>0)
        {
            const float er = fabs(pMP->mTrackProjXR-F.mvuRight[idx]);
            if(er>r*2)
                continue;
        }
        // std::cout<<F.descriptors.transpose(1,0)[idx]<<std::endl;
        // std::cout<<pMP->des<<std::endl;
        auto unsqueezed = F.descriptors.transpose(1,0)[idx].cpu();
        // std::cout<<"####"<<std::endl;
        // std::cout<<unsqueezed<<std::endl;
        // unsqueezed = unsqueezed.cpu();
        cv::Mat F_mat(256, 1, CV_32FC1, unsqueezed.data<float>());

        // std::cout<<pMP_mat.rows<<"  ,  "<<pMP_mat.cols<<"  ,  "<<F_mat.rows<<"  ,  "<<F_mat.cols<<std::endl;
        // std::cout<<F_mat-pMP_mat<<std::endl;
        float dist = cv::norm(pMP_mat-F_mat);
        // std::cout<<"DIST:: "<<idx<<"  ,  "<<dist<<std::endl;
        if(dist<bestDist)
        {
            bestDist=dist;
            bestIdx=idx;
        }
    }
    if(bestDist<=0.5)
    {
        F.mvpMapPoints[bestIdx]=pMP;
        nmatches++;
    }
  }
  return nmatches;
}

int Tracking::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th)
{
    int nmatches = 0;

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);

    const cv::Mat twc = -Rcw.t()*tcw;

    const cv::Mat Rlw = LastFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tlw = LastFrame.mTcw.rowRange(0,3).col(3);

    const cv::Mat tlc = Rlw*twc+tlw;

    for(int i=0; i<LastFrame.N; i++)
    {
        MapPoint* pMP = LastFrame.mvpMapPoints[i];

        if(pMP)
        {
          if(!LastFrame.mvbOutlier[i])
          {
            cv::Mat x3Dw = pMP->GetWorldPos();
            cv::Mat x3Dc = Rcw*x3Dw+tcw;

            const float xc = x3Dc.at<float>(0);
            const float yc = x3Dc.at<float>(1);
            const float invzc = 1.0/x3Dc.at<float>(2);

            if(invzc<0)
                continue;

            float u = CurrentFrame.fx*xc*invzc+CurrentFrame.cx;
            float v = CurrentFrame.fy*yc*invzc+CurrentFrame.cy;

            if(u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX)
                continue;
            if(v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY)
                continue;

            float radius = th*2.5;

            vector<size_t> vIndices2;

            vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius);

            if(vIndices2.empty())
                continue;

            auto unsqueezed_ = LastFrame.descriptors.transpose(1,0)[i].cpu();
            cv::Mat pMP_mat(256, 1, CV_32FC1, unsqueezed_.data<float>());

            // cv::Mat pMP_mat(256, 1, CV_32FC1, pMP->des.data<float>());

            int bestDist=256;
            int bestIdx =-1;

            for(vector<size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
            {
              const size_t i2 = *vit;
              if(CurrentFrame.mvpMapPoints[i2])
                  if(CurrentFrame.mvpMapPoints[i2]->Observations()>0)
                      continue;

              if(CurrentFrame.mvuRight[i2]>0)
              {
                  const float ur = u - CurrentFrame.mbf*invzc;
                  const float er = fabs(ur - CurrentFrame.mvuRight[i2]);
                  if(er>radius)
                      continue;
              }

              auto unsqueezed = mCurrentFrame.descriptors.transpose(1,0)[i2].cpu();
              cv::Mat F_mat(256, 1, CV_32FC1, unsqueezed.data<float>());

              float dist = cv::norm(pMP_mat-F_mat);

              if(dist<bestDist)
              {
                  bestDist=dist;
                  bestIdx=i2;
              }
            }
            if(bestDist<=0.5)
            {
                mCurrentFrame.mvpMapPoints[bestIdx]=pMP;
                nmatches++;
            }
          }
        }
    }

    return nmatches;
}

void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();
        int k=-1;
        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            k+=1;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                // std::cout<<pKF->descriptors.size(0)<<std::endl;
                auto unsqueezed = pKF->descriptors.transpose(1,0)[k].cpu();
                // std::cout<<k<<"  ,  "<<unsqueezed[k][0].item<float>()<<std::endl;
                pMP->des = unsqueezed;
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}

void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame*,int> keyframeCounter;
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad())
            {
                const map<KeyFrame*,size_t> observations = pMP->GetObservations();
                for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;
            }
        }
    }
    std::cout<<"KFC SIZE::  "<<keyframeCounter.size()<<std::endl;
    if(keyframeCounter.empty())
        return;

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;

        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }
    std::cout<<"mvpLocalKeyFrames SIZE::  "<<mvpLocalKeyFrames.size()<<std::endl;
    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80)
            break;

        KeyFrame* pKF = *itKF;

        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);
        // std::cout<<"vNEIGHS SIZE::  "<<vNeighs.size()<<std::endl;
        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }

    }

    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

bool Tracking::Relocalization()
{
    std::cout<<"Relocalization is performed when tracking is lost"<<std::endl<<std::endl;
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame*> vpCandidateKFs = mvpLocalKeyFrames;
    std::cout<<vpCandidateKFs.size()<<std::endl;
    if(vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    for(int i=0; i<nKFs; i++)
    {
        std::cout<<"1234"<<std::endl;
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            const vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();
            torch::Dict<std::string, Tensor> input;
            input.insert("image0", pKF->imgTnsr);
            input.insert("image1", mCurrentFrame.imgTnsr);
            input.insert("keypoints0", pKF->keypoints.unsqueeze(0));
            input.insert("keypoints1", mCurrentFrame.keypoints.unsqueeze(0));
            input.insert("scores0", pKF->scores.unsqueeze(0));
            input.insert("scores1", mCurrentFrame.scores.unsqueeze(0));
            input.insert("descriptors0",pKF->descriptors.unsqueeze(0));
            input.insert("descriptors1", mCurrentFrame.descriptors.unsqueeze(0));
            pred = toTensorDict(superglue.forward({input}));

            auto matches = pred.at("matches0")[0];
            auto valid = at::nonzero(matches > -1).squeeze();
            auto mkpts0 = pKF->keypoints.index_select(0, valid);
            auto mkpts1 = mCurrentFrame.keypoints.index_select(0, matches.index_select(0, valid));
            auto confidence = pred.at("matching_scores0")[0].index_select(0, valid);
            std::cout << "Image #0 keypoints: " << pKF->keypoints.size(0) << std::endl;
            std::cout << "Image #1 keypoints: " << mCurrentFrame.keypoints.size(0) << std::endl;
            // std::cout << "Valid match count: " << valid.size(0) << std::endl;
            std::cout<<"TOTAL:: "<<matches.size(0)<<"  ,  "<<confidence.size(0)<<"  ,  "<<mkpts0.size(0)<<"  ,  "<<mkpts1.size(0)<<std::endl;

            int nmatches = confidence.size(0);
            vvpMapPointMatches[i] = vector<MapPoint*>(mCurrentFrame.N,static_cast<MapPoint*>(NULL));
            std::cout<<vvpMapPointMatches[i].size()<<std::endl;
            int k=0;
            for(int j=0; j<matches.size(0); j++)
            {
              if(matches[j].item<int>()!=-1)
              {
                MapPoint* pMP = vpMapPointsKF[j];
                if(pMP)
                {
                  if(confidence[k].item().to<double>()>0.8)
                  {
                    // std::cout<<"hey hey  "<<matches[j].item<int>()<<"  ,  "<<mCurrentFrame.N<<"  ,  "<<pKF->N<<std::endl;
                    vvpMapPointMatches[i][matches[j].item<int>()]=pMP;

                  }
                }
                k++;
              }
            }

            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }
    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);
            std::cout<<"PIKA BOOM BOOM Tcw::  "<<Tcw<<std::endl;
            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);
                std::cout<<"PIKA BOOM BOOM::  "<<nGood<<std::endl;
                if(nGood<10)
                    continue;

                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50)
                {
                    int nadditional = SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,0.5);

                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional = SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3, 0.35);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }

                // If the pose is supported by enough inliers stop ransacs and continue

                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }
}

int Tracking::SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, const set<MapPoint*> &sAlreadyFound, const float th , const int dist)
{
    int nmatches = 0;

    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);
    const cv::Mat Ow = -Rcw.t()*tcw;

    // Rotation Histogram (to check rotation consistency)
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP)
        {
            if(!pMP->isBad() && !sAlreadyFound.count(pMP))
            {
                //Project
                cv::Mat x3Dw = pMP->GetWorldPos();
                cv::Mat x3Dc = Rcw*x3Dw+tcw;

                const float xc = x3Dc.at<float>(0);
                const float yc = x3Dc.at<float>(1);
                const float invzc = 1.0/x3Dc.at<float>(2);

                const float u = CurrentFrame.fx*xc*invzc+CurrentFrame.cx;
                const float v = CurrentFrame.fy*yc*invzc+CurrentFrame.cy;

                if(u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX)
                    continue;
                if(v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY)
                    continue;

                // Compute predicted scale level
                cv::Mat PO = x3Dw-Ow;
                float dist3D = cv::norm(PO);

                const float maxDistance = 10.0;
                const float minDistance = 0.0;

                // Depth must be inside the scale pyramid of the image
                if(dist3D<minDistance || dist3D>maxDistance)
                    continue;

                // Search in a window
                const float radius = th*2;

                const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius);

                if(vIndices2.empty())
                    continue;

                auto unsqueezed_ = pKF->descriptors.transpose(1,0)[i].cpu();
                cv::Mat dMP(256, 1, CV_32FC1, unsqueezed_.data<float>());

                int bestDist = 256;
                int bestIdx2 = -1;

                for(vector<size_t>::const_iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
                {
                    const size_t i2 = *vit;
                    if(CurrentFrame.mvpMapPoints[i2])
                        continue;

                    auto unsqueezed = CurrentFrame.descriptors.transpose(1,0)[i2].cpu();
                    cv::Mat d(256, 1, CV_32FC1, unsqueezed.data<float>());

                    const int dist = cv::norm(dMP,d);

                    if(dist<bestDist)
                    {
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                }

                if(bestDist<=dist)
                {
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
                    nmatches++;

                    float rot = pKF->mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle;
                    if(rot<0.0)
                        rot+=360.0f;
                    int bin = round(rot*factor);
                    if(bin==HISTO_LENGTH)
                        bin=0;
                    assert(bin>=0 && bin<HISTO_LENGTH);
                    rotHist[bin].push_back(bestIdx2);

                }

            }
        }
    }

    int ind1=-1;
    int ind2=-1;
    int ind3=-1;

    ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

    for(int i=0; i<HISTO_LENGTH; i++)
    {
        if(i!=ind1 && i!=ind2 && i!=ind3)
        {
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                CurrentFrame.mvpMapPoints[rotHist[i][j]]=NULL;
                nmatches--;
            }
        }
    }
    return nmatches;
}

void Tracking::ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1=0;
    int max2=0;
    int max3=0;

    for(int i=0; i<L; i++)
    {
        const int s = histo[i].size();
        if(s>max1)
        {
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(s>max2)
        {
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        }
        else if(s>max3)
        {
            max3=s;
            ind3=i;
        }
    }

    if(max2<0.1f*(float)max1)
    {
        ind2=-1;
        ind3=-1;
    }
    else if(max3<0.1f*(float)max1)
    {
        ind3=-1;
    }
}

void Tracking::Reset()
{
    cout << "System Reseting" << endl;
    // if(mpViewer)
    // {
    //     mpViewer->RequestStop();
    //     while(!mpViewer->isStopped())
    //         usleep(3000);
    // }

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();
    mVelocity = cv::Mat();
    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();
}

// void Tracking::ChangeCalibration(const string &strSettingPath)
// {
//     cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
//     float fx = fSettings["Camera.fx"];
//     float fy = fSettings["Camera.fy"];
//     float cx = fSettings["Camera.cx"];
//     float cy = fSettings["Camera.cy"];
//
//     cv::Mat K = cv::Mat::eye(3,3,CV_32F);
//     K.at<float>(0,0) = fx;
//     K.at<float>(1,1) = fy;
//     K.at<float>(0,2) = cx;
//     K.at<float>(1,2) = cy;
//     K.copyTo(mK);
//
//     cv::Mat DistCoef(4,1,CV_32F);
//     DistCoef.at<float>(0) = fSettings["Camera.k1"];
//     DistCoef.at<float>(1) = fSettings["Camera.k2"];
//     DistCoef.at<float>(2) = fSettings["Camera.p1"];
//     DistCoef.at<float>(3) = fSettings["Camera.p2"];
//     const float k3 = fSettings["Camera.k3"];
//     if(k3!=0)
//     {
//         DistCoef.resize(5);
//         DistCoef.at<float>(4) = k3;
//     }
//     DistCoef.copyTo(mDistCoef);
//
//     mbf = fSettings["Camera.bf"];
//
//     Frame::mbInitialComputations = true;
// }
//
// void Tracking::InformOnlyTracking(const bool &flag)
// {
//     mbOnlyTracking = flag;
// }



}
