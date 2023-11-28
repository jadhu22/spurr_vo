#include<ros/ros.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <chrono>
#include <experimental/filesystem>
#include <utility>

#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <nav_msgs/Odometry.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <Eigen/Eigen>
#include<opencv2/core/core.hpp>
#include<math.h>
#include"System.h"

using namespace std;
namespace fs = std::experimental::filesystem;
ros::Publisher VO_pub;
Eigen::Matrix4f oTr;
nav_msgs::Odometry odom;

class ImageGrabber
{
public:
    ofstream myfile;
    ImageGrabber(SPURR_VO::System* pSLAM):mpSLAM(pSLAM){
      myfile.open ("spurr_vo.txt");
    }

    void GrabRGBD(const sensor_msgs::ImageConstPtr& msgRGB,const sensor_msgs::ImageConstPtr& msgD);
    cv::Vec3f rotationMatrixToEulerAngles(cv::Mat &R);
    SPURR_VO::System* mpSLAM;


};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "RGBD");
    ros::start();

    torch::manual_seed(1);
    torch::autograd::GradMode::set_enabled(false);

    std::string spPath = "/home/jadoo/SuperGluePretrainedNetworkCPP/cpp/build/SuperPoint.zip";
    std::string sgPath = "/home/jadoo/SuperGluePretrainedNetworkCPP/cpp/build/SuperGlue.zip";

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    std::string settings = "/home/jadoo/ELEC845/project/ws/src/Spurr_VO/camera.yaml";
    SPURR_VO::System SLAM(settings, spPath, sgPath);

    ImageGrabber igb(&SLAM);

    ros::NodeHandle nh;

    message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh, "/camera/rgb/image_color", 1);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, "/camera/depth/image", 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), rgb_sub,depth_sub);
    VO_pub = nh.advertise<nav_msgs::Odometry>("VO_odom", 1);
    sync.registerCallback(boost::bind(&ImageGrabber::GrabRGBD,&igb,_1,_2));

    oTr << 0.0,  0.0, 1.0, 0.0,
          -1.0,  0.0, 0.0, 0.0,
           0.0, -1.0, 0.0, 0.0,
           0.0,  0.0, 0.0, 1.0;;

    ros::Rate r(100);
    while(ros::ok())
    {
        odom.header.stamp = ros::Time::now();
        odom.header.frame_id = "odom";
        VO_pub.publish(odom);
        ros::spinOnce();
        r.sleep();
    }

    ros::shutdown();

    return 0;
}

void ImageGrabber::GrabRGBD(const sensor_msgs::ImageConstPtr& msgRGB,const sensor_msgs::ImageConstPtr& msgD)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptrRGB;
    try
    {
        cv_ptrRGB = cv_bridge::toCvShare(msgRGB);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv_bridge::CvImageConstPtr cv_ptrD;
    try
    {
        cv_ptrD = cv_bridge::toCvShare(msgD);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    std::cout<<"RGBD GRABBER"<<std::endl;
    cv::Mat cTw = mpSLAM->TrackRGBD(cv_ptrRGB->image,cv_ptrD->image,cv_ptrRGB->header.stamp.toSec());
    std::cout<<cTw<<std::endl;
    if(!cTw.empty())
    {
      Eigen::Matrix4f cTr;
      cTr << cTw.at<float>(0,0), cTw.at<float>(0,1), cTw.at<float>(0,2), cTw.at<float>(0,3),
             cTw.at<float>(1,0), cTw.at<float>(1,1), cTw.at<float>(1,2), cTw.at<float>(1,3),
             cTw.at<float>(2,0), cTw.at<float>(2,1), cTw.at<float>(2,2), cTw.at<float>(2,3),
             0.0, 0.0, 0.0, 1.0;
      Eigen::Matrix4f rTc = cTr.inverse();

      rTc = oTr*rTc;

      cv::Mat wTc = cTw.inv();
      cv::Mat rot = cTw.rowRange(0,3).colRange(0,3);
      cv::Vec3f euler = rotationMatrixToEulerAngles(rot);
      float Yaw=0.0;
      if(euler[1]*180/M_PI<0)
          Yaw = -acos(rot.at<float>(2,2));
      if(euler[1]*180/M_PI>0)
          Yaw = acos(rot.at<float>(2,2));
      // std::cout<<"EULER::  "<<Yaw*180/M_PI<<"  ,  "<<euler[1]*180/M_PI<<"  ,  "<<euler[2]*180/M_PI<<std::endl;
      geometry_msgs::Quaternion odom_quat = tf::createQuaternionMsgFromYaw(Yaw);
      std::cout<<cv_ptrRGB->header.stamp<<"  ,  "<<rTc(0,3)<<"  ,  "<<rTc(1,3)<<"  ,  "<<0.0<<"  ,  "<<odom_quat.x<<"  ,  "<<odom_quat.y<<"  ,  "<<odom_quat.z<<"  ,  "<<odom_quat.w<<std::endl;
      myfile<<cv_ptrRGB->header.stamp<<" "<<float(rTc(0,3))<<" "<<float(rTc(1,3))<<" "<<0.0<<" "<<float(odom_quat.x)<<" "<<float(odom_quat.y)<<" "<<float(odom_quat.z)<<" "<<float(odom_quat.w)<<std::endl;
      //set the position
      odom.pose.pose.position.x = rTc(0,3);
      odom.pose.pose.position.y = rTc(1,3);
      odom.pose.pose.position.z = 0.0;
      odom.pose.pose.orientation = odom_quat;
    }
}

cv::Vec3f ImageGrabber::rotationMatrixToEulerAngles(cv::Mat &R)
{
    float sy = sqrt(R.at<float>(0,0) * R.at<float>(0,0) +  R.at<float>(1,0) * R.at<float>(1,0) );

    bool singular = sy < 1e-6; // If

    float x, y, z;
    if (!singular)
    {
        x = atan2(R.at<float>(2,1) , R.at<float>(2,2));
        y = atan2(-R.at<float>(2,0), sy);
        z = atan2(R.at<float>(1,0), R.at<float>(0,0));
    }
    else
    {
        x = atan2(-R.at<float>(1,2), R.at<float>(1,1));
        y = atan2(-R.at<float>(2,0), sy);
        z = 0;
    }
    return cv::Vec3f(x, y, z);
}
