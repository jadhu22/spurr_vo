# spurr_vo
Deep Feature based visual odometry

# Dependencies
- Opencv 4.0+
- LibTorch
- g2o (Build this in Thirdparty/g2o)
- C++ 14
- cv_bridge
- ROS Melodic (tested)

# Topics
- Subscriber
  - /camera/rgb/image_color (rgb image topic)
  - /camera/depth/image  (depth image topic)
- Publisher
  - VO_odom  (odometry topic)


