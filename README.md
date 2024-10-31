# Path_planning_v1.0
Outcome of path planning project first phase


Overview: 

This project develops an automated structured light 3D measurement system, integrating structured light technology with industrial robots to enhance precision, speed, and scalability in applications such as industrial inspection, reverse engineering, and biomedical analysis. The system addresses the challenges of manual viewpoint and path planning in traditional structured light 3D measurement systems, which are often time-consuming and reliant on expert input, resulting in inefficiencies.

The core of this project is a Generate & Test model-based method that automates the generation of measurement viewpoints. This method utilizes ellipsoidal measurement space constraints derived from model characteristics to produce candidate viewpoints, which are then refined based on scanner design parameters and criteria such as optical visibility, image quality, and surface coverage. This leads to a set of high-quality measurement viewpoints.

Further, the project explores automated path planning techniques from these viewpoints and validates the scanning process in practical settings. The developed system, implemented on Shining3d's Robotscan E0505 platform, automates the generation of measurement viewpoints and path planning from 3D model files, significantly enhancing efficiency and reducing reliance on manual processes. The system has been successfully applied in automated measurements of complex curved parts, demonstrating its effectiveness and efficiency in real-world operations.


Setup Instructions: 

For Viewpointsgeneration, you may need a python IDE like pycharm.
For coordinates transformation and experiment setup calibration, you need RoboDK or Robotstudio. You can also build one with ROS.
For Robotscan-E0505 control, you need a C++ IDE, like visual studio.



Contact Info: 

Xu Wang - wangxu0424whut@163.com

Stuttgart.Germany  2024.10.20
