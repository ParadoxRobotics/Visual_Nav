// Robot Navigation stack -> fisheye image to panoramic image
// Author : Munch Quentin, 2022.

#pragma once
// general lib
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <cmath>
#include <vector>
// OpenCV specific lib
#include <opencv2/opencv.hpp> // Include OpenCV API
#include <opencv2/imgproc.hpp> // Include basic image processing OpenCV API
#include <opencv2/calib3d/calib3d.hpp> // 3D/2D calibration CV API
#include <opencv2/core/utility.hpp> // Include basic CV utilities
#include <opencv2/core/types_c.h> // Include special CV structure
#include <opencv2/highgui.hpp> // Include basic CV GUI

// Convert a spherical fisheye greyscale image to a panoramic one with the correct size
cv::Mat Fisheye2Pano(cv::Mat Img, int RH, int RW){
  // Camera/image parameters (calibration or not)
  int H = Img.rows;
  int W = Img.cols;
  double r = H/2; // Circular image -> radius is H/2
  double cx = W/2;
  double cy = H/2;
  // Panoramic image parameters
  int Hp = (int)r;
  int Wp = (int)(2*cv::CV_PI*r);
  // Create empty panoramic image
  cv::Mat pano;
  pano.create(Hp, Wp, Img.type());
  // create panoramic image by projecting polar coordiante to cartesian
  for (int i = 0; i < pano.cols; i++){
    for (int j = 0; j < pano.rows; j++){
        // polar to cartesian
        double rp = j/Hp*r;
        double th = i/Wp*2.0*cv::CV_PI;
        int x = cx+rp*sin(th);
        int y = cy+rp*cos(th);
        pano.at<uchar>(cv::Point(i, j)) = fisheyeImage.at<uchar>(cv::Point(x, y));
    }
  }
  // return panoramic image
  return pano;
}
