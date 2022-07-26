// Robot Navigation stack -> Patch Norm pre-processing
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
#include <opencv2/core/utility.hpp> // Include basic CV utilities
#include <opencv2/core/types_c.h> // Include special CV structure
#include <opencv2/highgui.hpp> // Include basic CV GUI

// Patch normalization function
void PatchNorm(cv::Mat& Img, std::vector<int> PatchSize, std::vector<int> Padding){
  // Check if the image is valid
  if (!Img.data || Img.empty()){
    std::wcout << "Image Error" << std::endl;
    return EXIT_FAILURE;
  }
  // change the type of the image CV_8UC1 -> CV_32FC1
  Img.convertTo(Img, CV_32FC1);
  // Patch mean and std dev
  cv::Scalar mean, stddev;
  // Get the image size
  int H = Img.rows;
  int W = Img.cols;
  // Compute the number of patch given their size
  int row = (H / PatchSize[0]) + (H % PatchSize[0] ? 1 : 0);
  int col = (W / PatchSize[1]) + (W % PatchSize[1] ? 1 : 0);
  // perform local normalization P = P-Mean(P)/std(P)
  for (int y = 0; y < row; y++){
    for (int x = 0; x < col; x++){
      // Get the patch
      cv::Rect patchROI(x*PatchSize[1]-Padding[1], y*PatchSize[0]-Padding[0], PatchSize[1]+2*Padding[1], PatchSize[0]+2*Padding[0]);
      cv::Mat patch = Img(patchROI)
      // Normalize it
      cv::meanStdDev(patch, mean, stddev);
      patch = (patch - mean[0]) / stddev[0];
    }
  }
}
