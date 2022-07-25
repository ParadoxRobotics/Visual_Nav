// Robot Navigation stack
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
// Utils
#include <range.hpp> // Range operator (similar to python)

// Normalized Cross Correlation (NCC) for angular error correction
std::vector<float> NCC(cv::Mat RefImg, cv::Mat CurImg, float TemplateProportion, float VerticalCutOff){
  // init position error
  std::vector<float> posError{0.0, 0.0};
  // crop the reference image
  cv::Rect refROI(, 0, RefImg.);
  // crop the current image
  cv::Rect curROI(0, 0, CurImg.cols, (int)(CurImg.rows*VerticalCutOff));
  cv::Mat cropCur = CurImg(curROI);
  return posError
}
