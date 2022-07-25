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
  std::vector<float> posError = {0.0, 0.0};
  // crop the reference image
  cv::Rect refROI(RefImg.cols-(int)(RefImg.cols*TemplateProportion), 0, (int)(RefImg.rows*VerticalCutOff), (int)(RefImg.cols*TemplateProportion));
  cv::Mat cropRef = RefImg(refROI);
  // crop the current image
  cv::Rect curROI(0, 0, CurImg.cols, (int)(CurImg.rows*VerticalCutOff));
  cv::Mat cropCur = CurImg(curROI);
  // create bin position
  std::vector<int> positions = range(0, (int)(cropCur.cols-cropRef.cols), 1);
  // create difference bin
  std::vector<int> diff(positions.size(), 0);
  // NCC
  for(std::size_t i = 0; i < positions.size(); ++i){
    cv::Rect opROI(positions[i], 0, positions[i]+cropRef.cols, cropCur.rows);
    cv::Mat opCur = cropCur(opROI);
    diff[i] = cv::mean(cv::abs(opCur cropRef));
  }
  int idxMin = std::distance(diff.begin(), std::min_element(diff.begin(), diff.end()));
  // return position and error
  posError[0] = positions[idxMin] - (RefImg.cols-(int)(RefImg.cols*TemplateProportion));
  posError[1] = diff[idxMin];
  return posError
}
