#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>

#include "colors.h"

const int FLOWS_SIZE = 1000;

const int outputWidth =  1024;
const int outputHeight = 1024;

const int dataWidth =  64;
const int dataHeight = 64;

//const float widthScale =  static_cast<float>(dataWidth)/static_cast<float>(outputWidth);
//const float heightScale = static_cast<float>(dataHeight)/static_cast<float>(outputHeight);

const float widthScale =  static_cast<float>(outputWidth)/static_cast<float>(dataWidth);
const float heightScale = static_cast<float>(outputHeight)/static_cast<float>(dataHeight);

std::string type2str(int type) {
  // https://stackoverflow.com/questions/10167534/how-to-find-out-what-type-of-a-mat-object-is-with-mattype-in-opencv
  std::string r;
  
  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);
  
  switch (depth) {
    case CV_8U: r = "8U";
      break;
    case CV_8S: r = "8S";
      break;
    case CV_16U: r = "16U";
      break;
    case CV_16S: r = "16S";
      break;
    case CV_32S: r = "32S";
      break;
    case CV_32F: r = "32F";
      break;
    case CV_64F: r = "64F";
      break;
    default: r = "User";
      break;
  }
  
  r += "C";
  r += (chans + '0');
  
  return r;
}

template<typename T>
T rescale(T val, T inMin, T inMax, T outMin, T outMax){
  return outMin + (val - inMin) * ((outMax - outMin) / (inMax - inMin));
}

cv::Vec2f subpix(const cv::Mat &mat, cv::Point2f point) {
  // https://stackoverflow.com/questions/13299409/how-to-get-the-image-pixel-at-real-locations-in-opencv
  cv::Mat patch;
  cv::remap(mat, patch, cv::Mat(1, 1, CV_32FC2, &point), cv::noArray(),
            cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
  return patch.at<cv::Vec2f>(0, 0);
}

cv::Vec2f getRangeKutta(const cv::Mat &mat, const cv::Vec2f &point, const float dt) {
  cv::Vec2f k1 = subpix(mat, point) * dt;
  cv::Vec2f k2 = subpix(mat, point + k1 * 0.5f) * dt;
  cv::Vec2f k3 = subpix(mat, point + k2 * 0.5f) * dt;
  cv::Vec2f k4 = subpix(mat, point + k3) * dt;
  
  return point + ((k1 + (2. * k2) + (2. * k3) + k4) / 6.);
}

void curl(const cv::Mat2f &inMat, cv::Mat1f &outMat, cv::Mat3b &outMatRgb) {
  for (int row = 0; row < inMat.rows; row++) {
    for (int col = 0; col < inMat.cols; col++) {
      if (row > 0 && col > 0 && row < inMat.rows - 1 && col < inMat.cols - 1) {
        outMat.at<float>(row, col) =
            (inMat.at<cv::Point2f>(row, col - 1).y - inMat.at<cv::Point2f>(row, col + 1).y) - // dy
            (inMat.at<cv::Point2f>(row - 1, col).x - inMat.at<cv::Point2f>(row + 1, col).x);  // dx
      } else {
        outMat.at<float>(row, col) = 0.f;
      }
    }
    cv::Mat normalizedOutputMat = cv::Mat(outMat.rows, outMat.cols, CV_32FC1);
    cv::normalize(outMat, normalizedOutputMat, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    normalizedOutputMat.convertTo(normalizedOutputMat, CV_8UC1);
  
    
    resize(normalizedOutputMat, normalizedOutputMat, cv::Size(), widthScale, heightScale, cv::INTER_LINEAR);
//    resize(outMat, outMat, cv::Size(), widthScale, heightScale, cv::INTER_CUBIC);
  
    applyColorMap(normalizedOutputMat, outMatRgb, cv::COLORMAP_COOL);
  }
}

void drawArrows(const cv::Mat &mat, cv::Mat& outputMat){
  double min;
  double max;
  float scaleOverride = 10.f;
  cv::minMaxLoc(mat, &min, &max);
  const float arrowsScale = std::max(std::abs(min), std::abs(max)) / std::min(heightScale, widthScale);
  
  for (int row = 0; row < mat.rows; row++) {
    for (int col = 0; col < mat.cols; col++) {
      const cv::Point2f vector = mat.at<cv::Point2f>(row, col) * arrowsScale * scaleOverride;

      const cv::Point2f middle((col + 0.5f) * widthScale, (row + 0.5f) * heightScale);
      const cv::Point2f arrowStart = middle - vector;
      const cv::Point2f arrowEnd = middle + vector;

      cv::arrowedLine(outputMat, arrowStart, arrowEnd, Color::Orange, 1, 8, 0, 0.3);
    }
  }
}

void showFlowMat(const cv::Mat2f &mat/*, cv::Mat& outputMat*/) {
  std::cout << type2str(mat.type()) << std::endl;
  cv::Mat1f mat1 = cv::Mat1f::zeros(mat.rows, mat.cols);
  cv::Mat1f mat2 = cv::Mat1f::zeros(mat.rows, mat.cols);
  cv::Mat3f mat3 = cv::Mat3f::zeros(mat.rows, mat.cols);
  
  for (int row = 0; row < mat.rows; row++) {
    for (int col = 0; col < mat.cols; col++) {
      const cv::Vec2f& loc_pix = mat.at<cv::Vec2f>(row, col);
      mat1.at<float>(row, col) = loc_pix[0];
      mat2.at<float>(row, col) = loc_pix[1];
      mat3.at<cv::Vec3f>(row, col) = {loc_pix[0], loc_pix[1]};
    }
  }
  
  cv::imshow("FlowCh1", mat1);
  cv::imshow("FlowCh2", mat2);
  cv::imshow("FlowRGB", mat3);
  
  cv::Mat1f outputMat =    cv::Mat1f::zeros(mat.rows, mat.cols);
  cv::Mat3b outputMatRgb = cv::Mat3b::zeros(mat.rows, mat.cols);
  
  curl(mat, outputMat, outputMatRgb);
  drawArrows(mat, outputMatRgb);
  
  cv::imshow("Output", outputMat);
  cv::imshow("OutputRGB", outputMatRgb);
  
}


int main(int argc, const char **argv) {
  std::ostringstream oss;
  cv::FileStorage fs;
  cv::Mat2f flowMat;
  
  for (size_t i = 0; i < FLOWS_SIZE; i++) {
    oss.str("");
    oss << "./data/flow_field/u" << std::setw(5) << std::setfill('0') << i << ".yml";
    std::cout << oss.str() << std::endl;
    
    fs.open(oss.str(), cv::FileStorage::Mode::READ | cv::FileStorage::Mode::FORMAT_AUTO);
    fs["flow"] >> flowMat;
    fs.release();
  
    resize(flowMat, flowMat, cv::Size(), static_cast<float>(dataWidth)/static_cast<float>(flowMat.cols), static_cast<float>(dataHeight)/static_cast<float>(flowMat.rows), cv::INTER_CUBIC);
    showFlowMat(flowMat);
    cv::waitKey(20);
  }
}