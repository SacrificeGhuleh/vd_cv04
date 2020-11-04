#include <iostream>
#include <fstream>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <random>

#include "colors.h"

const int maxIterations = 1000;

const int outputWidth = 1024;
const int outputHeight = 1024;

const int dataWidth = 64;
const int dataHeight = 64;

const float widthScale = static_cast<float>(outputWidth) / static_cast<float>(dataWidth);
const float heightScale = static_cast<float>(outputHeight) / static_cast<float>(dataHeight);

const int numberOfPoints = 1000;
const int pointRadius = 3;

const float deltaT = 0.05;

std::array<cv::Point2f, numberOfPoints> points;
std::array<std::array<cv::Point2f, numberOfPoints>, maxIterations> pointsHistory;

int iteration = 0;

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
T rescale(T val, T inMin, T inMax, T outMin, T outMax) {
  return outMin + (val - inMin) * ((outMax - outMin) / (inMax - inMin));
}

template<typename T>
cv::Point_<T> subpix(const cv::Mat_<cv::Vec<T, 2>> &mat, cv::Point_<T> point) {
  // https://stackoverflow.com/questions/13299409/how-to-get-the-image-pixel-at-real-locations-in-opencv
  cv::Mat patch;
  cv::remap(mat, patch, cv::Mat(1, 1, CV_32FC2, &point), cv::noArray(),
            cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
  return patch.at<cv::Vec<T, 2>>(0, 0);
}

template<typename T>
cv::Point_<T> getRangeKutta(const cv::Mat2f &mat, const cv::Point_<T> &point) {
  cv::Point_<T> k1 = subpix(mat, point) * deltaT;
  cv::Point_<T> k2 = subpix(mat, point + k1 * 0.5f) * deltaT;
  cv::Point_<T> k3 = subpix(mat, point + k2 * 0.5f) * deltaT;
  cv::Point_<T> k4 = subpix(mat, point + k3) * deltaT;
  
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

void drawArrows(const cv::Mat &mat, cv::Mat &outputMat) {
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

void drawPoints(cv::Mat3b &mat) {
  int idx = 0;
  for (const cv::Point2f &point: points) {
    const cv::Point2f interpolatedPoint(point.x * widthScale, point.y * heightScale);
    pointsHistory.at(iteration).at(idx) = interpolatedPoint;
    cv::circle(mat, interpolatedPoint, pointRadius, colors[idx], cv::FILLED, cv::LINE_AA);
    idx++;
    idx %= colorsSize;
  }
}

void movePoints(const cv::Mat2f &mat) {
  for (cv::Point2f &point: points) {
    point = getRangeKutta(mat, point);
  }
}

void drawPointsHistory(cv::Mat3b &mat) {
  if (iteration < 1) return;
  for (int pointIdx = 0; pointIdx < numberOfPoints; pointIdx++) {
    const cv::Vec3b color = colors[pointIdx % colorsSize];
    for (int iter = 1; iter <= iteration; iter++) {
      const auto &p1 = pointsHistory.at(iter - 1).at(pointIdx);
      const auto &p2 = pointsHistory.at(iter).at(pointIdx);
      cv::line(mat, p1, p2, color, 1, cv::LINE_AA);
    }
  }
}

void showFlowMat(const cv::Mat2f &mat/*, cv::Mat& outputMat*/) {
  std::cout << type2str(mat.type()) << std::endl;
//  cv::Mat1f mat1 = cv::Mat1f::zeros(mat.rows, mat.cols);
//  cv::Mat1f mat2 = cv::Mat1f::zeros(mat.rows, mat.cols);
//  cv::Mat3f mat3 = cv::Mat3f::zeros(mat.rows, mat.cols);
//
//  for (int row = 0; row < mat.rows; row++) {
//    for (int col = 0; col < mat.cols; col++) {
//      const cv::Vec2f& loc_pix = mat.at<cv::Vec2f>(row, col);
//      mat1.at<float>(row, col) = loc_pix[0];
//      mat2.at<float>(row, col) = loc_pix[1];
//      mat3.at<cv::Vec3f>(row, col) = {loc_pix[0], loc_pix[1]};
//    }
//  }
//
//  cv::imshow("FlowCh1", mat1);
//  cv::imshow("FlowCh2", mat2);
//  cv::imshow("FlowRGB", mat3);
  
  cv::Mat1f outputMat = cv::Mat1f::zeros(mat.rows, mat.cols);
  cv::Mat3b outputMatRgb = cv::Mat3b::zeros(mat.rows, mat.cols);
  
  curl(mat, outputMat, outputMatRgb);
  drawArrows(mat, outputMatRgb);
  
  movePoints(mat);
  drawPoints(outputMatRgb);
  drawPointsHistory(outputMatRgb);

//  cv::imshow("Output", outputMat);
  cv::imshow("OutputRGB", outputMatRgb);
  
}

void generateRandomPoints() {
  std::array<cv::Point2f, dataWidth * dataHeight> allPoints;
  
  for (int y = 0; y < dataHeight; y++) {
    for (int x = 0; x < dataWidth; x++) {
      const int idx = y * dataWidth + x;
      allPoints.at(idx).x = x;
      allPoints.at(idx).y = y;
    }
  }
  
  std::shuffle(allPoints.begin(), allPoints.end(), std::mt19937(std::random_device()()));
  
  for (int i = 0; i < numberOfPoints; i++) {
    points.at(i) = allPoints.at(i);
  }
}

int main(int argc, const char **argv) {
  std::ostringstream oss;
  cv::FileStorage fs;
  cv::Mat2f flowMat;
  generateRandomPoints();
  
  for (iteration = 0; iteration < maxIterations; iteration++) {
    oss.str("");
    oss << "./data/flow_field/u" << std::setw(5) << std::setfill('0') << iteration << ".yml";
    std::cout << oss.str() << std::endl;
    
    fs.open(oss.str(), cv::FileStorage::Mode::READ | cv::FileStorage::Mode::FORMAT_AUTO);
    fs["flow"] >> flowMat;
    fs.release();
    
    resize(flowMat, flowMat, cv::Size(), static_cast<float>(dataWidth) / static_cast<float>(flowMat.cols),
           static_cast<float>(dataHeight) / static_cast<float>(flowMat.rows), cv::INTER_CUBIC);
    showFlowMat(flowMat);
    cv::waitKey(1);
  }
}