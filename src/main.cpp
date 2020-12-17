#include <iostream>
#include <fstream>
#include <algorithm>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <random>

#include "colors.h"

class Timer {
public:
  Timer() : beg_(clock_::now()) {}
  
  void reset() { beg_ = clock_::now(); }
  
  double elapsed() const {
    return std::chrono::duration_cast<second_>
        (clock_::now() - beg_).count();
  }

private:
  typedef std::chrono::high_resolution_clock clock_;
  typedef std::chrono::duration<double, std::ratio<1> > second_;
  std::chrono::time_point<clock_> beg_;
};


const int maxIterations = 1000;

const int outputWidth = 1024;
const int outputHeight = 1024;

const int dataWidth = 64;
const int dataHeight = 64;

const float widthScale = static_cast<float>(outputWidth) / static_cast<float>(dataWidth);
const float heightScale = static_cast<float>(outputHeight) / static_cast<float>(dataHeight);

const int numberOfPoints = 1000;
const int pointRadius = 3;
const int lineRadius = 1;

const float deltaT = 1;

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
cv::Point_<T> subpix(const cv::Mat_<cv::Vec<T, 2>> &mat, cv::Point_<T> point) {
  // https://stackoverflow.com/questions/13299409/how-to-get-the-image-pixel-at-real-locations-in-opencv
  cv::Mat patch;
  cv::remap(mat, patch, cv::Mat(1, 1, CV_32FC2, &point), cv::noArray(),
            cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
  return patch.at<cv::Vec<T, 2>>(0, 0);
}

template<typename T>
cv::Point_<T> getRungeKutta(const cv::Mat2f &mat, const cv::Point_<T> &point) {
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
  }
  
  cv::Mat normalizedOutputMat = cv::Mat(outMat.rows, outMat.cols, CV_32FC1);
  cv::normalize(outMat, normalizedOutputMat, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  normalizedOutputMat.convertTo(normalizedOutputMat, CV_8UC1);
  resize(normalizedOutputMat, normalizedOutputMat, cv::Size(), widthScale, heightScale, cv::INTER_LINEAR);
  applyColorMap(normalizedOutputMat, outMatRgb, cv::COLORMAP_COOL);
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
      
      cv::arrowedLine(outputMat, arrowStart, arrowEnd, cv::Vec3b(126,191,255), 1, cv::LINE_8, 0, 0.5);
    }
  }
}

void drawPoints(cv::Mat3b &mat) {
  int idx = 0;
  for (const cv::Point2f &point: points) {
    const cv::Point2f interpolatedPoint(point.x * widthScale, point.y * heightScale);
    pointsHistory.at(iteration).at(idx) = interpolatedPoint;
    cv::circle(mat, interpolatedPoint, pointRadius, colors[idx % colorsSize], cv::FILLED, cv::LINE_AA);
    idx++;
  }
}

void movePoints(const cv::Mat2f &mat) {
  for (cv::Point2f &point: points) {
    point = getRungeKutta(mat, point);
  }
}

void drawPointsHistory(cv::Mat3b &mat) {
  if (iteration < 1) return;
  for (int pointIdx = 0; pointIdx < numberOfPoints; pointIdx++) {
    const cv::Vec3b color = colors[pointIdx % colorsSize];
    for (int iter = 1; iter <= iteration; iter++) {
      const auto &p1 = pointsHistory.at(iter - 1).at(pointIdx);
      const auto &p2 = pointsHistory.at(iter).at(pointIdx);
      cv::line(mat, p1, p2, color, lineRadius, cv::LINE_AA);
    }
  }
}

void showFlowMat(const cv::Mat2f &flowMat, cv::Mat1f &outputMat, cv::Mat3b &outputMatRgb, cv::Mat3b& outputMatRgbWithLines) {
//  std::cout << type2str(flowMat.type()) << std::endl;
  
  curl(flowMat, outputMat, outputMatRgb);
  drawArrows(flowMat, outputMatRgb);
  
  movePoints(flowMat);
  drawPoints(outputMatRgb);
  
  outputMatRgbWithLines = outputMatRgb.clone();
  drawPointsHistory(outputMatRgbWithLines);

//  cv::imshow("Output", outputMat);
  cv::imshow("OutputRGB", outputMatRgb);
  cv::imshow("OutputRGBWithLines", outputMatRgbWithLines);
  
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
    points.at(i) = allPoints.at(i%allPoints.size());
  }
}

int main(int argc, const char **argv) {
  std::ostringstream oss;
  cv::FileStorage fs;
  
  cv::Mat2f flowMat;
  cv::Mat1f outputMat = cv::Mat1f::zeros(dataHeight, dataWidth);
  cv::Mat3b outputMatRgb;
  cv::Mat3b outputMatRgbWithLines;
  
  const std::filesystem::path outputPath = "out/out/";
  const std::filesystem::path outputRgbPath = "out/rgb/";
  const std::filesystem::path outputRgbLinesPath = "out/rgblines";
  
  if (!std::filesystem::is_directory(outputPath) || !std::filesystem::exists(outputPath)) {
    std::filesystem::create_directories(outputPath);
  }
  
  if (!std::filesystem::is_directory(outputRgbPath) || !std::filesystem::exists(outputRgbPath)) {
    std::filesystem::create_directories(outputRgbPath);
  }
  
  if (!std::filesystem::is_directory(outputRgbLinesPath) || !std::filesystem::exists(outputRgbLinesPath)) {
    std::filesystem::create_directories(outputRgbLinesPath);
  }
  
  
  generateRandomPoints();
  
  for (iteration = 0; iteration < maxIterations; iteration++) {
    {
      Timer loadingTimer;
      oss.str("");
      oss << "./data/flow_field/u" << std::setw(5) << std::setfill('0') << iteration << ".yml";
      if(!std::filesystem::exists(oss.str())){
        throw std::runtime_error("File does not exist");
      }
      
      fs.open(oss.str(), cv::FileStorage::Mode::READ | cv::FileStorage::Mode::FORMAT_AUTO);
      fs["flow"] >> flowMat;
      fs.release();
      std::cout << oss.str() << " loaded in " << loadingTimer.elapsed();
    }
    
    {
      Timer drawTimer;
      resize(flowMat, flowMat, cv::Size(), static_cast<float>(dataWidth) / static_cast<float>(flowMat.cols),
             static_cast<float>(dataHeight) / static_cast<float>(flowMat.rows), cv::INTER_CUBIC);
  
      showFlowMat(flowMat, outputMat, outputMatRgb, outputMatRgbWithLines);
      std::cout << ", rendered in " << drawTimer.elapsed() << std::endl;
    }
    
    {
      std::filesystem::path matOutputPath = outputPath;
      std::filesystem::path matOutputRgbPath = outputRgbPath;
      std::filesystem::path matOutputRgbLinesPath = outputRgbLinesPath;

      std::ostringstream ossfname;
      ossfname << std::setw(5) << std::setfill('0') << iteration << ".jpg";

      matOutputPath.append(ossfname.str());
      matOutputRgbPath.append(ossfname.str());
      matOutputRgbLinesPath.append(ossfname.str());

      cv::imwrite(matOutputPath.string(), outputMat);
      cv::imwrite(matOutputRgbPath.string(), outputMatRgb);
      cv::imwrite(matOutputRgbLinesPath.string(), outputMatRgbWithLines);
      
    }
    cv::waitKey(1);
  }
}