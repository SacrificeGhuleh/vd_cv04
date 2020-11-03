#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>

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

int main(int argc, const char **argv) {

}