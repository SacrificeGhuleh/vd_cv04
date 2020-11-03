#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>

const int FLOWS_SIZE = 1000;

std::string type2str(int type) {
  // https://stackoverflow.com/questions/10167534/how-to-find-out-what-type-of-a-mat-object-is-with-mattype-in-opencv
  std::string r;
  
  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);
  
  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }
  
  r += "C";
  r += (chans+'0');
  
  return r;
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

void showFlowMat(const cv::Mat& mat){
  std::cout << type2str(mat.type()) << std::endl;
  cv::Mat mat1 = cv::Mat::zeros(mat.rows, mat.cols, CV_32FC1);
  cv::Mat mat2 = cv::Mat::zeros(mat.rows, mat.cols, CV_32FC1);
  cv::Mat mat3 = cv::Mat::zeros(mat.rows, mat.cols, CV_32FC3);
  
  for (int row = 0; row < mat.rows; row++) {
    for (int col = 0; col < mat.cols; col++) {
      const cv::Vec2f loc_pix = mat.at<cv::Vec2f>(row, col);
      mat1.at<float>(row, col) = loc_pix[0];
      mat2.at<float>(row, col) = loc_pix[1];
      mat3.at<cv::Vec3f>(row, col) = {loc_pix[0], loc_pix[1]};
  
    }
  }

  cv::imshow("Flow1", mat1);
  cv::imshow("Flow2", mat2);
  cv::imshow("FlowRGB", mat3);
}

int main(int argc, const char **argv) {
  std::ostringstream oss;
  cv::FileStorage fs;
  cv::Mat flowMat;
  
  for(size_t i = 0; i < FLOWS_SIZE; i++){
    oss.str("");
    oss << "./data/flow_field/u" << std::setw(5) << std::setfill('0') << i << ".yml";
    std::cout << oss.str() << std::endl;
    
    fs.open(oss.str(), cv::FileStorage::Mode::READ | cv::FileStorage::Mode::FORMAT_AUTO);
    fs["flow"] >> flowMat;
    fs.release();
    
    showFlowMat(flowMat);
    cv::waitKey(20);
  }
}