//
// Created by richardzvonek on 10/21/20.
//

#ifndef VD_CV04_COLORS_H
#define VD_CV04_COLORS_H


#include <opencv2/core/matx.hpp>

// List of 20 Simple, Distinct Colors
// https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/

namespace Color {
  const cv::Vec3b Red(75, 25, 230);
  const cv::Vec3b Green(75, 180, 60);
  const cv::Vec3b Yellow(25, 225, 255);
  const cv::Vec3b Blue(200, 130, 0);
  const cv::Vec3b Orange(48, 130, 245);
  const cv::Vec3b Purple(180, 30, 145);
  const cv::Vec3b Cyan(240, 240, 70);
  const cv::Vec3b Magenta(230, 50, 240);
  const cv::Vec3b Lime(60, 245, 210);
  const cv::Vec3b Pink(190, 190, 250);
  const cv::Vec3b Teal(128, 128, 0);
  const cv::Vec3b Lavender(255, 190, 230);
  const cv::Vec3b Brown(40, 110, 170);
  const cv::Vec3b Beige(200, 250, 255);
  const cv::Vec3b Maroon(0, 0, 128);
  const cv::Vec3b Mint(195, 255, 170);
  const cv::Vec3b Olive(0, 128, 128);
  const cv::Vec3b Apricot(180, 215, 255);
  const cv::Vec3b Navy(128, 0, 0);
  const cv::Vec3b Grey(128, 128, 128);
  const cv::Vec3b Black(0, 0, 0);
  const cv::Vec3b White(255, 255, 255);
  
}

const cv::Vec3b colors[] = {
    Color::Red,
    Color::Green,
    Color::Yellow,
    Color::Blue,
    Color::Orange,
    Color::Purple,
    Color::Cyan,
    Color::Magenta,
    Color::Lime,
    Color::Pink,
    Color::Teal,
    Color::Lavender,
    Color::Brown,
    Color::Beige,
    Color::Maroon,
    Color::Mint,
    Color::Olive,
    Color::Apricot,
    Color::Navy,
    Color::Grey
};

constexpr uint16_t colorsSize = sizeof(colors) / sizeof(cv::Vec3b);

#endif //VD_CV04_COLORS_H
