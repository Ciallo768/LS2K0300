#ifndef RED_THRESHOLD_VISUAL_H
#define RED_THRESHOLD_VISUAL_H

#include <opencv2/opencv.hpp>


void RedThresholdNoGui_Init(const char* config_path = "red_threshold.txt");

void RedThresholdNoGui_Deinit();

bool RedThreshold_IsRedRGB(unsigned char r,
                           unsigned char g,
                           unsigned char b);

bool RedThreshold_IsRedBGR(unsigned char b,
                           unsigned char g,
                           unsigned char r);

void RedThreshold_MakeMask(const cv::Mat& frame_bgr,
                           cv::Mat& mask);

void RedThresholdNoGui_Update(const cv::Mat& frame_bgr,
                              cv::Mat& debug_show);

void RedThresholdNoGui_Save();

void RedThresholdNoGui_Load();

void RedThresholdNoGui_Print();

#endif