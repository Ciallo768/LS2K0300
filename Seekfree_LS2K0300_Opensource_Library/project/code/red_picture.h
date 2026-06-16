#ifndef RED_PICTURE_H
#define RED_PICTURE_H
#include "zf_common_headfile.hpp"
bool FindTargetRoiByFixedIpm(const cv::Mat& input_frame,
                             cv::Mat& target_roi,
                             struct Guaidian* red_rect_pts,      // 红块4角点，94×60
                             struct Guaidian* red_center_pt,     // 红块中心点，94×60
                             struct Guaidian* target_top_pts,    // 图片框上边2点，94×60
                             cv::Point2f* target_pts=nullptr,           // 保留：图片框4点，原图坐标
                             cv::Point2f* red_pts=nullptr,               // 保留：红块4点，原图坐标
                             float* debug_valid_ratio=nullptr,
                             bool* erase_pts_ready =nullptr);
extern Guaidian  red_point;
extern Guaidian red_L_L,red_R_L,picture_L_H,picture_R_H;//四个拐点
extern struct Guaidian red_rect_pts[4];    // 红块四个角点
extern struct Guaidian red_center_pt;      // 红块中心点
extern struct Guaidian target_top_pts[2];  // 图片框上面两个点
void DetectRedBlock(cv::Mat &src,int roi_x,int roi_y,int width,int height);
#endif