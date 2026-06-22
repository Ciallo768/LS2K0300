/***************************************************检测红色矩形块******************************************************/
#include "zf_common_headfile.hpp"
//角点排序函数 根据极点进行排序 
//输出顺序
//points[0] —— 左上角（Top-Left）
//points[1] —— 右上角（Top-Right）
//points[2] —— 右下角（Bottom-Right）
//points[3] —— 左下角（Bottom-Left）
Guaidian  red_point;
Guaidian red_L_L,red_R_L,picture_L_H,picture_R_H;//四个拐点
struct Guaidian red_rect_pts[4];    // 红块四个角点
struct Guaidian red_center_pt;      // 红块中心点
struct Guaidian target_top_pts[2];  // 图片框上面两个点
struct Guaidian whole_rect_pts[4];    // 整个图像区域的四个角点
 void OrderTargetStripQuad(cv::Point2f points[4])
{
    cv::Point2f center(0.0f, 0.0f);
    for (int i = 0; i < 4; ++i)
    {
        center += points[i];
    }
    center *= 0.25f;

    cv::Point2f ordered_cycle[4] = {points[0], points[1], points[2], points[3]};
    std::sort(ordered_cycle, ordered_cycle + 4, [&](const cv::Point2f &a, const cv::Point2f &b) {
        return std::atan2(a.y - center.y, a.x - center.x) <
               std::atan2(b.y - center.y, b.x - center.x);
    });

    float edge_len2[4];
    for (int i = 0; i < 4; ++i)
    {
        const cv::Point2f d = ordered_cycle[(i + 1) & 3] - ordered_cycle[i];
        edge_len2[i] = d.x * d.x + d.y * d.y;
    }

    const int long_edge_base = (edge_len2[0] + edge_len2[2] >= edge_len2[1] + edge_len2[3]) ? 0 : 1;
    const int opposite_edge = (long_edge_base + 2) & 3;
    const float base_mid_y = (ordered_cycle[long_edge_base].y + ordered_cycle[(long_edge_base + 1) & 3].y) * 0.5f;
    const float opposite_mid_y = (ordered_cycle[opposite_edge].y + ordered_cycle[(opposite_edge + 1) & 3].y) * 0.5f;
    const int top_edge = (base_mid_y <= opposite_mid_y) ? long_edge_base : opposite_edge;
    const int top_next = (top_edge + 1) & 3;
    const int bottom_next = (top_edge + 2) & 3;
    const int bottom_prev = (top_edge + 3) & 3;

    const bool top_forward =
        ordered_cycle[top_edge].x < ordered_cycle[top_next].x ||
        (std::fabs(ordered_cycle[top_edge].x - ordered_cycle[top_next].x) < 1.0e-3f &&
         ordered_cycle[top_edge].y <= ordered_cycle[top_next].y);
    if (top_forward)
    {
        points[0] = ordered_cycle[top_edge];
        points[1] = ordered_cycle[top_next];
        points[2] = ordered_cycle[bottom_next];
        points[3] = ordered_cycle[bottom_prev];
    }
    else
    {
        points[0] = ordered_cycle[top_next];
        points[1] = ordered_cycle[top_edge];
        points[2] = ordered_cycle[bottom_prev];
        points[3] = ordered_cycle[bottom_next];
    }
}
// 对整数参数做限幅，保证 value 始终落在 [low, high] 范围内。
int ClampTargetInt(int value, int low, int high)
{
    return std::max(low, std::min(value, high));
}
//判断值 是否在[low,high]之间
bool is_clamp(int value,int low,int high)
{
    return value>=low && value<=high;
}

//参数标定 ipm相机参数矩阵

#ifndef CAM_WIDTH
#define CAM_WIDTH 320
#endif

#ifndef CAM_HEIGHT
#define CAM_HEIGHT 240
#endif

struct IpmLutPoint
{
    float x;
    float y;
};

struct IpmLutTable
{
    IpmLutPoint points[CAM_HEIGHT][CAM_WIDTH];
};
/***************************************************红色标注******************************************************/
/*引入BEV鸟瞰图变换*/
// ===================== 固定 IPM 正变换参数：图像坐标 -> BEV 坐标 =====================
constexpr float kStaticIpmRot00 = 1.612538001f;
constexpr float kStaticIpmRot01 = -0.254463765f;
constexpr float kStaticIpmRot02 = -224.154440449f;
constexpr float kStaticIpmRot10 = -0.104520711f;
constexpr float kStaticIpmRot11 = -3.170461577f;
constexpr float kStaticIpmRot12 = 388.677685209f;
constexpr float kStaticIpmRot20 = -0.006968047f;
constexpr float kStaticIpmRot21 = 0.050865849f;
constexpr float kStaticIpmRot22 = 1.000000000f;

constexpr float kStaticIpmInv00 = 0.759965484f;
constexpr float kStaticIpmInv01 = 0.369279406f;
constexpr float kStaticIpmInv02 = 26.818973095f;
constexpr float kStaticIpmInv10 = 0.086256536f;
constexpr float kStaticIpmInv11 = -0.001676870f;
constexpr float kStaticIpmInv12 = 19.986547448f;
constexpr float kStaticIpmInv20 = 0.000907964f;
constexpr float kStaticIpmInv21 = 0.002658452f;
constexpr float kStaticIpmInv22 = 0.170243164f;

constexpr float kStaticIpmMinDivisor = 1e-4f;

constexpr float StaticAbs(float value)
{
    return value < 0.0f ? -value : value;
}

constexpr float StaticSafeDivisor(float value)
{
    return StaticAbs(value) < kStaticIpmMinDivisor
               ? (value >= 0.0f ? kStaticIpmMinDivisor : -kStaticIpmMinDivisor)
               : value;
}


// 原图坐标 -> BEV 坐标
constexpr IpmLutPoint BuildStaticCamToIpmPoint(int x, int y)
{
    const float xf = static_cast<float>(x);
    const float yf = static_cast<float>(y);

    const float d = StaticSafeDivisor(kStaticIpmRot20 * xf +
                                      kStaticIpmRot21 * yf +
                                      kStaticIpmRot22);

    return IpmLutPoint{
        (kStaticIpmRot00 * xf + kStaticIpmRot01 * yf + kStaticIpmRot02) / d,
        (kStaticIpmRot10 * xf + kStaticIpmRot11 * yf + kStaticIpmRot12) / d
    };
}
// 编译期生成查表：原图每个像素点对应一个 BEV 坐标
constexpr IpmLutTable BuildStaticCamToIpmLut()
{
    IpmLutTable lut{};

    for (int y = 0; y < CAM_HEIGHT; ++y)
    {
        for (int x = 0; x < CAM_WIDTH; ++x)
        {
            lut.points[y][x] = BuildStaticCamToIpmPoint(x, y);
        }
    }

    return lut;
}
constexpr auto kCamToIpmLut = BuildStaticCamToIpmLut();

inline cv::Point2f ImagePointToCamPoint(const cv::Point2f& image_point,
                                        const cv::Mat& input_frame)
{
    const float image_to_cam_x =
        static_cast<float>(CAM_WIDTH) /
        static_cast<float>(std::max(1, input_frame.cols));

    const float image_to_cam_y =
        static_cast<float>(CAM_HEIGHT) /
        static_cast<float>(std::max(1, input_frame.rows));

    return cv::Point2f(image_point.x * image_to_cam_x,
                       image_point.y * image_to_cam_y);
}


inline cv::Point2f CamPointToImagePoint(const cv::Point2f& cam_point,
                                        const cv::Mat& input_frame)
{
    const float cam_to_image_x =
        static_cast<float>(input_frame.cols) /
        static_cast<float>(std::max(1, CAM_WIDTH));

    const float cam_to_image_y =
        static_cast<float>(input_frame.rows) /
        static_cast<float>(std::max(1, CAM_HEIGHT));

    return cv::Point2f(cam_point.x * cam_to_image_x,
                       cam_point.y * cam_to_image_y);
}

// 查表：图像点 -> BEV 点
inline cv::Point2f LookupIpmPointFromCamLut(const cv::Point2f& cam_point)
{
    const int x = ClampTargetInt(static_cast<int>(std::lround(cam_point.x)),
                                 0,
                                 CAM_WIDTH - 1);

    const int y = ClampTargetInt(static_cast<int>(std::lround(cam_point.y)),
                                 0,
                                 CAM_HEIGHT - 1);

    const IpmLutPoint& point = kCamToIpmLut.points[y][x];

    return cv::Point2f(point.x, point.y);
}


// 反变换：BEV 点 -> 原图点
inline cv::Point2f StaticIpmToCamPoint(const cv::Point2f& ipm_point)
{
    const float x = ipm_point.x;
    const float y = ipm_point.y;

    float d = kStaticIpmInv20 * x +
              kStaticIpmInv21 * y +
              kStaticIpmInv22;

    if (std::fabs(d) < 1e-6f)
    {
        d = (d >= 0.0f) ? 1e-6f : -1e-6f;
    }

    const float cam_x = (kStaticIpmInv00 * x +
                         kStaticIpmInv01 * y +
                         kStaticIpmInv02) / d;

    const float cam_y = (kStaticIpmInv10 * x +
                         kStaticIpmInv11 * y +
                         kStaticIpmInv12) / d;

    return cv::Point2f(cam_x, cam_y);
}


// 当前图像点 -> BEV 点
inline cv::Point2f ImagePointToBevPoint(const cv::Point2f& image_point,
                                        const cv::Mat& input_frame)
{
    cv::Point2f cam_point = ImagePointToCamPoint(image_point, input_frame);
    return LookupIpmPointFromCamLut(cam_point);
}


// BEV 点 -> 当前图像点
inline cv::Point2f BevPointToImagePoint(const cv::Point2f& bev_point,
                                        const cv::Mat& input_frame)
{
    cv::Point2f cam_point = StaticIpmToCamPoint(bev_point);
    return CamPointToImagePoint(cam_point, input_frame);
}


//安全性检查 判断 转化到BEV上的点有没有越界
inline bool IsFinitePoint(const cv::Point2f& p)
{
    return std::isfinite(p.x) && std::isfinite(p.y);
}


//角度补偿函数
inline cv::Point2f RotateVector2D(const cv::Point2f& v, float angle_deg)
{
    const float rad = angle_deg * static_cast<float>(CV_PI) / 180.0f;

    const float c = std::cos(rad);
    const float s = std::sin(rad);

    return cv::Point2f(
        v.x * c - v.y * s,
        v.x * s + v.y * c
    );
}



 //rgb方式判断是否为红色像素点
//改为rgb方式 + 四领域 爬出轮廓
inline bool is_red_rgb(unsigned char r, unsigned char g ,unsigned char b)
{
    if(r > Flash.debug_rgb_r_min && r-g > Flash.debug_rgb_rg_diff && r-b > Flash.debug_rgb_rb_diff)
    return 1;//表示红色
    else return 0;
}

//快速访问图像中的单个点的rgb三色通道 并判断是否为红色
inline uchar get_red_rgb(
    const cv::Mat& img,
    int x,
    int y)
{   
    if(img.empty()){
        printf("get_red_rgb img empty!\n");
        return false;
    }
    if(img.channels()!=3)
    {
        return 0;
    }
    if(x >= img.cols || y>= img.rows || x < 0 || y<0) 
    {
        return 0;
    }
    const int channels = 3;
    const uchar* ptr = img.data + y * img.step;//获取起始行地址
    const uchar* pixel = ptr + x * channels;//获取三色通道值

    unsigned char b = pixel[0];
    unsigned char g = pixel[1];
    unsigned char r = pixel[2];

    return is_red_rgb(r,g,b) ? 255 : 0;
    //return RedThreshold_IsRedRGB(r, g, b) ? 1 : 0;
}
//用于判断黑白二值图像中某个点是否为白点
inline bool is_white(const cv::Mat& img,int x,int y)
{
    if(img.empty())return false;
    if(x >= img.cols || y>= img.rows || x < 0 || y<0) 
    {
        return false;
    }
    const uchar* row = img.ptr<uchar>(y);// 获取第 y 行的指针
    return row[x] != 0; // 二值图像中非零即为白色（255）

}


// 统计 3x3 邻域内有多少个红色点
inline int CountRed3x3(const cv::Mat& img, int x, int y)
{
    int cnt = 0;

    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            const int nx = x + dx;
            const int ny = y + dy;

            if (nx < 0 || nx >= img.cols || ny < 0 || ny >= img.rows)
            {
                continue;
            }

            if (get_red_rgb(img, nx, ny))
            {
                ++cnt;
            }
        }
    }

    return cnt;
}


// 统计当前点水平方向附近红色点数量
// half_window = 2 表示统计 x-2 ~ x+2，共 5 个点
inline int CountRedHorizontal(const cv::Mat& img, int x, int y, int half_window)
{
    int cnt = 0;

    for (int dx = -half_window; dx <= half_window; ++dx)
    {
        const int nx = x + dx;

        if (nx < 0 || nx >= img.cols || y < 0 || y >= img.rows)
        {
            continue;
        }

        if (get_red_rgb(img, nx, y))
        {
            ++cnt;
        }
    }

    return cnt;
}

inline bool FindSeedRedRun(const cv::Mat& img,
                           const cv::Point& seed,
                           int& run_x0,
                           int& run_x1)
{
    if (img.empty() || seed.x < 0 || seed.x >= img.cols ||
        seed.y < 0 || seed.y >= img.rows)
    {
        return false;
    }

    if (!get_red_rgb(img, seed.x, seed.y))
    {
        return false;
    }

    run_x0 = seed.x;
    run_x1 = seed.x;

    while (run_x0 - 1 >= 0 && get_red_rgb(img, run_x0 - 1, seed.y))
    {
        --run_x0;
    }

    while (run_x1 + 1 < img.cols && get_red_rgb(img, run_x1 + 1, seed.y))
    {
        ++run_x1;
    }

    return true;
}



cv::Rect rect_resize;
//快速判断图像中有无红色 粗找 在resizedframe中 用rgb 寻找 并返回一个roi矩阵 方便后续在ROI中爬线
//修改：不再返回 roi区域 避免因为roi区域跳动导致的处理时长问题
// 在 320*240 原图中跳行跳列粗找红色种子点
// 要求：
// 1. 只在 imgInfo.top 以下找
// 2. 只在赛道左右边界内找
// 3. 返回的 seed 是原图坐标
bool red_detect_rgb(const cv::Mat& src_img, cv::Point& seed)
{
    seed = cv::Point(-1, -1);

    if (src_img.empty() || src_img.channels() != 3)
    {
        return false;
    }

    const int small_h = 60;

    const float small_to_src_y =
        static_cast<float>(src_img.rows) / static_cast<float>(small_h);

    // 截止行：使用 imgInfo.top，表示只在这条线以下找红色
    const int small_cutoff_y = ClampTargetInt(imgInfo.top, 0, small_h - 1);

    const int search_top = ClampTargetInt(
        static_cast<int>(std::lround(small_cutoff_y * small_to_src_y)),
        0,
        src_img.rows - 1);

    // 搜索到原图最底部
    const int search_bottom = src_img.rows - 1;

    const int y_stride = 1;  // 每一行都找
    const int x_stride = 4;  // 每隔4列采样一次

    // 至少连续两个采样点是红色，才认为这一行有可靠红色段
    const int min_run_samples = 2;

    // 从底部往上找，优先找最靠下的红色块
    for (int y = search_bottom; y >= search_top; y -= y_stride)
    {
        int run_start_x = -1;
        int run_len = 0;

        int best_run_start_x = -1;
        int best_run_len = 0;

        // 不再判断左右边线，直接整行搜索
        for (int x = 0; x < src_img.cols; x += x_stride)
        {
            if (get_red_rgb(src_img, x, y))
            {
                if (run_start_x < 0)
                {
                    run_start_x = x;
                    run_len = 1;
                }
                else
                {
                    ++run_len;
                }
            }
            else
            {
                if (run_len > best_run_len)
                {
                    best_run_len = run_len;
                    best_run_start_x = run_start_x;
                }

                run_start_x = -1;
                run_len = 0;
            }
        }

        // 处理这一行最后一段红色
        if (run_len > best_run_len)
        {
            best_run_len = run_len;
            best_run_start_x = run_start_x;
        }

        // 如果这一行有足够长的红色段
        if (best_run_len >= min_run_samples && best_run_start_x >= 0)
        {
            const int seed_x =
                best_run_start_x + (best_run_len / 2) * x_stride;

            seed = cv::Point(
                ClampTargetInt(seed_x, 0, src_img.cols - 1),
                y);

            // 再确认种子点附近红色密度，防止单点误判
            if (CountRed3x3(src_img, seed.x, seed.y) >= 3)
            {
                // printf("粗找到种子点\n");
                return true;
            }
            else
            {
                // printf("3x3领域内种子点不足跳过\n");
            }
        }
    }

    return false;
}


// 依据最小旋转矩形的下底边 用长边的法向作为方向 取红色矩形最靠下的一小部分作为锚点 移动固定距离
//挑取一小部分红色矩形条作为旋转外接矩形的的轮廓 避免因为爬线爬出急救包 而导致矩形形状跳动
//问题很大
bool get_bottom_redRect(const std::vector<cv::Point>& component_points,
                                const cv::Size& image_size,
                                cv::RotatedRect& bottom_rect,
                                std::vector<cv::Point>* bottom_points_out = nullptr)
{
        bottom_rect = cv::RotatedRect();//清空元素

    if (component_points.size() < 20)
    {
        return false;
    }

    // --------------------------------------------------
    // 找完整连通域的最下方 y 坐标
    // --------------------------------------------------
    int max_y = component_points[0].y;
    int min_y = component_points[0].y;
    int min_x = component_points[0].x;
    int max_x = component_points[0].x;

    for (const cv::Point& p : component_points)//遍历轮廓中的每一个点
    {
        if (p.y > max_y) max_y = p.y;
        if (p.y < min_y) min_y = p.y;
        if (p.x < min_x) min_x = p.x;
        if (p.x > max_x) max_x = p.x;
    }

    const int component_h = max_y - min_y + 1;//定义最大高度
    const int component_w = max_x - min_x + 1;////定义最大宽度

    if (component_w < 6 || component_h < 2)
    {
        return false;
    }

    // --------------------------------------------------
    // 自适应选择底部带高度
    // --------------------------------------------------
    // 320x240 下建议 8~14 像素。
    // 如果红条较远，6 像素也能用；
    // 如果红条较近，最多取到 18 像素，避免包含太多上方急救包。
    int base_band_h = component_h / 4;
    base_band_h = ClampTargetInt(base_band_h, 6, 14);

    // 如果画面不是 240 高，按比例稍微调整
    if (image_size.height > 0)
    {
        int scale_band_h = image_size.height / 24;  // 240 高时约等于 10
        scale_band_h = ClampTargetInt(scale_band_h, 6, 14);
        base_band_h = std::max(base_band_h, scale_band_h);
    }

    // 最大允许向上取多少，太大会重新把急救包红色带进来
    const int max_band_h = ClampTargetInt(image_size.height / 10, 16, 28);

    std::vector<cv::Point> bottom_points;
    bottom_points.reserve(component_points.size());

    bool found_good_band = false;

    // --------------------------------------------------
    // 从较薄的底部带开始取，如果点太少或宽度不足，再逐步加厚
    // --------------------------------------------------
    for (int band_h = base_band_h; band_h <= max_band_h; band_h += 2)
    {
        bottom_points.clear();

        const int y_limit = max_y - band_h + 1;

        for (const cv::Point& p : component_points)
        {
            if (p.y >= y_limit)
            {
                bottom_points.push_back(p);
            }
        }

        if (bottom_points.size() < 12)
        {
            continue;
        }

        cv::Rect band_box = cv::boundingRect(bottom_points);

        // 底部红色定位条应该有一定横向宽度。
        // 如果因为红条倾斜，只取到一个角，band_box.width 会很小，
        // 此时继续加厚 band_h。
        if (band_box.width < std::max(8, component_w / 3))
        {
            continue;
        }

        found_good_band = true;
        break;
    }

    if (!found_good_band || bottom_points.size() < 12)
    {
        return false;
    }

    // --------------------------------------------------
    // 用底部一小条红色点计算旋转矩形
    // --------------------------------------------------
    bottom_rect = cv::minAreaRect(bottom_points);

    const float w = bottom_rect.size.width;
    const float h = bottom_rect.size.height;

    const float long_edge = std::max(w, h);
    const float short_edge = std::max(1.0f, std::min(w, h));
    const float ratio = long_edge / short_edge;

    // 底部红色定位条应当是长条状。
    // 如果 ratio 太小，说明取到的不像红色条，可能是噪声或急救包局部。
    if (long_edge < 8.0f || ratio < 1.8f)
    {
        return false;
    }

    // --------------------------------------------------
    // 可选输出底部点，方便调试显示
    // --------------------------------------------------
    if (bottom_points_out != nullptr)
    {
        *bottom_points_out = bottom_points;
    }

    return true;
}



//在图像中爬出红色轮廓并返回红色轮廓的最小外接矩形 (四领域) 
// 从粗找得到的红色种子点出发，用 4 邻域爬出完整红色连通域
// 输出红色连通域的最小外接旋转矩形
inline bool get_red_contour(const cv::Mat& src_img,
                            const cv::Point& seed,
                            cv::RotatedRect& rotated_rect)
{   
    rotated_rect = cv::RotatedRect();

    if (src_img.empty() || src_img.channels() != 3)
    {
        return false;
    }

    if (seed.x < 0 || seed.x >= src_img.cols ||
        seed.y < 0 || seed.y >= src_img.rows)
    {
        return false;
    }

    // 种子点必须是红色
    if (!get_red_rgb(src_img, seed.x, seed.y))
    {
        return false;
    }

    const int small_h = 60;
    const float small_to_src_y = static_cast<float>(src_img.rows) / static_cast<float>(small_h);

    // 细找仍然限制在 imgInfo.top 以下，避免爬到无关的上方区域
    const int search_top = ClampTargetInt(static_cast<int>(std::lround(imgInfo.top * small_to_src_y)),
                                          0,
                                          src_img.rows - 1);

    const int search_bottom = src_img.rows - 1;

    cv::Mat visited = cv::Mat::zeros(src_img.rows, src_img.cols, CV_8UC1);

    std::vector<cv::Point> stack;
    stack.reserve(2048);

    std::vector<cv::Point> component_points;
    component_points.reserve(2048);
    
    //保存当前点的相邻点
    std::vector<cv::Point> boundary_points;
    boundary_points.reserve(2048);

    stack.push_back(seed);
    visited.at<uchar>(seed.y, seed.x) = 255;// 后续优化为快速遍历

    const int dx[4] = {1, -1, 0, 0};
    const int dy[4] = {0, 0, 1, -1};

while (!stack.empty())
    {
        cv::Point p = stack.back();
        stack.pop_back();

        component_points.push_back(p);

        // 当前红点是否为边界点
        bool is_boundary = false;

        for (int i = 0; i < 4; ++i)
        {
            const int nx = p.x + dx[i];
            const int ny = p.y + dy[i];

            // 只要四邻域有一个方向越界，也认为当前点是边界
            if (nx < 0 || nx >= src_img.cols ||
                ny < search_top || ny > search_bottom)
            {
                is_boundary = true;
                continue;
            }

            // 先判断邻点是不是红色
            const bool neighbor_is_red = get_red_rgb(src_img, nx, ny);

            // 只要四邻域有一个点不是红点，当前点就是边界点
            if (!neighbor_is_red)
            {
                is_boundary = true;
                continue;
            }

            // 如果邻点是红色，但是已经访问过，就不用再入栈
            if (visited.at<uchar>(ny, nx) != 0)
            {
                continue;
            }

            // 邻点是红色且没有访问过，加入爬线
            visited.at<uchar>(ny, nx) = 255;
            stack.push_back(cv::Point(nx, ny));
        }

        // 新增：保存边界点
        if (is_boundary)
        {
            boundary_points.push_back(p);
        }
    }


    // 连通域太小，认为是噪声
    if (component_points.size() < 20)
    {
        return false;
    }
    // 边界点太少，也认为不可靠
    if (boundary_points.size() < 10)
    {
        return false;
    }
    rotated_rect = cv::minAreaRect(component_points);
    std::vector<cv::Point> bottom_points;

    static cv::Mat debug_mask;



    // debug_mask = cv::Mat::zeros(src_img.rows, src_img.cols, CV_8UC1);
    // debug_mask.setTo(cv::Scalar(0));

    // for (const cv::Point& p : component_points)
    // {
    //     if (p.x >= 0 && p.x < debug_mask.cols &&
    //     p.y >= 0 && p.y < debug_mask.rows)
    //     {
    //         debug_mask.at<uchar>(p.y, p.x) = 255;//调试用 画出细找出的红色轮廓
    //     }
    // }
//     rotated_rect = cv::minAreaRect(component_points);

    // printf("细找到红色轮廓\n");
    return true;
}


//裁剪 拉伸变换 
//根据得到的旋转矩形 获取旋转矩形的四个角点 
//用角点先找旋转矩形的长底边 以长底边的中垂线作为框的方向 同时要求框的中点也位于中垂线上
bool find_targetROI(const cv::Mat& input_frame, cv::RotatedRect& rotated_rect)
{
    rotated_rect = cv::RotatedRect();

    if (input_frame.empty())
    {
        return false;
    }

    // 图片区域：12cm x 12cm
    // 红块区域：12cm x 5cm
    const float TARGET_H_RATIO = 12.0f / 12.0f;
    const float RED_H_RATIO    = 5.0f  / 12.0f;

    cv::Point red_seed;
    cv::RotatedRect red_rect;

    // 1. 粗找红色种子点
    if (!red_detect_rgb(input_frame, red_seed))
    {
        return false;
    }

    // 2. 细找红色轮廓，得到红色矩形
    if (!get_red_contour(input_frame, red_seed, red_rect))
    {
        return false;
    }

    // 3. 获取红色旋转矩形的 4 个角点
    cv::Point2f pts[4];
    red_rect.points(pts);

    struct Edge
    {
        cv::Point2f a;
        cv::Point2f b;
        float len;
        float mid_y;
    };

    Edge edges[4];

    // 4. 计算 4 条边
    for (int i = 0; i < 4; i++)
    {
        edges[i].a = pts[i];
        edges[i].b = pts[(i + 1) % 4];
        edges[i].len = cv::norm(edges[i].b - edges[i].a);
        edges[i].mid_y = (edges[i].a.y + edges[i].b.y) * 0.5f;
    }

    // 5. 找最长边长度
    float max_len = 0.0f;

    for (int i = 0; i < 4; i++)
    {
        if (edges[i].len > max_len)
        {
            max_len = edges[i].len;
        }
    }

    if (max_len < 5.0f)
    {
        return false;
    }

    // 6. 找靠下的长边
    Edge bottom_long_edge;
    bool found_bottom_edge = false;

    for (int i = 0; i < 4; i++)
    {
        if (edges[i].len > max_len * 0.75f)
        {
            if (!found_bottom_edge || edges[i].mid_y > bottom_long_edge.mid_y)
            {
                bottom_long_edge = edges[i];
                found_bottom_edge = true;
            }
        }
    }

    if (!found_bottom_edge)
    {
        return false;
    }

    // 7. 长底边方向向量
    cv::Point2f v = bottom_long_edge.b - bottom_long_edge.a;
    float v_len = cv::norm(v);

    if (v_len < 1e-6f)
    {
        return false;
    }

    v.x /= v_len;
    v.y /= v_len;

    // 8. 求长底边的两个法向量
    cv::Point2f n1(-v.y, v.x);
    cv::Point2f n2(v.y, -v.x);

    // 图像坐标中 y 越小越靠上，所以选择 y 分量更小的法向量
    cv::Point2f n_up = (n1.y < n2.y) ? n1 : n2;

    // 9. 红块底边中点
    cv::Point2f bottom_mid =
        (bottom_long_edge.a + bottom_long_edge.b) * 0.5f;

    // 框宽度缩放系数
// 小于 1：框变窄
// 大于 1：框变宽
    const float ROI_W_SCALE = 0.88f;

// 框高度缩放系数
// 小于 1：框变矮
// 大于 1：框变高
    const float ROI_H_SCALE = 0.88f;

// 中心位置补偿
// 因为 n_up 是向上的方向：
// 正数：目标框整体向上移动
// 负数：目标框整体向下移动
    const float ROI_CENTER_UP_BIAS = -6.0f;

    // 10. 根据实际比例计算目标图片区域大小
    float target_w = max_len * ROI_W_SCALE;
    float target_h = max_len * TARGET_H_RATIO * ROI_H_SCALE;

    // 红块高度，理论上是红块长边的 5/12
    float red_h = max_len * RED_H_RATIO;

    // 11. 从红块底边向上推到目标图片中心
    float center_offset = red_h + target_h * 0.5f + ROI_CENTER_UP_BIAS;

    cv::Point2f target_center = bottom_mid + n_up * center_offset;

    // 12. 目标区域角度与红块长底边一致
    float angle = std::atan2(v.y, v.x) * 180.0f / static_cast<float>(CV_PI);

    rotated_rect = cv::RotatedRect(
        target_center,
        cv::Size2f(target_w, target_h),
        angle
    );

    return true;
}

//红条原图四点 在原图中找靠下长边 把红条底边映射到 BEV 在 BEV 中按照红块 12×5、目标图 12×12 的比例生成目标框
//参数 输入图像 红块的四个角点 输出BEV图像的图片的四个角点 ->通过计算BEV图像的中心坐标 加减宽度得来的
bool BuildTargetBevQuadFromRedRect(const cv::Mat& input_frame,
                                   const cv::Point2f red_img_pts[4],
                                   cv::Point2f target_bev_pts[4])
{
    if (input_frame.empty() ||
        red_img_pts == nullptr ||
        target_bev_pts == nullptr)
    {
        return false;
    }

    struct Edge
    {
        cv::Point2f a;
        cv::Point2f b;
        float len;
        float mid_y;
    };

    Edge edges[4];

    for (int i = 0; i < 4; i++)
    {
        edges[i].a = red_img_pts[i];
        edges[i].b = red_img_pts[(i + 1) % 4];

        edges[i].len = cv::norm(edges[i].b - edges[i].a);
        edges[i].mid_y = (edges[i].a.y + edges[i].b.y) * 0.5f;
    }

    // 1. 找红条最长边
    float max_len_img = 0.0f;

    for (int i = 0; i < 4; i++)
    {
        if (edges[i].len > max_len_img)
        {
            max_len_img = edges[i].len;
        }
    }

    if (max_len_img < 5.0f)
    {
        return false;
    }

    // 2. 选择图像中更靠下的长边作为红条底边
    Edge bottom_long_edge;
    bool found_bottom_edge = false;

    for (int i = 0; i < 4; i++)
    {
        if (edges[i].len > max_len_img * 0.75f)
        {
            if (!found_bottom_edge || edges[i].mid_y > bottom_long_edge.mid_y)
            {
                bottom_long_edge = edges[i];
                found_bottom_edge = true;
            }
        }
    }

    if (!found_bottom_edge)
    {
        return false;
    }

    // 3. 原图中的红条底边方向
    cv::Point2f v_img = bottom_long_edge.b - bottom_long_edge.a;
    float v_img_len = cv::norm(v_img);

    if (v_img_len < 1e-6f)
    {
        return false;
    }

    v_img.x /= v_img_len;
    v_img.y /= v_img_len;

    // 统一方向，尽量让底边方向从左到右
    if (v_img.x < 0.0f)
    {
        v_img.x = -v_img.x;
        v_img.y = -v_img.y;
        std::swap(bottom_long_edge.a, bottom_long_edge.b);
    }

    // 4. 原图中的向上法向
    cv::Point2f n1_img(-v_img.y, v_img.x);
    cv::Point2f n2_img(v_img.y, -v_img.x);

    // 图像坐标 y 越小越靠上
    cv::Point2f n_up_img = (n1_img.y < n2_img.y) ? n1_img : n2_img;

    cv::Point2f bottom_mid_img =
        (bottom_long_edge.a + bottom_long_edge.b) * 0.5f;

    // 5. 把底边两个端点和底边中点变换到 BEV
    cv::Point2f a_bev = ImagePointToBevPoint(bottom_long_edge.a, input_frame);
    cv::Point2f b_bev = ImagePointToBevPoint(bottom_long_edge.b, input_frame);
    cv::Point2f bottom_mid_bev = ImagePointToBevPoint(bottom_mid_img, input_frame);

    if (!IsFinitePoint(a_bev) ||
        !IsFinitePoint(b_bev) ||
        !IsFinitePoint(bottom_mid_bev))
    {
        return false;
    }

    // 6. BEV 中的红条宽度方向
    cv::Point2f v_bev = b_bev - a_bev;
    float anchor_w_bev = cv::norm(v_bev);//红块的长边变换到BEV后的长度

    if (anchor_w_bev < 1e-6f)
    {
        return false;
    }

    v_bev.x /= anchor_w_bev;
    v_bev.y /= anchor_w_bev;

    // 7. 用“原图向上一小段”映射到 BEV，确定 BEV 中哪边是上方
    cv::Point2f test_up_img = bottom_mid_img + n_up_img * 20.0f;
    cv::Point2f test_up_bev = ImagePointToBevPoint(test_up_img, input_frame);

    if (!IsFinitePoint(test_up_bev))
    {
        return false;
    }

    cv::Point2f n_ref_bev = test_up_bev - bottom_mid_bev;//计算测试边 - 底边的向量

    float n_ref_len = cv::norm(n_ref_bev);//计算其长度
    if (n_ref_len < 1e-6f)
    {
        return false;
    }

    n_ref_bev.x /= n_ref_len;
    n_ref_bev.y /= n_ref_len;

    // 8. 在 BEV 中构造与 v_bev 垂直的法向，并选择接近 n_ref_bev 的方向
    cv::Point2f n_bev_1(-v_bev.y, v_bev.x);
    cv::Point2f n_bev_2(v_bev.y, -v_bev.x);

    float dot1 = n_bev_1.x * n_ref_bev.x + n_bev_1.y * n_ref_bev.y;
    float dot2 = n_bev_2.x * n_ref_bev.x + n_bev_2.y * n_ref_bev.y;

    cv::Point2f n_up_bev = (dot1 > dot2) ? n_bev_1 : n_bev_2;

    // 9. 可调参数
    const float TARGET_H_RATIO = 12.0f / 12.0f;
    const float RED_H_RATIO    = 5.0f  / 12.0f;

    // 框大小缩放
    const float ROI_W_SCALE = 1.2f;
    const float ROI_H_SCALE = 1.2f;

    // 角度补偿，单位：度
    // 先用 0，后面如果图传上角度偏了，再试 2 或 -2
    const float ROI_ANGLE_BIAS_DEG = 0.0f;

    // 中心位置补偿，注意这里不是像素，是相对于红条宽度的比例
    // 负数：目标框向红条方向靠近
    // 正数：目标框远离红条
    // const float ROI_CENTER_UP_BIAS_RATIO = -0.05f;
    const float ROI_CENTER_UP_BIAS_RATIO = 0.1f;
    // 10. 角度补偿
    if (std::fabs(ROI_ANGLE_BIAS_DEG) > 1e-6f)
    {
        v_bev = RotateVector2D(v_bev, ROI_ANGLE_BIAS_DEG);

        float v_len = cv::norm(v_bev);
        if (v_len < 1e-6f)
        {
            return false;
        }

        v_bev.x /= v_len;
        v_bev.y /= v_len;

        // 重新生成法向，并保持方向和原来的 n_up_bev 大致一致
        cv::Point2f nn1(-v_bev.y, v_bev.x);
        cv::Point2f nn2(v_bev.y, -v_bev.x);

        float d1 = nn1.x * n_up_bev.x + nn1.y * n_up_bev.y;
        float d2 = nn2.x * n_up_bev.x + nn2.y * n_up_bev.y;

        n_up_bev = (d1 > d2) ? nn1 : nn2;
    }

    // 11. BEV 中目标框宽高
    float target_w_bev = anchor_w_bev * ROI_W_SCALE;//BEV中目标框的宽
    float target_h_bev = anchor_w_bev * TARGET_H_RATIO * ROI_H_SCALE;//BEV目标框的高

    float red_h_bev = anchor_w_bev * RED_H_RATIO;//红块的宽度

    float center_offset_bev =
        red_h_bev +
        target_h_bev * 0.5f +
        anchor_w_bev * ROI_CENTER_UP_BIAS_RATIO;//计算BEV中图片的中心的高度

    cv::Point2f target_center_bev =
        bottom_mid_bev + n_up_bev * center_offset_bev;//计算BEV图像中图片中心的坐标

    //在弯道进行左右补偿 通过红色定位条底边中点的 x 坐标 和 图像中心的 x 坐标 作比较 确定靠左还是靠右
    float side_norm =
    (bottom_mid_img.x - input_frame.cols * 0.5f) /
    (input_frame.cols * 0.5f);

    if (side_norm > 1.0f) side_norm = 1.0f;
    if (side_norm < -1.0f) side_norm = -1.0f;

    // 左侧 side_norm < 0，往右修
    // 右侧 side_norm > 0，往左修
    const float SIDE_COMP_GAIN = 0.5f;

    // 越靠边，越往上修一点
    const float SIDE_UP_GAIN = 0.06f;

    target_center_bev += v_bev * (-side_norm * anchor_w_bev * SIDE_COMP_GAIN);
    target_center_bev += n_ref_bev * (std::fabs(side_norm) * anchor_w_bev * SIDE_UP_GAIN);

    cv::Point2f half_w = v_bev * (target_w_bev * 0.5f);
    cv::Point2f half_h = n_up_bev * (target_h_bev * 0.5f);

    // 顺序：左上、右上、右下、左下
    target_bev_pts[0] = target_center_bev - half_w + half_h;//左上角点
    target_bev_pts[1] = target_center_bev + half_w + half_h;//右上角点
    target_bev_pts[2] = target_center_bev + half_w - half_h;
    target_bev_pts[3] = target_center_bev - half_w - half_h;

    return true;
}

//通过固定矩阵映射
//输入目标区域BEV图像的四个角点的坐标 输出透视图像 ROI区域 (图片区域)
//valid_ratio 
bool BuildTargetRoiByFixedIpmRemap(const cv::Mat& input_frame,
                                   const cv::Point2f target_bev_pts[4],
                                   cv::Mat& output_roi,
                                   float* valid_ratio = nullptr)
{
    output_roi.release();

    if (input_frame.empty() || target_bev_pts == nullptr)
    {
        return false;
    }

    const int TARGET_ROI_SIZE = 96;//
    const float MIN_VALID_RATIO = 0.60f;//

    static cv::Mat map_x;
    static cv::Mat map_y;

    map_x.create(TARGET_ROI_SIZE, TARGET_ROI_SIZE, CV_32FC1);
    map_y.create(TARGET_ROI_SIZE, TARGET_ROI_SIZE, CV_32FC1);

    int valid_count = 0;

    const float inv_size = 1.0f / static_cast<float>(TARGET_ROI_SIZE);

    for (int y = 0; y < TARGET_ROI_SIZE; y++)
    {
        float* map_x_row = map_x.ptr<float>(y);
        float* map_y_row = map_y.ptr<float>(y);

        const float v = (static_cast<float>(y) + 0.5f) * inv_size;

        for (int x = 0; x < TARGET_ROI_SIZE; x++)
        {
            const float u = (static_cast<float>(x) + 0.5f) * inv_size;

            // 1. ROI 中的点映射到 BEV 目标四边形中
            cv::Point2f top =
                target_bev_pts[0] * (1.0f - u) +
                target_bev_pts[1] * u;

            cv::Point2f bottom =
                target_bev_pts[3] * (1.0f - u) +
                target_bev_pts[2] * u;

            cv::Point2f bev_point =
                top * (1.0f - v) +
                bottom * v;

            // 2. BEV 点反变换回当前图像坐标
            cv::Point2f image_point = BevPointToImagePoint(bev_point,
                                                           input_frame);

            // 3. 写入 remap 表
            if (IsFinitePoint(image_point) &&
                image_point.x >= 0.0f &&
                image_point.x <= static_cast<float>(input_frame.cols - 1) &&
                image_point.y >= 0.0f &&
                image_point.y <= static_cast<float>(input_frame.rows - 1))
            {
                map_x_row[x] = image_point.x;
                map_y_row[x] = image_point.y;
                valid_count++;
            }
            else
            {
                map_x_row[x] = -1.0f;
                map_y_row[x] = -1.0f;
            }
        }
    }

    const float ratio =
        static_cast<float>(valid_count) /
        static_cast<float>(TARGET_ROI_SIZE * TARGET_ROI_SIZE);

    if (valid_ratio != nullptr)
    {
        *valid_ratio = ratio;
    }

    if (ratio < MIN_VALID_RATIO)
    {
        return false;
    }

    cv::Mat sampled;

    cv::remap(input_frame,
              sampled,
              map_x,
              map_y,
              cv::INTER_LINEAR,
              cv::BORDER_CONSTANT,
              cv::Scalar(128, 128, 128, 128));

    if (sampled.empty())
    {
        return false;
    }

    // 统一输出为 BGR 三通道
    if (sampled.channels() == 3)
    {
        output_roi = sampled.clone();
    }
    else if (sampled.channels() == 1)
    {
        cv::cvtColor(sampled, output_roi, cv::COLOR_GRAY2BGR);
    }
    else if (sampled.channels() == 2)
    {
        cv::cvtColor(sampled, output_roi, cv::COLOR_BGR5652BGR);
    }
    else if (sampled.channels() == 4)
    {
        cv::cvtColor(sampled, output_roi, cv::COLOR_BGRA2BGR);
    }
    else
    {
        return false;
    }

    return !output_roi.empty();
}


void red_init(){
    red_L_L.row = 0;//红块左下
    red_L_L.column = 0;
    red_R_L.row = 0;//红块右下
    red_R_L.column =0;

    picture_L_H.row = 0;//图片左上
    picture_L_H.column = 0;
    picture_R_H.row = 0;//图片右上
    picture_R_H.column = 0;
}

//points[0] —— 左上角（Top-Left）
//points[1] —— 右上角（Top-Right）
//points[2] —— 右下角（Bottom-Right）
//points[3] —— 左下角（Bottom-Left）

//红块+图片部分角点映射 与 排序 
//参数1 红块的四个角点 参2 图片部分四个角点
//逻辑 画出由这四个点组成图像的最小外接矩形
void map_and_sort(cv::Point2f pts[4],cv::Point2f pts_picture[4]){
    red_init();//清理四个拐点

    cv::Point2f pts_remove[4];
    pts_remove[0] = cv::Point2f(pts_picture[0].x/3.4,pts_picture[0].y/4);//左上
    pts_remove[1] = cv::Point2f(pts_picture[1].x/3.4,pts_picture[1].y/4);//右上
    pts_remove[2] = cv::Point2f(pts_picture[2].x/3.4,pts_picture[2].y/4);//右下
    pts_remove[3] = cv::Point2f(pts_picture[3].x/3.4,pts_picture[3].y/4);//左下

    // 将 pts_remove 转成 vector
    std::vector<cv::Point2f> points;
    for (int i = 0; i < 4; ++i)
    {
        points.emplace_back(pts_remove[i]);
    }

    // 计算正外接矩形（轴对齐）
    cv::Rect rect = cv::boundingRect(points);

    // 返回正框的四个角点（float）
    cv::Point2f rect_pts[4];

    rect_pts[0] = cv::Point2f(rect.x, rect.y);                 // 左上
    rect_pts[1] = cv::Point2f(rect.x + rect.width, rect.y);    // 右上
    rect_pts[2] = cv::Point2f(rect.x + rect.width,
                           rect.y + rect.height);               // 右下
    rect_pts[3] = cv::Point2f(rect.x, rect.y + rect.height);   // 左下

    for(int i = 0; i < 4; i++)
    {
    int y = static_cast<int>(rect_pts[i].y);
      if(rect_pts[i].x > LCDW_1||rect_pts[i].y > LCDH_1||rect_pts[i].x <= 0||rect_pts[i].y <= 0)  return;

      if(rect_pts[i].x > Right_Sideline[y] || rect_pts[i].x < Left_Sideline[y]) return;
    }


    picture_L_H.column = static_cast<uint8_t>(pts_remove[0].x);// 左上
    picture_L_H.row = static_cast<uint8_t>(pts_remove[0].y);

    picture_R_H.column = static_cast<uint8_t>(pts_remove[1].x)+2;// 右上
    picture_R_H.row = static_cast<uint8_t>(pts_remove[1].y)+2;

    red_R_L.column = static_cast<uint8_t>(pts_remove[2].x);// 右下
    red_R_L.row = static_cast<uint8_t>(pts_remove[2].y);

    red_L_L.column = static_cast<uint8_t>(pts_remove[3].x)-2;// 左下
    red_L_L.row = static_cast<uint8_t>(pts_remove[3].y)-2;

}


// 把原图 320x240 上的四边形点映射到 94x60 二值图上
static bool SrcQuadToSmallQuad(const cv::Point2f src_quad[4],
                               const cv::Size& src_size,
                               cv::Point small_quad[4])
{
    if (src_quad == nullptr || src_size.width <= 1 || src_size.height <= 1)
    {
        return false;
    }

    for (int i = 0; i < 4; i++)
    {
        if (!IsFinitePoint(src_quad[i]))
        {
            return false;
        }

        int x = static_cast<int>(std::lround(
            src_quad[i].x * static_cast<float>(LCDW_1 - 1) /
            static_cast<float>(src_size.width - 1)));

        int y = static_cast<int>(std::lround(
            src_quad[i].y * static_cast<float>(LCDH_1 - 1) /
            static_cast<float>(src_size.height - 1)));

        x = ClampTargetInt(x, 0, LCDW_1 - 1);
        y = ClampTargetInt(y, 0, LCDH_1 - 1);

        small_quad[i] = cv::Point(x, y);
    }

    return true;
}

//bin_img和 image共享同一块内存
// 修改 bin_img，等于直接修改 image
// 在 94x60 二值图中抹掉一个四边形区域
static void EraseQuadOnBinary(uint8_t image[LCDH_1][LCDW_1],
                              const cv::Point2f src_quad[4],
                              const cv::Size& src_size,
                              int pad)
{
    cv::Point small_quad[4];

    if (!SrcQuadToSmallQuad(src_quad, src_size, small_quad))
    {
        return;
    }

    // Image_Use 本身是连续内存，可以直接包装成 Mat
    cv::Mat bin_img(LCDH_1, LCDW_1, CV_8UC1, image);

    static cv::Mat mask;
    mask.create(LCDH_1, LCDW_1, CV_8UC1);
    mask.setTo(cv::Scalar(0));

    cv::fillConvexPoly(mask,
                       small_quad,
                       4,
                       cv::Scalar(255),//白色
                       cv::LINE_8);//填充四边形

    // 稍微扩大一点，避免边缘残留
    if (pad > 0)
    {
        cv::dilate(mask,
                   mask,
                   cv::Mat(),
                   cv::Point(-1, -1),
                   pad);
    }

    // 这里设置为 white，表示把图片/红块区域当成赛道白色区域处理
    bin_img.setTo(cv::Scalar(white), mask);
}

//bin_img和 image共享同一块内存
// 修改 bin_img，等于直接修改 image
// 在 94x60 二值图中抹掉一个四边形区域
struct EraseQuadCache
{
    cv::Point2f red[4];
    cv::Point2f target[4];
    bool has_last = false;
    int lost_cnt = 0;
};

static EraseQuadCache g_erase_cache;


//判断是否为有效四边形
static bool IsQuadValidForErase(const cv::Point2f quad[4],
                                const cv::Size& img_size)
{
    if (quad == nullptr || img_size.width <= 0 || img_size.height <= 0)
    {
        return false;
    }

    std::vector<cv::Point2f> q;
    q.reserve(4);

    for (int i = 0; i < 4; i++)
    {
        if (!IsFinitePoint(quad[i]))//安全性检查 判断 转化到BEV上的点有没有越界
        {
            return false;
        }

        q.push_back(quad[i]);
    }

    float area = std::fabs(static_cast<float>(cv::contourArea(q)));

    // 原图坐标下，面积太小说明点异常
    if (area < 20.0f)
    {
        return false;
    }

    cv::Rect2f box = cv::boundingRect(q);

    // 四边形完全跑出图像外，不使用
    if (box.x > img_size.width - 1 ||
        box.y > img_size.height - 1 ||
        box.x + box.width < 0 ||
        box.y + box.height < 0)
    {
        return false;
    }

    return true;
}

static bool AreEraseQuadsValid(const cv::Point2f red_quad[4],
                               const cv::Point2f target_quad[4],
                               const cv::Size& img_size)
{
    return IsQuadValidForErase(red_quad, img_size) &&
           IsQuadValidForErase(target_quad, img_size);//只有当红色矩形块和图像区域都有效时才返回真
}

static void SaveEraseCache(const cv::Point2f red_quad[4],
                           const cv::Point2f target_quad[4])
{
    for (int i = 0; i < 4; i++)
    {
        g_erase_cache.red[i] = red_quad[i];
        g_erase_cache.target[i] = target_quad[i];
    }

    g_erase_cache.has_last = true;
    g_erase_cache.lost_cnt = 0;
}

// 同时抹掉红色矩形块和上方图片区域
static void EraseTargetAndRedOnBinary(uint8_t image[LCDH_1][LCDW_1],
                                      const cv::Point2f red_pts[4],
                                      const cv::Point2f target_pts[4],
                                      const cv::Size& src_size)
{
    // 红色矩形块
    EraseQuadOnBinary(image, red_pts, src_size, 1);

    // 上方图片区域
    EraseQuadOnBinary(image, target_pts, src_size, 1);
}
//初始化代码
static inline void ClearGuaidianPoint(struct Guaidian& p)
{
    p.row = 0;
    p.column = 0;
    p.flag = 0;
}
//辅助函数 用于将320*240的图像转化到94*60
static inline void ImagePointToSmallGuaidian(const cv::Point2f& img_pt,
                                             const cv::Mat& input_frame,
                                             struct Guaidian& out_pt)
{
    const float src_to_small_x = 94.0f / static_cast<float>(input_frame.cols);
    const float src_to_small_y = 60.0f / static_cast<float>(input_frame.rows);

    int small_x = static_cast<int>(std::lround(img_pt.x * src_to_small_x));
    int small_y = static_cast<int>(std::lround(img_pt.y * src_to_small_y));

    small_x = ClampTargetInt(small_x, 0, 93);
    small_y = ClampTargetInt(small_y, 0, 59);

    out_pt.column = static_cast<uint8>(small_x);
    out_pt.row = static_cast<uint8>(small_y);
    out_pt.flag = 1;
}

//对图片区域做防抖处理
// 小于这个像素差，认为只是抖动，不更新图片框
static const float TARGET_QUAD_STABLE_PX = 3.0f;

// 防抖函数：对图片区域 target_bev_pts 做稳定处理
static void StabilizeTargetQuad(const cv::Mat& input_frame,
                                cv::Point2f target_bev_pts[4])
{
    static bool has_last = false;
    static cv::Point2f last_bev_pts[4];
    static cv::Point2f last_img_pts[4];

    cv::Point2f now_img_pts[4];

    // 当前 BEV 四点转回原图坐标，用于和上一帧比较
    for (int i = 0; i < 4; i++)
    {
        now_img_pts[i] = BevPointToImagePoint(target_bev_pts[i], input_frame);
    }

    OrderTargetStripQuad(now_img_pts);

    // 第一帧直接记录
    if (!has_last)
    {
        for (int i = 0; i < 4; i++)
        {
            last_bev_pts[i] = target_bev_pts[i];
            last_img_pts[i] = now_img_pts[i];
        }

        has_last = true;
        return;
    }

    // 计算当前帧和上一帧四个角点的平均距离
    float dist_sum = 0.0f;

    for (int i = 0; i < 4; i++)
    {
        float dx = now_img_pts[i].x - last_img_pts[i].x;
        float dy = now_img_pts[i].y - last_img_pts[i].y;
        dist_sum += std::sqrt(dx * dx + dy * dy);
    }

    float avg_dist = dist_sum / 4.0f;

    // 如果差距很小，认为是抖动，直接使用上一帧
    if (avg_dist < TARGET_QUAD_STABLE_PX)
    {
        for (int i = 0; i < 4; i++)
        {
            target_bev_pts[i] = last_bev_pts[i];
        }

        return;
    }

    // 如果差距明显，认为是真实移动，更新上一帧
    for (int i = 0; i < 4; i++)
    {
        last_bev_pts[i] = target_bev_pts[i];
        last_img_pts[i] = now_img_pts[i];
    }
}


//target_roi 输出的目标图片区域
//修改：新增参数 作为灰度图抹除红块的断点标记 防止因为返回false 而导致整帧不抹
//修改返回最小外接矩形的四个点 以及图片框的上面两个点
//red_rect_pts 红块四个角点
//红块中心点
//修改:返回整个图片加红块区域的最小外接矩形的四个角点
bool FindTargetRoiByFixedIpm(const cv::Mat& input_frame,
                             cv::Mat& target_roi,
                             struct Guaidian* red_rect_pts,      // 红块4角点，94×60
                             struct Guaidian* red_center_pt,     // 红块中心点，94×60
                             struct Guaidian* target_top_pts,    // 图片框上边2点，94×60
                             struct Guaidian*  whole_rect_pts,
                             cv::Point2f* target_pts,            // 保留：图片框4点，原图坐标
                             cv::Point2f* red_pts,               // 保留：红块4点，原图坐标
                             float* debug_valid_ratio,
                             bool* erase_pts_ready)    //新增：整个图像最小外接矩形的四个角点
{
    // 1. 初始化输出
    if (red_rect_pts != nullptr)
    {
        for (int i = 0; i < 4; i++)
        {
            ClearGuaidianPoint(red_rect_pts[i]);//初始化四点
        }
    }

    if (red_center_pt != nullptr)
    {
        ClearGuaidianPoint(*red_center_pt);
    }

      if (whole_rect_pts != nullptr)
    {
        for (int i = 0; i < 4; i++)
        {
            ClearGuaidianPoint(whole_rect_pts[i]);//初始化四点
        }
    }

    if (target_top_pts != nullptr)
    {
        for (int i = 0; i < 2; i++)
        {
            ClearGuaidianPoint(target_top_pts[i]);
        }
    }

    if (target_pts != nullptr)
    {
        for (int i = 0; i < 4; i++)
        {
            target_pts[i] = cv::Point2f(0.0f, 0.0f);
        }
    }

    if (red_pts != nullptr)
    {
        for (int i = 0; i < 4; i++)
        {
            red_pts[i] = cv::Point2f(0.0f, 0.0f);
        }
    }

    if (erase_pts_ready != nullptr)
    {
        *erase_pts_ready = false;
    }

    Flag.Redblock = 0;
    target_roi.release();

    if (input_frame.empty())
    {
        return false;
    }

    cv::Point red_seed;
    cv::RotatedRect red_rect;

    // 2. 粗找红色种子点
    if (!red_detect_rgb(input_frame, red_seed))
    {
        return false;
    }

    Flag.Redblock = 1;

    // 3. 细找底部红块 / 红色定位条
    if (!get_red_contour(input_frame, red_seed, red_rect))
    {
        return false;
    }

    // 4. 获取底部红块最小外接矩形四点
    cv::Point2f red_img_pts[4];
    red_rect.points(red_img_pts);

    // 排序后：
    // red_img_pts[0] 左上
    // red_img_pts[1] 右上
    // red_img_pts[2] 右下
    // red_img_pts[3] 左下
    OrderTargetStripQuad(red_img_pts);

    // 5. 保留 red_pts：原图坐标下的红块四点
    if (red_pts != nullptr)
    {
        for (int i = 0; i < 4; i++)
        {
            red_pts[i] = red_img_pts[i];
        }
    }

    // 6. 返回红块四个角点，映射到 94×60
    if (red_rect_pts != nullptr)
    {
        for (int i = 0; i < 4; i++)
        {
            ImagePointToSmallGuaidian(red_img_pts[i],
                                      input_frame,
                                      red_rect_pts[i]);
        }
    }

    // 7. 计算底部红块中心点
    cv::Point2f red_center_img(0.0f, 0.0f);

    for (int i = 0; i < 4; i++)
    {
        red_center_img += red_img_pts[i];
    }

    red_center_img *= 0.25f;

    // 8. 红块中心点映射到 94×60，并做防抖
    if (red_center_pt != nullptr)
    {
        struct Guaidian now_center;
        ClearGuaidianPoint(now_center);

        ImagePointToSmallGuaidian(red_center_img,
                                  input_frame,
                                  now_center);

        static bool has_last_center = false;
        static int last_center_x = 0;
        static int last_center_y = 0;

        const int RED_CENTER_DEAD_ZONE = 1;

        int now_x = static_cast<int>(now_center.column);
        int now_y = static_cast<int>(now_center.row);

        if (!has_last_center)
        {
            last_center_x = now_x;
            last_center_y = now_y;
            has_last_center = true;
        }
        else
        {
            int dx = now_x - last_center_x;
            int dy = now_y - last_center_y;

            int dist2 = dx * dx + dy * dy;
            int dead_zone2 = RED_CENTER_DEAD_ZONE * RED_CENTER_DEAD_ZONE;

            if (dist2 > dead_zone2)
            {
                last_center_x = now_x;
                last_center_y = now_y;
            }
        }

        red_center_pt->column =
            static_cast<uint8>(ClampTargetInt(last_center_x, 0, 93));
        red_center_pt->row =
            static_cast<uint8>(ClampTargetInt(last_center_y, 0, 59));
        red_center_pt->flag = 1;
    }

    // 9. 根据红块四点，在 BEV 中生成图片框四点
    cv::Point2f target_bev_pts[4];

    if (!BuildTargetBevQuadFromRedRect(input_frame,
                                       red_img_pts,
                                       target_bev_pts))
    {
        return false;
    }

    //新增图片区域防抖
    StabilizeTargetQuad(input_frame,target_bev_pts);

    // 10. 将 BEV 图片框四点反变换回原图
    cv::Point2f target_img_pts[4];

    for (int i = 0; i < 4; i++)
    {
        target_img_pts[i] = BevPointToImagePoint(target_bev_pts[i],
                                                 input_frame);
    }

    // 排序后：
    // target_img_pts[0] 左上
    // target_img_pts[1] 右上
    // target_img_pts[2] 右下
    // target_img_pts[3] 左下
    OrderTargetStripQuad(target_img_pts);

    if (whole_rect_pts != nullptr)
{
    std::vector<cv::Point2f> whole_points;//定义轮廓
    whole_points.reserve(8);

    // 加入红色矩形块4点
    for (int i = 0; i < 4; i++)
    {
        if (!IsFinitePoint(red_img_pts[i]))//安全性检查 判断 转化到BEV上的点有没有越界
        {
            return false;
        }

        whole_points.push_back(red_img_pts[i]);
    }

    // 加入图片区域4点
    for (int i = 0; i < 4; i++)
    {
        if (!IsFinitePoint(target_img_pts[i]))
        {
            return false;
        }

        whole_points.push_back(target_img_pts[i]);
    }

    // 计算同时包住图片区域和红块区域的最小旋转矩形
    cv::RotatedRect whole_rect = cv::minAreaRect(whole_points);

    cv::Point2f whole_img_pts[4];
    whole_rect.points(whole_img_pts);

    // 排序成：左上、右上、右下、左下
    OrderTargetStripQuad(whole_img_pts);

    // 映射到 94×60
    for (int i = 0; i < 4; i++)
        {
        ImagePointToSmallGuaidian(whole_img_pts[i],
                                  input_frame,
                                  whole_rect_pts[i]);
        }
    }

    // 11. 保留 target_pts：原图坐标下的图片框四点
    if (target_pts != nullptr)
    {
        for (int i = 0; i < 4; i++)
        {
            target_pts[i] = target_img_pts[i];
        }
    }

    // 12. 返回图片框上边两个点，映射到 94×60
    if (target_top_pts != nullptr)
    {
        ImagePointToSmallGuaidian(target_img_pts[0],
                                  input_frame,
                                  target_top_pts[0]);

        ImagePointToSmallGuaidian(target_img_pts[1],
                                  input_frame,
                                  target_top_pts[1]);
    }

    // 13. 运行到这里说明：
    // red_pts、target_pts、红块小图点、图片框上边小图点都已经准备好
    // 可以用于灰度图抹除
    if (red_rect_pts != nullptr &&
        red_center_pt != nullptr &&
        target_top_pts != nullptr &&
        target_pts != nullptr &&
        red_pts != nullptr &&
        erase_pts_ready != nullptr)
    {
        *erase_pts_ready = true;
    }

    // 14. 使用固定 IPM 反变换 + remap 裁剪固定大小 ROI
    // 注意：即使这里失败，erase_pts_ready 也已经可能为 true
    if (!BuildTargetRoiByFixedIpmRemap(input_frame,
                                       target_bev_pts,
                                       target_roi,
                                       debug_valid_ratio))
    {
        return false;
    }

    return true;
}

 std::vector<cv::Point2f> orderPoints(cv::Point2f pts[4])
{
    std::vector<cv::Point2f> p(pts, pts + 4);

    // 先按y排序
    std::sort(p.begin(), p.end(),
    [](const cv::Point2f& a, const cv::Point2f& b)
    {
        return a.y < b.y;
    });

    std::vector<cv::Point2f> top = {p[0], p[1]};
    std::vector<cv::Point2f> bottom = {p[2], p[3]};

    // 上面两个按x排序
    std::sort(top.begin(), top.end(),
    [](const cv::Point2f& a, const cv::Point2f& b)
    {
        return a.x < b.x;
    });

    // 下面两个按x排序
    std::sort(bottom.begin(), bottom.end(),
    [](const cv::Point2f& a, const cv::Point2f& b)
    {
        return a.x < b.x;
    });

    return {
        top[0],      // 左上
        top[1],      // 右上
        bottom[1],   // 右下
        bottom[0]    // 左下
    };
}

// 截图函数 用于拍摄数据集
// 参数1：src       拍摄区域，比如 target_roi 或 lq_frame
// 参数2：count     需要保存的总张数
// 参数3：name      命名方式/类别名，比如 "supplies"、"vehicle"、"weapon"
// 参数4：save_path 保存路径，比如 "./dataset/supplies"
void snapshot(const cv::Mat& src,
              int count,
              const std::string& name,
              const std::string& save_path)
{
    if (src.empty())
    {
        return;
    }

    if (count <= 0)
    {
        return;
    }

    if (name.empty() || save_path.empty())
    {
        return;
    }

    static int saved_count = 0;
    static std::string last_name = "";
    static std::string last_path = "";

    // 如果换了类别名或保存路径，就重新计数
    if (last_name != name || last_path != save_path)
    {
        saved_count = 0;
        last_name = name;
        last_path = save_path;
    }

    // 已经保存够了，就不再保存
    if (saved_count >= count)
    {
        return;
    }

    cv::Mat save_img;

    // 防止 src 是摄像头缓冲区或 ROI 引用，先 clone
    if (src.channels() == 3)
    {
        save_img = src.clone();
    }
    else if (src.channels() == 1)
    {
        cv::cvtColor(src, save_img, cv::COLOR_GRAY2BGR);
    }
    else if (src.channels() == 4)
    {
        cv::cvtColor(src, save_img, cv::COLOR_BGRA2BGR);
    }
    else
    {
        printf("snapshot: 不支持的图像通道数: %d\n", src.channels());
        return;
    }

    char file_path[256];

    // 判断路径末尾有没有 /
    if (save_path.back() == '/')
    {
        snprintf(file_path,
                 sizeof(file_path),
                 "%s%s_%06d.jpg",
                 save_path.c_str(),
                 name.c_str(),
                 saved_count + 1);
    }
    else
    {
        snprintf(file_path,
                 sizeof(file_path),
                 "%s/%s_%06d.jpg",
                 save_path.c_str(),
                 name.c_str(),
                 saved_count + 1);
    }

    if (cv::imwrite(file_path, save_img))
    {
        saved_count++;
        system_delay_ms(100);
        printf("保存图片成功: %s  当前数量: %d / %d\n",
               file_path,
               saved_count,
               count);
    }
    else
    {
        printf("保存图片失败: %s\n", file_path);
    }
}
//为加寻找位置最靠下的红块
//只在赛道内部找红块 
//contour 没必要
bool findRedrect(cv::Mat &src, cv::RotatedRect &rect)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    if(src.empty()) return false;

    int best_id = -1;
    int best_bottom_y = -1;//依据位置进行筛选 选择位置最靠下的
    double max_area = 0;

    cv::Mat hsv, mask1, mask2, mask;
    std::vector<std::vector<cv::Point>> contours;

    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    cv::inRange(hsv, cv::Scalar(0, 120, 70), cv::Scalar(10, 255, 255), mask1);
    cv::inRange(hsv, cv::Scalar(160, 120, 70), cv::Scalar(179, 255, 255), mask2);

    mask = mask1 | mask2;

    // cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
    //     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));//不做卷积处理
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    cv::Point bestCenter;//存储轮廓的质心坐标
    cv::Rect bestRect;//存储最小外接矩形

    int cx,cy = 0;

    if(contours.empty()) return false;

    for(size_t i = 0; i < contours.size(); i++)
    {
         int bottom_y = -1;
        for(const auto &p : contours[i])
        {
            if(p.y > bottom_y)
            bottom_y = p.y;
        }

        if(bottom_y > best_bottom_y)
        {
            best_bottom_y = bottom_y;
            best_id = i;
        }
        
    }
    //cy    cx  Right_Sideline[L_h_guai.row]
    

    if(best_id < 0) return false;

    cv::Rect rect1 = cv::boundingRect(contours[best_id]);
    cx = rect1.x + rect1.width;
    cy = rect1.y + rect1.height;
    if(cx > Right_Sideline[cy] || cx < Left_Sideline[cy]) return false;
    rect = cv::minAreaRect(contours[best_id]);
   // camera_server.update_frame_mat(mask);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_ms = end_time - start_time;
    printf("findRedrect函数耗时: %.2f ms\n",elapsed_ms.count());
    return true;
}


bool findRedrect1(cv::Mat &src, cv::RotatedRect &rect, std::vector<cv::Point> &contour)
{   
    auto start_time = std::chrono::high_resolution_clock::now();
    if(src.empty()) return false;

    cv::Mat hsv, mask1, mask2, mask;

    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    cv::inRange(hsv,
                cv::Scalar(0, 120, 70),
                cv::Scalar(10, 255, 255),
                mask1);

    cv::inRange(hsv,
                cv::Scalar(160, 120, 70),
                cv::Scalar(179, 255, 255),
                mask2);

    mask = mask1 | mask2;

    static cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT,
        cv::Size(3, 3)
    );

    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 1);
    // cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 1);

    const int rows = mask.rows;
    const int cols = mask.cols;

    if(rows < 3 || cols < 3)
        return false;


    cv::Point seed(-1, -1);

    for(int y = rows - 2; y >= 1; y--)
    {
        const uchar *ptr = mask.ptr<uchar>(y);

        for(int x = 1; x < cols - 1; x++)
        {
            if(ptr[x] > 0)
            {
                seed = cv::Point(x, y);
                break;
            }
        }

        if(seed.x >= 0)
            break;
    }

    if(seed.x < 0)
        return false;


    std::vector<cv::Point> stack_points; // 待爬点
    std::vector<cv::Point> pixels;       // 爬到的红色矩形块所有像素

    stack_points.reserve(1024);
    pixels.reserve(1024);

    cv::Mat mask_origin = mask;  // 保存原始 mask，用来判断真实连接
    cv::Mat work = mask.clone();         // work 用来标记访问过的点

    stack_points.push_back(seed);

    work.ptr<uchar>(seed.y)[seed.x] = 0;

    while(!stack_points.empty())
    {
        cv::Point p = stack_points.back();
        stack_points.pop_back();

        pixels.push_back(p);

        for(int dy = -1; dy <= 1; dy++)
        {
            for(int dx = -1; dx <= 1; dx++)
            {
                if(dx == 0 && dy == 0)
                    continue;

                int nx = p.x + dx;
                int ny = p.y + dy;

                if(nx <= 0 || nx >= cols - 1 || ny <= 0 || ny >= rows - 1)
                    continue;

                uchar *work_ptr = work.ptr<uchar>(ny);

                if(work_ptr[nx] == 0)
                    continue;


                if(dx != 0 && dy != 0)
                {
                    const uchar *row_p  = mask_origin.ptr<uchar>(p.y);
                    const uchar *row_ny = mask_origin.ptr<uchar>(ny);

                    bool side1 = row_p[nx] > 0;   // 当前行的左右辅助点
                    bool side2 = row_ny[p.x] > 0; // 目标行的上下辅助点

                    if(!side1 || !side2)
                        continue;
                }


                int white_count = 0;

                for(int yy = -1; yy <= 1; yy++)
                {
                    const uchar *check_ptr = mask_origin.ptr<uchar>(ny + yy);

                    for(int xx = -1; xx <= 1; xx++)
                    {
                        if(check_ptr[nx + xx] > 0)
                            white_count++;
                    }
                }

                if(white_count < 4)
                    continue;

                stack_points.push_back(cv::Point(nx, ny));

                // 标记访问过
                work_ptr[nx] = 0;
            }
        }
    }

    if(pixels.size() < 20)
        return false;

    
    double cx = 0.0;
    double cy = 0.0;

    for(const auto &p : pixels)
    {
        cx += p.x;
        cy += p.y;
    }

    cx /= pixels.size();
    cy /= pixels.size();


    cv::Mat compMask = cv::Mat::zeros(mask.size(), CV_8UC1);

    for(const auto &p : pixels)
    {
        compMask.ptr<uchar>(p.y)[p.x] = 255;
    }

    contour.clear();
    contour.reserve(pixels.size());

    for(const auto &p : pixels)
    {
        bool is_boundary = false;

        for(int dy = -1; dy <= 1; dy++)
        {
            for(int dx = -1; dx <= 1; dx++)
            {
                if(dx == 0 && dy == 0)
                    continue;

                int nx = p.x + dx;
                int ny = p.y + dy;

                if(nx < 0 || nx >= cols || ny < 0 || ny >= rows)
                {
                    is_boundary = true;
                    continue;
                }

                if(compMask.ptr<uchar>(ny)[nx] == 0)
                {
                    is_boundary = true;
                }
            }
        }

        if(is_boundary)
        {
            contour.push_back(p);
        }
    }

    if(contour.size() < 4)
        return false;

    std::sort(contour.begin(), contour.end(),
              [cx, cy](const cv::Point &a, const cv::Point &b)
              {
                  double angle_a = std::atan2(a.y - cy, a.x - cx);
                  double angle_b = std::atan2(b.y - cy, b.x - cx);
                  return angle_a < angle_b;
              });


    double u20 = 0.0;
    double u02 = 0.0;
    double u11 = 0.0;

    for(const auto &p : pixels)
    {
        double x = p.x - cx;
        double y = p.y - cy;

        u20 += x * x;
        u02 += y * y;
        u11 += x * y;
    }

    double angle = 0.5 * std::atan2(2.0 * u11, u20 - u02);

    cv::Point2f dir_x(
        static_cast<float>(std::cos(angle)),
        static_cast<float>(std::sin(angle))
    );

    cv::Point2f dir_y(
        static_cast<float>(-std::sin(angle)),
        static_cast<float>( std::cos(angle))
    );

    double min_x =  1e9;
    double max_x = -1e9;
    double min_y =  1e9;
    double max_y = -1e9;

    for(const auto &p : pixels)
    {
        double rx = p.x - cx;
        double ry = p.y - cy;

        double px = rx * dir_x.x + ry * dir_x.y;
        double py = rx * dir_y.x + ry * dir_y.y;

        if(px < min_x) min_x = px;
        if(px > max_x) max_x = px;

        if(py < min_y) min_y = py;
        if(py > max_y) max_y = py;
    }

    cv::Point2f center(
        static_cast<float>(cx),
        static_cast<float>(cy)
    );

    float width = static_cast<float>(max_x - min_x);
    float height = static_cast<float>(max_y - min_y);

    rect = cv::RotatedRect(
        center,
        cv::Size2f(width, height),
        static_cast<float>(angle * 180.0 / CV_PI)
    );
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_ms = end_time - start_time;
    printf("findRedrect1函数耗时: %.2f ms\n",elapsed_ms.count());
    return true;
}



//计算轮廓长宽比 用于判断轮廓
double maxEdgeRatio(const std::vector<cv::Point2f>& pts){
    if(pts.size() != 4) return 0.0 ;
    float edges[4];
    for(int i = 0; i < 4; i++){
        edges[i] = cv::norm(pts[i] - pts[(i + 1) % 4]);//依次计算了了两个点的距离
    }

    float maxLen = *std::max_element(edges, edges + 4);
    float minLen = *std::min_element(edges, edges + 4);

    if(minLen < 1e-5f) return FLT_MAX;

    return maxLen / minLen;

}
//计算点到直线的距离
static float pointLineDistance(const cv::Point2f& p,
                               const cv::Point2f& a,
                               const cv::Point2f& b)
{
    cv::Point2f ab = b - a;
    cv::Point2f ap = p - a;

    float len = cv::norm(ab);
    if(len < 1e-6f)
        return 1e9f;

    return std::fabs(ab.x * ap.y - ab.y * ap.x) / len;
}


//计算两条直线的交点
static bool lineIntersection(const cv::Vec4f& l1,
                             const cv::Vec4f& l2,
                             cv::Point2f& out)
{
    float vx1 = l1[0];
    float vy1 = l1[1];
    float x1  = l1[2];
    float y1  = l1[3];

    float vx2 = l2[0];
    float vy2 = l2[1];
    float x2  = l2[2];
    float y2  = l2[3];

    float d = vx1 * vy2 - vy1 * vx2;

    if(std::fabs(d) < 1e-5f)
        return false;

    float t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / d;

    out.x = x1 + t * vx1;
    out.y = y1 + t * vy1;

    return true;
}

static void sortCornersTLTRBRBL(std::vector<cv::Point2f>& corners)
{
    if(corners.size() != 4)
        return;

    std::sort(corners.begin(), corners.end(),
              [](const cv::Point2f& a, const cv::Point2f& b)
              {
                  return a.y < b.y;
              });

    std::vector<cv::Point2f> top = {corners[0], corners[1]};
    std::vector<cv::Point2f> bottom = {corners[2], corners[3]};

    std::sort(top.begin(), top.end(),
              [](const cv::Point2f& a, const cv::Point2f& b)
              {
                  return a.x < b.x;
              });

    std::sort(bottom.begin(), bottom.end(),
              [](const cv::Point2f& a, const cv::Point2f& b)
              {
                  return a.x < b.x;
              });

    corners.clear();

    corners.push_back(top[0]);      // 左上
    corners.push_back(top[1]);      // 右上
    corners.push_back(bottom[1]);   // 右下
    corners.push_back(bottom[0]);   // 左下
}

bool getCorners(const std::vector<cv::Point>& contour,
                std::vector<cv::Point2f>& corners)
{
    corners.clear();

    if(contour.size() < 10)
        return false;


    std::vector<cv::Point> hull;
    cv::convexHull(contour, hull, true);//凸包运算

    if(hull.size() < 4)
        return false;

    double peri = cv::arcLength(hull, true);

    std::vector<cv::Point> approx;



    for(double k = 0.008; k <= 0.08; k += 0.004)
    {
        cv::approxPolyDP(hull, approx, k * peri, true);

        if(approx.size() == 4)
            break;
    }

    if(approx.size() != 4)
        return false;

    std::vector<cv::Point2f> rough;
    rough.reserve(4);

    for(const auto& p : approx)
        rough.emplace_back((float)p.x, (float)p.y);

    sortCornersTLTRBRBL(rough);

    cv::Point2f lt = rough[0];
    cv::Point2f rt = rough[1];
    cv::Point2f rb = rough[2];
    cv::Point2f lb = rough[3];

    std::vector<cv::Point2f> top_pts;
    std::vector<cv::Point2f> right_pts;
    std::vector<cv::Point2f> bottom_pts;
    std::vector<cv::Point2f> left_pts;

    float w1 = cv::norm(rt - lt);
    float w2 = cv::norm(rb - lb);
    float h1 = cv::norm(lb - lt);
    float h2 = cv::norm(rb - rt);

    float avg_len = (w1 + w2 + h1 + h2) * 0.25f;
    float dist_th = std::max(2.0f, avg_len * 0.08f);

    for(const auto& p0 : contour)
    {
        cv::Point2f p((float)p0.x, (float)p0.y);

        float d_top    = pointLineDistance(p, lt, rt);
        float d_right  = pointLineDistance(p, rt, rb);
        float d_bottom = pointLineDistance(p, lb, rb);
        float d_left   = pointLineDistance(p, lt, lb);

        float d_min = std::min(std::min(d_top, d_right),
                               std::min(d_bottom, d_left));

        if(d_min > dist_th)
            continue;

        if(d_min == d_top)
            top_pts.push_back(p);
        else if(d_min == d_right)
            right_pts.push_back(p);
        else if(d_min == d_bottom)
            bottom_pts.push_back(p);
        else
            left_pts.push_back(p);
    }

    if(top_pts.size() < 2 || right_pts.size() < 2 ||
       bottom_pts.size() < 2 || left_pts.size() < 2)
    {
        // 如果拟合点不够，就退回 approxPolyDP 的结果
        corners = rough;
        return true;
    }



    cv::Vec4f top_line;
    cv::Vec4f right_line;
    cv::Vec4f bottom_line;
    cv::Vec4f left_line;

    cv::fitLine(top_pts,    top_line,    cv::DIST_L2, 0, 0.01, 0.01);
    cv::fitLine(right_pts,  right_line,  cv::DIST_L2, 0, 0.01, 0.01);
    cv::fitLine(bottom_pts, bottom_line, cv::DIST_L2, 0, 0.01, 0.01);
    cv::fitLine(left_pts,   left_line,   cv::DIST_L2, 0, 0.01, 0.01);

  

    cv::Point2f p_lt, p_rt, p_rb, p_lb;

    if(!lineIntersection(top_line,    left_line,  p_lt)) return false;
    if(!lineIntersection(top_line,    right_line, p_rt)) return false;
    if(!lineIntersection(bottom_line, right_line, p_rb)) return false;
    if(!lineIntersection(bottom_line, left_line,  p_lb)) return false;

    corners = {p_lt, p_rt, p_rb, p_lb};

    sortCornersTLTRBRBL(corners);

    return true;
}
//计算两个向量的夹角的sin值 
static inline float cross_angle(const cv::Point2f& vx,
                                const cv::Point2f& vy)
{
    return vx.x * vy.y - vx.y * vy.x;
}
//计算两个向量的夹角的sin值 
static float parallelError(const cv::Point2f& a,
                           const cv::Point2f& b)
{
    float na = cv::norm(a);
    float nb = cv::norm(b);

    if(na < 1e-6f || nb < 1e-6f)
        return 1.0f;

    // 返回值越接近 0，说明两条边越平行
    return std::fabs(cross_angle(a, b)) / (na * nb);
}

//  void DetectRedBlock(cv::Mat &src)
// {   
//     auto start_time = std::chrono::high_resolution_clock::now();
//     if(src.empty()){
//         return;
//     }

//     red_init();

//     auto checkQuadParallel = [](const std::vector<cv::Point2f>& q,
//                             float parallel_th) -> bool
// {
//     if(q.size() != 4)
//         return false;

//     for(const auto& p : q)
//     {
//         if(!std::isfinite(p.x) || !std::isfinite(p.y))
//             return false;
//     }

//     cv::Point2f top    = q[1] - q[0];  // 上边：左上 -> 右上
//     cv::Point2f bottom = q[2] - q[3];  // 下边：左下 -> 右下

//     cv::Point2f left   = q[3] - q[0];  // 左边：左上 -> 左下
//     cv::Point2f right  = q[2] - q[1];  // 右边：右上 -> 右下

//     float err_tb = parallelError(top, bottom);
//     float err_lr = parallelError(left, right);

//     if(err_tb > parallel_th)
//         return false;

//     if(err_lr > parallel_th)
//         return false;

//     return true;
// };

// auto stabilizeQuad = [&](std::vector<cv::Point2f>& q,
//                          std::vector<cv::Point2f>& last_q,
//                          bool& has_last,
//                          float deadband_px,
//                          float parallel_th) -> bool
// {
//     if(q.size() != 4)
//         return false;

//     // 1. 当前四边形两组对边不近似平行
//     if(!checkQuadParallel(q, parallel_th))
//     {
//         // 如果有上一帧，就继续使用上一帧，避免出现不规则四边形
//         // if(has_last)
//         // {
//         //     q = last_q;
//         //     return true;
//         // }

//         return false;
//     }

//     // 2. 第一帧，直接保存
//     if(!has_last)
//     {
//         last_q = q;
//         has_last = true;
//         return true;
//     }

//     // 3. 计算当前帧和上一帧四个角点的最大距离
//     float max_dist = 0.0f;

//     for(int i = 0; i < 4; i++)
//     {
//         float d = cv::norm(q[i] - last_q[i]);

//         if(d > max_dist)
//             max_dist = d;
//     }

//     // 4. 小于一定像素值，不更新
//     if(max_dist < deadband_px)
//     {
//         q = last_q;
//         return true;
//     }

//     // 5. 超过死区，认为是真实移动，更新上一帧
//     last_q = q;
//     return true;
//     };

//     cv::RotatedRect rect_min,rect_max;
//     //在小在图找色块
//     std::vector<cv::Point> contour;//原图

//     if(!findRedrect(src, rect_min)) return;

//     cv::Point2f rect_points_resize[4];
//     cv::Point2f rect_points[4];

// // if(Flag.infer == 1){
    
//     rect_min.points(rect_points_resize);
//     std::vector<cv::Point2f> points_resize = orderPoints(rect_points_resize);//对四个角点排序
//     //映射回320*240
//     for(size_t i = 0; i < 4; i++){
//         points_resize[i].x = points_resize[i].x * 3.4f;
//         points_resize[i].y = points_resize[i].y * 4.0f;
//     }
 
//     cv::Rect Rect_roi = cv::boundingRect(points_resize);//在原图画四个角点的最小外接矩形
//     int pad = 8;
//     Rect_roi.x = Rect_roi.x - pad;
//     Rect_roi.y = Rect_roi.y - pad;
//     Rect_roi.width = Rect_roi.width + pad * 2;
//     Rect_roi.height = Rect_roi.height + pad * 2;

//     cv::Rect img_rect(0,0,lq_frame.cols,lq_frame.rows);
//     Rect_roi = Rect_roi & img_rect;
//     if(Rect_roi.width <= 0 || Rect_roi.height <= 0) return;

//     //cv::rectangle(lq_frame, Rect_roi, cv::Scalar(0, 255, 0), 1);

//     cv::Mat roi = lq_frame(Rect_roi);//找轮廓的ROI区域
//     if(roi.empty()) return;

//     std::vector<cv::Point2f> red_corners;
//     std::vector<cv::Point> red_contour;
//     if(!findRedrect1(roi, rect_max, red_contour)) return;

//     if(!getCorners(red_contour,red_corners)) return;



//     // ROI 坐标映射回原图坐标
//     for(size_t i = 0; i < red_corners.size(); i++)
//     {
//         red_corners[i].x += Rect_roi.x;
//         red_corners[i].y += Rect_roi.y;
//     }

//     struct Edge{
//         cv::Point2f a;
//         cv::Point2f b;
//         float len;
//         float mid_y;
//     };
//     std::vector<Edge> edges;
//     // 检查角点是否越界 同时计算边长
//     for(size_t i = 0; i < red_corners.size(); i++)
//     {
//     if(red_corners[i].x < 0 || red_corners[i].x >= lq_frame.cols ||
//        red_corners[i].y < 0 || red_corners[i].y >= lq_frame.rows)
//     {
//         return;
//     }
//         Edge e;
//         e.a = red_corners[i];
//         e.b = red_corners[(i+1)%red_corners.size()];
//         e.len = cv::norm(e.b - e.a);
//         e.mid_y = (e.a.y + e.b.y) * 0.5f;
//         edges.push_back(e);
//     }


//     std::sort(edges.begin(),edges.end(),[](const Edge& e1,const Edge& e2){
//         return e1.len > e2.len;
//     });
//     Edge topEdge;//位置靠上的长边
//     Edge bottomEdge;//位置靠下的长边
//     Edge longEdge1 = edges[0];
//     Edge longEdge2 = edges[1];//拿到两条长边
//     Edge shortEdge1 = edges[2];//短边
//     Edge shortEdge2 = edges[3];//短边
//     //任意长边与短边之比小于2 就丢弃
//     if(longEdge1.len/shortEdge1.len < 2.0f||longEdge2.len/shortEdge2.len < 2.0f||longEdge1.len/shortEdge2.len < 2.0f||longEdge2.len/shortEdge1.len < 2.0f)
//     {
//         printf("长宽比异常\n");
//         return; 
//     }

//     cv::Point2f left_top,left_bottom,right_top,right_bottom;//定义四个角点
//     cv::Point2f last_left_top,last_right_top,last_right_bottom,last_left_bottom;

//     //为找位置最靠上长的红边
//     if(longEdge1.mid_y < longEdge2.mid_y)
//     {
//         topEdge = longEdge1;
//         bottomEdge = longEdge2;
//     }
//     else
//     {
//         topEdge = longEdge2;
//         bottomEdge = longEdge1;
//     }

//     if(topEdge.a.x<topEdge.b.x) {
//         left_top = topEdge.a;
//         right_top = topEdge.b;
//     }
//     else
//     {
//         left_top = topEdge.b;
//         right_top = topEdge.a;
//     }

//     if(bottomEdge.a.x<bottomEdge.b.x) {
//         left_bottom = bottomEdge.a;
//         right_bottom = bottomEdge.b;
//     }
//     else
//     {
//         left_bottom = bottomEdge.b;
//         right_bottom = bottomEdge.a;
//     }

//     // cv::circle(lq_frame, left_top, 1, cv::Scalar(0, 255, 0), -1);


//     // cv::circle(lq_frame, left_top, 1, cv::Scalar(0, 255, 0), -1);


//     // cv::circle(lq_frame, left_top, 1, cv::Scalar(0, 255, 0), -1);

//     // cv::circle(lq_frame, right_top, 1, cv::Scalar(255, 0, 0), -1);

//     // cv::circle(lq_frame, left_bottom, 1, cv::Scalar(0, 255, 0), -1);

//     // cv::circle(lq_frame, right_bottom, 1, cv::Scalar(255, 0, 0), -1);
//     std::vector<cv::Point2f> red_real_pts;
//     red_real_pts.push_back(cv::Point2f(0.0f,  0.0f));  // 红块左上
//     red_real_pts.push_back(cv::Point2f(12.0f, 0.0f));  // 红块右上
//     red_real_pts.push_back(cv::Point2f(12.0f, 5.0f));  // 红块右下
//     red_real_pts.push_back(cv::Point2f(0.0f,  5.0f));  // 红块左下


//     std::vector<cv::Point2f> red_img_pts;
//     red_img_pts.push_back(left_top);
//     red_img_pts.push_back(right_top);
//     red_img_pts.push_back(right_bottom);
//     red_img_pts.push_back(left_bottom);



//     // 红块四点上一帧缓存
//     static bool has_last_red_quad = false;
//     static std::vector<cv::Point2f> last_red_quad(4);

// // 稳定红块四点
//     if(!stabilizeQuad(red_img_pts,
//                   last_red_quad,
//                   has_last_red_quad,
//                   2.0f,     // 小于 1 像素不更新
//                   0.40f))   // 对边平行阈值
//     {
//         return;
//     }

// // 使用稳定后的红块四点
//     left_top     = red_img_pts[0];
//     right_top    = red_img_pts[1];
//     right_bottom = red_img_pts[2];
//     left_bottom  = red_img_pts[3];

//     cv::circle(lq_frame, left_top, 1, cv::Scalar(0, 255, 0), -1);

//     cv::circle(lq_frame, right_top, 1, cv::Scalar(255, 0, 0), -1);

//     cv::Mat H = cv::getPerspectiveTransform(red_real_pts, red_img_pts);


//     std::vector<cv::Point2f> card_real_pts;
//     card_real_pts.push_back(cv::Point2f(0.0f,  -14.0f)); // 白色区域左上
//     card_real_pts.push_back(cv::Point2f(12.0f, -14.0f)); // 白色区域右上
//     card_real_pts.push_back(cv::Point2f(12.0f,  0.0f));  // 白色区域右下，也就是红块右上
//     card_real_pts.push_back(cv::Point2f(0.0f,   0.0f));  // 白色区域左下，也就是红块左上

//     std::vector<cv::Point2f> card_img_pts;
//     cv::perspectiveTransform(card_real_pts, card_img_pts, H);

//     cv::Point2f left_top1  = card_img_pts[0];  // 图片左上角点
//     cv::Point2f right_top1 = card_img_pts[1];  // 图片右上角点
//     cv::Point2f right_bottom1 = card_img_pts[2];
//     cv::Point2f left_bottom1  = card_img_pts[3];




//     // cv::line(lq_frame, left_bottom1, left_top1,
//     //          cv::Scalar(0, 255, 0), 1);

//     // cv::line(lq_frame, right_bottom1, right_top1,
//     //          cv::Scalar(0, 255, 0), 1);

//     // cv::line(lq_frame, left_top1, right_top1,
//     //          cv::Scalar(0, 255, 0), 1);

//     // cv::line(lq_frame, left_bottom1, right_bottom1,
//     //          cv::Scalar(0, 255, 0), 1);


//     std::vector<cv::Point2f> src_pts = {
//         left_top1,
//         right_top1,
//         right_top,
//         left_top
//     };

// // 最终透视区域上一帧缓存
//     static bool has_last_src_quad = false;
//     static std::vector<cv::Point2f> last_src_quad(4);

// // 稳定最终透视四边形
//     if(!stabilizeQuad(src_pts,
//                   last_src_quad,
//                   has_last_src_quad,
//                   0.0f,     // 小于 2 像素不更新
//                   0.40f))   // 外推后的区域稍微放宽一些
//     {
//         return;
//     }

//     cv::line(lq_frame, src_pts[0], src_pts[1], cv::Scalar(0, 255, 0), 1);
//     cv::line(lq_frame, src_pts[1], src_pts[2], cv::Scalar(0, 255, 0), 1);
//     cv::line(lq_frame, src_pts[2], src_pts[3], cv::Scalar(0, 255, 0), 1);
//     cv::line(lq_frame, src_pts[3], src_pts[0], cv::Scalar(0, 255, 0), 1);

//     std::vector<cv::Point2f> dst_pts = {
//     {0,0},
//     {59,0},
//     {59,59},
//     {0,59}
//     };

//     auto start_time3 = std::chrono::high_resolution_clock::now();
//     cv::Mat m = cv::getPerspectiveTransform(src_pts, dst_pts);//60*60    透视矩阵 把图片拉伸成60*60
//     auto end_time3 = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double, std::milli> elapsed_ms1 = end_time3 - start_time3;
//     printf("获取透视矩阵耗时: %.2f ms\n",elapsed_ms1.count());
    
//     cv::Mat roi_img;
//     auto start_time2 = std::chrono::high_resolution_clock::now();
//     cv::warpPerspective(lq_frame, roi_img, m, cv::Size(60, 60));
//     auto end_time2 = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double, std::milli> elapsed_ms2 = end_time2 - start_time2;
//     printf("透视耗时: %.2f ms\n",elapsed_ms2.count());

//     if(roi_img.empty()||roi_img.rows<=0||roi_img.cols<=0)
//     {
//         return;
//     }
//     float confidence = 0.0;
//     auto start_time1 = std::chrono::high_resolution_clock::now();
//     std::string result = classifier.Infer(roi_img,confidence);
//     auto end_time1 = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double, std::milli> elapsed_ms3 = end_time1 - start_time1;
//     printf("推理耗时: %.2f ms\n",elapsed_ms3.count());
//    // printf("置信度：%f\n",confidence);
//    // cv::Mat roi_img = lq_frame(roi_rect);//截图送入模型
// // }

// // camera_server.update_frame_mat(roi_img);
// auto end_time = std::chrono::high_resolution_clock::now();
// std::chrono::duration<double, std::milli> elapsed_ms = end_time - start_time;
// printf("函数总耗时: %.2f ms\n",elapsed_ms.count());


// }


//标注ROI区域
// bool GenerateROI(const cv::Point &center, cv::Rect &roi, const cv::Mat &src)
// {
//     if (center.x < 0 || center.y < 0) {
//         return false;  // 点坐标无效
//     }
    
//     if (src.empty()) {
//         return false;  // 图像为空
//     }
//     int half = ROI_SIZE/2;
//     roi_x_1 = center.x - half;
//    // roi_y_1 = center.y;
//     roi_y_1 = center.y - (int)half*2;//把中心的y坐标稍微向上移动一些

//     if (roi_x_1 < 0 || roi_y_1 < 0)
//         return false;

//     if (roi_x_1 + ROI_SIZE > src.cols)
//         return false;

//     if (roi_y_1+ ROI_SIZE > src.rows)
//         return false;

//     roi = cv::Rect(roi_x_1, roi_y_1, ROI_SIZE, ROI_SIZE);
//     return true;
// }


 std::vector<RedObject> red_objects;
 
 int red_area = 0;
  #define MIN_RED_AREA 0//最小红色区域面积 用于滤除红色噪点
 #define MIN_AREA_FOR_BARRIER 250 //障碍物面积
int red_points_num;
int alpha = 0.1;
cv::Point center(-1,-1);
cv::Point last_center(-1,-1);
 void DetectRedBlock(cv::Mat &src,int roi_x,int roi_y,int width,int height)
{
//     red_objects.clear();//先清元素
//     red_area = 0;
//     red_points_num = 0;
//     long long sum_x = 0;
//     long long sum_y = 0;
//     if(src.empty()) 
//     {
//         printf("DetectRedBlock src is empty\n");
//         return;
//     }
//     if(roi_x<0 || roi_y<0||roi_x >= src.cols || roi_y >= src.rows)  return;
//     if(width <= 0 || height <= 0 || width>src.cols||height>src.rows) return;
//     if(roi_x + width > src.cols || roi_y + height > src.rows) return;
//     cv::Rect roi(roi_x,roi_y,width,height);

//     cv::Rect roi_rect;
//     cv::Mat roi_src = src(roi);
//     cv::Mat mask;
//     mask.create(roi_src.size(), CV_8UC1);


//     for (int y = 0; y < roi_src.rows; ++y)
//     {
//         const cv::Vec3b* src_ptr = roi_src.ptr<cv::Vec3b>(y);
//         uchar* mask_ptr = mask.ptr<uchar>(y);

//         for (int x = 0; x < roi_src.cols; ++x)
//         {
//             int b = src_ptr[x][0];
//             int g = src_ptr[x][1];
//             int r = src_ptr[x][2];

//             if (r > Flash.debug_rgb_r_min && (r - g) > Flash.debug_rgb_rg_diff && (r - b) > Flash.debug_rgb_rb_diff){
//                 Image_Use[y+roi_y][x+roi_x]=white;
//                 red_points_num++;
//                 sum_x += x;
//                 sum_y += y;
//                 mask_ptr[x] = white;
//             }
//             else{
//                 mask_ptr[x] = 0;
//             }
//         }
//     }

  
// if (red_points_num >= 5)
//     {
//         int cx = sum_x / red_points_num;
//         int cy = sum_y / red_points_num;
        
//         resize_cx = cx + roi_x;
//         resize_cy = cy + roi_y;
//         // 映射到原图坐标
//         if(resize_cx*3.4>320) return;
//         if(resize_cy*4>240) return;
//         center= cv::Point((resize_cx)*3.4, (resize_cy)*4);//修改1

//         if(last_center != cv::Point(-1, -1))
//         {
//             int dx = center.x - last_center.x;
//             int dy = center.y - last_center.y;
//             double dist = sqrt(dx * dx + dy * dy);
//             // if(dist>80){
//             //     printf("本帧无效\n");
//             //     return;
//             // }
//             if(dist<16)
//             {
//                 center = last_center;
//             }
//         }
//         last_center = center;
//         red_area = red_points_num;
//         red_objects.push_back({center,red_area});

//         // 可选：画点
//         //cv::circle(lq_frame, center, 3, cv::Scalar(0, 255, 0), -1);
//         //printf("检测到红色块\n");
//        // printf("Red Center: (%d, %d), Area: %d,red_points_num: %d, sum_x:%d\n",center.x, center.y, red_area,red_points_num,sum_x);
//     }
//     else{
//         center = cv::Point(-1,-1);
//         last_center = cv::Point(-1,-1);
//         resize_cx = 0;
//         resize_cy = 0;
//     }

//     if(!red_objects.empty()&&Flag.infer ==1)
//     {   
//    auto &obj = red_objects[0];

//         if (obj.center.x < 0 || obj.center.y < 0 || 
//         obj.center.x >= lq_frame.cols || obj.center.y >= lq_frame.rows||obj.center.y < 0||obj.center.x < 0) 
//         {
//         return;
//         }

//         if(!GenerateROI(obj.center, roi_rect, lq_frame)){
//             return;
//         }
//         if (roi_rect.x < 0 || roi_rect.y < 0 ||
//             roi_rect.width <= 0 || roi_rect.height <= 0 ||
//             roi_rect.x + roi_rect.width > lq_frame.cols ||
//             roi_rect.y + roi_rect.height > lq_frame.rows)
//         {
//                 std::cout << "BAD ROI: "
//               << roi_rect << " | img: "
//               << lq_frame.cols << "x"
//               << lq_frame.rows << std::endl;
//                 return;
//         }
//             cv::Mat roi_img = lq_frame(roi_rect);//截图送入模型
//               if(roi_img.empty()||roi_img.rows<=0||roi_img.cols<=0)
//                  {
//                     return;
//                  }
//                 // cv::rectangle(lq_frame, roi_rect, cv::Scalar(255, 0, 0), 1);
//                  float confidence;
//                  auto start_time = std::chrono::high_resolution_clock::now();

//                  real_picture_distance=real_distance[MAX(L_h_guai.row,R_h_guai.row)];
//                  printf("real_picture_distance:%f\n",real_picture_distance);

//                  std::string result = classifier.Infer(roi_img,confidence);
//                 auto end_time = std::chrono::high_resolution_clock::now();
//                 std::chrono::duration<double, std::milli> elapsed_ms = end_time - start_time;
//                  printf("检测结果:%s, 置信度: %.1f,推理耗时: %.2f ms\n",result.c_str(),confidence,elapsed_ms.count());


//                 if(result == "supply" && confidence>40)
//                 {
//                     Flag.supply++;
//                     Flag.weapon = 0;
//                     Flag.vehicle = 0;
//                 }
//                 if(result == "weapon" && confidence>40)
//                 {
//                     Flag.supply = 0;
//                     Flag.weapon++;
//                     Flag.vehicle = 0;
//                 }
//                 if(result == "vehicle" && confidence>40)
//                 {
//                     Flag.supply = 0;
//                     Flag.weapon = 0;
//                     Flag.vehicle++;
//                 }
        
//     }

}

/***************************************************红色标注******************************************************/