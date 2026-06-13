#include "zf_common_headfile.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>

#include <unistd.h>
#include <termios.h>
#include <sys/select.h>
/*
 * 文件功能：红色阈值可视化调试程序
 * 作用：用于手动调整 RGB 红色识别阈值，观察阈值效果
 */
static std::string g_config_path = "red_threshold.txt";

static int R_MIN = 120;
static int R_MAX = 255;

static int G_MIN = 0;
static int G_MAX = 120;

static int B_MIN = 0;
static int B_MAX = 120;

static int RG_DIFF = 30;
static int RB_DIFF = 30;

static int USE_DIFF = 1;
static int USE_OPEN = 1;
static int OPEN_ITER = 1;

static int g_selected = 1;

static struct termios g_old_termios;
static bool g_termios_saved = false;

static int clamp255(int v)
{
    return std::max(0, std::min(255, v));
}

static void normalizeThreshold()
{
    R_MIN = clamp255(R_MIN);
    R_MAX = clamp255(R_MAX);

    G_MIN = clamp255(G_MIN);
    G_MAX = clamp255(G_MAX);

    B_MIN = clamp255(B_MIN);
    B_MAX = clamp255(B_MAX);

    RG_DIFF = clamp255(RG_DIFF);
    RB_DIFF = clamp255(RB_DIFF);

    USE_DIFF = USE_DIFF ? 1 : 0;
    USE_OPEN = USE_OPEN ? 1 : 0;

    OPEN_ITER = std::max(0, std::min(5, OPEN_ITER));
}

static void setTerminalRawMode()
{
    if (tcgetattr(STDIN_FILENO, &g_old_termios) == 0)
    {
        g_termios_saved = true;

        struct termios new_termios = g_old_termios;
        new_termios.c_lflag &= ~(ICANON | ECHO);

        tcsetattr(STDIN_FILENO, TCSANOW, &new_termios);
    }
}

static void restoreTerminalMode()
{
    if (g_termios_saved)
    {
        tcsetattr(STDIN_FILENO, TCSANOW, &g_old_termios);
        g_termios_saved = false;
    }
}

static int readKeyNonBlock()
{
    fd_set set;
    struct timeval timeout;

    FD_ZERO(&set);
    FD_SET(STDIN_FILENO, &set);

    timeout.tv_sec = 0;
    timeout.tv_usec = 0;

    int ret = select(STDIN_FILENO + 1, &set, NULL, NULL, &timeout);

    if (ret > 0)
    {
        char ch = 0;
        if (read(STDIN_FILENO, &ch, 1) > 0)
        {
            return static_cast<int>(ch);
        }
    }

    return -1;
}

static int getSelectedValue()
{
    switch (g_selected)
    {
    case 1: return R_MIN;
    case 2: return R_MAX;
    case 3: return G_MIN;
    case 4: return G_MAX;
    case 5: return B_MIN;
    case 6: return B_MAX;
    case 7: return RG_DIFF;
    case 8: return RB_DIFF;
    case 9: return OPEN_ITER;
    default: return 0;
    }
}

static void setSelectedValue(int value)
{
    if (g_selected == 9)
    {
        value = std::max(0, std::min(5, value));
    }
    else
    {
        value = clamp255(value);
    }

    switch (g_selected)
    {
    case 1: R_MIN = value; break;
    case 2: R_MAX = value; break;
    case 3: G_MIN = value; break;
    case 4: G_MAX = value; break;
    case 5: B_MIN = value; break;
    case 6: B_MAX = value; break;
    case 7: RG_DIFF = value; break;
    case 8: RB_DIFF = value; break;
    case 9: OPEN_ITER = value; break;
    default: break;
    }
}

void RedThresholdNoGui_Print()
{
    std::cout << std::endl;
    std::cout << "========== 红色阈值 ==========" << std::endl;
    std::cout << "1 R_MIN  = " << R_MIN << std::endl;
    std::cout << "2 R_MAX  = " << R_MAX << std::endl;
    std::cout << "3 G_MIN  = " << G_MIN << std::endl;
    std::cout << "4 G_MAX  = " << G_MAX << std::endl;
    std::cout << "5 B_MIN  = " << B_MIN << std::endl;
    std::cout << "6 B_MAX  = " << B_MAX << std::endl;
    std::cout << "7 RG_DIFF = " << RG_DIFF << std::endl;
    std::cout << "8 RB_DIFF = " << RB_DIFF << std::endl;
    std::cout << "9 OPEN_ITER = " << OPEN_ITER << std::endl;
    std::cout << "USE_DIFF = " << USE_DIFF << std::endl;
    std::cout << "USE_OPEN = " << USE_OPEN << std::endl;
    std::cout << "当前选择: " << g_selected << std::endl;
    std::cout << "按键: 1~9选择, +/-调节, s保存, l加载, p打印" << std::endl;
    std::cout << "==============================" << std::endl;
}

void RedThresholdNoGui_Save()
{
    normalizeThreshold();

    std::ofstream ofs(g_config_path.c_str());

    if (!ofs.is_open())
    {
        std::cout << "保存失败: " << g_config_path << std::endl;
        return;
    }

    ofs << R_MIN << " " << R_MAX << std::endl;
    ofs << G_MIN << " " << G_MAX << std::endl;
    ofs << B_MIN << " " << B_MAX << std::endl;
    ofs << RG_DIFF << " " << RB_DIFF << std::endl;
    ofs << USE_DIFF << std::endl;
    ofs << USE_OPEN << " " << OPEN_ITER << std::endl;

    ofs.close();

    std::cout << "红色阈值已保存: " << g_config_path << std::endl;
    RedThresholdNoGui_Print();
}

void RedThresholdNoGui_Load()
{
    std::ifstream ifs(g_config_path.c_str());

    if (!ifs.is_open())
    {
        std::cout << "未找到阈值文件，使用默认阈值: "
                  << g_config_path << std::endl;
        return;
    }

    ifs >> R_MIN >> R_MAX;
    ifs >> G_MIN >> G_MAX;
    ifs >> B_MIN >> B_MAX;
    ifs >> RG_DIFF >> RB_DIFF;
    ifs >> USE_DIFF;
    ifs >> USE_OPEN >> OPEN_ITER;

    ifs.close();

    normalizeThreshold();

    std::cout << "红色阈值已加载: " << g_config_path << std::endl;
    RedThresholdNoGui_Print();
}

void RedThresholdNoGui_Init(const char* config_path)
{
    if (config_path != NULL)
    {
        g_config_path = config_path;
    }

    RedThresholdNoGui_Load();

    setTerminalRawMode();

    std::cout << "无GUI红色阈值调节启动。" << std::endl;
    std::cout << "图传显示: 原图 + Mask + Result" << std::endl;
    RedThresholdNoGui_Print();
}

void RedThresholdNoGui_Deinit()
{
    restoreTerminalMode();
}

bool RedThreshold_IsRedRGB(unsigned char r,
                           unsigned char g,
                           unsigned char b)
{
    int r_min = std::min(R_MIN, R_MAX);
    int r_max = std::max(R_MIN, R_MAX);

    int g_min = std::min(G_MIN, G_MAX);
    int g_max = std::max(G_MIN, G_MAX);

    int b_min = std::min(B_MIN, B_MAX);
    int b_max = std::max(B_MIN, B_MAX);

    bool range_ok =
        r >= r_min && r <= r_max &&
        g >= g_min && g <= g_max &&
        b >= b_min && b <= b_max;

    if (!range_ok)
    {
        return false;
    }

    if (USE_DIFF)
    {
        int rg = static_cast<int>(r) - static_cast<int>(g);
        int rb = static_cast<int>(r) - static_cast<int>(b);

        if (rg < RG_DIFF)
        {
            return false;
        }

        if (rb < RB_DIFF)
        {
            return false;
        }
    }

    return true;
}

bool RedThreshold_IsRedBGR(unsigned char b,
                           unsigned char g,
                           unsigned char r)
{
    return RedThreshold_IsRedRGB(r, g, b);
}

void RedThreshold_MakeMask(const cv::Mat& frame_bgr,
                           cv::Mat& mask)
{
    if (frame_bgr.empty() || frame_bgr.channels() != 3)
    {
        mask.release();
        return;
    }

    mask.create(frame_bgr.rows, frame_bgr.cols, CV_8UC1);

    for (int y = 0; y < frame_bgr.rows; y++)
    {
        const cv::Vec3b* src_row = frame_bgr.ptr<cv::Vec3b>(y);
        unsigned char* mask_row = mask.ptr<unsigned char>(y);

        for (int x = 0; x < frame_bgr.cols; x++)
        {
            unsigned char b = src_row[x][0];
            unsigned char g = src_row[x][1];
            unsigned char r = src_row[x][2];

            mask_row[x] = RedThreshold_IsRedBGR(b, g, r) ? 255 : 0;
        }
    }

    if (USE_OPEN && OPEN_ITER > 0)
    {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,
                                                   cv::Size(3, 3));

        cv::morphologyEx(mask,
                         mask,
                         cv::MORPH_OPEN,
                         kernel,
                         cv::Point(-1, -1),
                         OPEN_ITER);
    }
}

static void handleKey()
{
    int key = readKeyNonBlock();

    if (key < 0)
    {
        return;
    }

    if (key >= '1' && key <= '9')
    {
        g_selected = key - '0';
        RedThresholdNoGui_Print();
    }
    else if (key == '+' || key == '=')
    {
        int value = getSelectedValue();
        setSelectedValue(value + 1);
        RedThresholdNoGui_Print();
    }
    else if (key == '-' || key == '_')
    {
        int value = getSelectedValue();
        setSelectedValue(value - 1);
        RedThresholdNoGui_Print();
    }
    else if (key == 's' || key == 'S')
    {
        RedThresholdNoGui_Save();
    }
    else if (key == 'l' || key == 'L')
    {
        RedThresholdNoGui_Load();
    }
    else if (key == 'p' || key == 'P')
    {
        RedThresholdNoGui_Print();
    }
    else if (key == 'd' || key == 'D')
    {
        USE_DIFF = !USE_DIFF;
        RedThresholdNoGui_Print();
    }
    else if (key == 'o' || key == 'O')
    {
        USE_OPEN = !USE_OPEN;
        RedThresholdNoGui_Print();
    }
}

static void drawInfo(cv::Mat& img)
{
    if (img.empty())
    {
        return;
    }

    std::string text1 =
        "R:[" + std::to_string(R_MIN) + "," + std::to_string(R_MAX) + "] " +
        "G:[" + std::to_string(G_MIN) + "," + std::to_string(G_MAX) + "] " +
        "B:[" + std::to_string(B_MIN) + "," + std::to_string(B_MAX) + "]";

    std::string text2 =
        "RG:" + std::to_string(RG_DIFF) +
        " RB:" + std::to_string(RB_DIFF) +
        " SEL:" + std::to_string(g_selected) +
        " +/- adjust s save";

    cv::putText(img,
                text1,
                cv::Point(5, 20),
                cv::FONT_HERSHEY_SIMPLEX,
                0.45,
                cv::Scalar(0, 255, 0),
                1);

    cv::putText(img,
                text2,
                cv::Point(5, 40),
                cv::FONT_HERSHEY_SIMPLEX,
                0.45,
                cv::Scalar(0, 255, 0),
                1);
}

void RedThresholdNoGui_Update(const cv::Mat& frame_bgr,
                              cv::Mat& debug_show)
{
    if (frame_bgr.empty())
    {
        debug_show.release();
        return;
    }

    normalizeThreshold();

    handleKey();

    cv::Mat mask;
    cv::Mat mask_bgr;
    cv::Mat result;
    cv::Mat src_show;

    RedThreshold_MakeMask(frame_bgr, mask);

    cv::cvtColor(mask, mask_bgr, cv::COLOR_GRAY2BGR);

    result = cv::Mat::zeros(frame_bgr.size(), frame_bgr.type());
    frame_bgr.copyTo(result, mask);

    src_show = frame_bgr.clone();

    drawInfo(src_show);
    drawInfo(mask_bgr);
    drawInfo(result);

    cv::hconcat(src_show, mask_bgr, debug_show);
    cv::hconcat(debug_show, result, debug_show);
}