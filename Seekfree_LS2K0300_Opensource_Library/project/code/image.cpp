#include "zf_common_headfile.hpp"
#include "ww_transmission.h"
#define UVC_PATH       "/dev/video0"  
#define MODEL_INPUT_WIDTH         40      // 模型输入图像宽度（根据实际模型调整）
#define MODEL_INPUT_HEIGHT        40      // 模型输入图像高度（根据实际模型调整）
#define MODEL_INPUT_CHANNEL       3       // 模型输入图像通道数（RGB=3）
#define MODEL_OUTPUT_CLASS_NUM    3       // 模型输出类别数（根据实际任务调整）
#define MODEL_INPUT_SIZE          (MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * MODEL_INPUT_CHANNEL)  // 输入总元素数
#define TFLITE_OP_RESOLVER_MAX_NUM 20     // 算子解析器最大支持算子数
#define TENSOR_ARENA_SIZE         (128 * 1024)  // 张量空间大小（单位：字节，根据模型大小调整）
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];
static tflite::MicroInterpreter* interpreter = nullptr;
static tflite::MicroMutableOpResolver<20> resolver;
const char* class_labels[] = {"materials","traffic","weapon"}; //需要与train.py提示顺序一致
zf_device_uvc uvc_dev;//初始化摄像头对象
uint8_t Image_Zip[LCDH_1][LCDW_1] = {0}; //压缩后的图像数组
uint8_t Image_Use[LCDH_1][LCDW_1] = {0};//(经过大津法，膨胀，腐蚀后的)二值化的图像
cv::VideoCapture cap;
unsigned char Image_IFS[LCDH_1][LCDW_1];
LQ_NCNN classifier;
uint8_t Threshold = 0;  //大津法求出的阈值
struct imageInformation imgInfo;
struct YuanSu Flag;
cv::Mat frame, grayFrame, binaryFrame,resizedFrame,flippedFrame,translatedFrame,translationMatrix,float_img,resized_img;
cv::Mat target_roi;//存储目标图片区域
TransmissionStreamServer camera_server;
int roi_x_1= 0;
int roi_y_1 = 0;
int ROI_SIZE = 60;
char txt[80];
bool zf_init_flag = false;
bool first_frame_logged = false;
#define ENABLE_FIRST_FRAME_DEBUG     true
int picture_cnt = 0;
bool first_flag = true;
int roix1,roiy1 = 0;
using namespace cv;
cv::Mat lq_frame;
//绕行相关
bool picture_yaw_init = false;//记录第一次进入绕行逻辑的标志位
float Yaw_picture = 0;//记录检测到图片时的初始yaw值
float Yaw_picture_diff = 0;//当前的YAW值相较于Yaw_picture的差值
float Yaw_picture_err = 0;
float Yaw_picture_target = 0;//绕行的目标偏差值
float encoder_val = 0;//用于记录编码器的值 以实现分阶段运行
int resize_cx,resize_cy = 0;
cv::Point2f target_pts[4] = {
    cv::Point2f(0.0f, 0.0f),
    cv::Point2f(0.0f, 0.0f),
    cv::Point2f(0.0f, 0.0f),
    cv::Point2f(0.0f, 0.0f)
};
cv::Point2f red_pts[4];
float valid_ratio = 0.0f;
bool erase_pts_ready = false;

unsigned char R_TH = 140;
unsigned char G_TH = 125;
unsigned char B_TH = 125;
unsigned char R_G_TH = 50;
unsigned char R_B_TH = 50;
Guaidian  red_point;
Guaidian red_L_L,red_R_L,picture_L_H,picture_R_H;//四个拐点
//处理陀螺仪角度跳变
float Yaw_correct(float current_yaw,float target_yaw)
{
    float diff = current_yaw - target_yaw;
    if(diff >=180)
    {
        diff =  diff - 360;
    }
    if (diff<=-180)
    {
        diff =  diff + 360;
    }
    return diff;
    
}

int halfside_wan[60] =
{
    33, 33, 33, 33, 33, 33, 32, 32, 32, 32,
    32, 32, 32, 31, 31, 31, 31, 31, 31, 30,
    30, 30, 30, 29, 29, 29, 30, 30, 31, 31,
    32, 32, 32, 33, 33, 34, 35, 35, 35, 36,
    36, 37, 38, 39, 40, 40, 41, 41, 42, 42,
    43, 43, 43, 44, 44, 45, 46, 46, 46, 46,
};
// int halfside_wan[60] =
// {
//     33, 33, 33, 33, 33, 33, 32, 32, 32, 32,
//     32, 32, 32, 31, 31, 31, 31, 31, 31, 30,
//     30, 30, 30, 29, 29, 29, 30, 30, 31, 31,
//     32, 32, 32, 33, 33, 34, 35, 1.7, 1.65, 1.5,
//     1.45, 1.4, 1.35, 1.3, 1.25, 1.2, 1.15, 1.1, 1.05, 1,
//     0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5,
// };


const float atan_deg_tab[49] = 
{
    0.00f, 45.00f, 63.43f, 71.57f, 75.96f, 78.69f, 80.54f, 81.87f, 82.87f, 83.66f,
    84.29f, 84.81f, 85.24f, 85.60f, 85.91f, 86.19f, 86.42f, 86.63f, 86.82f, 86.99f,
    87.14f, 87.27f, 87.40f, 87.51f, 87.61f, 87.71f, 87.80f, 87.88f, 87.95f, 88.03f,
    88.09f, 88.15f, 88.21f, 88.26f, 88.32f, 88.36f, 88.41f, 88.45f, 88.49f, 88.53f,
    88.57f, 88.60f, 88.64f, 88.67f, 88.70f, 88.73f, 88.75f, 88.78f, 88.81f
};
float actan_err(float err)
{
    float actan_err;
    if(err>0) actan_err=atan_deg_tab[(int)err];
    if(err<=0) actan_err=-atan_deg_tab[-(int)err];
  return actan_err;
}

float real_distance[60] =
{
     485,410, 345, 292.0, 244.5, 213.8, 184.0, 161.0, 141.5, 127.0,
     115.0, 104.5, 95.6, 88.0, 82.0, 77.0, 71.7,  67.0,  62.5,  58.0,
     54.7,  51.50,  48.2,  45.3,  42.5,  40.3,  38.0,  36.5,  34.6,  32.5,
     31.0,  29.5,  27.2,  26.5,  24.8,  23.3,  22.1,  21.0,  20.0,  19.0,
     18.0,  17.0,  16.0,  14.9,  14.0,  13.10,  12.6,  12.0,  11.0,  10.5, 
     10.0,  9.3,  8.7,  8.2,  7.5,  6.9,  6.5,  6.0,  5.5,  5,
};

int real_distance_to_row(float distance) {
    float min_diff = 10000.0f;  // 初始化为一个很大的数
    int nearest_index = 0;
    
    for (int i = 0; i < 60; i++) {
        float diff = fabsf(real_distance[i] - distance);
        if (diff < min_diff) {
            min_diff = diff;
            nearest_index = i;
        }
    }
    
    return nearest_index;
}

void imgInfoInit(void)
{
    imgInfo.bottom = LCDH_1 - 1;
    imgInfo.lastMid = LCDW_1 / 2;
    imgInfo.top = 0;
    imgInfo.L_loselineSum = 0;
    imgInfo.R_loselineSum = 0;
}

void debug_log_printf_callback(const char* s)
{
    if (s == NULL) return;
    printf("%s", s);
}

void zf_model_init()
{
    RegisterDebugLogCallback(debug_log_printf_callback);

    tflite::InitializeTarget();

    const tflite::Model* model = ::tflite::GetModel(loong_cnn_model_simple_tflite);
    TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

    // 注册算子
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddMaxPool2D();
    resolver.AddFullyConnected();
    resolver.AddRelu6();
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddShape();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddCast();
    resolver.AddSqueeze();
    resolver.AddExpandDims();
    resolver.AddConcatenation();
    resolver.AddTranspose();
    resolver.AddStridedSlice();
    resolver.AddPack();
    resolver.AddLogistic();
    resolver.AddMean();
    resolver.AddAdd();

    // ✅ 正确创建 interpreter
    interpreter = new tflite::MicroInterpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE);

    TfLiteStatus status = interpreter->AllocateTensors();

    if (status != kTfLiteOk)
    {
        printf("❌ AllocateTensors 失败！\n");
        while (1);
    }

    printf("✅ TFLite 初始化完成\n");
}
//-------------------------------------------------------------------------------------------------------------------
//  @brief      优化的大津法
//  @param      image  图像数组
//  @param      clo    宽
//  @param      row    高
//  @param      pixel_threshold 阈值分离
//  @return     uint8
//  @since      2021.6.23
//  Sample usage:
//-------------------------------------------------------------------------------------------------------------------
uint8_t Threshold_deal(uint8 image[LCDH_1][LCDW_1],uint16 col,uint16 row) {
  #define GrayScale 255
  uint16 width = col;
  uint16 height = row;
  int pixelCount[GrayScale];
  float pixelPro[GrayScale];
  int i, j, pixelSum = width * height;
//  uint8 threshold = 0;
  uint8* data =(uint8*)  image;  //指向像素数据的指针
  for (i = 0; i < GrayScale; i++) {
    pixelCount[i] = 0;
    pixelPro[i] = 0;
  }

  uint32 gray_sum = 0;
  //统计灰度级中 每个像素在整幅图像中的个数
  for (i = 0; i < height; i += 1) {
    for (j = 0; j < width; j += 1) {
      // if((sun_mode&&data[i*width+j]<pixel_threshold)||(!sun_mode))
      //{
      pixelCount[(int)data[i * width + j]]++;  //将当前的点的像素值作为计数数组的下标
      gray_sum += (int)data[i * width + j];  //灰度值总和
      //}
    }
  }

  //计算每个像素值的点在整幅图像中的比例
  for (i = 0; i < GrayScale; i++) {
    pixelPro[i] = (float)pixelCount[i] / pixelSum;
  }


  //遍历灰度级[0,255]
  float w0, w1, u0tmp, u1tmp, u0, u1, u, deltaTmp, deltaMax = 0;
  w0 = w1 = u0tmp = u1tmp = u0 = u1 = u = deltaTmp = 0;
  for (j = 0; j < 180; j++) {
    w0 +=
        pixelPro[j];  //背景部分每个灰度值的像素点所占比例之和 即背景部分的比例
    u0tmp += j * pixelPro[j];  //背景部分 每个灰度值的点的比例 *灰度值

    w1 = 1 - w0;
    u1tmp = gray_sum / pixelSum - u0tmp;

    u0 = u0tmp / w0;    //背景平均灰度
    u1 = u1tmp / w1;    //前景平均灰度
    u = u0tmp + u1tmp;  //全局平均灰度
    deltaTmp = w0 * pow((u0 - u), 2) + w1 * pow((u1 - u), 2);
    if (deltaTmp > deltaMax) {
      deltaMax = deltaTmp;
      Threshold = (uint8)j;
    }
    if (deltaTmp < deltaMax) break;
  }

  return (uint8)Threshold;
}


/******************************************************图像二值化************************************************/


uint8 Threshold_static = 70;
void Get01change_dajin() {
//    if(run_flag==0){
        Threshold = Threshold_deal(Image_Use, LCDW_1, LCDH_1);
//    }
  if (Threshold < Threshold_static)
    Threshold = (uint8)Threshold_static;
  uint8 i, j = 0;
  int thre;
  for (i = 0; i < LCDH_1; i++) {
    for (j = 0; j < LCDW_1; j++) {
      if (j <= 18)
        thre = Threshold - 10;
      else if ((j > 82 && j <= 88))
        thre = Threshold - 10;
      else if (j >= 76)
        thre = Threshold - 10;
      else
        thre = Threshold;

      if (Image_Use[i][j] >(thre))         //数值越大，显示的内容越多，较浅的图像也能显示出来
          Image_Use[i][j] = 255;  //白
      else
          Image_Use[i][j] = 0;  //黑
    }
  }

//   for(int i = red_L_L.column; i < red_R_L.column; i++)
//   {
//     for(int j = red_L_L.row; j > picture_R_H.row; j--)
//     {
//       Image_Use[j][i] = white;
//     }
//   }


}


void my_sobel(unsigned char imageIn[LCDH_1][LCDW_1], unsigned char imageOut[LCDH_1][LCDW_1])
{
    short KERNEL_SIZE = 3;
    short xStart = KERNEL_SIZE / 2;
    short xEnd = LCDW_1 - KERNEL_SIZE / 2;
    short yStart = KERNEL_SIZE / 2;
    short yEnd = LCDH_1 - KERNEL_SIZE / 2;
    short i, j;
    short temp[2];
    short temp1 ,temp2 ;
    //for(i = 0; i < Compress_H; i++)//算的更慢不过对比了全局图像
    for (i = yStart; i < yEnd; i++)   //有点的跳跃
       {
           //for(j = 0; j < Compress_W; j++)//算的更慢不过对比了全局图像
           for (j = xStart; j < xEnd; j++)  //有点的跳跃
           {
               /* 计算不同方向梯度幅值  */
               temp[0] = -(short) imageIn[i - 1][j - 1] + (short) imageIn[i - 1][j + 1]     //{{-1, 0, 1},
               - (short) 2*imageIn[i][j - 1] + (short) 2*imageIn[i][j + 1]       // {-2, 0, 2},
               - (short) imageIn[i + 1][j - 1] + (short) imageIn[i + 1][j + 1];    // {-1, 0, 1}};

               temp[1] = -(short) imageIn[i - 1][j - 1] + (short) imageIn[i + 1][j - 1]     //{{-1, -2, -1},
               - (short) 2*imageIn[i - 1][j] + (short) 2*imageIn[i + 1][j]       // { 0,  0,  0},
               - (short) imageIn[i - 1][j + 1] + (short) imageIn[i + 1][j + 1];    // { 1,  2,  1}};

               temp[0] = fabs(temp[0]);
               temp[1] = fabs(temp[1]);

               temp1 = temp[0] + temp[1] ;

               temp2 =  (short) imageIn[i - 1][j - 1] + (short)2* imageIn[i - 1][j] + (short) imageIn[i - 1][j + 1]
                       + (short)2* imageIn[i][j - 1] + (short) imageIn[i][j] + (short) 2*imageIn[i][j + 1]
                       + (short) imageIn[i + 1][j - 1] + (short) 2*imageIn[i + 1][j] + (short) imageIn[i + 1][j + 1];

               if (temp1 > temp2 / 12.0f)
               {
                   imageOut[i][j] = black;
               }
              else
              {
                  imageOut[i][j] = white;
              }
           }
       }

}

/******************************************************图像二值化************************************************/




void my_sobel_dajin(unsigned char imageIn[LCDH_1][LCDW_1], unsigned char imageOut[LCDH_1][LCDW_1])
{       

    //    if(run_flag==0){
        Threshold = Threshold_deal(Image_Use, LCDW_1, LCDH_1);
//    }
  if (Threshold < Threshold_static)
    Threshold = (uint8)Threshold_static;

    short KERNEL_SIZE = 3;
    short xStart = KERNEL_SIZE / 2;
    short xEnd = LCDW_1 - KERNEL_SIZE / 2;
    short yStart = KERNEL_SIZE / 2;
    short yEnd = LCDH_1 - KERNEL_SIZE / 2;
    short i, j;
    short temp[2];
    short temp1 ,temp2 ;
    //for(i = 0; i < Compress_H; i++)//算的更慢不过对比了全局图像
    for (i = yStart; i < yEnd; i++)   //有点的跳跃
       {
           //for(j = 0; j < Compress_W; j++)//算的更慢不过对比了全局图像
           for (j = xStart; j < xEnd; j++)  //有点的跳跃
           {
               /* 计算不同方向梯度幅值  */
               temp[0] = -(short) imageIn[i - 1][j - 1] + (short) imageIn[i - 1][j + 1]     //{{-1, 0, 1},
               - (short) 2*imageIn[i][j - 1] + (short) 2*imageIn[i][j + 1]       // {-2, 0, 2},
               - (short) imageIn[i + 1][j - 1] + (short) imageIn[i + 1][j + 1];    // {-1, 0, 1}};

               temp[1] = -(short) imageIn[i - 1][j - 1] + (short) imageIn[i + 1][j - 1]     //{{-1, -2, -1},
               - (short) 2*imageIn[i - 1][j] + (short) 2*imageIn[i + 1][j]       // { 0,  0,  0},
               - (short) imageIn[i - 1][j + 1] + (short) imageIn[i + 1][j + 1];    // { 1,  2,  1}};

               temp[0] = fabs(temp[0]);
               temp[1] = fabs(temp[1]);

               temp1 = temp[0] + temp[1] ;

            //    temp2 =  (short) imageIn[i - 1][j - 1] + (short)2* imageIn[i - 1][j] + (short) imageIn[i - 1][j + 1]
            //            + (short)2* imageIn[i][j - 1] + (short) imageIn[i][j] + (short) 2*imageIn[i][j + 1]
            //            + (short) imageIn[i + 1][j - 1] + (short) 2*imageIn[i + 1][j] + (short) imageIn[i + 1][j + 1];

               if (temp1 > Threshold*1.0)
               {
                   imageOut[i][j] = 0;
               }
              else
              {
                  imageOut[i][j] = 255;
              }
           }
       }

  int thre;
  for (i = 0; i < LCDH_1; i++) {
    for (j = 0; j < LCDW_1; j++) {
      if (j <= 18)
        thre = Threshold - 10;
      else if ((j > 82 && j <= 88))
        thre = Threshold - 10;
      else if (j >= 76)
        thre = Threshold - 10;
      else
        thre = Threshold;

      if (imageIn[i][j] <(thre))         //数值越大，显示的内容越多，较浅的图像也能显示出来
          imageOut[i][j] = 0;  //hei
    //   else
    //    imageOut[i][j] = 1;  //hei
    }
  }
}


/***************************************************给图像两边画黑框*************************************************/

void Draw_BlackSideline(uint8_t Image_1[LCDH_1][LCDW_1])
{
    int i = 0;
        for(i = imgInfo.bottom - 1; i > imgInfo.top; i--)
        {
            Image_1[i][0] = black;
            Image_1[i][LCDW_1 - 1] = black;
        }
}
/***************************************************获取最长白列***************************************************/

int Mid_Line[LCDH_0] = {LCDW_0 / 2};     // 中线数组，初始化所有元素为屏幕宽度的一半
int Last_Mid_Line[LCDH_0] = {0};           // 上一次中线数组，初始化所有元素为0

/* 获取截止行和最长白列 */
void Get_ImageTop(void)
{

    int8 len;                            // 临时变量，用于存储白色像素列的长度
    uint8 j;                             // 循环变量，用于遍历列
    imgInfo.white_num = 0;               // 初始化最长白色像素列的长度为0
    imgInfo.top = 0;                     // 初始化图像的有效顶部行号为0
    imgInfo.max_column = 0;              // 初始化最长白色像素列的列号为0

    uint8 l_side=0 , r_side=0 ;                    // 默认左边界（对应循环终止条件 i==2）
                                                   // 默认右边界（对应循环终止条件 i==LCDW_1-3）

    static uint8 flag_m = 0;             // 静态标志变量，用于判断是否是第一次执行

    if(flag_m == 0)                      // 如果是第一次执行
    {
        Last_Mid_Line[imgInfo.bottom - 1] = LCDW_1 / 2; // 设置上一次中线位置为屏幕长度的一半
        flag_m = 1;                      // 将标志变量设置为1，表示已执行过一次
    }

//    if(Mode_1)                           // 如果处于模式1
//    {
        imgInfo.bottom = LCDH_1 - 1;     // 设置图像的底部行号为屏幕高度减1

        if(Last_Mid_Line[imgInfo.bottom - 1] > LCDW_1 - 1 || Last_Mid_Line[imgInfo.bottom - 1] < 0)
        {                                // 如果上一次中线位置超出屏幕宽度范围
            Last_Mid_Line[imgInfo.bottom - 1] = LCDW_1 / 2; // 重新设置上一次中线位置为屏幕宽度的一半
        }

        // 从上一次中线位置向左搜索白色区域的左边界
        for(int i = Last_Mid_Line[imgInfo.bottom - 1]; i > 1; i--)
        {
           if((Image_Use[imgInfo.bottom - 1][i] == white
               && Image_Use[imgInfo.bottom - 1][i - 1] == black
               && Image_Use[imgInfo.bottom - 1][i - 2] == black)
                   || (i==2))//寻找到白黑黑跳变点 或者找到最左边（会在图像四周画上图像，所以i=2为最左边）
           {
               l_side = (uint8)i;               // 找到左边界，退出循环
               break;
           }
        }

        // 从上一次中线位置向右搜索白色区域的右边界
        for(int i = Last_Mid_Line[imgInfo.bottom - 1]; i < LCDW_1 - 2; i++)
        {
           if((Image_Use[imgInfo.bottom - 1][i] == white
               && Image_Use[imgInfo.bottom - 1][i + 1] == black
               && Image_Use[imgInfo.bottom - 1][i + 2] == black)
                   || (i==LCDW_1 - 3))
           {
               r_side =(uint8) i;               // 找到右边界，退出循环
               break;
           }
        }
            // 在左右边界之间，每隔4列检查一次最长白色像素列
            for (j = l_side; j <= r_side; j += 1)
            {
                for (len = imgInfo.bottom - 1; len >= imgInfo.top && Image_Use[len][j]; len--);
                                             // 从底部向上搜索，直到找到非白色像素或到达顶部

                len = LCDH_1 - len;          // 计算白色像素列的长度（从底部到顶部的距离）
                if (imgInfo.white_num < len)
                {
                    imgInfo.white_num = len; // 更新最长白色像素列的长度
                    imgInfo.max_column = j;  // 更新最长白色像素列的列号
                }
            }
        // 确定截止行（图像有效部分的顶部）
        if ((LCDH_1 - imgInfo.white_num) >= 0)
            imgInfo.top = LCDH_1 - imgInfo.white_num + 1; // 根据最长白色像素列的长度计算截止行
        else
            imgInfo.top = 0;             // 如果计算出的截止行小于0，则设置为0
//    }
}

int Find_Top(int line)
{
    for(int i = imgInfo.bottom - 1; i <= imgInfo.top; i--)
    {
        if(Image_Use[i - 1][line] == white && Image_Use[i][line] == black && Image_Use[i + 1][line] == black)
        {
            return i;
        }
    }
    return 0;
}


/**
 * @name: xielv_sideline
 * @details: 计算斜率
 * @param {uint8 x1, uint8 y1    第一个点的坐标
 *         uint8 x2, uint8 y2    第二个点的坐标
 *         uint8 data            返回的值 斜率 k 或 常数 b，只能给大小写的k或b，给其他值则返回k
 *          }
 * @return: 返回的值 斜率 k 或 常数 b
 **/
float xielv_sideline(int x1, int y1, int x2, int y2, char data)
{
    float k = 0.0, b = 0.0;

    k = (float)(y2 - y1) / (x2 - x1);
    b = (float)(y1 - k * x1);

    if (data == 'b' || data == 'B')
        return b;
    else if (data == 'k' || data == 'K')
        return k;
    else
        return k;
}

int Left_Sideline[LCDH_0] = {0};    //左边线数组
unsigned char Left_Sideline_flag[LCDH_0] = {0};   //左边线标志位

int Right_Sideline[LCDH_0] = {LCDH_0 - 2};   //右边线数组
unsigned char Right_Sideline_flag[LCDH_0] = {0};  //右边线标志位

unsigned char white_width[LCDH_0] = {0}; //每行的白点数（宽度）
unsigned char starith_white_width[LCDH_0] = {0}; //每行的白点数（宽度）
/* 寻找边线 */

void Find_Sideline(uint8 Start_row, uint8 End_row)
{
    int i = 0, j = 0;
    imgInfo.L_loselineSum = 0; // 初始化左侧丢失线（未检测到边线）的计数器为0

//    else if(Mode_1) // 如果处于模式1
//    {
        for(i = Start_row; i >= End_row; i--) // 从起始行向结束行遍历（逆序）
        {
            Left_Sideline_flag[i] = 0; // 初始化当前行的左侧边线标志为0（未检测到）
            Right_Sideline_flag[i] = 0; // 初始化当前行的右侧边线标志为0（未检测到）

            if(Image_Use[i][imgInfo.max_column] == black) // 如果最长白列的当前行像素为黑色（可能是阴影或障碍物）
            {
                Left_Sideline[i] = Left_Sideline[i + 1]; // 使用上一行的左侧边线位置
                Right_Sideline[i] = Right_Sideline[i + 1]; // 使用上一行的右侧边线位置
                break; // 跳出循环，因为当前行被认为是无效的
            }

            // 左边线检测 从最长白列往左找线
            for(j = imgInfo.max_column; j > 1; j--)
            {
                /* 正常白黑黑跳变点 或者白黑白但距离上一次记录的左侧边线位置小于3列，则认为找到了左侧边线 因为用的压缩图像 可能会有噪点*///
                if((Image_Use[i][j] == white && Image_Use[i][j - 1] == black && Image_Use[i][j - 2] == black)//&& (i<= (imgInfo.top+10)||(i> (imgInfo.top+10)&&(j - Left_Sideline[i+1] < 30)))
                    || (Image_Use[i][j] == white && Image_Use[i][j - 1] == black && Image_Use[i][j - 2] == white && j - Left_Sideline[i+1] < 3))//
                {
                    Left_Sideline[i] = j; // 记录左侧边线的列号
                    Left_Sideline_flag[i] = 1; // 设置左侧边线标志为1（已检测到）
                    break; // 找到左侧边线后退出循环
                }
            }
            if(j == 1 && Left_Sideline_flag[i] == 0) // 如果遍历到第一列仍未检测到左侧边线
            {
                Left_Sideline[i] = 1; // 将左侧边线设置为第一列
                Left_Sideline_flag[i] = 0; // 左侧边线标志保持为0（未真实检测到）
                imgInfo.L_loselineSum ++; // 左侧丢失线计数器加1
            }

            for(j = imgInfo.max_column; j < LCDW_1 - 2; j++)//
            {
                if((Image_Use[i][j] == white && Image_Use[i][j + 1] == black && Image_Use[i][j + 2] == black)//&& (i<=(imgInfo.top+10)||(i> (imgInfo.top+10)&&(Right_Sideline[i + 1] - j < 30)))
                    || (Image_Use[i][j] == white && Image_Use[i][j + 1] == black && Image_Use[i][j + 2] == white && Right_Sideline[i + 1] - j < 3))
                {
                    Right_Sideline[i] = j; // 记录右侧边线的列号
                    Right_Sideline_flag[i] = 1; // 设置右侧边线标志为1（已检测到）
                    break; // 找到右侧边线后退出循环
                }
            }
            if(j == LCDW_1 - 2 && Right_Sideline_flag[i] == 0) // 如果遍历到最后一列前两列仍未检测到右侧边线
            {
                Right_Sideline[i] = LCDW_1 - 2; // 将右侧边线设置为最后一列前两列（通常是屏幕边界前的安全位置）
                Right_Sideline_flag[i] = 0; // 右侧边线标志保持为0（未真实检测到）
                imgInfo.R_loselineSum ++; // 右侧丢失线计数器加1
            }

            white_width[i] = Right_Sideline[i] - Left_Sideline[i]; // 计算并记录当前行的白色区域宽度
        }
      //  printf("Before Avoid: %d\n", Left_Sideline[40]);
//    }
}

void Find_left_Sideline(uint8 Start_row, uint8 End_row)
{
    imgInfo.L_loselineSum = 0; // 初始化左侧丢失线（未检测到边线）的计数器为0
    int i = 0, j = 0;

        for(i = Start_row; i >= End_row; i--) // 从起始行向结束行遍历（逆序）
        {
                       if(Image_Use[i][Left_Sideline[i+1]+7] == white)
            {
            Left_Sideline_flag[i] = 0; // 初始化当前行的左侧边线标志为0（未检测到）

            // if(Image_Use[i][max_column] == black) // 如果最长白列的当前行像素为黑色（可能是阴影或障碍物）
            // {
            //     Left_Sideline[i] = Left_Sideline[i + 1]; // 使用上一行的左侧边线位置
            //     break; // 跳出循环，因为当前行被认为是无效的
            // }

            // 左边线检测 从最长白列往左找线
 
            for(j = Left_Sideline[i+1]+7; j > 1; j--)
            {
                /* 正常白黑黑跳变点 或者白黑白但距离上一次记录的左侧边线位置小于3列，则认为找到了左侧边线 因为用的压缩图像 可能会有噪点*///
                if((Image_Use[i][j] == white && Image_Use[i][j - 1] == black && Image_Use[i][j - 2] == black)//&& (i<= (imgInfo.top+10)||(i> (imgInfo.top+10)&&(j - Left_Sideline[i+1] < 30)))
                    || (Image_Use[i][j] == white && Image_Use[i][j - 1] == black && Image_Use[i][j - 2] == white && j - Left_Sideline[i+1] < 3))//
                {
                    Left_Sideline[i] = j; // 记录左侧边线的列号
                    Left_Sideline_flag[i] = 1; // 设置左侧边线标志为1（已检测到）
                    break; // 找到左侧边线后退出循环
                }
            }
            if(j == 1 && Left_Sideline_flag[i] == 0) // 如果遍历到第一列仍未检测到左侧边线
            {
                Left_Sideline[i] = 1; // 将左侧边线设置为第一列
                Left_Sideline_flag[i] = 0; // 左侧边线标志保持为0（未真实检测到）
            }
           }

        }
        for(int i=imgInfo.bottom-1;i>=(imgInfo.top+ 1);i--)
        {
            if( Left_Sideline_flag[i] ==0)              
              imgInfo.L_loselineSum ++; // 左侧丢失线计数器加1
             white_width[i] = Right_Sideline[i] - Left_Sideline[i]; // 计算并记录当前行的白色区域宽度
        }
}

void Find_right_Sideline(uint8 Start_row, uint8 End_row)
{
    imgInfo.R_loselineSum = 0; // 初始化右侧丢失线计数器为0
    int i = 0, j = 0;

    for(i = Start_row; i >= End_row; i--) // 从起始行向结束行逆序遍历
    {
        // ===================== 修复 1：防止 j 为负数（最关键！）=====================
        int start_j = Right_Sideline[i+1] - 7;
        if (start_j < 0) start_j = 0;  // 强制 >=0，不越界

        if(Image_Use[i][start_j] == white)
        {
            Right_Sideline_flag[i] = 0;

            // ===================== 修复 2：安全起始点，不会负数 =====================
            for(j = start_j; j < LCDW_1 - 2; j++)
            {
                // ===================== 修复 3：右边线正确跳变判断 =====================
                if((Image_Use[i][j] == white && Image_Use[i][j + 1] == black && Image_Use[i][j + 2] == black)
                    || (Image_Use[i][j] == white && Image_Use[i][j + 1] == black && Image_Use[i][j + 2] == white && Right_Sideline[i + 1] - j < 3))
                {
                    Right_Sideline[i] = j;
                    Right_Sideline_flag[i] = 1;
                    break;
                }
            }

            // ===================== 修复 4：统一的未检测到边界处理 =====================
            if(j >= LCDW_1 - 2 && Right_Sideline_flag[i] == 0)
            {
                Right_Sideline[i] = LCDW_1 - 2;
                Right_Sideline_flag[i] = 0;
            }
        }
    }

    // ===================== 修复 5：丢行统计 移到外面（和左边完全一致）=====================
    for(int i = imgInfo.bottom - 1; i >= imgInfo.top + 1; i--)
    {
        if(Right_Sideline_flag[i] == 0)
            imgInfo.R_loselineSum ++;

        white_width[i] = Right_Sideline[i] - Left_Sideline[i];
    }
}

struct Guaidian L_l_guai, L_h_guai, R_l_guai, R_h_guai;  //拐点信息结构体
int Guai_row = 0;

void Find_Guaidian(void)
{
    L_l_guai.flag = 0;
    L_h_guai.flag = 0;
    R_l_guai.flag = 0;
    R_h_guai.flag = 0;


    if(L_l_guai.flag == 0)
        L_l_guai.row = LCDH_1 - 1;
    if(R_l_guai.flag == 0)
        R_l_guai.row = LCDH_1 - 1;

    //找左上，右上拐点
    for(int i = imgInfo.top + 2; i <= imgInfo.bottom - 5; i++)
    {

        /**************左上拐点**************/
        if(
             Left_Sideline[i] > 3 &&
             (
                  /*    *0
                      0100
                           */
                 (Image_Use[i][Left_Sideline[i] + 0] && Image_Use[i + 1][Left_Sideline[i] + 0]
                  && Image_Use[i + 1][Left_Sideline[i] - 1] && !Image_Use[i + 1][Left_Sideline[i] - 2] && Image_Use[i + 1][Left_Sideline[i] - 3])
                  ||
                  /*    *0
                      0000
                           */
                  (Image_Use[i][Left_Sideline[i] + 0] && Image_Use[i + 1][Left_Sideline[i] + 0] && Image_Use[i + 1][Left_Sideline[i] - 1]
                   && Image_Use[i + 1][Left_Sideline[i] - 2] && Image_Use[i + 1][Left_Sideline[i] - 3])
                  ||
                  /*    *0
                      0010
                           */
                   (Image_Use[i][Left_Sideline[i] + 0] && Image_Use[i + 1][Left_Sideline[i] + 0] && !Image_Use[i + 1][Left_Sideline[i] - 1]
                    && Image_Use[i + 1][Left_Sideline[i] - 2] && Image_Use[i + 1][Left_Sideline[i] - 3])
                  ||
                  /*    *0
                      00010
                           */
                   (Image_Use[i][Left_Sideline[i] + 0]&& Image_Use[i + 1][Left_Sideline[i] + 1] && !Image_Use[i + 1][Left_Sideline[i] + 0] && Image_Use[i + 1][Left_Sideline[i] - 1]
                    && Image_Use[i + 1][Left_Sideline[i] - 2] && Image_Use[i + 1][Left_Sideline[i] - 3])
//                   ||/*    *0
//                          01000
//                            */
// (Image_Use[i][Left_Sideline[i] + 0] && Image_Use[i + 1][Left_Sideline[i] + 0]
// && Image_Use[i + 1][Left_Sideline[i] - 1] && !Image_Use[i + 1][Left_Sideline[i] - 3]
// && Image_Use[i + 1][Left_Sideline[i] - 2])
               
             )
             && white_width[i + 2] - white_width[i - 0] > 12
             && xielv_sideline(i, Left_Sideline[i], i + 1, Left_Sideline[i + 1], 'k') < 1
             && white_width[i + 3] > white_width[i] && white_width[i + 4] > white_width[i]
             //&& abs(Left_Sideline[i - 1] - Left_Sideline[i + 1]) > 8
//             && abs(xielv_sideline(i, Left_Sideline[i], i - 3, Left_Sideline[i - 3], 'k') - xielv_sideline(i, Left_Sideline[i], i + 2, Left_Sideline[i + 2], 'k')) > 0.5
//             && xielv_sideline(i, Left_Sideline[i], i + 2, Left_Sideline[i + 2], 'k') < 1
             && L_h_guai.flag == 0
             && i < L_l_guai.row
        )
        {
            L_h_guai.row = i + 1;
            L_h_guai.column = (uint8)Left_Sideline[i];
            L_h_guai.flag = 1;

//            if(R_h_guai.flag)
//                break;

        }

        /**************右上拐点**************/
        if(
             Right_Sideline[i] < LCDW_1 - 3 &&
             (
                   /*  0*
                       0010
                            */
                  (Image_Use[i][Right_Sideline[i] - 0] && Image_Use[i + 1][Right_Sideline[i] - 0] && Image_Use[i + 1][Right_Sideline[i] + 1]
                   && !Image_Use[i + 1][Right_Sideline[i] + 2] && Image_Use[i + 1][Right_Sideline[i] + 3])
                   ||
                   /*  0*
                       0000
                            */
                  (Image_Use[i][Right_Sideline[i] - 0] && Image_Use[i + 1][Right_Sideline[i] - 0] && Image_Use[i + 1][Right_Sideline[i] + 1]
                   && Image_Use[i + 1][Right_Sideline[i] + 2] && Image_Use[i + 1][Right_Sideline[i] + 3])
                   ||
                   /*  0*   
                       0100
                            */
                  (Image_Use[i][Right_Sideline[i] - 0] && Image_Use[i + 1][Right_Sideline[i] - 0] && !Image_Use[i + 1][Right_Sideline[i] + 1]
                   && Image_Use[i + 1][Right_Sideline[i] + 2] && Image_Use[i + 1][Right_Sideline[i] + 3])
                   ||
                   /*  0*
                      01000
                            */
                  (Image_Use[i][Right_Sideline[i] - 0] && Image_Use[i + 1][Right_Sideline[i] - 1] && !Image_Use[i + 1][Right_Sideline[i] - 0] && Image_Use[i + 1][Right_Sideline[i] + 1]
                   && Image_Use[i + 1][Right_Sideline[i] + 2] && Image_Use[i + 1][Right_Sideline[i] + 3])
//                    ||/*    0*
//                          00010
//                     */
// (Image_Use[i][Right_Sideline[i] - 0] && Image_Use[i + 1][Right_Sideline[i] - 0]
// && Image_Use[i + 1][Right_Sideline[i] + 1] && !Image_Use[i + 1][Right_Sideline[i] + 3]
// && Image_Use[i + 1][Right_Sideline[i] + 2])

             )
             && white_width[i + 2] - white_width[i - 0] > 12
             && white_width[i + 3] > white_width[i] && white_width[i + 4] > white_width[i]
             //&& abs(Right_Sideline[i - 1] - Right_Sideline[i + 1]) > 8
//             && abs(xielv_sideline(i, Right_Sideline[i], i - 3, Right_Sideline[i - 3], 'k') - xielv_sideline(i, Right_Sideline[i], i + 2, Right_Sideline[i + 2], 'k')) > 0.5
             && xielv_sideline(i, Right_Sideline[i], i + 1, Right_Sideline[i + 1], 'k') > -1
             && R_h_guai.flag == 0
             && i < R_l_guai.row
           )
        {
            R_h_guai.row = i + 1;
            R_h_guai.column = (uint8)Right_Sideline[i];
            R_h_guai.flag = 1;

//            if(L_h_guai.flag)
//                break;
        }

    }


//    else if(Mode_1)
//    {
        //找左下，右下拐点
        for(int i = imgInfo.bottom - 3; i >= imgInfo.top + 3 && Flag.Huandao_L != 3 && Flag.Huandao_R != 3; i--)
        {
            if(L_h_guai.flag == 0)
                L_h_guai.row = 1;
            if(R_h_guai.flag == 0)
                R_h_guai.row = 1;
            /**************左下拐点**************/
            if(
                 Left_Sideline[i] > 3 &&
                 (
                      /*  0000
                            *0
                                */
                     (Image_Use[i][Left_Sideline[i] +0] && Image_Use[i - 1][Left_Sideline[i] + 0]
                      && Image_Use[i - 1][Left_Sideline[i] - 1] && Image_Use[i - 1][Left_Sideline[i] - 2]  && Image_Use[i - 1][Left_Sideline[i] -3])
                     ||
                     /*  0100
                           *0
                               */
                     (Image_Use[i][Left_Sideline[i] + 0] && Image_Use[i - 1][Left_Sideline[i] + 0] && Image_Use[i - 1][Left_Sideline[i] - 1]
                      && !Image_Use[i - 1][Left_Sideline[i] - 2] && Image_Use[i - 1][Left_Sideline[i] - 3])
                      ||
                      /*  0010
                            *0
                                */
                      (Image_Use[i][Left_Sideline[i] + 0] && Image_Use[i - 1][Left_Sideline[i] + 0] && !Image_Use[i - 1][Left_Sideline[i] - 1]
                       && Image_Use[i - 1][Left_Sideline[i] - 2] && Image_Use[i - 1][Left_Sideline[i] - 3])
                 )
                 /*上面三行的白行都比这一行多*/
                 && white_width[i - 2] - white_width[i + 1] > 10
                 && white_width[i - 2] > white_width[i] && white_width[i - 3] > white_width[i] && white_width[i - 4] > white_width[i]
                 && i > L_h_guai.row
                 && L_l_guai.flag == 0
            )
            {
                L_l_guai.row = i - 1;
                L_l_guai.column = (uint8)Left_Sideline[i];
                L_l_guai.flag = 1;

//                if(R_l_guai.flag)
//                    break;

            }

            /**************右下拐点**************/
            if(
                 Right_Sideline[i] < LCDW_1 - 3 &&
                 (
                         /*
                              0000
                              0*
                          */
                         (Image_Use[i][Right_Sideline[i] - 0] && Image_Use[i - 1][Right_Sideline[i] - 0]
                         && Image_Use[i - 1][Right_Sideline[i] + 1] && Image_Use[i - 1][Right_Sideline[i] + 2] && Image_Use[i - 1][Right_Sideline[i] + 3])
                         ||
                         /*
                              0010
                              0*
                          */
                         (Image_Use[i][Right_Sideline[i] - 0] && Image_Use[i - 1][Right_Sideline[i] - 0] && Image_Use[i - 1][Right_Sideline[i] + 1]
                          && !Image_Use[i - 1][Right_Sideline[i] + 2] && Image_Use[i - 1][Right_Sideline[i] + 3])
                          ||
                          /*
                               0001
                               0*
                           */
                          (Image_Use[i][Right_Sideline[i] - 0] && Image_Use[i - 1][Right_Sideline[i] - 0] && Image_Use[i - 1][Right_Sideline[i] + 1]
                           && Image_Use[i - 1][Right_Sideline[i] + 2] && !Image_Use[i - 1][Right_Sideline[i] + 3])
                 )
                 /*上面三行的白行都比这一行多*/
                 && white_width[i - 2] > white_width[i] && white_width[i - 3] > white_width[i]
                 && white_width[i - 2] - white_width[i + 1] > 10
//                 && abs(xielv_sideline(i, Right_Sideline[i], i - 3, Right_Sideline[i - 3], 'k') - xielv_sideline(i, Right_Sideline[i], i + 2, Right_Sideline[i + 2], 'k')) > 0.5
//                 && xielv_sideline(i, Right_Sideline[i], i - 3, Right_Sideline[i - 3], 'k') > 1
                 && i > R_h_guai.row
                 && R_l_guai.flag == 0
            )
            {
                R_l_guai.row = i - 1;
                R_l_guai.column = (uint8)Right_Sideline[i];
                R_l_guai.flag = 1;

//                if(L_l_guai.flag)
//                    break;
            }
        }
    
//    }
int l_guai_max=0,r_guai_max=93;//,l_guai_line,r_guai_line
if(Flag.Huandao_R == 4)
{
    for(int i = imgInfo.top + 2; i <54; i++)
    {
            if(Left_Sideline[i]> l_guai_max)//
            {
                l_guai_max = Left_Sideline[i];
                L_l_guai.row = i ;
            }
    }

                L_l_guai.column = (uint8)Left_Sideline[L_l_guai.row]+3;
                L_l_guai.flag = 1;
}
if(Flag.Huandao_L == 4)
{
    for(int i = imgInfo.top + 2; i <54; i++)
    {
            if(Right_Sideline[i]< r_guai_max)//
            {
                r_guai_max = Right_Sideline[i];
                R_l_guai.row = i ;
            }
    }

                R_l_guai.column = (uint8)Right_Sideline[R_l_guai.row]-3;
                R_l_guai.flag = 1;
}
        // for(int i = imgInfo.bottom - 3; i >= imgInfo.top + 3 && Flag.Huandao_L != 3 && Flag.Huandao_R != 3; i--)



}


void Find_Guaidian1(void)
{
                        // printf("Fla:%d\n", white_width[37 + 2] - white_width[37 - 0]);
    L_l_guai.flag = 0;
    L_h_guai.flag = 0;
    R_l_guai.flag = 0;
    R_h_guai.flag = 0;


    if(L_l_guai.flag == 0)
        L_l_guai.row = LCDH_1 - 1;
    if(R_l_guai.flag == 0)
        R_l_guai.row = LCDH_1 - 1;

    //找左上，右上拐点
    for(int i = imgInfo.top + 2; i <= imgInfo.bottom - 5; i++)
    {

        /**************左上拐点**************/
        if(
             Left_Sideline[i] > 3 &&
             (
                  /*    *0
                      0100
                           */
                 (Image_Use[i][Left_Sideline[i] + 0] && Image_Use[i + 1][Left_Sideline[i] + 0]
                  && Image_Use[i + 1][Left_Sideline[i] - 1] && !Image_Use[i + 1][Left_Sideline[i] - 2] && Image_Use[i + 1][Left_Sideline[i] - 3])
                  ||
                  /*    *0
                      0000
                           */
                  (Image_Use[i][Left_Sideline[i] + 0] && Image_Use[i + 1][Left_Sideline[i] + 0] && Image_Use[i + 1][Left_Sideline[i] - 1]
                   && Image_Use[i + 1][Left_Sideline[i] - 2] && Image_Use[i + 1][Left_Sideline[i] - 3])
                  ||
                  /*    *0
                      0010
                           */
                   (Image_Use[i][Left_Sideline[i] + 0] && Image_Use[i + 1][Left_Sideline[i] + 0] && !Image_Use[i + 1][Left_Sideline[i] - 1]
                    && Image_Use[i + 1][Left_Sideline[i] - 2] && Image_Use[i + 1][Left_Sideline[i] - 3])
                  ||
                  /*    *0
                      00010
                           */
                   (Image_Use[i][Left_Sideline[i] + 0]&& Image_Use[i + 1][Left_Sideline[i] + 1] && !Image_Use[i + 1][Left_Sideline[i] + 0] && Image_Use[i + 1][Left_Sideline[i] - 1]
                    && Image_Use[i + 1][Left_Sideline[i] - 2] && Image_Use[i + 1][Left_Sideline[i] - 3])
//                   ||/*    *0
//                          01000
//                            */
// (Image_Use[i][Left_Sideline[i] + 0] && Image_Use[i + 1][Left_Sideline[i] + 0]
// && Image_Use[i + 1][Left_Sideline[i] - 1] && !Image_Use[i + 1][Left_Sideline[i] - 3]
// && Image_Use[i + 1][Left_Sideline[i] - 2])
               
             )
             && white_width[i + 2] - white_width[i - 0] > 12
             && xielv_sideline(i, Left_Sideline[i], i + 1, Left_Sideline[i + 1], 'k') < 1
             && white_width[i + 3] > white_width[i] && white_width[i + 4] > white_width[i]
             //&& abs(Left_Sideline[i - 1] - Left_Sideline[i + 1]) > 8
//             && abs(xielv_sideline(i, Left_Sideline[i], i - 3, Left_Sideline[i - 3], 'k') - xielv_sideline(i, Left_Sideline[i], i + 2, Left_Sideline[i + 2], 'k')) > 0.5
//             && xielv_sideline(i, Left_Sideline[i], i + 2, Left_Sideline[i + 2], 'k') < 1
            //  && L_h_guai.flag == 0
            //  && i < L_l_guai.row
        )
        {
            L_h_guai.row = i + 1;
            L_h_guai.column = (uint8)Left_Sideline[i];
            L_h_guai.flag = 1;

//            if(R_h_guai.flag)
//                break;

        }

        /**************右上拐点**************/
        if(
             Right_Sideline[i] < LCDW_1 - 3 &&
             (
                   /*  0*
                       0010
                            */
                  (Image_Use[i][Right_Sideline[i] - 0] && Image_Use[i + 1][Right_Sideline[i] - 0] && Image_Use[i + 1][Right_Sideline[i] + 1]
                   && !Image_Use[i + 1][Right_Sideline[i] + 2] && Image_Use[i + 1][Right_Sideline[i] + 3])
                   ||
                   /*  0*
                       0000
                            */
                  (Image_Use[i][Right_Sideline[i] - 0] && Image_Use[i + 1][Right_Sideline[i] - 0] && Image_Use[i + 1][Right_Sideline[i] + 1]
                   && Image_Use[i + 1][Right_Sideline[i] + 2] && Image_Use[i + 1][Right_Sideline[i] + 3])
                   ||
                   /*  0*   
                       0100
                            */
                  (Image_Use[i][Right_Sideline[i] - 0] && Image_Use[i + 1][Right_Sideline[i] - 0] && !Image_Use[i + 1][Right_Sideline[i] + 1]
                   && Image_Use[i + 1][Right_Sideline[i] + 2] && Image_Use[i + 1][Right_Sideline[i] + 3])
                   ||
                   /*  0*
                      01000
                            */
                  (Image_Use[i][Right_Sideline[i] - 0] && Image_Use[i + 1][Right_Sideline[i] - 1] && !Image_Use[i + 1][Right_Sideline[i] - 0] && Image_Use[i + 1][Right_Sideline[i] + 1]
                   && Image_Use[i + 1][Right_Sideline[i] + 2] && Image_Use[i + 1][Right_Sideline[i] + 3])
//                    ||/*    0*
//                          00010
//                     */
// (Image_Use[i][Right_Sideline[i] - 0] && Image_Use[i + 1][Right_Sideline[i] - 0]
// && Image_Use[i + 1][Right_Sideline[i] + 1] && !Image_Use[i + 1][Right_Sideline[i] + 3]
// && Image_Use[i + 1][Right_Sideline[i] + 2])

             )
             && white_width[i + 2] - white_width[i - 0] > 12
             && white_width[i + 3] > white_width[i] && white_width[i + 4] > white_width[i]
             //&& abs(Right_Sideline[i - 1] - Right_Sideline[i + 1]) > 8
//             && abs(xielv_sideline(i, Right_Sideline[i], i - 3, Right_Sideline[i - 3], 'k') - xielv_sideline(i, Right_Sideline[i], i + 2, Right_Sideline[i + 2], 'k')) > 0.5
             && xielv_sideline(i, Right_Sideline[i], i + 1, Right_Sideline[i + 1], 'k') > -1
            //  && R_h_guai.flag == 0
            //  && i < R_l_guai.row
           )
        {
            R_h_guai.row = i + 1;
            R_h_guai.column = (uint8)Right_Sideline[i];
            R_h_guai.flag = 1;

//            if(L_h_guai.flag)
//                break;
        }

    }


//    else if(Mode_1)
//    {
        //找左下，右下拐点
        for(int i = imgInfo.bottom - 3; i >= imgInfo.top + 3 && Flag.Huandao_L != 3 && Flag.Huandao_R != 3; i--)
        {
            if(L_h_guai.flag == 0)
                L_h_guai.row = 1;
            if(R_h_guai.flag == 0)
                R_h_guai.row = 1;
            /**************左下拐点**************/
            if(
                 Left_Sideline[i] > 3 &&
                 (
                      /*  0000
                            *0
                                */
                     (Image_Use[i][Left_Sideline[i] +0] && Image_Use[i - 1][Left_Sideline[i] + 0]
                      && Image_Use[i - 1][Left_Sideline[i] - 1] && Image_Use[i - 1][Left_Sideline[i] - 2]  && Image_Use[i - 1][Left_Sideline[i] -3])
                     ||
                     /*  0100
                           *0
                               */
                     (Image_Use[i][Left_Sideline[i] + 0] && Image_Use[i - 1][Left_Sideline[i] + 0] && Image_Use[i - 1][Left_Sideline[i] - 1]
                      && !Image_Use[i - 1][Left_Sideline[i] - 2] && Image_Use[i - 1][Left_Sideline[i] - 3])
                      ||
                      /*  0010
                            *0
                                */
                      (Image_Use[i][Left_Sideline[i] + 0] && Image_Use[i - 1][Left_Sideline[i] + 0] && !Image_Use[i - 1][Left_Sideline[i] - 1]
                       && Image_Use[i - 1][Left_Sideline[i] - 2] && Image_Use[i - 1][Left_Sideline[i] - 3])
                 )
                 /*上面三行的白行都比这一行多*/
                 && white_width[i - 2] - white_width[i + 1] > 5
                 && white_width[i - 2] > white_width[i] && white_width[i - 3] > white_width[i] && white_width[i - 4] > white_width[i]
                 && i > L_h_guai.row
                 && L_l_guai.flag == 0
            )
            {
                L_l_guai.row = i - 1;
                L_l_guai.column = (uint8)Left_Sideline[i];
                L_l_guai.flag = 1;

//                if(R_l_guai.flag)
//                    break;

            }

            /**************右下拐点**************/
            if(
                 Right_Sideline[i] < LCDW_1 - 3 &&
                 (
                         /*
                              0000
                              0*
                          */
                         (Image_Use[i][Right_Sideline[i] - 0] && Image_Use[i - 1][Right_Sideline[i] - 0]
                         && Image_Use[i - 1][Right_Sideline[i] + 1] && Image_Use[i - 1][Right_Sideline[i] + 2] && Image_Use[i - 1][Right_Sideline[i] + 3])
                         ||
                         /*
                              0010
                              0*
                          */
                         (Image_Use[i][Right_Sideline[i] - 0] && Image_Use[i - 1][Right_Sideline[i] - 0] && Image_Use[i - 1][Right_Sideline[i] + 1]
                          && !Image_Use[i - 1][Right_Sideline[i] + 2] && Image_Use[i - 1][Right_Sideline[i] + 3])
                          ||
                          /*
                               0001
                               0*
                           */
                          (Image_Use[i][Right_Sideline[i] - 0] && Image_Use[i - 1][Right_Sideline[i] - 0] && Image_Use[i - 1][Right_Sideline[i] + 1]
                           && Image_Use[i - 1][Right_Sideline[i] + 2] && !Image_Use[i - 1][Right_Sideline[i] + 3])
                 )
                 /*上面三行的白行都比这一行多*/
                 && white_width[i - 2] > white_width[i] && white_width[i - 3] > white_width[i]
                 && white_width[i - 2] - white_width[i + 1] > 5
//                 && abs(xielv_sideline(i, Right_Sideline[i], i - 3, Right_Sideline[i - 3], 'k') - xielv_sideline(i, Right_Sideline[i], i + 2, Right_Sideline[i + 2], 'k')) > 0.5
//                 && xielv_sideline(i, Right_Sideline[i], i - 3, Right_Sideline[i - 3], 'k') > 1
                 && i > R_h_guai.row
                 && R_l_guai.flag == 0
            )
            {
                R_l_guai.row = i - 1;
                R_l_guai.column = (uint8)Right_Sideline[i];
                R_l_guai.flag = 1;

//                if(L_l_guai.flag)
//                    break;
            }
        }
    
//    }
int l_guai_max=0,r_guai_max=93;//,l_guai_line,r_guai_line
if(Flag.Huandao_R == 4)
{
    for(int i = imgInfo.top + 2; i <54; i++)
    {
            if(Left_Sideline[i]> l_guai_max)//
            {
                l_guai_max = Left_Sideline[i];
                L_l_guai.row = i ;
            }
    }

                L_l_guai.column = (uint8)Left_Sideline[L_l_guai.row]+3;
                L_l_guai.flag = 1;
}
if(Flag.Huandao_L == 4)
{
    for(int i = imgInfo.top + 2; i <54; i++)
    {
            if(Right_Sideline[i]< r_guai_max)//
            {
                r_guai_max = Right_Sideline[i];
                R_l_guai.row = i ;
            }
    }

                R_l_guai.column = (uint8)Right_Sideline[R_l_guai.row]-3;
                R_l_guai.flag = 1;
}
        // for(int i = imgInfo.bottom - 3; i >= imgInfo.top + 3 && Flag.Huandao_L != 3 && Flag.Huandao_R != 3; i--)



}


struct Guaidian  L_h_guai1,  R_h_guai1;  //拐点信息结构体

void Find_l_h_Guaidian(void)
{
    L_h_guai1.flag = 0;
    R_h_guai1.flag = 0;
            L_h_guai1.row = 0;
            // L_h_guai1.column =0;
    //找左上，右上拐点
    for(int i = imgInfo.top + 2; i <= imgInfo.bottom - 5; i++)
    {

        /**************左上拐点**************/
        if(
             Left_Sideline[i] > 3 &&
             (
                  /*    *0
                      0100
                           */
                 (Image_Use[i][Left_Sideline[i] + 0] && Image_Use[i + 1][Left_Sideline[i] + 0]
                  && Image_Use[i + 1][Left_Sideline[i] - 1] && !Image_Use[i + 1][Left_Sideline[i] - 2] && Image_Use[i + 1][Left_Sideline[i] - 3])
                  ||
                  /*    *0
                      0000
                           */
                  (Image_Use[i][Left_Sideline[i] + 0] && Image_Use[i + 1][Left_Sideline[i] + 0] && Image_Use[i + 1][Left_Sideline[i] - 1]
                   && Image_Use[i + 1][Left_Sideline[i] - 2] && Image_Use[i + 1][Left_Sideline[i] - 3])
                  ||
                  /*    *0
                      0010
                           */
                   (Image_Use[i][Left_Sideline[i] + 0] && Image_Use[i + 1][Left_Sideline[i] + 0] && !Image_Use[i + 1][Left_Sideline[i] - 1]
                    && Image_Use[i + 1][Left_Sideline[i] - 2] && Image_Use[i + 1][Left_Sideline[i] - 3])
                  ||
                  /*    *0
                      00010
                           */
                   (Image_Use[i][Left_Sideline[i] + 0]&& Image_Use[i + 1][Left_Sideline[i] + 1] && !Image_Use[i + 1][Left_Sideline[i] + 0] && Image_Use[i + 1][Left_Sideline[i] - 1]
                    && Image_Use[i + 1][Left_Sideline[i] - 2] && Image_Use[i + 1][Left_Sideline[i] - 3])
             )
             && white_width[i + 2] - white_width[i - 0] > 9
             && xielv_sideline(i, Left_Sideline[i], i + 1, Left_Sideline[i + 1], 'k') < 1
             && white_width[i + 3] > white_width[i] && white_width[i + 4] > white_width[i]
             //&& abs(Left_Sideline[i - 1] - Left_Sideline[i + 1]) > 8
//             && abs(xielv_sideline(i, Left_Sideline[i], i - 3, Left_Sideline[i - 3], 'k') - xielv_sideline(i, Left_Sideline[i], i + 2, Left_Sideline[i + 2], 'k')) > 0.5
//             && xielv_sideline(i, Left_Sideline[i], i + 2, Left_Sideline[i + 2], 'k') < 1
             && L_h_guai1.flag == 0
             && i < L_l_guai.row
        )
        {
            L_h_guai1.row = i + 1;
            L_h_guai1.column = (uint8)Left_Sideline[i];
            L_h_guai1.flag = 1;


        }


    }
}


void Find_r_h_Guaidian(void)
{
    R_h_guai1.flag = 0;
    L_h_guai1.flag = 0;
            R_h_guai1.row = 0;
            // R_h_guai1.column =0;
    //找左上，右上拐点
    for(int i = imgInfo.top + 2; i <= imgInfo.bottom - 5; i++)
    {
        /**************右上拐点**************/
        if(
             Right_Sideline[i] < LCDW_1 - 3 &&
             (
                   /*  0*
                       0010
                            */
                  (Image_Use[i][Right_Sideline[i] - 0] && Image_Use[i + 1][Right_Sideline[i] - 0] && Image_Use[i + 1][Right_Sideline[i] + 1]
                   && !Image_Use[i + 1][Right_Sideline[i] + 2] && Image_Use[i + 1][Right_Sideline[i] + 3])
                   ||
                   /*  0*
                       0000
                            */
                  (Image_Use[i][Right_Sideline[i] - 0] && Image_Use[i + 1][Right_Sideline[i] - 0] && Image_Use[i + 1][Right_Sideline[i] + 1]
                   && Image_Use[i + 1][Right_Sideline[i] + 2] && Image_Use[i + 1][Right_Sideline[i] + 3])
                   ||
                   /*  0*
                       0100
                            */
                  (Image_Use[i][Right_Sideline[i] - 0] && Image_Use[i + 1][Right_Sideline[i] - 0] && !Image_Use[i + 1][Right_Sideline[i] + 1]
                   && Image_Use[i + 1][Right_Sideline[i] + 2] && Image_Use[i + 1][Right_Sideline[i] + 3])
                   ||
                   /*  0*
                      01000
                            */
                  (Image_Use[i][Right_Sideline[i] - 0] && Image_Use[i + 1][Right_Sideline[i] - 1] && !Image_Use[i + 1][Right_Sideline[i] - 0] && Image_Use[i + 1][Right_Sideline[i] + 1]
                   && Image_Use[i + 1][Right_Sideline[i] + 2] && Image_Use[i + 1][Right_Sideline[i] + 3])
             )
             && white_width[i + 2] - white_width[i - 0] > 9
             && white_width[i + 3] > white_width[i] && white_width[i + 4] > white_width[i]
             //&& abs(Right_Sideline[i - 1] - Right_Sideline[i + 1]) > 8
//             && abs(xielv_sideline(i, Right_Sideline[i], i - 3, Right_Sideline[i - 3], 'k') - xielv_sideline(i, Right_Sideline[i], i + 2, Right_Sideline[i + 2], 'k')) > 0.5
             && xielv_sideline(i, Right_Sideline[i], i + 1, Right_Sideline[i + 1], 'k') > -1
             && R_h_guai1.flag == 0
             && i < R_l_guai.row
           )
        {
            R_h_guai1.row = i + 1;
            R_h_guai1.column = (uint8)Right_Sideline[i];
            R_h_guai1.flag = 1;

//            if(L_h_guai.flag)
//                break;
        }


    }
}

/***************************************************找中线**********************************************************/
void Find_Midline(){
//   if(imgInfo.L_loselineSum==0&&imgInfo.R_loselineSum==0){

    for(uint16 i = imgInfo.bottom-1;i>imgInfo.top;i--){
      Mid_Line[i] = (Left_Sideline[i] + Right_Sideline[i])/2;
    }
//   }
        for(uint16 i = imgInfo.bottom - 1; i > imgInfo.top; i--)
        {
            Last_Mid_Line[i] = Mid_Line[i];
        }
}


int top_white_num;
int right_num = 0,r_num,left_num = 0,l_num = 0,R_l_lsoe=0,L_l_lose=0;
int L_loseline_l ,L_loseline_h,R_loseline_l ,R_loseline_h;
float k1,kL,kR;
uint16_t maxkuan_line;
void straight_judge(void)
{

// if(L_h_guai.flag||R_l_guai.flag)
// {

// }
           maxkuan_line=15;
       uint16_t kuan_max = 0;
        for(int i = imgInfo.top;i < 52; i++)
        {
            if((uint16_t)white_width[i]> kuan_max&&white_width[i]<70)//
            {
                kuan_max = (uint16_t)white_width[i];
                maxkuan_line=(uint16_t)i;
            }
        }

        top_white_num=0;
    //    for(int i = imgInfo.top-2;i<imgInfo.top;i++)
    //    {
            for(int j = Left_Sideline[imgInfo.top + 2];j<Right_Sideline[imgInfo.top + 2];j++)
            {
                if(Image_Use[MAX(imgInfo.top-1,0)][j] == white)
                    top_white_num+=1;
            }
//        }
    int count=0;
    for(int i = imgInfo.bottom ;i>imgInfo.top+1;i--)
    {

        if(Left_Sideline_flag[i] == 0)
        {
            count++;
            if(count==1)
            L_loseline_h = i;
            L_loseline_l = i;
        }
        if(count==0)
        {
            L_loseline_h = 60;
            L_loseline_l = 60;
        }

    }
    count=0;
    for(int i = imgInfo.bottom ;i>imgInfo.top+1;i--)
    {

        if(Right_Sideline_flag[i] == 0)
        {
            count++;
            if(count==1)
            R_loseline_h = i;
            R_loseline_l = i;//最低点不更新
        }
        if(count==0)
        {
            R_loseline_h = 60;
            R_loseline_l = 60;
        }

    }
    r_num=0;
    float k = xielv_sideline(24, Right_Sideline[24], 34, Right_Sideline[34], 'k');
    float b = xielv_sideline(24, Right_Sideline[24], 34, Right_Sideline[34], 'b');

//    if(Flag.Huandao_R==5)
        kR=k;

        if(Flag.Huandao_R==6||Flag.Huandao_L==6)
        {
            for(int i = 18 ;i<45 ;i++)
            {
            if(abs(Right_Sideline[i]-k*i-b)>3)
            {
                r_num++;
            }
            }
        }

        else
        {
    for(int i = 12 ;i<36 ;i++)
    {   
    if(abs(Right_Sideline[i]-k*i-b)>3)
    {
        r_num++;
    }
    }
        }

    l_num=0;
     k = xielv_sideline(24, Left_Sideline[24], 34, Left_Sideline[34], 'k');
     b = xielv_sideline(24, Left_Sideline[24], 34, Left_Sideline[34], 'b');

//     if(Flag.Huandao_L==5)
     kL=k;

     if(Flag.Huandao_R==6||Flag.Huandao_L==6)
     {
         for(int i = 18;i<45 ;i++)
         {
             if(abs(Left_Sideline[i]-k*i-b)>3)
         {
             l_num++;
         }
         }
     }



     else
     {
    for(int i = 12 ;i<36 ;i++)
    {
        if(abs(Left_Sideline[i]-k*i-b)>3)
    {
        l_num++;
    }
    }
     }

    if(r_num<3)
    {
        imgInfo.R_straight_flag=1;
    }
    else
    {
        imgInfo.R_straight_flag=0;
    }
    if(l_num<3)
    {
        imgInfo.L_straight_flag=1;
    }
    else
    {
        imgInfo.L_straight_flag=0;
    }



       imgInfo.Both_lose=0;
       for(int i = imgInfo.top +5;i<=45 ;i++)
       {
           if(Left_Sideline_flag[i]==0&&Right_Sideline_flag[i]==0)
           {
                imgInfo.Both_lose++;
           }
       }
}


/***************************************************线性回归计算中线斜率**********************************************/
float B,A;
void regression(int startline,int endline)
{

  int i=0,SumX=0,SumY=0,SumLines = 0;
  float SumUp=0,SumDown=0,avrX=0,avrY=0;
  SumLines=endline-startline;   // startline 为开始行， //endline 结束行 //SumLines

  for(i=startline;i<endline;i++)
  {
    SumX+=Mid_Line[i];//列号（X坐标）
    SumY+=i;    //行号（Y坐标）
  }
  avrX=(float)SumX/SumLines;     //X平均值
  avrY=(float)SumY/SumLines;     //Y平均值
  SumUp=0;
  SumDown=0;
  for(i=startline;i<endline;i++)
  {
    SumUp+=(Mid_Line[i]-avrX)*(i-avrY);
    SumDown+=(i-avrY)*(i-avrY);
  }
  if(SumDown==0)
    B=0;
  else{
    B=(float)(SumUp/SumDown);
    A=avrX - B*avrY;  //截距
  }

}
/***************************************************曲率计算********************************************************/
float curvature;
void calculateCurvature(float x1,float y1,float x2,float y2,float x3,float y3){
  
  // 计算三角形的边长
  float a = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
  float b = sqrt(pow(x3 - x2, 2) + pow(y3 - y2, 2));
  float c = sqrt(pow(x3 - x1, 2) + pow(y3 - y1, 2));

  // 计算三角形的半周长
  float s = (a + b + c) / 2;

  // 计算三角形的面积（使用海伦公式）
  float area = sqrt(s * (s - a) * (s - b) * (s - c));

  // 计算外接圆半径
  float radius = (a * b * c) / (4 * area);

  // 计算曲率（曲率是半径的倒数）
  curvature =(float) 1 / radius*100;  

}

float Yaw_Huandao,Yaw_Huandao_err,yaw_correct,distance_HUAN1;//1m=35000

void Huandao_L_imu()
{
    if(Flag.Huandao_L >1)
    {
    huandao_yaw_correct();
    }
    if(Flag.Huandao_L == 0 && Flag.Huandao_R == 0&&Flag.picture!=2&&Flag.picture!=3&&Flag.picture!=4&&Flag.picture!=5)
    {

    if(imgInfo.top <= 12&&r_num<3&& imgInfo.L_loselineSum -imgInfo.R_loselineSum >0&&R_l_guai.flag==0&&R_h_guai.flag==0&&L_l_guai.flag==1&&L_l_guai.row>20&&imgInfo.Both_lose==0)//
    {
        distance_HUAN1=real_distance[L_l_guai.row]/100*66;
        Flag.Huandao_L = 1;

    }
    }
    /*左环岛*/
    /* 进入条件 左环岛标志位为0 右环岛标志位为0 左下拐点存在 右边丢线行数小于10行 左边丢线行数大于0行 右边没有拐点 截止行在图像较上面（前面不在弯道）*/
    /*此处受图像影响 待图像畸变较小时可以加上左上拐点的上面两行没有边线，便于识别*/


    if(Flag.Huandao_L == 1)
       {
           //进入左环岛判定后进行第二次左环岛判定，左边上下两个拐点都存在
           if(L_l_guai.flag == 1 && L_h_guai.flag == 1)
           {
               //要求左下拐点至少要在图像下方一点
               if(L_l_guai.row > LCDH_1 / 2)
               {
                   float k = xielv_sideline(L_l_guai.row, L_l_guai.column, L_h_guai.row, L_h_guai.column, 'k');
                   float b = xielv_sideline(L_l_guai.row, L_l_guai.column, L_h_guai.row, L_h_guai.column, 'b');


                   //从左下拐点遍历到左上拐点，通过斜率进行补线
                   for (int i = L_l_guai.row; i >= L_h_guai.row; i--)
                   {
                       if (!Image_Use[i][(int8)(k * i + b)])
                           continue;
                       else
                           Left_Sideline[i] = k * i + b;
                   }
                   Flag.Buxian = 1;
               }
               //若左下拐点在图像偏中上部分，则以左上拐点到图像下方第五列来进行斜率补线
               else if(L_l_guai.row <= LCDH_1 / 2)
               {
                   float k = xielv_sideline(L_h_guai.row, L_h_guai.column, LCDH_1 - 2, 5, 'k');
                   float b = xielv_sideline(L_h_guai.row, L_h_guai.column, LCDH_1 - 2, 5, 'b');


                   for (int i = LCDH_1 - 2; i >= L_h_guai.row; i--)
                   {

                       if(!Image_Use[i][(int8)(k * i + b)])
                           continue;
                       else
                           Left_Sideline[i] = k * i + b;
                   }
               }

           }
           //若进入了左环岛二级判断仍只有左下拐点
           else if(L_l_guai.flag == 1)
           {
               //从左下拐点到顶部边线进行补线
               float k = xielv_sideline(L_l_guai.row, L_l_guai.column, imgInfo.bottom - 1, Left_Sideline[imgInfo.bottom - 1], 'k');
               float b = xielv_sideline(L_l_guai.row, L_l_guai.column, imgInfo.bottom - 1, Left_Sideline[imgInfo.bottom - 1], 'b');

               for (int i = imgInfo.bottom - 1; i > imgInfo.top; i--)
               {
                   if (!Image_Use[i][(int8)(k * i + b)])
                       continue;
                   else
                       Left_Sideline[i] = k * i + b;

               }
               Flag.Buxian = 1;
           }
           else if(R_l_guai.flag == 1 || R_h_guai.flag == 1 && imgInfo.R_loselineSum > 25)//
           {
               Flag.Huandao_L = 0;
           }
           //若圆环二级判断左上左下拐点都没有
           else
           {
               //顶部到底部进行补线
               float k = xielv_sideline(imgInfo.top + 1, Left_Sideline[imgInfo.top + 1], LCDH_1 - 2, 5, 'k');
               float b = xielv_sideline(imgInfo.top + 1, Left_Sideline[imgInfo.top + 1], LCDH_1 - 2, 5, 'b');

               for (int i = LCDH_1 - 2; i >= imgInfo.top + 1; i--)
               {

                   if(!Image_Use[i][(int8)(k * i + b)])
                       continue;
                   else
                       Left_Sideline[i] = k * i + b;
               }
           }
           //若是左环岛二级判断底部三行左边线丢线了 且左边丢线行数大于25行，说明此时车身已经靠在圆环出口那里了
           if((Left_Sideline_flag[LCDH_1 - 4] == 0 && Left_Sideline_flag[LCDH_1 - 5] == 0) && imgInfo.L_loselineSum > 10&&distance>distance_HUAN1)
           {
               Flag.Huandao_L = 2;
               Yaw_Huandao=icm_data.yaw;
               distance=0;
               distance_HUAN1=0;
           }

       }

         if(Flag.Huandao_L == 2)
        {
              float k ;
              float b ;

              if( L_h_guai.flag == 1)
              {
              k = xielv_sideline(L_h_guai.row, L_h_guai.column, MIN(L_h_guai.row+5,57), Right_Sideline[MIN(L_h_guai.row+5,57)], 'k');
              b = xielv_sideline(L_h_guai.row, L_h_guai.column, MIN(L_h_guai.row+5,57), Right_Sideline[MIN(L_h_guai.row+5,57)], 'b');

             for (int i = imgInfo.bottom - 1; i > imgInfo.top; i--)
             {
                 if(!Image_Use[i][(int8)(k * i + b)])
                    continue;
                 else
                 {
                     for(int m = 0; m <= 7; m++)
                     {
                         if(((k * i + b) + m) >= LCDW_1 - 1)
                         {
                             break;
                         }
                         else
                             Image_Use[i][(int8)(k * i + b) + m] = black;
                     }
                 }
             }

             Flag.Buxian = 1;
              }
            if( L_l_guai.flag == 1)//L_h_guai.flag == 1 &&
            {

                k=-0.5;
                b=L_l_guai.column-k*L_l_guai.row;
                for (int i = imgInfo.bottom-1; i >= L_l_guai.row; i--)
                {

                    if(!Image_Use[i][(int8)(k * i + b)])
                        continue;
                    else
                    {
                        for(int m = 0; m >=-7; m--)
                        {
                            if(((k * i + b) + m) <=0)
                            {
                                break;
                            }
                            else
                                Image_Use[i][(int8)(k * i + b) + m] = black;
                        }
                    }
                }



                Flag.Buxian = 1;

            }


            else
            {
                float k = xielv_sideline(3, 40, LCDH_1 - 2, 5, 'k');
                float b = xielv_sideline(3, 40, LCDH_1 - 2, 5, 'b');

                for (int i = LCDH_1 - 2; i >= imgInfo.top + 1; i--)
                {

                    if(!Image_Use[i][(int8)(k * i + b)])
                        continue;
                    else
                        Left_Sideline[i] = k * i + b;
                }

                k = xielv_sideline(3 , 38, LCDH_1 - 1, LCDW_1 - 25, 'k');
                b = xielv_sideline(3 , 38, LCDH_1 - 1, LCDW_1 - 25, 'b');

                for (int i = LCDH_1 - 2; i >= imgInfo.top + 1; i--)
                {

                    if(!Image_Use[i][(int8)(k * i + b)])
                        continue;
                    else
                        Right_Sideline[i] = k * i + b;
                }

                    Flag.Buxian = 1;
                }


            if(((Left_Sideline_flag[LCDH_1 - 7] == 1 && Left_Sideline_flag[LCDH_1 - 8] == 1 
                    && L_h_guai.flag) )&&distance>10)//&&distance>10|| !L_l_guai.flag
            {
                Flag.Huandao_L = 3;
                distance=0;
            }   
            Get_ImageTop();
            Find_Sideline(imgInfo.bottom - 1, imgInfo.top + 1);
        }

         if(Flag.Huandao_L == 3)
            {
             if( L_h_guai.flag == 1)//L_h_guai.flag == 1 &&
             {
                    float k = xielv_sideline(L_h_guai.row, L_h_guai.column, MIN(L_h_guai.row+20,57),Right_Sideline[MIN(L_h_guai.row+20,57)], 'k');
                    float b = xielv_sideline(L_h_guai.row, L_h_guai.column, MIN(L_h_guai.row+20,57),Right_Sideline[MIN(L_h_guai.row+20,57)], 'b');

                    for (int i = imgInfo.bottom - 1; i > imgInfo.top; i--)
                    {
                        if(!Image_Use[i][(int8)(k * i + b)])
                           continue;
                        else
                        {
                            for(int m = 0; m <= 7; m++)
                            {
                                if(((k * i + b) + m) >= LCDW_1 - 1)
                                {
                                    break;
                                }
                                else
                                    Image_Use[i][(int8)(k * i + b) + m] = black;
                            }
                            Flag.Buxian = 1;
                        }
                    }
                }
            //  else if(  L_h_guai.flag == 0&&R_h_guai.flag == 1)//L_h_guai.flag == 1 &&
            //  {
            //         float k = xielv_sideline(R_h_guai.row, R_h_guai.column, imgInfo.bottom - 1, LCDW_1 - 20, 'k');
            //         float b = xielv_sideline(R_h_guai.row, R_h_guai.column, imgInfo.bottom - 1, LCDW_1 - 20, 'b');

            //         for (int i = imgInfo.bottom - 1; i > imgInfo.top; i--)
            //         {
            //             if(!Image_Use[i][(int8)(k * i + b)])
            //                continue;
            //             else
            //             {
            //                 for(int m = 0; m <= 7; m++)
            //                 {
            //                     if(((k * i + b) + m) >= LCDW_1 - 1)
            //                     {
            //                         break;
            //                     }
            //                     else
            //                         Image_Use[i][(int8)(k * i + b) + m] = black;
            //                 }
            //                 Flag.Buxian = 1;
            //             }
            //         }
            //     }
//             else
//             {
//                 Eerr_flag=1;
////                    float k = xielv_sideline(1, 0, imgInfo.bottom - 1, LCDW_1 - 20, 'k');
////                    float b = xielv_sideline(1, 0, imgInfo.bottom - 1, LCDW_1 - 20, 'b');
////
////                    for (int i = imgInfo.bottom - 1; i > imgInfo.top; i--)
////                    {
////                        if(!Image_Use[i][(int8)(k * i + b)])
////                           continue;
////                        else
////                        {
////                            for(int m = 0; m <= 7; m++)
////                            {
////                                if(((k * i + b) + m) >= LCDW_1 - 1)
////                                {
////                                    break;
////                                }
////                                else
////                                    Image_Use[i][(int8)(k * i + b) + m] = black;
////                            }
//                            Flag.Buxian = 1;
////                        }
////                    }
//                }
                //环岛3是直接把USE图像数组改了，所以重新找截止行来确定手动补线后的截止行和边线，这时找到的图像就是一个大弯拐进环岛
                Get_ImageTop();
                Find_Sideline(imgInfo.bottom - 1, imgInfo.top + 1);

                //进入弯道
                if(Yaw_Huandao_err<-105)//(distance>30)&&imgInfo.Both_lose==0
                {
                    Flag.Huandao_L = 4;
//                    distance=0;
                }
            }

             if(Flag.Huandao_L == 4)
            {
                //找到右下拐点，即说明已经到了快要出弯的地方
                if(R_l_guai.flag)
                {
                    uint8 temp = 0;
                    //找到右赛道的最顶的边线，从那个点补到右下拐点上
//                    for(int m = imgInfo.bottom - 4; m > imgInfo.top + 3; m--)
//                    {
//                        if(!Left_Sideline_flag[m + 1] && !Left_Sideline_flag[m + 2] && !Left_Sideline_flag[m + 3] && !Left_Sideline_flag[m + 4] && !Left_Sideline_flag[m + 5] && Left_Sideline_flag[m - 1] && Left_Sideline_flag[m - 2])
//                        {
//                            temp =(uint8) m;
//                            break;
//                        }
//                    }
//                    if(temp == 0)
//                    {
                        temp = imgInfo.top + 1;
//                    }
                    float k = xielv_sideline(R_l_guai.row, R_l_guai.column, temp, 30, 'k');
                    float b = xielv_sideline(R_l_guai.row, R_l_guai.column, temp, 30, 'b');

                    for (int i = R_l_guai.row; i > imgInfo.top; i--)
                    {
                        for(int m = 0; m <= 13; m++)
                        {
                            if(((k * i + b) + m) >= LCDW_1 - 1)
                            {
                                break;
                            }
                            else
                            {
                                Image_Use[i][(uint8)(k * i + b) + m] = black;
                            }

                        }
                    }
                    Flag.Buxian = 1;
                }
            //    else if((!R_l_guai.flag) && abs(Left_Sideline[imgInfo.top+2]-Right_Sideline[imgInfo.top+2])>30 )
            //    {
            //        uint8 temp = 0;

            //         for(int m = imgInfo.bottom - 4; m > imgInfo.top + 1; m--)
            //         {
            //             if(!Left_Sideline_flag[m + 1] && !Left_Sideline_flag[m + 2] && !Left_Sideline_flag[m + 3] && !Left_Sideline_flag[m + 4] && !Left_Sideline_flag[m + 5] && Left_Sideline_flag[m - 1])
            //             {
            //                 temp =(uint8) m;
            //                 break;
            //             }
            //         }
            //         if(temp == 0)
            //         {
            //             temp = imgInfo.top + 1;
            //         }
            //        float k = xielv_sideline(LCDH_1 - 2, LCDW_1 - 2, temp, 30, 'k');
            //        float b = xielv_sideline(LCDH_1 - 2, LCDW_1 - 2, temp, 30, 'b');

            //        for (int i = LCDH_1 - 2; i >= temp - 2; i--)
            //        {
            //            for(int m = 0; m <= 4; m++)
            //            {
            //                if(((k * i + b) + m) >= LCDW_1 - 1)
            //                {
            //                    break;
            //                }
            //                else
            //                {
            //                    Image_Use[i][(uint8)(k * i + b) + m] = black;
            //                }

            //            }
            //        }
            //        Flag.Buxian = 1;
            //    }
                else
                {
                    Flag.Buxian = 1;
                }
                //没有右下拐点但还有右丢线情况，说明车已经跨过右拐点那里，这是从m点补到右下角
                // if(
                //         R_l_guai.flag == 0 && imgInfo.R_loselineSum > 15 && !Right_Sideline_flag[LCDH_1 - 3] && !Right_Sideline_flag[LCDH_1 - 4]
                //         && !Right_Sideline_flag[LCDH_1 - 5] && !Right_Sideline_flag[LCDH_1 - 6]
                //         && !Right_Sideline_flag[LCDH_1 - 7] && !Right_Sideline_flag[LCDH_1 - 8]
                // )
                // {
                //     Flag.Huandao_L = 5;
                // }

                if(Yaw_Huandao_err<110&&Yaw_Huandao_err>100)//(L_h_guai.flag==1&&imgInfo.R_straight_flag==1)&&imgInfo.top<10
                {
                    Flag.Huandao_L = 5;
                }
                Get_ImageTop();
                Find_Sideline(imgInfo.bottom - 1, imgInfo.top + 1);
                imgInfo.R_loselineSum = 0;
            }


              if(Flag.Huandao_L == 5)
                 {
                     uint8 temp = 0;
                     //找圆弧内白到黑的交界点
                     for(int m = imgInfo.bottom - 4; m > imgInfo.top + 3; m--)
                     {
                         if(!Left_Sideline_flag[m + 1] && !Left_Sideline_flag[m + 2] && !Left_Sideline_flag[m + 3]
                            && !Left_Sideline_flag[m + 4] && !Left_Sideline_flag[m + 5] && Left_Sideline_flag[m - 1])
                         {
                             temp = (uint8)m;
                             break;
                         }
                     }
                     if(temp == 0)
                     {
                         temp = imgInfo.top + 3;
                     }
                     float k = xielv_sideline(LCDH_1 - 2, LCDW_1 - 2, temp, 40, 'k');
                     float b = xielv_sideline(LCDH_1 - 2, LCDW_1 - 2, temp, 40, 'b');

                     //只要有右丢线情况就进行补线
                     if(imgInfo.R_loselineSum > 1)
                     {
                         for (int i = LCDH_1 - 2; i >= temp - 2; i--)
                         {
                             for(int m = 0; m <= 4; m++)
                             {
                                 if(((k * i + b) + m) >= LCDW_1 - 1)
                                 {
                                     break;
                                 }
                                 else
                                     Image_Use[i][(uint8)(k * i + b) + m] = black;
                             }
                         }
                         Flag.Buxian = 1;
                     }

                     //直到右边没有丢线情况并且左边出现拐点只后
                     if(Yaw_Huandao_err<25)//(L_h_guai.flag==1&&imgInfo.R_straight_flag==1)&&imgInfo.top<10
                     {
                         Flag.Huandao_L = 6;
                         Flag.Buxian = 0;
                     }
                     Get_ImageTop();
                     Find_Sideline(imgInfo.bottom - 1, imgInfo.top + 1);
                 }

                  if(Flag.Huandao_L == 6)
                 {
                     if(L_h_guai.flag == 1)
                     {
                         //从左拐点向下补线，防止再次入环
                         float k = xielv_sideline(L_h_guai.row + 5, L_h_guai.column, LCDH_1 - 2, 8, 'k');
                         float b = xielv_sideline(L_h_guai.row + 5, L_h_guai.column, LCDH_1 - 2, 8, 'b');

                         for (int i = LCDH_1 - 2; i >= L_h_guai.row; i--)
                         {

                             if(!Image_Use[i][(uint8)(k * i + b)])
                                 continue;
                             else
                                 Left_Sideline[i] = k * i + b;
                         }
                     }
                     else if(L_h_guai.flag == 0)
                     {
                         float k = xielv_sideline(imgInfo.top + 10, Left_Sideline[imgInfo.top + 10], LCDH_1 - 2, 5, 'k');
                         float b = xielv_sideline(imgInfo.top + 10, Left_Sideline[imgInfo.top + 10], LCDH_1 - 2, 5, 'b');

                         for (int i = LCDH_1 - 2; i >= imgInfo.top + 1; i--)
                         {

                             if(!Image_Use[i][(uint8)(k * i + b)])
                                 continue;
                             else
                                 Left_Sideline[i] = k * i + b;
                         }
                     }
                     else
                         Flag.Buxian = 0;

                     if(imgInfo.L_straight_flag==1&&distance>=10&&L_h_guai.flag==0)//imgInfo.R_loselineSum < 1//Left_Sideline_flag[imgInfo.bottom - 6] && Left_Sideline_flag[imgInfo.bottom - 7] && Left_Sideline_flag[imgInfo.bottom - 8]&&
                     {
                         Flag.Huandao_L = 0;
                         distance=0;
                     }
                 }

                  if((imgInfo.L_straight_flag==1&& imgInfo.R_straight_flag==1)&&L_h_guai.flag==0&&L_l_guai.flag==0&&R_h_guai.flag==0&&R_l_guai.flag==0&&abs(imgInfo.R_loselineSum-imgInfo.L_loselineSum)==0)//(Flag.Huandao_R>4&&  imgInfo.R_straight_flag==1)||
                  {
                                               Flag.Huandao_L = 0;
                  }
//                 if(Flag.Huandao_L && imgInfo.L_loselineSum < 10 && imgInfo.R_loselineSum < 10)
//                 {
//                     for(int i = imgInfo.bottom - 1; i > imgInfo.top + 1; i--)
//                     {
//                         if(i > imgInfo.top)
//                         {
//                             //右边界不发生突变且呈直线状态
//                             if(Right_Sideline[i] - Right_Sideline[i - 1] < 3 && Right_Sideline[i] >= Right_Sideline[i - 1])
//                             {
//                                 right_num ++;
//                             }
//                             //左边界不发生突变且呈直线状态
//                             if(Left_Sideline[i] - Left_Sideline[i - 1] < 3 && Left_Sideline[i] >= Left_Sideline[i - 1])
//                             {
//                                 left_num ++;
//                             }
//                         }
//                     }
//
//                     if(right_num > 50 && left_num > 50)
//                     {
//                         Flag.Huandao_L = 0;
//                         left_num = 0;
//                         right_num = 0;
//
//                     }
//                     else
//                     {
//                         left_num = 0;
//                         right_num = 0;
//                     }
//                 }


}










void Huandao_R_imu()
{

    if(Flag.Huandao_R >1)
    {
        huandao_yaw_correct();
    }
    if(Flag.Huandao_R == 0 && Flag.Huandao_L == 0&&Flag.picture!=2&&Flag.picture!=3&&Flag.picture!=4&&Flag.picture!=5)
    {

    if(imgInfo.top <= 12&&l_num<3 && imgInfo.R_loselineSum -imgInfo.L_loselineSum >0&&L_l_guai.flag==0&&L_h_guai.flag==0&&R_l_guai.flag==1&&R_l_guai.row>20&&imgInfo.Both_lose==0)//
    {
        distance_HUAN1=real_distance[R_l_guai.row]/100*66;
        Flag.Huandao_R = 1;

    }
    }
    /*右环岛*/
    /* 进入条件 左环岛标志位为0 右环岛标志位为0 左下拐点存在 右边丢线行数小于10行 左边丢线行数大于0行 右边没有拐点 截止行在图像较上面（前面不在弯道）*/
    /*此处受图像影响 待图像畸变较小时可以加上左上拐点的上面两行没有边线，便于识别*/


    if(Flag.Huandao_R == 1)
       {
           //进入左环岛判定后进行第二次左环岛判定，左边上下两个拐点都存在
           if(R_l_guai.flag == 1 && R_h_guai.flag == 1)
           {
               //要求左下拐点至少要在图像下方一点
               if(R_l_guai.row > LCDH_1 / 2)
               {
                   float k = xielv_sideline(R_l_guai.row, R_l_guai.column, R_h_guai.row, R_h_guai.column, 'k');
                   float b = xielv_sideline(R_l_guai.row, R_l_guai.column, R_h_guai.row, R_h_guai.column, 'b');


                   //从左下拐点遍历到左上拐点，通过斜率进行补线
                   for (int i = R_l_guai.row; i >= R_h_guai.row; i--)
                   {
                       if (!Image_Use[i][(int8)(k * i + b)])
                           continue;
                       else
                           Right_Sideline[i] = k * i + b;
                   }
                   Flag.Buxian = 1;
               }
               //若左下拐点在图像偏中上部分，则以左上拐点到图像下方第五列来进行斜率补线
               else if(R_l_guai.row <= LCDH_1 / 2)
               {
                   float k = xielv_sideline(R_h_guai.row, R_h_guai.column, LCDH_1 - 2, 88, 'k');
                   float b = xielv_sideline(R_h_guai.row, R_h_guai.column, LCDH_1 - 2, 88, 'b');


                   for (int i = LCDH_1 - 2; i >= R_h_guai.row; i--)
                   {

                       if(!Image_Use[i][(int8)(k * i + b)])
                           continue;
                       else
                           Right_Sideline[i] = k * i + b;
                   }
               }

           }
           //若进入了左环岛二级判断仍只有左下拐点
           else if(R_l_guai.flag == 1)
           {
               //从左下拐点到顶部边线进行补线
               float k = xielv_sideline(R_l_guai.row, R_l_guai.column, imgInfo.bottom - 1, Right_Sideline[imgInfo.bottom - 1], 'k');
               float b = xielv_sideline(R_l_guai.row, R_l_guai.column, imgInfo.bottom - 1, Right_Sideline[imgInfo.bottom - 1], 'b');

               for (int i = imgInfo.bottom - 1; i > imgInfo.top; i--)
               {
                   if (!Image_Use[i][(int8)(k * i + b)])
                       continue;
                   else
                       Right_Sideline[i] = k * i + b;

               }
               Flag.Buxian = 1;
           }
           else if(L_l_guai.flag == 1 || L_h_guai.flag == 1 && imgInfo.L_loselineSum > 25)
           {
               Flag.Huandao_R = 0;
           }
           //若圆环二级判断左上左下拐点都没有
           else
           {
               //顶部到底部进行补线
               float k = xielv_sideline(imgInfo.top + 1, Right_Sideline[imgInfo.top + 1], LCDH_1 - 2, 88, 'k');
               float b = xielv_sideline(imgInfo.top + 1, Right_Sideline[imgInfo.top + 1], LCDH_1 - 2, 88, 'b');

               for (int i = LCDH_1 - 2; i >= imgInfo.top + 1; i--)
               {

                   if(!Image_Use[i][(int8)(k * i + b)])
                       continue;
                   else
                       Right_Sideline[i] = k * i + b;
               }
           }
           //若是左环岛二级判断底部三行左边线丢线了 且左边丢线行数大于25行，说明此时车身已经靠在圆环出口那里了
           if((Right_Sideline_flag[LCDH_1 - 4] == 0 && Right_Sideline_flag[LCDH_1 - 5] == 0) && imgInfo.R_loselineSum > 10&&distance>distance_HUAN1)
           {
               distance=0;
               distance_HUAN1=0;
               Flag.Huandao_R = 2;
               Yaw_Huandao=icm_data.yaw;
           }
       }

         if(Flag.Huandao_R == 2)
        {
                float k ;
                float b ;

                if( R_h_guai.flag == 1)//L_h_guai.flag == 1 &&
                {
                    k = xielv_sideline(R_h_guai.row, R_h_guai.column, MIN(R_h_guai.row+5,57), Left_Sideline[MIN(R_h_guai.row+5,57)], 'k');
                    b = xielv_sideline(R_h_guai.row, R_h_guai.column,MIN(R_h_guai.row+5,57), Left_Sideline[MIN(R_h_guai.row+5,57)], 'b');

                   for (int i = imgInfo.bottom - 1; i > imgInfo.top; i--)
                   {
                       if(!Image_Use[i][(int8)(k * i + b)])
                          continue;
                       else
                       {
                           for(int m = 0; m >=-7; m--)
                           {
                               if(((k * i + b) + m) <=0)
                               {
                                   break;
                               }
                               else
                                   Image_Use[i][(int8)(k * i + b) + m] = black;
                           }
                       }
                   }

                   Flag.Buxian = 1;
                }
            //左上拐点存在，继续补线
            if( R_l_guai.flag == 1)//L_h_guai.flag == 1 &&
            {

                k=0.5;
                b=R_l_guai.column-k*R_l_guai.row;
                for (int i = imgInfo.bottom-1; i >= R_l_guai.row; i--)
                {

                    if(!Image_Use[i][(int8)(k * i + b)])
                        continue;
                    else
                    {
                        for(int m = 0; m <=7; m++)
                        {
                            if(((k * i + b) + m) >=93)
                            {
                                break;
                            }
                            else
                                Image_Use[i][(int8)(k * i + b) + m] = black;
                        }
                    }
                }



                Flag.Buxian = 1;


            }

            else
            {
                float k = xielv_sideline(3, 53, LCDH_1 - 2, 88, 'k');
                float b = xielv_sideline(3, 53, LCDH_1 - 2, 88, 'b');

                for (int i = LCDH_1 - 2; i >= imgInfo.top + 1; i--)
                {

                    if(!Image_Use[i][(int8)(k * i + b)])
                        continue;
                    else
                        Right_Sideline[i] = k * i + b;
                }

                k = xielv_sideline(3 , 55, LCDH_1 - 1, 24, 'k');
                b = xielv_sideline(3 , 55, LCDH_1 - 1, 24, 'b');

                for (int i = LCDH_1 - 2; i >= imgInfo.top + 1; i--)
                {

                    if(!Image_Use[i][(int8)(k * i + b)])
                        continue;
                    else
                        Left_Sideline[i] = k * i + b;
                }

                    Flag.Buxian = 1;
                }


            if(((Right_Sideline_flag[LCDH_1 - 7] == 1 && Right_Sideline_flag[LCDH_1 - 8] == 1 
                    && R_h_guai.flag) )&&distance>10)//|| !R_l_guai.flag
            {
                Flag.Huandao_R = 3;
                distance=0;
            }
            Get_ImageTop();
            Find_Sideline(imgInfo.bottom - 1, imgInfo.top + 1);

        }

         if(Flag.Huandao_R == 3)
            {

             if( R_h_guai.flag == 1)//L_h_guai.flag == 1 &&
             {
                    float k = xielv_sideline(R_h_guai.row, R_h_guai.column, MIN(R_h_guai.row+20,57), Left_Sideline[MIN(R_h_guai.row+20,57)], 'k');
                    float b = xielv_sideline(R_h_guai.row, R_h_guai.column, MIN(R_h_guai.row+20,57), Left_Sideline[MIN(R_h_guai.row+20,57)], 'b');

                    for (int i = imgInfo.bottom - 1; i > imgInfo.top; i--)
                    {
                        if(!Image_Use[i][(int8)(k * i + b)])
                           continue;
                        else
                        {
                            for(int m = 0; m >=-7; m--)
                            {
                                if(((k * i + b) + m) <=0)
                                {
                                    break;
                                }
                                else
                                    Image_Use[i][(int8)(k * i + b) + m] = black;
                            }
                            Flag.Buxian = 1;
                        }
                    }
                }
            //  else if(  R_h_guai.flag == 0&&L_h_guai.flag == 1)//L_h_guai.flag == 1 &&
            //  {
            //         float k = xielv_sideline(L_h_guai.row, L_h_guai.column, imgInfo.bottom - 1,19, 'k');
            //         float b = xielv_sideline(L_h_guai.row, L_h_guai.column, imgInfo.bottom - 1, 19, 'b');

            //         for (int i = imgInfo.bottom - 1; i > imgInfo.top; i--)
            //         {
            //             if(!Image_Use[i][(int8)(k * i + b)])
            //                continue;
            //             else
            //             {
            //                 for(int m = 0; m >=- 7; m--)
            //                 {
            //                     if(((k * i + b) + m) <=0)
            //                     {
            //                         break;
            //                     }
            //                     else
            //                         Image_Use[i][(int8)(k * i + b) + m] = black;
            //                 }
            //                 Flag.Buxian = 1;
            //             }
            //         }   
            //     }

//             else
//             {
//                 Eerr_flag=1;
//
////                    float k = xielv_sideline(1, 0, imgInfo.bottom - 1, 19, 'k');
////                    float b = xielv_sideline(1, 0, imgInfo.bottom - 1, 19, 'b');
////
////                    for (int i = imgInfo.bottom - 1; i > imgInfo.top; i--)
////                    {
////                        if(!Image_Use[i][(int8)(k * i + b)])
////                           continue;
////                        else
////                        {
////                            for(int m = 0; m >= -7; m--)
////                            {
////                                if(((k * i + b) + m) <=0)
////                                {
////                                    break;
////                                }
////                                else
////                                    Image_Use[i][(int8)(k * i + b) + m] = black;
////                            }
//                            Flag.Buxian = 1;
//                        }
//                    }
//                }

                //环岛3是直接把USE图像数组改了，所以重新找截止行来确定手动补线后的截止行和边线，这时找到的图像就是一个大弯拐进环岛
                Get_ImageTop();
                Find_Sideline(imgInfo.bottom - 1, imgInfo.top + 1);

                //进入弯道
                if(Yaw_Huandao_err>105)//(distance>30)&&imgInfo.Both_lose==0
                {
                   Flag.Huandao_R = 4;
//                    distance=0;
                }
            }

             if(Flag.Huandao_R == 4)
            {
                //找到右下拐点，即说明已经到了快要出弯的地方
                if(L_l_guai.flag)
                {
                    uint8 temp = 0;
                        temp = imgInfo.top + 1;
                    float k = xielv_sideline(L_l_guai.row, L_l_guai.column, temp, 63, 'k');
                    float b = xielv_sideline(L_l_guai.row, L_l_guai.column, temp, 63, 'b');

                    for (int i = L_l_guai.row; i > imgInfo.top; i--)
                    {
                        for(int m = 0; m >=-13; m--)
                        {
                            if(((k * i + b) + m) <= 0)
                            {
                                break;
                            }
                            else
                            {
                                Image_Use[i][(uint8)(k * i + b) + m] = black;
                            }

                        }
                    }
                    Flag.Buxian = 1;
                }
            //    else if((!L_l_guai.flag) &&  abs(Left_Sideline[imgInfo.top+2]-Right_Sideline[imgInfo.top+2])>30 )
            //    {
            //        uint8 temp = 0;

            //     //    for(int m = imgInfo.bottom - 4; m > imgInfo.top + 1; m--)
            //     //    {
            //     //        if(!Right_Sideline_flag[m + 1] && !Right_Sideline_flag[m + 2] && !Right_Sideline_flag[m + 3] && !Right_Sideline_flag[m + 4] && !Right_Sideline_flag[m + 5] && Right_Sideline_flag[m - 1])
            //     //        {
            //     //            temp =(uint8) m;
            //     //            break;
            //     //        }
            //     //    }
            //     //    if(temp == 0)
            //     //    {
            //            temp = imgInfo.top + 1;
            //     //    }
            //        float k = xielv_sideline(LCDH_1 - 2, 1, temp, 63, 'k');
            //        float b = xielv_sideline(LCDH_1 - 2, 1, temp, 63, 'b');

            //        for (int i = LCDH_1 - 2; i >= temp - 2; i--)
            //        {
            //            for(int m = 0; m >=-4; m--)
            //            {
            //                if(((k * i + b) + m) <= 0)
            //                {
            //                    break;
            //                }
            //                else
            //                {
            //                    Image_Use[i][(uint8)(k * i + b) + m] = black;
            //                }

            //            }
            //        }
            //        Flag.Buxian = 1;
            //    }
                else
                {
                                        Flag.Buxian = 1;
                }
                // if(
                //        imgInfo.L_loselineSum > 15 && !Left_Sideline_flag[LCDH_1 - 3] && !Left_Sideline_flag[LCDH_1 - 4]
                //         && !Left_Sideline_flag[LCDH_1 - 5] && !Left_Sideline_flag[LCDH_1 - 6]
                //         && !Left_Sideline_flag[LCDH_1 - 7] && !Left_Sideline_flag[LCDH_1 - 8]
                // )
                // {   
                //     Flag.Huandao_R = 5;
                // }

                if(Yaw_Huandao_err>-110&&Yaw_Huandao_err<-100)//(L_h_guai.flag==1&&imgInfo.R_straight_flag==1)&&imgInfo.top<10
                {
                    Flag.Huandao_R = 5;
                }

                Get_ImageTop();
                Find_Sideline(imgInfo.bottom - 1, imgInfo.top + 1);
                imgInfo.L_loselineSum = 0;
            }


              if(Flag.Huandao_R == 5)
                 {
                     uint8 temp = 0;
                     //找圆弧内白到黑的交界点
                     for(int m = imgInfo.bottom - 4; m > imgInfo.top + 3; m--)
                     {
                         if(!Right_Sideline_flag[m + 1] && !Right_Sideline_flag[m + 2] && !Right_Sideline_flag[m + 3]
                            && !Right_Sideline_flag[m + 4] && !Right_Sideline_flag[m + 5] && Right_Sideline_flag[m - 1])
                         {
                             temp = (uint8)m;
                             break;
                         }
                     }
                     if(temp == 0)
                     {
                         temp = imgInfo.top + 3;
                     }
                     float k = xielv_sideline(LCDH_1 - 2, 1, temp, 53, 'k');
                     float b = xielv_sideline(LCDH_1 - 2, 1, temp, 53, 'b');

                     //只要有右丢线情况就进行补线
                     if(imgInfo.L_loselineSum > 1)
                     {
                         for (int i = LCDH_1 - 2; i >= temp - 2; i--)
                         {
                             for(int m = 0; m >=-4; m--)
                             {
                                 if(((k * i + b) + m) <= 0)
                                 {
                                     break;
                                 }
                                 else
                                     Image_Use[i][(uint8)(k * i + b) + m] = black;
                             }
                         }
                         Flag.Buxian = 1;
                     }

                     //直到右边没有丢线情况并且左边出现拐点只后
                     if(Yaw_Huandao_err>-25)//(R_h_guai.flag==1&&imgInfo.L_straight_flag==1)&&imgInfo.top<10
                     {
                         Flag.Huandao_R = 6;
                         Flag.Buxian = 0;
                     }
                     Get_ImageTop();
                     Find_Sideline(imgInfo.bottom - 1, imgInfo.top + 1);
                 }

                  if(Flag.Huandao_R == 6)
                 {
                     if(R_h_guai.flag == 1)
                     {
                         //从左拐点向下补线，防止再次入环
                         float k = xielv_sideline(R_h_guai.row + 5, R_h_guai.column, LCDH_1 - 2, 85, 'k');
                         float b = xielv_sideline(R_h_guai.row + 5, R_h_guai.column, LCDH_1 - 2, 85, 'b');

                         for (int i = LCDH_1 - 2; i >= R_h_guai.row; i--)
                         {

                             if(!Image_Use[i][(uint8)(k * i + b)])
                                 continue;
                             else
                                 Right_Sideline[i] = k * i + b;
                         }
                     }
                     else if(R_h_guai.flag == 0)
                     {
                         float k = xielv_sideline(imgInfo.top + 10, Right_Sideline[imgInfo.top + 10], LCDH_1 - 2, 88, 'k');
                         float b = xielv_sideline(imgInfo.top + 10, Right_Sideline[imgInfo.top + 10], LCDH_1 - 2, 88, 'b');

                         for (int i = LCDH_1 - 2; i >= imgInfo.top + 1; i--)
                         {

                             if(!Image_Use[i][(uint8)(k * i + b)])
                                 continue;
                             else
                                 Right_Sideline[i] = k * i + b;
                         }
                     }
                     else
                         Flag.Buxian = 0;

                     if(imgInfo.R_straight_flag==1&&distance>=10&&R_h_guai.flag==0)//imgInfo.R_loselineSum < 1//Left_Sideline_flag[imgInfo.bottom - 6] && Left_Sideline_flag[imgInfo.bottom - 7] && Left_Sideline_flag[imgInfo.bottom - 8]&&
                     {
                         Flag.Huandao_R = 0;
                         distance=0;
                     }
                 }

                  if((imgInfo.L_straight_flag==1&& imgInfo.R_straight_flag==1)&&L_h_guai.flag==0&&L_l_guai.flag==0&&R_h_guai.flag==0&&R_l_guai.flag==0&&abs(imgInfo.R_loselineSum-imgInfo.L_loselineSum)==0)//(Flag.Huandao_R>4&&  imgInfo.R_straight_flag==1)||
                  {
                                               Flag.Huandao_R = 0;
                  }
//                 if(Flag.Huandao_R && imgInfo.R_loselineSum < 10 && imgInfo.L_loselineSum < 10)
//                 {
//                     for(int i = imgInfo.bottom - 1; i > imgInfo.top + 1; i--)
//                     {
//                         if(i > imgInfo.top)
//                         {
//                             //右边界不发生突变且呈直线状态
//                             if(Left_Sideline[i] - Left_Sideline[i - 1] < 3 && Left_Sideline[i] >= Left_Sideline[i - 1])
//                             {
//                                 right_num ++;
//                             }
//                             //左边界不发生突变且呈直线状态
//                             if(Right_Sideline[i] - Right_Sideline[i - 1] < 3 && Right_Sideline[i] >= Right_Sideline[i - 1])
//                             {
//                                 right_num ++;
//                             }
//                         }
//                     }
//
//                     if(left_num > 50 && right_num > 50)
//                     {
//                         Flag.Huandao_R = 0;
//                         left_num = 0;
//                         right_num = 0;
//
//                     }
//                     else
//                     {
//                         left_num = 0;
//                         right_num = 0;
//                     }
//                 }


}

/* 补线 */
void Buxian(void)
{

  
    Flag.Buxian = 0;
//    else if(Mode_1)
//    {
        //四个拐点都找到
        if(L_l_guai.flag && L_h_guai.flag && R_l_guai.flag && R_h_guai.flag && abs(R_h_guai.column - L_h_guai.column) > 5)//
        {

            float kl, kr, bl, br;

            if(L_h_guai.row < L_l_guai.row)
            {
                kl = xielv_sideline(L_l_guai.row, L_l_guai.column, L_h_guai.row, L_h_guai.column, 'k');
                bl = xielv_sideline(L_l_guai.row, L_l_guai.column, L_h_guai.row, L_h_guai.column, 'b');

                for (int i = L_h_guai.row; i <= L_l_guai.row; i++)
                {
                    Left_Sideline[i] = kl * i + bl;
                    Left_Sideline_flag[i] = 1;
                }

                Flag.Buxian = 1;
            }

            if(R_h_guai.row < R_l_guai.row)
            {
                kr = xielv_sideline(R_l_guai.row, R_l_guai.column, R_h_guai.row, R_h_guai.column, 'k');
                br = xielv_sideline(R_l_guai.row, R_l_guai.column, R_h_guai.row, R_h_guai.column, 'b');

                for (int i = R_h_guai.row; i <= R_l_guai.row; i++)
                {
                    Right_Sideline[i] = kr * i + br;
                    Right_Sideline_flag[i] = 1;
                }

                Flag.Buxian = 1;
            }

//            Flag.Shizi=1;
        }

//        if(Flag.Shizi==1)
//        {
//            if(distance>30){Flag.Shizi=0;distance=0;}
        //找到左上，左下，右上
         if(L_l_guai.flag && L_h_guai.flag && R_h_guai.flag && abs(R_h_guai.column - L_h_guai.column) > 5)//
        {
            float kl, kr, bl, br;

            if(L_h_guai.row < L_l_guai.row)
            {
                kl = xielv_sideline(L_l_guai.row, L_l_guai.column, L_h_guai.row, L_h_guai.column, 'k');
                bl = xielv_sideline(L_l_guai.row, L_l_guai.column, L_h_guai.row, L_h_guai.column, 'b');

                for (int i = L_h_guai.row; i <= L_l_guai.row; i++)
                {
                    Left_Sideline[i] = kl * i + bl;
                    Left_Sideline_flag[i] = 1;
                }

                Flag.Buxian = 1;
            }

            if(R_h_guai.row < L_l_guai.row)
            {
                kr = xielv_sideline(R_h_guai.row, R_h_guai.column, imgInfo.bottom - 5, Right_Sideline[imgInfo.bottom - 5], 'k');
                br = xielv_sideline(R_h_guai.row, R_h_guai.column, imgInfo.bottom - 5, Right_Sideline[imgInfo.bottom - 5], 'b');

                for (int i = R_h_guai.row; i <= imgInfo.bottom - 5; i++)
                {
                    Right_Sideline[i] = kr * i + br;
                    Right_Sideline_flag[i] = 1;
                    if (
                            (Right_Sideline[i] == Right_Sideline[i + 1] && Right_Sideline[i] == Right_Sideline[i + 2])
                            || !Image_Use[i + 1][(uint8)(kr * (i + 1) + br)]
                        )
                        break;
                }

                Flag.Buxian = 1;
            }
        }


        else if(R_l_guai.flag && R_h_guai.flag && L_h_guai.flag && abs(R_h_guai.column - L_h_guai.column) > 5)//
        {
            float kl, kr, bl, br;

            if(L_h_guai.row < R_l_guai.row)
            {
                kl = xielv_sideline(L_h_guai.row, L_h_guai.column, imgInfo.bottom - 5, Left_Sideline[imgInfo.bottom - 5], 'k');
                bl = xielv_sideline(L_h_guai.row, L_h_guai.column, imgInfo.bottom - 5, Left_Sideline[imgInfo.bottom - 5], 'b');

                for (int i = L_h_guai.row; i <= imgInfo.bottom - 5; i++)
                {
                    Left_Sideline[i] = kl * i + bl;
                    Left_Sideline_flag[i] = 1;
                    if(
                            (Left_Sideline[i] == Left_Sideline[i + 1] && Left_Sideline[i] == Left_Sideline[i + 2])
                            || !Image_Use[i + 1][(uint8)(kl * (i + 1) + bl)]
                        )
                        break;
                }

                Flag.Buxian = 1;
            }

            if(R_h_guai.row < R_l_guai.row)
            {
                kr = xielv_sideline(R_l_guai.row, R_l_guai.column, R_h_guai.row, R_h_guai.column, 'k');
                br = xielv_sideline(R_l_guai.row, R_l_guai.column, R_h_guai.row, R_h_guai.column, 'b');

                for (int i = R_h_guai.row; i <= R_l_guai.row; i++)
                {
                    Right_Sideline[i] = kr * i + br;
                    Right_Sideline_flag[i] = 1;
                }

                Flag.Buxian = 1;
            }

        }

        else if(R_h_guai.flag && L_h_guai.flag && abs(R_h_guai.column - L_h_guai.column) > 5)//
        {
            float kl, kr, bl, br;

            kl = xielv_sideline(L_h_guai.row, L_h_guai.column, imgInfo.bottom - 5, Left_Sideline[imgInfo.bottom - 5] - 3, 'k');
            bl = xielv_sideline(L_h_guai.row, L_h_guai.column, imgInfo.bottom - 5, Left_Sideline[imgInfo.bottom - 5] - 3, 'b');

            for (int i = L_h_guai.row; i <= imgInfo.bottom - 5; i++)
            {
                Left_Sideline[i] = kl * i + bl;
                Left_Sideline_flag[i] = 1;
                if(
                        (Left_Sideline[i] == Left_Sideline[i + 1] && Left_Sideline[i] == Left_Sideline[i + 2])
                        || !Image_Use[i + 1][(uint8)(kl * (i + 1) + bl)]
                    )
                    break;
            }

            kr = xielv_sideline(R_h_guai.row, R_h_guai.column, imgInfo.bottom - 5, Right_Sideline[imgInfo.bottom - 5] - 3, 'k');
            br = xielv_sideline(R_h_guai.row, R_h_guai.column, imgInfo.bottom - 5, Right_Sideline[imgInfo.bottom - 5] - 3, 'b');

            for (int i = R_h_guai.row; i < imgInfo.bottom - 5; i++)
            {
                Right_Sideline[i] = kr * i + br;
                Right_Sideline_flag[i] = 1;
                if (
                        (Right_Sideline[i] == Right_Sideline[i + 1] && Right_Sideline[i] ==  Right_Sideline[i + 2])
                        || !Image_Use[i + 1][(uint8)(kr * (i + 1) + br)]
                    )
                    break;
            }

            Flag.Buxian = 1;
        }

        else if(L_l_guai.flag && L_h_guai.flag)
        {
            float kl, bl;
            if(L_h_guai.row < L_l_guai.row)
            {
                kl = xielv_sideline(L_l_guai.row, L_l_guai.column, L_h_guai.row, L_h_guai.column, 'k');
                bl = xielv_sideline(L_l_guai.row, L_l_guai.column, L_h_guai.row, L_h_guai.column, 'b');

                for (int i = L_h_guai.row; i <= L_l_guai.row; i++)
                {
                    Left_Sideline[i] = kl * i + bl;
                    Left_Sideline_flag[i] = 1;
                }
            }

            Flag.Buxian = 1;
        }
        else if(R_l_guai.flag && R_h_guai.flag)
        {
            float kr, br;

            if(R_h_guai.row < R_l_guai.row)
            {
                kr = xielv_sideline(R_l_guai.row, R_l_guai.column, R_h_guai.row, R_h_guai.column, 'k');
                br = xielv_sideline(R_l_guai.row, R_l_guai.column, R_h_guai.row, R_h_guai.column, 'b');

                for (int i = R_h_guai.row; i <= R_l_guai.row; i++)
                {
                    Right_Sideline[i] = kr * i + br;
                    Right_Sideline_flag[i] = 1;
                }
            }

            Flag.Buxian = 1;
        }


       //只找到左上
       
       else if(L_h_guai.flag&&imgInfo.Both_lose>0)
       {
           float kl, bl;

           kl = xielv_sideline(L_h_guai.row, L_h_guai.column, imgInfo.bottom - 3, Left_Sideline[imgInfo.bottom - 3], 'k');
           bl = xielv_sideline(L_h_guai.row, L_h_guai.column, imgInfo.bottom - 3, Left_Sideline[imgInfo.bottom - 3], 'b');

           for (int i = L_h_guai.row; i <= imgInfo.bottom - 5; i++)
           {
               Left_Sideline[i] = kl * i + bl;
               Left_Sideline_flag[i] = 1;
               if(
                       (Left_Sideline[i] == Left_Sideline[i + 1] && Left_Sideline[i] == Left_Sideline[i + 2])
                       || !Image_Use[i + 1][(uint8)(kl * (i + 1) + bl)]
                  )
                  break;
            }

            Flag.Buxian = 1;
       }

       //只找到右上
       else if(R_h_guai.flag&&imgInfo.Both_lose>0)
       {
               float kr, br;

               kr = xielv_sideline(R_h_guai.row, R_h_guai.column, imgInfo.bottom - 3, Right_Sideline[imgInfo.bottom - 3], 'k');
               br = xielv_sideline(R_h_guai.row, R_h_guai.column, imgInfo.bottom - 3, Right_Sideline[imgInfo.bottom - 3], 'b');

               for (int i = R_h_guai.row; i < imgInfo.bottom - 5; i++)
               {
                   Right_Sideline[i] = kr * i + br;
                   Right_Sideline_flag[i] = 1;
                   if (
                           (Right_Sideline[i] == Right_Sideline[i + 1] && Right_Sideline[i] ==  Right_Sideline[i + 2])
                           || !Image_Use[i + 1][(uint8)(kr * (i + 1) + br)]
                       )
                       break;
               }

               Flag.Buxian = 1;
       }

//        else if(L_l_guai.flag && R_l_guai.flag)
//        {
//            float kr, br;
//
//            kr = xielv_sideline(R_l_guai.row, R_l_guai.column, imgInfo.top + 5, Right_Sideline[imgInfo.top + 5], 'k');
//            br = xielv_sideline(R_l_guai.row, R_l_guai.column, imgInfo.top + 5, Right_Sideline[imgInfo.top + 5], 'b');
//
//            for (int i = R_l_guai.row; i > imgInfo.top + 5; i--)
//            {
//                Right_Sideline[i] = kr * i + br;
//                Right_Sideline_flag[i] = 1;
//                if (
//                        (Right_Sideline[i] == Right_Sideline[i + 1] && Right_Sideline[i] ==  Right_Sideline[i + 2])
//                        || !Image_Use[i + 1][(uint8)(kr * (i + 1) + br)]
//                    )
//                    break;
//            }
//
//
//            kr = xielv_sideline(L_l_guai.row, L_l_guai.column, imgInfo.top + 5, Left_Sideline[imgInfo.top + 5], 'k');
//            br = xielv_sideline(L_l_guai.row, L_l_guai.column, imgInfo.top + 5, Left_Sideline[imgInfo.top + 5], 'b');
//
//            for (int i = L_l_guai.row; i > imgInfo.top + 5; i--)
//            {
//                Left_Sideline[i] = kr * i + br;
//                Left_Sideline_flag[i] = 1;
//                if (
//                        (Left_Sideline[i] == Left_Sideline[i + 1] && Left_Sideline[i] ==  Left_Sideline[i + 2])
//                        || !Image_Use[i + 1][(uint8)(kr * (i + 1) + br)]
//                    )
//                    break;
//            }
//
//            Flag.Buxian = 1;
//        }

        else
            Flag.Buxian = 0;
//    }
//        else
//            Flag.Buxian = 0;
}


/***************************************************动态前瞻********************************************************/

int forward,forward1; 
float B_near,A_near,B_far,A_far,BA_ratio;
void dynamic_forward(){

regression(imgInfo.top+21,imgInfo.top+1);
B_far = B;
A_far = A;
regression(imgInfo.bottom-21,imgInfo.bottom-1);
B_near = B;
A_near = A;

regression(imgInfo.top + 1,imgInfo.bottom - 1);
//左弯
if(B<0){calculateCurvature((float)Left_Sideline[imgInfo.bottom - 1], (float)imgInfo.bottom - 1, (float)Left_Sideline[imgInfo.bottom - 11], (float)imgInfo.bottom - 11, (float)Left_Sideline[imgInfo.bottom - 21], (float)imgInfo.bottom - 21);}
//右弯
if(B>0){calculateCurvature((float)Right_Sideline[imgInfo.bottom - 1], (float)imgInfo.bottom - 1, (float)Right_Sideline[imgInfo.bottom - 11], (float)imgInfo.bottom - 11, (float)Right_Sideline[imgInfo.bottom - 21], (float)imgInfo.bottom - 21);}

float base_forward = 15;
// 根据曲率和近处斜率增大 forward（弯越急，看得越近）
float k_curv = 500.0f;  // 曲率增益
float k_slope = 80.0f;  // 斜率增益
float forward_candidate = base_forward  + k_curv * curvature  + k_slope * fabsf(B_near);
// 限幅到有效范围
if (forward_candidate < imgInfo.top+1) {
  forward = imgInfo.top+1;
} else if (forward_candidate > imgInfo.bottom-1) {
  forward = imgInfo.bottom-1;
  } 
else {
  forward = (int)forward_candidate;
  }
  // 极端急弯强制看最近
if (curvature > 0.12f || fabsf(B_near) > 0.8f) {
  forward = imgInfo.bottom-1;
  }
}

/***************************************************误差计算********************************************************/
float Dir_err = 0, Last_Dir_err = 0,Dir_Err[60],D_ERR;  //图像误差
void Err_Sum(void)
{
     esc_duty=1000;
    if(run_flag==1)
    {   //33100/2 16550         15*35  
                forward1=35;
                                // forward1=40;
                forward = forward1-Now_Speed/40;//23 
                speed_goal=10*75;
    //      Image.Kp=3.5*(0.2143*imgInfo.top+0.4286);
    //  if(Image.Kp<=1.3)Image.Kp=1.3;
    //  if(Image.Kp>=3.5)Image.Kp=3.5;
    //   //  speed_goal=10*30;
        // if(Flag.picture==2)speed_goal=50;
    //   if(Flag.picture==2&&real_distance[MAX(R_h_guai.row,L_h_guai.row)]>50)speed_goal=(real_distance[MAX(R_h_guai.row,L_h_guai.row)]-25)*9;
      if(Flag.picture==2)
      {
             speed_goal=100;
    //     if(real_distance[MAX(R_h_guai.row,L_h_guai.row)]<recognize_distance2)
    //     {
    //   Pos_Cal(&picture_distance,recognize_distance2,real_distance[MAX(R_h_guai.row,L_h_guai.row)]);
    //   speed_goal=picture_distance.output;
    //     }
 
      }
        if(Flag.ramp==2)speed_goal=150;
    
       
// if(Flag.Zebra_cross>1)
// {
// Image.Kp=14;
//                 Velocity_L.kd=Image.Kp*20;
//                 Velocity_R.kd=Velocity_L.kd;
//                 Dis_1.Kp=Velocity_L.kd*0.01;
//                 Dis_1.Kd=Dis_1.Kp*0.25;
float image_kp=3.0;
float image_kd=20;
if(real_distance[imgInfo.top]<150||Flag.Huandao_R!=0||Flag.Huandao_L!=0||Flag.picture!=0)//
{

                Image.Kp=3.5*MaX_Speed/400*image_kp/1;//MIN(MaX_Speed,speed_goal)
                Velocity_L.kd=Image.Kp*image_kd;
                Velocity_R.kd=Velocity_L.kd;
                Dis_1.Kp=Velocity_L.kd*0.01;
                Dis_1.Kd=Dis_1.Kp*0.25;
               esc_pwm.set_duty(esc_duty); 

                Velocity_L.ki= Velocity_L.kd;
                 Velocity_R.ki= Velocity_R.kd;
}
else
{

                Image.Kp=3.5*MaX_Speed/400*image_kp/1;//MIN(MaX_Speed,speed_goal)
                Velocity_L.kd=Image.Kp*image_kd;
                Velocity_R.kd=Velocity_L.kd;
                Dis_1.Kp=Velocity_L.kd*0.01;
                Dis_1.Kd=Dis_1.Kp*0.25;
                esc_pwm.set_duty(MIN(750,esc_duty));                
}
// }


if(real_distance[imgInfo.top]>200&&Flag.Huandao_R==0&&Flag.Huandao_L==0)
{

                Image.Kp=3.5*MaX_Speed/400*image_kp/2;
                Velocity_L.kd=Image.Kp*image_kd*2;
                Velocity_R.kd=Velocity_L.kd;
                Dis_1.Kp=Velocity_L.kd*0.01;
                Dis_1.Kd=Dis_1.Kp*0.25;

            //         speed_add=0;
            // speed_add=(real_distance[imgInfo.top]-200)*0;
            // if(speed_add>0)
            // speed_goal+=speed_add;

        
}
if(Flag.Zebra_cross==0)
{

                Image.Kp=3.5*MaX_Speed/400*image_kp/2;
                Velocity_L.kd=Image.Kp*image_kd*2;
                Velocity_R.kd=Velocity_L.kd;
                Dis_1.Kp=Velocity_L.kd*0.01;
                Dis_1.Kd=Dis_1.Kp*0.25;
                esc_pwm.set_duty(esc_duty);                 
}




if(  Image.Kp>20)  Image.Kp=20;


// if(Flag.Zebra_cross<=1)
// {     
//                 Image.Kp=3.5*MaX_Speed/400*1.5/1;
//                 Velocity_L.kd=Image.Kp*20;
//                 Velocity_R.kd=Velocity_L.kd;
//                 Dis_1.Kp=Velocity_L.kd*0.01;
//                 Dis_1.Kd=Dis_1.Kp*0.2;      
//                 esc_pwm.set_duty(1000);
// }





    }


    //     forward1=33;
    //     forward = forward1-Now_Speed/40;//23 
    // if(Flag.Huandao_R==2||Flag.Huandao_L==2)forward=forw ard1-Now_Speed/40-1;
                // forward1=40+5;
                // forward = forward1-Now_Speed/20;//23 

                    //  forward = 30-Master_Speed/60;//23
                //                 forward1=27;
                // forward = forward1-Now_Speed/60;//23 
    if(Flag.Huandao_R==2||Flag.Huandao_L==2)forward-=1;
    if(Flag.Huandao_R==3||Flag.Huandao_L==3)forward-=1;
    if(Flag.Huandao_R==5||Flag.Huandao_L==5)   forward+=2;

for(int i=imgInfo.top + 1;i<imgInfo.bottom - 1;i++)
{
    Dir_Err[i]=(float)(LCDW_1/2-(((float)Left_Sideline[i]+(float)Right_Sideline[i])/2));//*(3-0.05*i)*(2.0-0.025*i)
    // if(Left_Sideline_flag[i]==0||Right_Sideline_flag[i]==0)Dir_Err[i]*=2;
    //  if(!( Dir_Err[i]<100&& Dir_Err[i]>-100)) Dir_Err[i]=0;
}



    BA_ratio=0;
    for(int i=imgInfo.top + 1;i<imgInfo.bottom - 1;i++)
    {
        Dir_Err[i] = (1-BA_ratio)*Dir_Err[i] + BA_ratio*(B*i + A);
    }

        //  if(Flag.Huandao_R==2||Flag.Huandao_L==2)forward=-1;
        //  if(Flag.Huandao_R==3||Flag.Huandao_L==3)forward=-1;
        //  if(Flag.Huandao_R==5||Flag.Huandao_L==5)   forward+=1;

     if(forward<imgInfo.top+1)forward=imgInfo.top+1;
     if(forward>50)forward=50;
     
        Dir_err=Dir_Err[forward];

    // if(Flag.small_rock==1)Dir_err-=10;
    // if(Flag.small_rock==2)Dir_err+=10;

    // if(Flag.Zebra_cross==1)Dir_err=Dir_Err[maxkuan_line]*90/(120-maxkuan_line);
    // if(Flag.Zebra_cross==2)Dir_err=Dir_Err[maxkuan_line]*90/(120-maxkuan_line);
    // if(Flag.Zebra_cross==4)Dir_err=Dir_Err[maxkuan_line]*90/(120-maxkuan_line);
    
    // if(Flag.picture==3)Dir_err =Dir_Err[maxkuan_line]*90/(120-maxkuan_line)+30;//
        // if(Flag.picture==2)Dir_err =(float)(LCDW_1/2-((float)Left_Sideline[forward]));//
        if(Flag.picture==3)Dir_err =(float)(LCDW_1/2-((float)Left_Sideline[forward]-10));//
        if(Flag.picture==4)Dir_err =(float)(LCDW_1/2-((float)Right_Sideline[forward]+10));//
        // if(Flag.picture==3){Dir_err = Dir_err+30;}//[maxkuan_line]*90/(120-maxkuan_line)+20

// Dir_err =(float)(LCDW_1/2-((float)Left_Sideline[forward]));

    // if(!(Dir_err<100&&Dir_err>-100))Dir_err=0;
    // if(Dir_err>150)Dir_err=150;    
    // if(Dir_err<-150)Dir_err=-150;



            if(Flag.small_rock ==1)
    {
        Dir_err-=10;    
    }

    if(Flag.small_rock ==2)
    {
        Dir_err+=10;
    }
    // if(Dir_err<-47)Dir_err=-47;
    // if(Dir_err>47)Dir_err=47;


                    D_ERR=Dir_err-Last_Dir_err;

                // if(Flag.picture==2||Flag.picture==3)
                // {
                //                     if((D_ERR)>1)Dir_err=Last_Dir_err+4;
                // if((D_ERR)<-1)Dir_err=Last_Dir_err-4;
                // }
                // else{
                // if(Flag.picture==0)
                // {

                // if((D_ERR)>=5)Dir_err=Last_Dir_err+5.0f;
                // else if((D_ERR)<-5)Dir_err=Last_Dir_err-5.0f;
                //                 if(Flag.picture==3)
                // {
                // if((D_ERR)>=1)Dir_err=Last_Dir_err+1.0f;
                // else if((D_ERR)<-1)Dir_err=Last_Dir_err-1.0f;
                // }
                
            //     // }
            //     // }
            Last_Dir_err = Dir_err;

        // Image.Kp = 3.5*3.0;//  0.8; //250*0.015
        // if(Image.Kp>4)Image.Kp=4;

        // if(Now_Speed<200)Image.Kp=3.5*2;        
        // if(Now_Speed<150)Image.Kp=3.5*Now_Speed*0.007;
        // if(fabs(Dir_err)<10)Image.Kp=3.5*fabs(Dir_err)*0.3;
        // if(MAX(encoder_L.speed,encoder_R.speed)<speed_goal/1)
        // {

        // if(Image.Kp<3.5*0.5)Image.Kp=3.5*0.5;
        // Image.Kd=Image.Kp*0.3;
        // Dis_1.Kp=Image.Kp/3.5/3;
        //  Dis_1.Kp = Master_Speed/400*60;//
        // if(MAX(encoder_L.speed,encoder_R.speed)<speed_goal/1.5)Image.Kp=3.5*MAX(encoder_L.speed,encoder_R.speed)/400*2.5;
        // if(MAX(encoder_L.speed,encoder_R.speed)<speed_goal/2)Image.Kp=3.5*MAX(encoder_L.speed,encoder_R.speed)/400*1.5;
        // if(MAX(encoder_L.speed,encoder_R.speed)<speed_goal/3)Image.Kp=3.5*MAX(encoder_L.speed,encoder_R.speed)/400*1;
        // if(MAX(encoder_L.speed,encoder_R.speed)<speed_goal/4)Image.Kp=3.5*MAX(encoder_L.speed,encoder_R.speed)/400*0.5;
        // if(MAX(encoder_L.speed,encoder_R.speed)<speed_goal/5)Image.Kp=0;
        // if(fabs(Dir_err)<15)Image.Kp*=0.75;
        // else if(fabs(Dir_err)<5)Image.Kp*=0.5;
        // else if(fabs(Dir_err)<5)Image.Kp*=0.25;
        // }
        // if(MAX(encoder_L.speed,encoder_R.speed)<speed_goal/2)Image.Kp=3.5*MAX(encoder_L.speed,encoder_R.speed)/speed_goal*3;
        // if(MAX(encoder_L.speed,encoder_R.speed)<speed_goal/3)Image.Kp=3.5*MAX(encoder_L.speed,encoder_R.speed)/speed_goal*2;
        // if(MAX(encoder_L.speed,encoder_R.speed)<speed_goal/4)Image.Kp=3.5*MAX(encoder_L.speed,encoder_R.speed)/speed_goal*1;
//         float K=0.4;

//         Image.Kp=(Master_Speed*0.035+1.5)*K;
// if(Flag.Huandao_L==0||Flag.Huandao_R==0&&(real_distance[imgInfo.top]<120))
// {
//     Image.Kp*=2;
//         // if(real_distance[imgInfo.top]>70)Image.Kp*=0.85;
//         // else if(real_distance[imgInfo.top]>90)Image.Kp*=0.7;
//         // else if(real_distance[imgInfo.top]>120)Image.Kp*=0.55;
//         // else if(real_distance[imgInfo.top]>160)Image.Kp*=0.4;
//         // else if(real_distance[imgInfo.top]>210)Image.Kp*=0.25;
//         // else if(real_distance[imgInfo.top]>270)Image.Kp*=0.10;
//         // else if(real_distance[imgInfo.top]>340)Image.Kp*=0.3;
// }


        // float K=0.25;

        // Image.Kp=(MaX_Speed*0.03+4)*K;   
                // if(MaX_Speed>speed_goal)MaX_Speed=speed_goal;

                // if(MaX_Speed>=speed_goal)Image.Kp=3.5*speed_goal/400*3.0/1;
                // Image.Kp=7;
// Image.Kp=3.5*Master_Speed/400*2.7;
// if(Image.Kp<5)Image.Kp=5;
// if(Image.Kp>15)Image.Kp=15;
if(Flag.Huandao_L==0||Flag.Huandao_R==0)
{
    // Image.Kp*=2;
        // if(real_distance[imgInfo.top]>100)Image.Kp*=0.85;
        // else if(real_distance[imgInfo.top]>150)Image.Kp*=0.7;
        // else if(real_distance[imgInfo.top]>200)Image.Kp*=0.55;
        // else if(real_distance[imgInfo.top]>250)Image.Kp*=0.4;
        // else if(real_distance[imgInfo.top]>300)Image.Kp*=0.25;
        // if(real_distance[imgInfo.top]>120)Image.Kp=3.5*Master_Speed/400*2.75;
        // // // if(Now_Speed<0)Image.Kp=0;  
        // if(real_distance[imgInfo.top]>200)Image.Kp=3.5*Master_Speed/400*2.5;
        // if(real_distance[imgInfo.top]>250)Image.Kp=3.5*Master_Speed/400*2.0;
        // if(real_distance[imgInfo.top]>300)Image.Kp=3.5*Master_Speed/400*1.5;
}

        // Image.Kp=3.5*Master_Speed/400*3.0;
        // if(Image.Kp<3.5*0.5)Image.Kp=3.5*0.5;
        // Image.Kd=Image.Kp*0.3;
        // Dis_1.Kp=Image.Kp/3.5/3;
        //  Dis_1.Kp = Master_Speed/400*60;//
        // if(MAX(encoder_L.speed,encoder_R.speed)<speed_goal/1.5)Image.Kp=3.5*MAX(encoder_L.speed,encoder_R.speed)/400*2.5;
        // if(MAX(encoder_L.speed,encoder_R.speed)<speed_goal/2)Image.Kp=3.5*MAX(encoder_L.speed,encoder_R.speed)/400*1.5;
        // if(MAX(encoder_L.speed,encoder_R.speed)<speed_goal/3)Image.Kp=3.5*MAX(encoder_L.speed,encoder_R.speed)/400*1;
        // if(MAX(encoder_L.speed,encoder_R.speed)<speed_goal/4)Image.Kp=3.5*MAX(encoder_L.speed,encoder_R.speed)/400*0.5;
        // if(MAX(encoder_L.speed,encoder_R.speed)<speed_goal/5)Image.Kp=0;
        // if(fabs(Dir_err)<15)Image.Kp*=0.75;
        // else if(fabs(Dir_err)<5)Image.Kp*=0.5;
        // else if(fabs(Dir_err)<5)Image.Kp*=0.25;
        // }
        // if(MAX(encoder_L.speed,encoder_R.speed)<speed_goal/2)Image.Kp=3.5*MAX(encoder_L.speed,encoder_R.speed)/speed_goal*3;
        // if(MAX(encoder_L.speed,encoder_R.speed)<speed_goal/3)Image.Kp=3.5*MAX(encoder_L.speed,encoder_R.speed)/speed_goal*2;
        // if(MAX(encoder_L.speed,encoder_R.speed)<speed_goal/4)Image.Kp=3.5*MAX(encoder_L.speed,encoder_R.speed)/speed_goal*1;
        


// if(Left_Sideline_flag[forward]==1&&Right_Sideline_flag[forward]==1)
// {
//     Image.Kp*=0.5;
// }
// if(speed_add>0)Image.Kp*=0.25;
        // if(Flag.picture==2)Image.Kp=3.5*1.5;
        // if(Flag.picture==3)Image.Kp=3.5*1.5;
        // if(Flag.picture==4)Image.Kp=3.5*1.5;

        Image_out =Image_PID_Calculate(&Image,Dir_err,0);//-icm_data.gyro_z//Image_E2
}



int red_find_x,red_find_y,red_find_y1;
int picture_first_num,picture_second_num,picture_third_num,maxlong_colume,colume_long[94], long_max,picture_white=0,picture_black=0,jump_point1;
float black_ratio;
// int red_find_x,red_find_y,red_find_y1;
// int picture_first_num,picture_second_num,picture_third_num,maxlong_colume,colume_long[94], long_max,picture_white=0,picture_black=0,jump_point1;
// float black_ratio;
int red_x_mid,red_y_mid,x_err_red;
float err_picture;
int red_left,red_right;
float real_picture_distance,recognize_distance,recognize_distance2;

void picture(void)
{
        picture_white=0;
        picture_black=0;
        jump_point1=0;
        black_ratio=0;
        err_picture=100;
        x_err_red=200;
        red_x_mid=0;
        red_y_mid=0;


                   float K1 = xielv_sideline(55, Left_Sideline[55], 45, Left_Sideline[45], 'k');
                   float B1 = xielv_sideline(55, Left_Sideline[55], 45, Left_Sideline[45], 'b');
                    float K2 = xielv_sideline(55, Right_Sideline[55], 45, Right_Sideline[45], 'k');
                   float B2 = xielv_sideline(55, Right_Sideline[55], 45, Right_Sideline[45], 'b');
                   red_left=K1*MAX(R_h_guai.row,L_h_guai.row)+B1;
                   red_right=K2*MAX(R_h_guai.row,L_h_guai.row)+B2;

                // recognize_distance=Now_Speed*0.10+40;

                recognize_distance=75;
                recognize_distance2=30;



    if(Flag.picture==0)
{



            //    cv::Rect rect(red_find_x,red_find_y,Right_Sideline[L_h_guai.row],red_find_y1 - red_find_y);
            //    cv::rectangle(resizedFrame, rect, cv::Scalar(0, 255, 0), 1);

                Flag.Redblock=0;   
            if(!red_objects.empty())
               {
                // Flag.Redblock=1;
                red_x_mid=resize_cx;
                red_y_mid=resize_cy;
                // err_picture=fabs(real_distance[red_y_mid]-real_distance[L_h_guai.row]);
                x_err_red=abs((Right_Sideline[red_y_mid]+Left_Sideline[red_y_mid])/2-red_x_mid);
                // if(err_picture<15)//&&abs(center.x-L_h_guai.column)<20
                Flag.Redblock=1;
               }

    




               if(Flag.Redblock==1&&real_distance[red_y_mid]<100&&Left_Sideline[red_y_mid]<red_x_mid&&red_x_mid<Right_Sideline[red_y_mid]){Flag.picture=2;}
    //         if(x_err_red>5&&imgInfo.R_straight_flag==0&&imgInfo.R_straight_flag==1)
    //         {
    //            Flag.small_rock =1;
    //         }

    //         if(x_err_red>5&&imgInfo.R_straight_flag==1&&imgInfo.R_straight_flag==0)
    //         {
    //            Flag.small_rock =2;
    //         }

    // if(Flag.small_rock ==1||Flag.small_rock ==2)
    // {
    //     if(distance>10)
    //     {
    //         distance=0;
    //         Flag.small_rock =0;
    //     }
    // }

}





    if(Flag.picture==1){
        
    }
    if(Flag.picture==2){


               
                      Flag.Redblock=0;      
            if(!red_objects.empty())
               { 
                // Flag.Redblock=1;
                red_x_mid=resize_cx;
                red_y_mid=resize_cy;
                // err_picture=fabs(real_distance[red_y_mid]-real_distance[L_h_guai.row]);
                x_err_red=abs((Right_Sideline[red_y_mid]+Left_Sideline[red_y_mid])/2-red_x_mid);
                // if(err_picture<15)//&&abs(center.x-L_h_guai.column)<20
                Flag.Redblock=1;
               }
               if(real_distance[red_y_mid]<35) Flag.infer = 1;

    




                if(Flag.weapon>=1)
                {
                    printf("检测到 weapon\n");//左绕
                    Flag.infer = 0;
                    Flag.supply = 0;
                    Flag.vehicle = 0;
                    Flag.weapon = 0;
                    Flag.picture = 3;

                }

                if(Flag.supply>=1)
                {   printf("检测到 supply\n");//右绕
                    Flag.infer = 0;
                    Flag.supply= 0;
                    Flag.vehicle = 0;
                    Flag.weapon = 0;
                    Flag.picture = 4;

                }


                if(Flag.vehicle>=1)
                {   printf("检测到 vehicle\n");//交通工具
                    Flag.infer = 0;
                    Flag.supply = 0;
                    Flag.vehicle = 0;
                    Flag.weapon = 0;
                    Flag.picture = 5;

                    
                }



        
    }

    if(Flag.picture==3){

        if(distance_picture>40)
        {
            distance_picture=0;
            //    Image.Kp=3.5*3.0;
            //结束绕行
            Flag.picture = 6;//清除标志位
            distance_picture=0;
            printf("left\n");

        
        }
    }

    if(Flag.picture==4){
       
        if(distance_picture>40)
        {
            distance_picture=0;
            //    Image.Kp=3.5*3.0;
            //结束绕行
            Flag.picture = 6;//清除标志位
            distance_picture=0;
            printf("右行结束\n");

        }
    }

        if(Flag.picture==5){
       
        if(distance_picture>40)
        {
            distance_picture=0;
            // Image.Kp=3.5*3.0;
            //结束绕行
            Flag.picture = 6;//清除标志位
            distance_picture=0;
            printf("straight行结束\n");

        }
    }

            if(Flag.picture==6){
        if(distance_picture>40)
        {
            distance_picture=0;
            // Image.Kp=3.5*3.0;
            //结束绕行
            Flag.picture = 0;//清除标志位
            distance_picture=0;
            printf("picture结束\n");

        }
            }

}



void picture1(void)
{
        picture_white=0;
        picture_black=0;
        jump_point1=0;
        black_ratio=0;
        err_picture=100;
        x_err_red=200;



                   float K1 = xielv_sideline(55, Left_Sideline[55], 45, Left_Sideline[45], 'k');
                   float B1 = xielv_sideline(55, Left_Sideline[55], 45, Left_Sideline[45], 'b');
                    float K2 = xielv_sideline(55, Right_Sideline[55], 45, Right_Sideline[45], 'k');
                   float B2 = xielv_sideline(55, Right_Sideline[55], 45, Right_Sideline[45], 'b');
                   red_left=K1*MAX(R_h_guai.row,L_h_guai.row)+B1;
                   red_right=K2*MAX(R_h_guai.row,L_h_guai.row)+B2;

                // recognize_distance=Now_Speed*0.10+40;

                recognize_distance=Now_Speed*0.1+40;
                recognize_distance2=30;
    if(Flag.picture==0)
{

      if(R_h_guai.flag==1&&L_h_guai.flag==1)//&&(real_distance[imgInfo.top]-real_distance[R_h_guai.row])>25&&imgInfo.Both_lose==0 
{
 
 //           &&(real_distance[imgInfo.top]-real_distance[MAX(R_h_guai.row,L_h_guai.row)])>60&&real_distance[imgInfo.top]>100&&MAX(R_h_guai.row,L_h_guai.row)<real_distance_to_row(40)
            if(MAX(R_h_guai.row,L_h_guai.row)>real_distance_to_row(recognize_distance))
            {
                    Flag.Redblock=0;red_x_mid=0;red_y_mid=0;
               red_find_x=red_left;//94//320
               red_find_y=real_distance_to_row(real_distance[MAX(R_h_guai.row,L_h_guai.row)]+12);//red_find_y到R_h_guai.row（*4)
                              if(red_find_y<imgInfo.top)red_find_y=imgInfo.top;
               red_find_y1=MAX(R_h_guai.row,L_h_guai.row);
               DetectRedBlock(resizedFrame,red_find_x,red_find_y,red_right- red_find_x,red_find_y1 - red_find_y);
                // cv::Rect rect(red_find_x,red_find_y,Right_Sideline[MAX(R_h_guai.row,L_h_guai.row)],red_find_y1 - red_find_y);
                // cv::rectangle(resizedFrame, rect, cv::Scalar(0, 255, 0), 1);
        
                //cv::circle(resizedFrame, bestCenter, 1, cv::Scalar(0, 255, 0), -1);
            if(!red_objects.empty())
               {
                // Flag.Redblock=1;
                red_x_mid=resize_cx;
                red_y_mid=resize_cy;
                err_picture=fabs(real_distance[red_y_mid]-real_distance[MAX(R_h_guai.row,L_h_guai.row)]);
                x_err_red=abs((Right_Sideline[red_y_mid]+Left_Sideline[red_y_mid])/2-red_x_mid);
                // if(err_picture<20)//&&abs(center.x-R_h_guai.column)<20)
                Flag.Redblock=1;
               }
            } 


                
}




     else if(R_h_guai.flag==1){
        //&&(real_distance[imgInfo.top]-real_distance[R_h_guai.row])>60&&real_distance[imgInfo.top]>100&&R_h_guai.row<real_distance_to_row(40)
      if(R_h_guai.row>real_distance_to_row(recognize_distance)
      )
{


                    Flag.Redblock=0;red_x_mid=0;red_y_mid=0;
               red_find_x=red_left;//94//320
               red_find_y=real_distance_to_row(real_distance[R_h_guai.row]+12);//red_find_y到R_h_guai.row（*4)
               if(red_find_y<imgInfo.top)red_find_y=imgInfo.top;
               red_find_y1=R_h_guai.row;
               DetectRedBlock(resizedFrame,red_find_x,red_find_y,red_right - red_find_x,red_find_y1 - red_find_y);
                // cv::Rect rect(red_find_x,red_find_y,Right_Sideline[R_h_guai.row],red_find_y1 - red_find_y);
                // cv::rectangle(resizedFrame, rect, cv::Scalar(0, 255, 0), 1);
        
                //cv::circle(resizedFrame, bestCenter, 1, cv::Scalar(0, 255, 0), -1);
                             
            if(!red_objects.empty())
               {
                // Flag.Redblock=1;
                red_x_mid=resize_cx;
                red_y_mid=resize_cy;
                err_picture=fabs(real_distance[red_y_mid]-real_distance[R_h_guai.row]);
                x_err_red=abs((Right_Sideline[red_y_mid]+Left_Sideline[red_y_mid])/2-red_x_mid);
                // if(err_picture<15)//&&abs(center.x-R_h_guai.column)<20
                Flag.Redblock=1;
               }


}
}

      else if(L_h_guai.flag==1){
        //&&(real_distance[imgInfo.top]-real_distance[L_h_guai.row])>60&&real_distance[imgInfo.top]>100&&L_h_guai.row<real_distance_to_row(40)
       if(L_h_guai.row>real_distance_to_row(recognize_distance)
      )
{



                    Flag.Redblock=0;red_x_mid=0;red_y_mid=0;
               red_find_x=red_left;//Right_Sideline[L_h_guai.row]
               red_find_y=real_distance_to_row(real_distance[L_h_guai.row]+12);//
                              if(red_find_y<imgInfo.top)red_find_y=imgInfo.top;
                            red_find_y1=L_h_guai.row;
               DetectRedBlock(resizedFrame,red_find_x,red_find_y,red_right - red_find_x,red_find_y1 - red_find_y);
            //    cv::Rect rect(red_find_x,red_find_y,Right_Sideline[L_h_guai.row],red_find_y1 - red_find_y);
            //    cv::rectangle(resizedFrame, rect, cv::Scalar(0, 255, 0), 1);

                            
            if(!red_objects.empty())
               { 
                // Flag.Redblock=1;
                red_x_mid=resize_cx;
                red_y_mid=resize_cy;
                err_picture=fabs(real_distance[red_y_mid]-real_distance[L_h_guai.row]);
                x_err_red=abs((Right_Sideline[red_y_mid]+Left_Sideline[red_y_mid])/2-red_x_mid);
                // if(err_picture<15)//&&abs(center.x-L_h_guai.column)<20
                Flag.Redblock=1;
               }

}}






               if(Flag.Redblock==1){Flag.picture=2;}
    //         if(x_err_red>5&&imgInfo.R_straight_flag==0&&imgInfo.R_straight_flag==1)
    //         {
    //            Flag.small_rock =1;
    //         }

    //         if(x_err_red>5&&imgInfo.R_straight_flag==1&&imgInfo.R_straight_flag==0)
    //         {
    //            Flag.small_rock =2;
    //         }

    // if(Flag.small_rock ==1||Flag.small_rock ==2)
    // {
    //     if(distance>10)
    //     {
    //         distance=0;
    //         Flag.small_rock =0;
    //     }
    // }

}





    if(Flag.picture==1){
        
    }
    if(Flag.picture==2){

        Flag.infer = 1;
        if((L_h_guai.flag==1&&real_distance[MAX(R_h_guai.row,L_h_guai.row)]<30)||(R_h_guai.flag==1&&abs(recognize_distance2-real_distance[MAX(R_h_guai.row,L_h_guai.row)])<2))//&&fabs(Now_Speed)<10
        {
    //    Image.Kp/=2.5;
           if(R_h_guai.flag==1&&L_h_guai.flag==1)//&&(real_distance[imgInfo.top]-real_distance[R_h_guai.row])>25&&imgInfo.Both_lose==0 
            {
 
            // if(MAX(R_h_guai.row,L_h_guai.row)>real_distance_to_row(80)&&(real_distance[imgInfo.top]-real_distance[MAX(R_h_guai.row,L_h_guai.row)])>60&&real_distance[imgInfo.top]>100) 
            // {
                Flag.Redblock=0;red_x_mid=0;red_y_mid=0;
               red_find_x=red_left;//94//320
               red_find_y=real_distance_to_row(real_distance[MAX(R_h_guai.row,L_h_guai.row)]+12);//red_find_y到R_h_guai.row（*4)
                              if(red_find_y<imgInfo.top)red_find_y=imgInfo.top;
               red_find_y1=MAX(R_h_guai.row,L_h_guai.row);
               DetectRedBlock(resizedFrame,red_find_x,red_find_y,red_right - red_find_x,red_find_y1 - red_find_y);
                // cv::Rect rect(red_find_x,red_find_y,Right_Sideline[MAX(R_h_guai.row,L_h_guai.row)],red_find_y1 - red_find_y);
                // cv::rectangle(resizedFrame, rect, cv::Scalar(0, 255, 0), 1);
        
                //cv::circle(resizedFrame, bestCenter, 1, cv::Scalar(0, 255, 0), -1);
            if(!red_objects.empty())
               {
                // Flag.Redblock=1;
                red_x_mid=resize_cx;
                red_y_mid=resize_cy;
                err_picture=fabs(real_distance[red_y_mid]-real_distance[MAX(R_h_guai.row,L_h_guai.row)]);
                x_err_red=abs((Right_Sideline[red_y_mid]+Left_Sideline[red_y_mid])/2-red_x_mid);
                // if(err_picture<20)//&&abs(center.x-R_h_guai.column)<20)
                // Flag.Redblock=1;
               }
            // } 


                
}





      else if(R_h_guai.flag==1&&abs(recognize_distance2-real_distance[R_h_guai.row])<2)//&&fabs(Now_Speed)<10&&R_h_guai.row>real_distance_to_row(80)&&(real_distance[imgInfo.top]-real_distance[R_h_guai.row])>60&&real_distance[imgInfo.top]>100
{


                Flag.Redblock=0;red_x_mid=0;red_y_mid=0;
               red_find_x=red_left;//94//320
               red_find_y=real_distance_to_row(real_distance[R_h_guai.row]+12);//red_find_y到R_h_guai.row（*4)
                              if(red_find_y<imgInfo.top)red_find_y=imgInfo.top;
               red_find_y1=R_h_guai.row;
               DetectRedBlock(resizedFrame,red_find_x,red_find_y,red_right - red_find_x,red_find_y1 - red_find_y);
                // cv::Rect rect(red_find_x,red_find_y,Right_Sideline[R_h_guai.row],red_find_y1 - red_find_y);
                // cv::rectangle(resizedFrame, rect, cv::Scalar(0, 255, 0), 1);
        
                //cv::circle(resizedFrame, bestCenter, 1, cv::Scalar(0, 255, 0), -1);
                             
            if(!red_objects.empty())
               {
                // Flag.Redblock=1;
                red_x_mid=resize_cx;
                red_y_mid=resize_cy;
                err_picture=fabs(real_distance[red_y_mid]-real_distance[R_h_guai.row]);
                x_err_red=abs((Right_Sideline[red_y_mid]+Left_Sideline[red_y_mid])/2-red_x_mid);
                // if(err_picture<15)//&&abs(center.x-R_h_guai.column)<20
                // Flag.Redblock=1;
               }


}

      else if(L_h_guai.flag==1&&abs(recognize_distance2-real_distance[L_h_guai.row])<2)//&&fabs(Now_Speed)<10&&L_h_guai.row>real_distance_to_row(80)&&(real_distance[imgInfo.top]-real_distance[L_h_guai.row])>60&&real_distance[imgInfo.top]>100
{



                Flag.Redblock=0;red_x_mid=0;red_y_mid=0;
               red_find_x=red_left;//Right_Sideline[L_h_guai.row]
               red_find_y=real_distance_to_row(real_distance[L_h_guai.row]+12);//
                              if(red_find_y<imgInfo.top)red_find_y=imgInfo.top;
                            red_find_y1=L_h_guai.row;
               DetectRedBlock(resizedFrame,red_find_x,red_find_y,red_right - red_find_x,red_find_y1 - red_find_y);
            //    cv::Rect rect(red_find_x,red_find_y,Right_Sideline[L_h_guai.row],red_find_y1 - red_find_y);
            //    cv::rectangle(resizedFrame, rect, cv::Scalar(0, 255, 0), 1);

                            
            if(!red_objects.empty())
               {
                // Flag.Redblock=1;
                red_x_mid=resize_cx;
                red_y_mid=resize_cy;
                err_picture=fabs(real_distance[red_y_mid]-real_distance[L_h_guai.row]);
                x_err_red=abs((Right_Sideline[red_y_mid]+Left_Sideline[red_y_mid])/2-red_x_mid);
                // if(err_picture<15)//&&abs(center.x-L_h_guai.column)<20
                // Flag.Redblock=1;
               }
}

                if(Flag.weapon>=1)
                {
                    printf("检测到 weapon\n");//左绕
                    Flag.infer = 0;
                    Flag.supply = 0;
                    Flag.vehicle = 0;
                    Flag.weapon = 0;
                    Flag.picture = 3;

                }

                if(Flag.supply>=1)
                {   printf("检测到 supply\n");//右绕
                    Flag.infer = 0;
                    Flag.supply= 0;
                    Flag.vehicle = 0;
                    Flag.weapon = 0;
                    Flag.picture = 4;

                }


                if(Flag.vehicle>=1)
                {   printf("检测到 vehicle\n");//交通工具
                    Flag.infer = 0;
                    Flag.supply = 0;
                    Flag.vehicle = 0;
                    Flag.weapon = 0;
                    Flag.picture = 5;

                    
                }



        }
    }

    if(Flag.picture==3){

        if(distance_picture>30)
        {
            distance_picture=0;
            //    Image.Kp=3.5*3.0;
            //结束绕行
            Flag.picture = 6;//清除标志位
            distance_picture=0;
            printf("left\n");

        
        }
    }

    if(Flag.picture==4){
       
        if(distance_picture>30)
        {
            distance_picture=0;
            //    Image.Kp=3.5*3.0;
            //结束绕行
            Flag.picture = 6;//清除标志位
            distance_picture=0;
            printf("右行结束\n");

        }
    }

        if(Flag.picture==5){
       
        if(distance_picture>30)
        {
            distance_picture=0;
            // Image.Kp=3.5*3.0;
            //结束绕行
            Flag.picture = 6;//清除标志位
            distance_picture=0;
            printf("straight行结束\n");

        }
    }

            if(Flag.picture==6){
        if(distance_picture>40)
        {
            distance_picture=0;
            // Image.Kp=3.5*3.0;
            //结束绕行
            Flag.picture = 0;//清除标志位
            distance_picture=0;
            printf("picture结束\n");

        }
            }

}



void protect(void)
{
// if(Flag.Zebra_cross != 1&&Flag.Zebra_cross != 4&&run_flag==1)//&&Flash.mtv_exposure_time>=100
// {
//     if(imgInfo.top >=57||fabs(icm_data.gyro_z)>30)//&&Flag.Zhangai != 1
//     {
//         run_flag =2;
//         printf("异常触发保护\r\n");
//     }//||Speed_Encoder_l>2000||Speed_Encoder_r>2000||Speed_now>1500


// }
}
lq_camera_ex cam(320,240,156,LQ_CAMERA_0CPU_MJPG,LQ_CAMERA_PATH);
void image_init(void)
{  
    
    std::string model_param = "tiny_classifier_fp32.ncnn.param";//tiny_classifier_fp32.ncnn.param
    std::string model_bin   = "tiny_classifier_fp32.ncnn.bin";//tiny_classifier_fp32.ncnn.bin
    int input_width    = 64;
    int input_height   = 64;
    std::vector<std::string> labels = {"supply", "vehicle", "weapon"};
        // 归一化参数（ImageNet标准）
     float mean_vals[3] = {123.675f, 116.28f, 103.53f};
     float norm_vals[3] = {0.01712475f, 0.017507f, 0.01742919f};

    classifier.SetModelPath(model_param, model_bin);
    classifier.SetInputSize(input_width, input_height);
    classifier.SetLabels(labels);
    classifier.SetNormalize(mean_vals, norm_vals);
    classifier.Init();

    cam.start_collect();
    if(cam.is_cam_opened())
    {
        printf("Camera opened successfully!\n");
    }
    else{
        printf("Camera opened failed!\n");
        return;
    }
    cam.set_exposure_manual(60);
    printf("龙邱摄像头宽度:%d\n",cam.get_camera_width());
    printf("龙邱摄像头高度:%d\n",cam.get_camera_height());
    printf("龙邱摄像头帧率:%d\n",cam.get_camera_fps());
    //预分配内存
    grayFrame.create(LCDH_0, LCDW_0, CV_8UC1);
    resizedFrame.create(LCDH_1, LCDW_1, CV_8UC1);
    translatedFrame.create(LCDH_1, LCDW_1, CV_8UC1);
    binaryFrame.create(LCDH_1, LCDW_1, CV_8UC1);
    translationMatrix = (cv::Mat_<float>(2, 3) << 1, 0, 0, 0, 1, 0);//将捕获图像向右平移3个像素点
   camera_server.start_server(8080);//打开图传服务器

}


void zf_camera_init(){

}


uint16_t jump_point,finish_flag;
float distance_cross,distance,distance_picture;


void distance_judge(void)//1m=66
{
           if(Flag.Huandao_L==1||Flag.Huandao_R==1||Flag.Huandao_L==2||Flag.Huandao_R==2||Flag.Huandao_R==6||Flag.Huandao_L==6
                  ||Flag.small_rock||Flag.Zebra_cross==0||Flag.Zebra_cross==2||Flag.Zebra_cross==4||Flag.ramp!=0)//||Flag.Huandao_R==3||Flag.Huandao_L==3
    distance+=(float)(encoder_L.count_now+encoder_R.count_now)/2/350;
    if(Flag.picture==3||Flag.picture==4||Flag.picture==5||Flag.picture==6)
    {
    distance_picture+=(float)(encoder_L.count_now+encoder_R.count_now)/2/350;
    }
}


 void zebra_corssing(void)
 {


    if(Flag.Zebra_cross==0)
    {
    // jump_point=0;
    // for(int i = 25;i < 44; i++)
    // {
    //     for(int j = 30; j < 64; j++)
    //     {
    //         if(Image_Use[i][j] == 255 && Image_Use[i][j+1] == 0)
    //         {
    //             jump_point ++;
    //         }
    //     }
    // }

    // if(jump_point>25)//&&top_white_num>0&&fabs(real_distance[imgInfo.top]-real_distance[MAX(R_h_guai.row,L_h_guai.row)])<20&&imgInfo.top>30
    // {
    //             Flag.Zebra_cross=1;
    //         // }

    // }
    // }

    // if(Flag.Zebra_cross==1)
    // {

        if(distance>100)
        {
            Flag.Zebra_cross=2;
            distance=0;
        }
    }

    if(Flag.Zebra_cross==2)
    {
        if(distance>30)
        {
            Flag.Zebra_cross=3;
            distance=0;
        }
    }

    if(Flag.Zebra_cross==3)
    {
        jump_point=0; 
        for(int i = 25;i < 44; i++)
        {
            for(int j = 30; j < 64; j++)
            {
                if(Image_Use[i][j] == 255 && Image_Use[i][j+1] == 0)
                {
                    jump_point ++;
                }
            }
        }


        if(jump_point>25&&Flag.ramp==0&&Flag.picture==0)//&&top_white_num>0&&fabs(real_distance[imgInfo.top]-real_distance[MAX(R_h_guai.row,L_h_guai.row)])<20&&imgInfo.top>30
        {
            Flag.Zebra_cross=4;
        }
    }

    if(Flag.Zebra_cross==4)
    {
        if(distance>10)
        {
            Flag.Zebra_cross=5;
            run_flag=2;
            esc_pwm.set_duty(500);
        }

    }

 }






uint16_t small_rock_r_num,small_rock_l_num;
void small_rock(void)
{
    if(Flag.small_rock ==0)
    {
                   float K1 = xielv_sideline(55, Left_Sideline[55], 45, Left_Sideline[45], 'k');
                   float B1 = xielv_sideline(55, Left_Sideline[55], 45, Left_Sideline[45], 'b');
                    float K2 = xielv_sideline(55, Right_Sideline[55], 45, Right_Sideline[45], 'k');
                   float B2 = xielv_sideline(55, Right_Sideline[55], 45, Right_Sideline[45], 'b');
                   red_left=K1*MAX(R_h_guai.row,L_h_guai.row)+B1;
                   red_right=K2*MAX(R_h_guai.row,L_h_guai.row)+B2;


      if(R_h_guai.flag==1&&L_h_guai.flag==1)//&&(real_distance[imgInfo.top]-real_distance[R_h_guai.row])>25&&imgInfo.Both_lose==0 
{
 
            if(MAX(R_h_guai.row,L_h_guai.row)>real_distance_to_row(40)&&MAX(R_h_guai.row,L_h_guai.row)<real_distance_to_row(10)&&
            (real_distance[imgInfo.top]-real_distance[MAX(R_h_guai.row,L_h_guai.row)])>60&&real_distance[imgInfo.top]>100) 
            {
                    Flag.Redblock=0;red_x_mid=0;red_y_mid=0;
               red_find_x=red_left;//94//320
               red_find_y=real_distance_to_row(real_distance[MAX(R_h_guai.row,L_h_guai.row)]+12);//red_find_y到R_h_guai.row（*4)
               red_find_y1=MAX(R_h_guai.row,L_h_guai.row);
               DetectRedBlock(resizedFrame,red_find_x,red_find_y,red_right- red_find_x,red_find_y1 - red_find_y);
                // cv::Rect rect(red_find_x,red_find_y,Right_Sideline[MAX(R_h_guai.row,L_h_guai.row)],red_find_y1 - red_find_y);
                // cv::rectangle(resizedFrame, rect, cv::Scalar(0, 255, 0), 1);
        
                //cv::circle(resizedFrame, bestCenter, 1, cv::Scalar(0, 255, 0), -1);
            if(!red_objects.empty())
               {
                // Flag.Redblock=1;
                red_x_mid=resize_cx;
                red_y_mid=resize_cy;
                err_picture=fabs(real_distance[red_y_mid]-real_distance[MAX(R_h_guai.row,L_h_guai.row)]);
                x_err_red=abs((Right_Sideline[red_y_mid]+Left_Sideline[red_y_mid])/2-red_x_mid);
                // if(err_picture<20)//&&abs(center.x-R_h_guai.column)<20)
                // Flag.Redblock=1;
               }
            } 


                
}





      if(R_h_guai.flag==1&&R_h_guai.row>real_distance_to_row(40)&&R_h_guai.row<real_distance_to_row(10)&&
      (real_distance[imgInfo.top]-real_distance[R_h_guai.row])>60&&real_distance[imgInfo.top]>100)
{


                    Flag.Redblock=0;red_x_mid=0;red_y_mid=0;
               red_find_x=red_left;//94//320
               red_find_y=real_distance_to_row(real_distance[R_h_guai.row]+12);//red_find_y到R_h_guai.row（*4)
               red_find_y1=R_h_guai.row;
               DetectRedBlock(resizedFrame,red_find_x,red_find_y,red_right - red_find_x,red_find_y1 - red_find_y);
                // cv::Rect rect(red_find_x,red_find_y,Right_Sideline[R_h_guai.row],red_find_y1 - red_find_y);
                // cv::rectangle(resizedFrame, rect, cv::Scalar(0, 255, 0), 1);
        
                //cv::circle(resizedFrame, bestCenter, 1, cv::Scalar(0, 255, 0), -1);
                             
            if(!red_objects.empty())
               {
                // Flag.Redblock=1;
                red_x_mid=resize_cx;
                red_y_mid=resize_cy;
                err_picture=fabs(real_distance[red_y_mid]-real_distance[R_h_guai.row]);
                x_err_red=abs((Right_Sideline[red_y_mid]+Left_Sideline[red_y_mid])/2-red_x_mid);
                // if(err_picture<15)//&&abs(center.x-R_h_guai.column)<20
                // Flag.Redblock=1;
               }


}

      else if(L_h_guai.flag==1&&L_h_guai.row>real_distance_to_row(40)&&L_h_guai.row<real_distance_to_row(10)
      &&(real_distance[imgInfo.top]-real_distance[L_h_guai.row])>60&&real_distance[imgInfo.top]>100)
{



                    Flag.Redblock=0;red_x_mid=0;red_y_mid=0;
               red_find_x=red_left;//Right_Sideline[L_h_guai.row]
               red_find_y=real_distance_to_row(real_distance[L_h_guai.row]+12);//
                            red_find_y1=L_h_guai.row;
               DetectRedBlock(resizedFrame,red_find_x,red_find_y,red_right - red_find_x,red_find_y1 - red_find_y);
            //    cv::Rect rect(red_find_x,red_find_y,Right_Sideline[L_h_guai.row],red_find_y1 - red_find_y);
            //    cv::rectangle(resizedFrame, rect, cv::Scalar(0, 255, 0), 1);

                            
            if(!red_objects.empty())
               { 
                // Flag.Redblock=1;
                red_x_mid=resize_cx;
                red_y_mid=resize_cy;
                err_picture=fabs(real_distance[red_y_mid]-real_distance[L_h_guai.row]);
                x_err_red=abs((Right_Sideline[red_y_mid]+Left_Sideline[red_y_mid])/2-red_x_mid);
                // if(err_picture<15)//&&abs(center.x-L_h_guai.column)<20
                // Flag.Redblock=1;
               }







                


}




            if(x_err_red>5&&imgInfo.R_straight_flag==0&&imgInfo.R_straight_flag==1)
            {
               Flag.small_rock =1;
            }

            if(x_err_red>5&&imgInfo.R_straight_flag==1&&imgInfo.R_straight_flag==0)
            {
               Flag.small_rock =2;
            }



    }


    if(Flag.small_rock ==1||Flag.small_rock ==2)
    {
        if(distance>10)
        {
            distance=0;
            Flag.small_rock =0;
        }
    }
}


int ramp_white_num;
float ramp_line,ramp_err;
void ramp(void)
{

//    float k = xielv_sideline(30, (int)Dir_Err[30], 45, (int)Dir_Err[45], 'k');
//    float b = xielv_sideline(30, (int)Dir_Err[30], 45, (int)Dir_Err[45], 'b');
if(Flag.ramp==0)
{
     
    if(imgInfo.top<10)
  {
        ramp_err=0;
            //  ramp_err=50;
            ramp_line=0;
       for(int i = imgInfo.top+5;i<50 ;i++)
    {
            //   if(fabs(Dir_Err[i]-k*i-b)>2)
            // {
            //        if(i<50||(Right_Sideline_flag[i]==1&&Left_Sideline_flag[i]==1))
            //            ramp_line++;
            // }
            if(Left_Sideline_flag[i]==0||Right_Sideline_flag[i]==0)
            {
                ramp_line++;
            }

            ramp_white_num+=white_width[i];
        //   ramp_err+=Dir_Err[i];
    }
       if(ramp_white_num>500&&ramp_line==0)//ramp_line==0&&&&B<0.2dl1x_distance_raw>50&&dl1x_distance_raw<1500&&fabs(ramp_err)<30
    {
           Flag.ramp=2;

    }

  }
}
// if(Flag.ramp==1)
// {
//     if(distance>30)
//  {
//         Flag.ramp=2;
//  }
// }
if(Flag.ramp==2)
{
    if(distance>100)
 {
        Flag.ramp=3;
        distance=0;

 }
}
if(Flag.ramp==3)
{
    if(distance>150)
 {
        Flag.ramp=0;
        distance=0;

 }
}
}
 /***************************************************检测红色矩形块******************************************************/


//角点排序函数 根据极点进行排序 
//输出顺序
//points[0] —— 左上角（Top-Left）
//points[1] —— 右上角（Top-Right）
//points[2] —— 右下角（Bottom-Right）
//points[3] —— 左下角（Bottom-Left）
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
    if(x > img.cols || y> img.rows || x < 0 || y<0) 
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
    if(x > img.cols || y> img.rows || x < 0 || y<0) 
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

//粗找红块 并返回一个roi区域 兼容94*60

//转hsv找色块 如果色块超出赛道边界 或 超过截至行 return false
// inline bool hsv_get_roi(const cv::Mat& img,cv::Rect& rect)
// {
//     if(img.empty) return false;
//     if(img.width < 0 || img.height < 0) return false;

//     cv::Mat mask1,mask2,mask;
//     std::vector<cv::Point> contours;
    
//     cv::inRange(hsv, cv::Scalar(0, 120, 70), cv::Scalar(10, 255, 255), mask1);//红色阈值
//     cv::inRange(hsv, cv::Scalar(160, 120, 70), cv::Scalar(179, 255, 255), mask2);//红色阈值

//     mask = mask1| mask2;
    
//     cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

//     int bottom_y = -1;
//     cv::Point bestCenter;
//     cv::Rect bestRect;
//     //找出位置最靠下的红色矩形块
//     for(auto& c : contours)
//     {
//         cv::Rect rect = cv::boundingRect(c);
//         int cx = rect.x + rect.width / 2;
//         int cy = rect.y + rect.height / 2;

//         int area = rect.area();

//         if(area < 10) continue;//面积太小

//         if(cx < Left_Sideline[cy] || cx > Right_Sideline[cy] || cy < imgInfo.top) return false;//超出赛道边界 或在截至行以上


//         if (cy > bottom_y)
//         {
//             bottom_y = cy;
//             bestCenter = cv::Point(cx, cy);
//             bestRect = rect;
//         }
//     }

//     if (bottom_y >= 0)
//     {
//         bestCenter.y += roi_y;
//         bestRect.y += roi_y;
//     }
    

// }

// inline bool rgb_get_roi(const cv::Mat& img,cv::Rect& rect)
// {
//     if(img.empty) return false;
//     if(img.width < 0 || img.height < 0) return false;

// }

// //参数：输入图像 输出roi矩形 mode模式选择 0-转hsv处理 1-rgb扫线处理
// inline bool get_red_roi(const cv::Mat& img,cv::Rect& roi_src,int model)
// {
//     roi_src =cv::Rect();
//     if(img.empty) return false;
//     if(img.width < 0 || src.height < 0) return false;

//     //转hsv处理 调用相关函数
//     if(model == 0)
//     {

//     }

//     //rgb扫线处理 调用相关函数
//     if(model == 1)
//     {

//     }
// }

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

    const int small_w = 94;
    const int small_h = 60;

    const float small_to_src_x = static_cast<float>(src_img.cols) / static_cast<float>(small_w);
    const float small_to_src_y = static_cast<float>(src_img.rows) / static_cast<float>(small_h);
    const float src_to_small_y = static_cast<float>(small_h) / static_cast<float>(src_img.rows);

    const int small_search_top = ClampTargetInt(imgInfo.top, 0, small_h - 1);
    const int small_search_bottom = ClampTargetInt(imgInfo.bottom - 1, small_search_top, small_h - 1);

    const int search_top = ClampTargetInt(
        static_cast<int>(std::lround(small_search_top * small_to_src_y)),
        0,
        src_img.rows - 1);

    const int search_bottom = ClampTargetInt(
        static_cast<int>(std::lround((small_search_bottom + 1) * small_to_src_y)) - 1,
        search_top,
        src_img.rows - 1);
  
    const int y_stride = 1;//向内收缩 y坐标每两行
    const int x_stride = 4;//向内收缩 x坐标每四列找一次

    // 至少连续两个采样点是红色，才认为这一行有可靠红色段
    const int min_run_samples = 2;

    for (int y = search_bottom; y >= search_top; y -= y_stride)
    {
        const int small_y = ClampTargetInt(
            static_cast<int>(std::lround(y * src_to_small_y)),
            0,
            small_h - 1);

        int lx_small = Left_Sideline[small_y];
        int rx_small = Right_Sideline[small_y];

        // 左右边界无效时直接跳过，不退化为整行搜索
        // 这样可以减少误把赛道外红色物体当成目标的概率
        if (lx_small < 0 || lx_small >= small_w ||
            rx_small < 0 || rx_small >= small_w)
        {   printf("本行左右边界无效跳过\n");
            continue;
        }

        if (lx_small > rx_small)
        {
            std::swap(lx_small, rx_small);
        }

        if (rx_small - lx_small < 3)
        {
            continue;
        }

        int lx = static_cast<int>(std::floor(lx_small * small_to_src_x));//映射到大图
        int rx = static_cast<int>(std::ceil((rx_small + 1) * small_to_src_x)) - 1;//映射到大图

        lx = ClampTargetInt(lx, 0, src_img.cols - 1);
        rx = ClampTargetInt(rx, 0, src_img.cols - 1);

        // 向赛道内部收一点，避免贴边噪声
        const int lane_inner_pad = 4;
        lx = ClampTargetInt(lx + lane_inner_pad, 0, src_img.cols - 1);
        rx = ClampTargetInt(rx - lane_inner_pad, 0, src_img.cols - 1);

        if (lx > rx)
        {
            printf("大图中左边线>右边线跳过\n");
            continue;
        }

        int run_start_x = -1;
        int run_len = 0;

        int best_run_start_x = -1;
        int best_run_len = 0;

        for (int x = lx; x <= rx; x += x_stride)
        {
            if (get_red_rgb(src_img, x, y))//快速访问图像中的单个点的rgb三色通道 并判断是否为红色
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

        if (run_len > best_run_len)
        {
            best_run_len = run_len;
            best_run_start_x = run_start_x;
        }

        if (best_run_len >= min_run_samples && best_run_start_x >= 0)
        {
            
            const int seed_x = best_run_start_x + (best_run_len / 2) * x_stride;
            seed = cv::Point(ClampTargetInt(seed_x, 0, src_img.cols - 1), y);

            // 再确认种子点附近红色密度，防止单点误判
            if (CountRed3x3(src_img, seed.x, seed.y) >= 3)// 统计 3x3 邻域内有多少个红色点
            {   
                // printf("粗找到种子点");
                return true;
            }
            else{
                printf("3x3领域内种子点不足跳过\n");
            }
        }
        // else{
        //     printf("小于两个红点\n");
        // }
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
    const float ROI_W_SCALE = 0.92f;
    const float ROI_H_SCALE = 0.92f;

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

    const int TARGET_ROI_SIZE = 64;//
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


//target_roi 输出的目标图片区域
//修改：新增参数 作为灰度图抹除红块的断点标记 防止因为返回false 而导致整帧不抹
bool FindTargetRoiByFixedIpm(const cv::Mat& input_frame,
                             cv::Mat& target_roi,
                             struct Guaidian* red_point,
                             cv::Point2f* debug_target_img_pts = nullptr,
                             cv::Point2f* debug_red_img_pts = nullptr,
                             float* debug_valid_ratio = nullptr,
                             bool* erase_pts_ready = nullptr)
{

    if (red_point != nullptr)
    {
        red_point->column = -1;
        red_point->row = -1;
    }

    if (erase_pts_ready != nullptr)
    {
        *erase_pts_ready = false;
    }

    target_roi.release();

    if (input_frame.empty())
    {
        return false;
    }

    cv::Point red_seed;
    cv::RotatedRect red_rect;

    // 1. 粗找红色种子点
    if (!red_detect_rgb(input_frame, red_seed))
    {
        //printf("未找到种子点\n");
        return false;
    }
    // cv::circle(lq_frame, red_seed, 1, cv::Scalar(0, 255, 0), -1);//调试用 看种子点对不对
    // 2. 细找红色定位条
    if (!get_red_contour(input_frame, red_seed, red_rect))
    {   
        //printf("未找到定位矩形块\n");
        return false;
    }

    // 3. 获取红条原图四点
    cv::Point2f red_img_pts[4];
    red_rect.points(red_img_pts);

    // 修改:对红条四个角点排序
    //points[0] —— 左上角（Top-Left）
    //points[1] —— 右上角（Top-Right）
    //points[2] —— 右下角（Bottom-Right）
    //points[3] —— 左下角（Bottom-Left）
    OrderTargetStripQuad(red_img_pts);

    float bottom_mid_x = (red_img_pts[2].x + red_img_pts[3].x) * 0.5f;
    float bottom_mid_y = (red_img_pts[2].y + red_img_pts[3].y) * 0.5f;

    const float src_to_small_x = 94.0f / static_cast<float>(input_frame.cols);
    const float src_to_small_y = 60.0f / static_cast<float>(input_frame.rows);
    //返回红条的最小外接旋转矩形的下底边的中心点并映射到94*60

    int small_x = static_cast<int>(std::lround(bottom_mid_x * src_to_small_x));
    int small_y = static_cast<int>(std::lround(bottom_mid_y * src_to_small_y));

    small_x = ClampTargetInt(small_x, 0, 93);
    small_y = ClampTargetInt(small_y, 0, 59);

    static bool has_last_red_point = false;
    static int last_red_x = -1;
    static int last_red_y = -1;

    const int RED_POINT_DEAD_ZONE = 1;   // 小抖动不更新
    const int RED_POINT_MAX_JUMP = 15;   // 突然跳太远也不更新，防误检

    if (!has_last_red_point)
    {
        last_red_x = small_x;
        last_red_y = small_y;
        has_last_red_point = true;
    }
    else
    {
        int dx = small_x - last_red_x;
        int dy = small_y - last_red_y;

        int dist2 = dx * dx + dy * dy;

        int dead_zone2 = RED_POINT_DEAD_ZONE * RED_POINT_DEAD_ZONE;
        int max_jump2 = RED_POINT_MAX_JUMP * RED_POINT_MAX_JUMP;

        if (dist2 <= dead_zone2)
        {
        // 变化太小，认为是抖动，不更新
        }
        // else if (dist2 >= max_jump2)
        // {
        // // 跳变太大，可能是误检，不更新
        // }
        else
        {
        // 正常变化，更新
        last_red_x = small_x;
        last_red_y = small_y;
        }
    }

    red_point->column = last_red_x;
    red_point->row = last_red_y;


    if (debug_red_img_pts != nullptr)
    {
        for (int i = 0; i < 4; i++)
        {
            debug_red_img_pts[i] = red_img_pts[i];
        }
    }

    // 4. 根据红条原图四点，在 BEV 中生成目标框四点
    cv::Point2f target_bev_pts[4];

    if (!BuildTargetBevQuadFromRedRect(input_frame,
                                       red_img_pts,
                                       target_bev_pts))
    {
        return false;
    }

    // 5. 调试用：把 BEV 目标框反变换回原图，用于图传画绿色框
    if (debug_target_img_pts != nullptr)
    {
        for (int i = 0; i < 4; i++)
        {
            debug_target_img_pts[i] =
                BevPointToImagePoint(target_bev_pts[i], input_frame);
        }
        //printf("成功返回目标框坐标\n");
        OrderTargetStripQuad(debug_target_img_pts);//对图片四点排序
    }


    //程序运行到此处时 说明已经得到所需坐标 可以抹除红块了
    if (debug_red_img_pts != nullptr &&
    debug_target_img_pts != nullptr &&
    erase_pts_ready != nullptr)
    {
    *erase_pts_ready = true;
    }


    // 6. 使用固定 IPM 反变换 + remap 裁剪出固定大小 ROI
    if (!BuildTargetRoiByFixedIpmRemap(input_frame,
                                       target_bev_pts,
                                       target_roi,
                                       debug_valid_ratio))
    {
        return false;
    }


    // map_and_sort(red_img_pts,debug_target_img_pts);
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

 void DetectRedBlock(cv::Mat &src)
{   
    auto start_time = std::chrono::high_resolution_clock::now();
    if(src.empty()){
        return;
    }

    red_init();

    auto checkQuadParallel = [](const std::vector<cv::Point2f>& q,
                            float parallel_th) -> bool
{
    if(q.size() != 4)
        return false;

    for(const auto& p : q)
    {
        if(!std::isfinite(p.x) || !std::isfinite(p.y))
            return false;
    }

    cv::Point2f top    = q[1] - q[0];  // 上边：左上 -> 右上
    cv::Point2f bottom = q[2] - q[3];  // 下边：左下 -> 右下

    cv::Point2f left   = q[3] - q[0];  // 左边：左上 -> 左下
    cv::Point2f right  = q[2] - q[1];  // 右边：右上 -> 右下

    float err_tb = parallelError(top, bottom);
    float err_lr = parallelError(left, right);

    if(err_tb > parallel_th)
        return false;

    if(err_lr > parallel_th)
        return false;

    return true;
};

auto stabilizeQuad = [&](std::vector<cv::Point2f>& q,
                         std::vector<cv::Point2f>& last_q,
                         bool& has_last,
                         float deadband_px,
                         float parallel_th) -> bool
{
    if(q.size() != 4)
        return false;

    // 1. 当前四边形两组对边不近似平行
    if(!checkQuadParallel(q, parallel_th))
    {
        // 如果有上一帧，就继续使用上一帧，避免出现不规则四边形
        // if(has_last)
        // {
        //     q = last_q;
        //     return true;
        // }

        return false;
    }

    // 2. 第一帧，直接保存
    if(!has_last)
    {
        last_q = q;
        has_last = true;
        return true;
    }

    // 3. 计算当前帧和上一帧四个角点的最大距离
    float max_dist = 0.0f;

    for(int i = 0; i < 4; i++)
    {
        float d = cv::norm(q[i] - last_q[i]);

        if(d > max_dist)
            max_dist = d;
    }

    // 4. 小于一定像素值，不更新
    if(max_dist < deadband_px)
    {
        q = last_q;
        return true;
    }

    // 5. 超过死区，认为是真实移动，更新上一帧
    last_q = q;
    return true;
    };

    cv::RotatedRect rect_min,rect_max;
    //在小在图找色块
    std::vector<cv::Point> contour;//原图

    if(!findRedrect(src, rect_min)) return;

    cv::Point2f rect_points_resize[4];
    cv::Point2f rect_points[4];

// if(Flag.infer == 1){
    
    rect_min.points(rect_points_resize);
    std::vector<cv::Point2f> points_resize = orderPoints(rect_points_resize);//对四个角点排序
    //映射回320*240
    for(size_t i = 0; i < 4; i++){
        points_resize[i].x = points_resize[i].x * 3.4f;
        points_resize[i].y = points_resize[i].y * 4.0f;
    }
 
    cv::Rect Rect_roi = cv::boundingRect(points_resize);//在原图画四个角点的最小外接矩形
    int pad = 8;
    Rect_roi.x = Rect_roi.x - pad;
    Rect_roi.y = Rect_roi.y - pad;
    Rect_roi.width = Rect_roi.width + pad * 2;
    Rect_roi.height = Rect_roi.height + pad * 2;

    cv::Rect img_rect(0,0,lq_frame.cols,lq_frame.rows);
    Rect_roi = Rect_roi & img_rect;
    if(Rect_roi.width <= 0 || Rect_roi.height <= 0) return;

    //cv::rectangle(lq_frame, Rect_roi, cv::Scalar(0, 255, 0), 1);

    cv::Mat roi = lq_frame(Rect_roi);//找轮廓的ROI区域
    if(roi.empty()) return;

    std::vector<cv::Point2f> red_corners;
    std::vector<cv::Point> red_contour;
    if(!findRedrect1(roi, rect_max, red_contour)) return;

    if(!getCorners(red_contour,red_corners)) return;



    // ROI 坐标映射回原图坐标
    for(size_t i = 0; i < red_corners.size(); i++)
    {
        red_corners[i].x += Rect_roi.x;
        red_corners[i].y += Rect_roi.y;
    }

    struct Edge{
        cv::Point2f a;
        cv::Point2f b;
        float len;
        float mid_y;
    };
    std::vector<Edge> edges;
    // 检查角点是否越界 同时计算边长
    for(size_t i = 0; i < red_corners.size(); i++)
    {
    if(red_corners[i].x < 0 || red_corners[i].x >= lq_frame.cols ||
       red_corners[i].y < 0 || red_corners[i].y >= lq_frame.rows)
    {
        return;
    }
        Edge e;
        e.a = red_corners[i];
        e.b = red_corners[(i+1)%red_corners.size()];
        e.len = cv::norm(e.b - e.a);
        e.mid_y = (e.a.y + e.b.y) * 0.5f;
        edges.push_back(e);
    }


    std::sort(edges.begin(),edges.end(),[](const Edge& e1,const Edge& e2){
        return e1.len > e2.len;
    });
    Edge topEdge;//位置靠上的长边
    Edge bottomEdge;//位置靠下的长边
    Edge longEdge1 = edges[0];
    Edge longEdge2 = edges[1];//拿到两条长边
    Edge shortEdge1 = edges[2];//短边
    Edge shortEdge2 = edges[3];//短边
    //任意长边与短边之比小于2 就丢弃
    if(longEdge1.len/shortEdge1.len < 2.0f||longEdge2.len/shortEdge2.len < 2.0f||longEdge1.len/shortEdge2.len < 2.0f||longEdge2.len/shortEdge1.len < 2.0f)
    {
        printf("长宽比异常\n");
        return; 
    }

    cv::Point2f left_top,left_bottom,right_top,right_bottom;//定义四个角点
    cv::Point2f last_left_top,last_right_top,last_right_bottom,last_left_bottom;

    //为找位置最靠上长的红边
    if(longEdge1.mid_y < longEdge2.mid_y)
    {
        topEdge = longEdge1;
        bottomEdge = longEdge2;
    }
    else
    {
        topEdge = longEdge2;
        bottomEdge = longEdge1;
    }

    if(topEdge.a.x<topEdge.b.x) {
        left_top = topEdge.a;
        right_top = topEdge.b;
    }
    else
    {
        left_top = topEdge.b;
        right_top = topEdge.a;
    }

    if(bottomEdge.a.x<bottomEdge.b.x) {
        left_bottom = bottomEdge.a;
        right_bottom = bottomEdge.b;
    }
    else
    {
        left_bottom = bottomEdge.b;
        right_bottom = bottomEdge.a;
    }

    // cv::circle(lq_frame, left_top, 1, cv::Scalar(0, 255, 0), -1);


    // cv::circle(lq_frame, left_top, 1, cv::Scalar(0, 255, 0), -1);


    // cv::circle(lq_frame, left_top, 1, cv::Scalar(0, 255, 0), -1);

    // cv::circle(lq_frame, right_top, 1, cv::Scalar(255, 0, 0), -1);

    // cv::circle(lq_frame, left_bottom, 1, cv::Scalar(0, 255, 0), -1);

    // cv::circle(lq_frame, right_bottom, 1, cv::Scalar(255, 0, 0), -1);
    std::vector<cv::Point2f> red_real_pts;
    red_real_pts.push_back(cv::Point2f(0.0f,  0.0f));  // 红块左上
    red_real_pts.push_back(cv::Point2f(12.0f, 0.0f));  // 红块右上
    red_real_pts.push_back(cv::Point2f(12.0f, 5.0f));  // 红块右下
    red_real_pts.push_back(cv::Point2f(0.0f,  5.0f));  // 红块左下


    std::vector<cv::Point2f> red_img_pts;
    red_img_pts.push_back(left_top);
    red_img_pts.push_back(right_top);
    red_img_pts.push_back(right_bottom);
    red_img_pts.push_back(left_bottom);



    // 红块四点上一帧缓存
    static bool has_last_red_quad = false;
    static std::vector<cv::Point2f> last_red_quad(4);

// 稳定红块四点
    if(!stabilizeQuad(red_img_pts,
                  last_red_quad,
                  has_last_red_quad,
                  2.0f,     // 小于 1 像素不更新
                  0.40f))   // 对边平行阈值
    {
        return;
    }

// 使用稳定后的红块四点
    left_top     = red_img_pts[0];
    right_top    = red_img_pts[1];
    right_bottom = red_img_pts[2];
    left_bottom  = red_img_pts[3];

    cv::circle(lq_frame, left_top, 1, cv::Scalar(0, 255, 0), -1);

    cv::circle(lq_frame, right_top, 1, cv::Scalar(255, 0, 0), -1);

    cv::Mat H = cv::getPerspectiveTransform(red_real_pts, red_img_pts);


    std::vector<cv::Point2f> card_real_pts;
    card_real_pts.push_back(cv::Point2f(0.0f,  -14.0f)); // 白色区域左上
    card_real_pts.push_back(cv::Point2f(12.0f, -14.0f)); // 白色区域右上
    card_real_pts.push_back(cv::Point2f(12.0f,  0.0f));  // 白色区域右下，也就是红块右上
    card_real_pts.push_back(cv::Point2f(0.0f,   0.0f));  // 白色区域左下，也就是红块左上

    std::vector<cv::Point2f> card_img_pts;
    cv::perspectiveTransform(card_real_pts, card_img_pts, H);

    cv::Point2f left_top1  = card_img_pts[0];  // 图片左上角点
    cv::Point2f right_top1 = card_img_pts[1];  // 图片右上角点
    cv::Point2f right_bottom1 = card_img_pts[2];
    cv::Point2f left_bottom1  = card_img_pts[3];




    // cv::line(lq_frame, left_bottom1, left_top1,
    //          cv::Scalar(0, 255, 0), 1);

    // cv::line(lq_frame, right_bottom1, right_top1,
    //          cv::Scalar(0, 255, 0), 1);

    // cv::line(lq_frame, left_top1, right_top1,
    //          cv::Scalar(0, 255, 0), 1);

    // cv::line(lq_frame, left_bottom1, right_bottom1,
    //          cv::Scalar(0, 255, 0), 1);


    std::vector<cv::Point2f> src_pts = {
        left_top1,
        right_top1,
        right_top,
        left_top
    };

// 最终透视区域上一帧缓存
    static bool has_last_src_quad = false;
    static std::vector<cv::Point2f> last_src_quad(4);

// 稳定最终透视四边形
    if(!stabilizeQuad(src_pts,
                  last_src_quad,
                  has_last_src_quad,
                  0.0f,     // 小于 2 像素不更新
                  0.40f))   // 外推后的区域稍微放宽一些
    {
        return;
    }

    cv::line(lq_frame, src_pts[0], src_pts[1], cv::Scalar(0, 255, 0), 1);
    cv::line(lq_frame, src_pts[1], src_pts[2], cv::Scalar(0, 255, 0), 1);
    cv::line(lq_frame, src_pts[2], src_pts[3], cv::Scalar(0, 255, 0), 1);
    cv::line(lq_frame, src_pts[3], src_pts[0], cv::Scalar(0, 255, 0), 1);

    std::vector<cv::Point2f> dst_pts = {
    {0,0},
    {59,0},
    {59,59},
    {0,59}
    };

    auto start_time3 = std::chrono::high_resolution_clock::now();
    cv::Mat m = cv::getPerspectiveTransform(src_pts, dst_pts);//60*60    透视矩阵 把图片拉伸成60*60
    auto end_time3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_ms1 = end_time3 - start_time3;
    printf("获取透视矩阵耗时: %.2f ms\n",elapsed_ms1.count());
    
    cv::Mat roi_img;
    auto start_time2 = std::chrono::high_resolution_clock::now();
    cv::warpPerspective(lq_frame, roi_img, m, cv::Size(60, 60));
    auto end_time2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_ms2 = end_time2 - start_time2;
    printf("透视耗时: %.2f ms\n",elapsed_ms2.count());

    if(roi_img.empty()||roi_img.rows<=0||roi_img.cols<=0)
    {
        return;
    }
    float confidence = 0.0;
    auto start_time1 = std::chrono::high_resolution_clock::now();
    std::string result = classifier.Infer(roi_img,confidence);
    auto end_time1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_ms3 = end_time1 - start_time1;
    printf("推理耗时: %.2f ms\n",elapsed_ms3.count());
   // printf("置信度：%f\n",confidence);
   // cv::Mat roi_img = lq_frame(roi_rect);//截图送入模型
// }

// camera_server.update_frame_mat(roi_img);
auto end_time = std::chrono::high_resolution_clock::now();
std::chrono::duration<double, std::milli> elapsed_ms = end_time - start_time;
printf("函数总耗时: %.2f ms\n",elapsed_ms.count());


}


//标注ROI区域
bool GenerateROI(const cv::Point &center, cv::Rect &roi, const cv::Mat &src)
{
    if (center.x < 0 || center.y < 0) {
        return false;  // 点坐标无效
    }
    
    if (src.empty()) {
        return false;  // 图像为空
    }
    int half = ROI_SIZE/2;
    roi_x_1 = center.x - half;
   // roi_y_1 = center.y;
    roi_y_1 = center.y - (int)half*2;//把中心的y坐标稍微向上移动一些

    if (roi_x_1 < 0 || roi_y_1 < 0)
        return false;

    if (roi_x_1 + ROI_SIZE > src.cols)
        return false;

    if (roi_y_1+ ROI_SIZE > src.rows)
        return false;

    roi = cv::Rect(roi_x_1, roi_y_1, ROI_SIZE, ROI_SIZE);
    return true;
}


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
    red_objects.clear();//先清元素
    red_area = 0;
    red_points_num = 0;
    long long sum_x = 0;
    long long sum_y = 0;
    if(src.empty()) 
    {
        printf("DetectRedBlock src is empty\n");
        return;
    }
    if(roi_x<0 || roi_y<0||roi_x >= src.cols || roi_y >= src.rows)  return;
    if(width <= 0 || height <= 0 || width>src.cols||height>src.rows) return;
    if(roi_x + width > src.cols || roi_y + height > src.rows) return;
    cv::Rect roi(roi_x,roi_y,width,height);

    cv::Rect roi_rect;
    cv::Mat roi_src = src(roi);
    cv::Mat mask;
    mask.create(roi_src.size(), CV_8UC1);


    for (int y = 0; y < roi_src.rows; ++y)
    {
        const cv::Vec3b* src_ptr = roi_src.ptr<cv::Vec3b>(y);
        uchar* mask_ptr = mask.ptr<uchar>(y);

        for (int x = 0; x < roi_src.cols; ++x)
        {
            int b = src_ptr[x][0];
            int g = src_ptr[x][1];
            int r = src_ptr[x][2];

            if (r > Flash.debug_rgb_r_min && (r - g) > Flash.debug_rgb_rg_diff && (r - b) > Flash.debug_rgb_rb_diff){
                Image_Use[y+roi_y][x+roi_x]=white;
                red_points_num++;
                sum_x += x;
                sum_y += y;
                mask_ptr[x] = white;
            }
            else{
                mask_ptr[x] = 0;
            }
        }
    }

  
if (red_points_num >= 5)
    {
        int cx = sum_x / red_points_num;
        int cy = sum_y / red_points_num;
        
        resize_cx = cx + roi_x;
        resize_cy = cy + roi_y;
        // 映射到原图坐标
        if(resize_cx*3.4>320) return;
        if(resize_cy*4>240) return;
        center= cv::Point((resize_cx)*3.4, (resize_cy)*4);//修改1

        if(last_center != cv::Point(-1, -1))
        {
            int dx = center.x - last_center.x;
            int dy = center.y - last_center.y;
            double dist = sqrt(dx * dx + dy * dy);
            // if(dist>80){
            //     printf("本帧无效\n");
            //     return;
            // }
            if(dist<16)
            {
                center = last_center;
            }
        }
        last_center = center;
        red_area = red_points_num;
        red_objects.push_back({center,red_area});

        // 可选：画点
        //cv::circle(lq_frame, center, 3, cv::Scalar(0, 255, 0), -1);
        //printf("检测到红色块\n");
       // printf("Red Center: (%d, %d), Area: %d,red_points_num: %d, sum_x:%d\n",center.x, center.y, red_area,red_points_num,sum_x);
    }
    else{
        center = cv::Point(-1,-1);
        last_center = cv::Point(-1,-1);
        resize_cx = 0;
        resize_cy = 0;
    }

    if(!red_objects.empty()&&Flag.infer ==1)
    {   
   auto &obj = red_objects[0];

        if (obj.center.x < 0 || obj.center.y < 0 || 
        obj.center.x >= lq_frame.cols || obj.center.y >= lq_frame.rows||obj.center.y < 0||obj.center.x < 0) 
        {
        return;
        }

        if(!GenerateROI(obj.center, roi_rect, lq_frame)){
            return;
        }
        if (roi_rect.x < 0 || roi_rect.y < 0 ||
            roi_rect.width <= 0 || roi_rect.height <= 0 ||
            roi_rect.x + roi_rect.width > lq_frame.cols ||
            roi_rect.y + roi_rect.height > lq_frame.rows)
        {
                std::cout << "BAD ROI: "
              << roi_rect << " | img: "
              << lq_frame.cols << "x"
              << lq_frame.rows << std::endl;
                return;
        }
            cv::Mat roi_img = lq_frame(roi_rect);//截图送入模型
              if(roi_img.empty()||roi_img.rows<=0||roi_img.cols<=0)
                 {
                    return;
                 }
                // cv::rectangle(lq_frame, roi_rect, cv::Scalar(255, 0, 0), 1);
                 float confidence;
                 auto start_time = std::chrono::high_resolution_clock::now();

                 real_picture_distance=real_distance[MAX(L_h_guai.row,R_h_guai.row)];
                 printf("real_picture_distance:%f\n",real_picture_distance);

                 std::string result = classifier.Infer(roi_img,confidence);
                auto end_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> elapsed_ms = end_time - start_time;
                 printf("检测结果:%s, 置信度: %.1f,推理耗时: %.2f ms\n",result.c_str(),confidence,elapsed_ms.count());


                if(result == "supply" && confidence>40)
                {
                    Flag.supply++;
                    Flag.weapon = 0;
                    Flag.vehicle = 0;
                }
                if(result == "weapon" && confidence>40)
                {
                    Flag.supply = 0;
                    Flag.weapon++;
                    Flag.vehicle = 0;
                }
                if(result == "vehicle" && confidence>40)
                {
                    Flag.supply = 0;
                    Flag.weapon = 0;
                    Flag.vehicle++;
                }
        
    }

}




/***************************************************红色标注******************************************************/
enum ClassType{
    CLASS_SUPPLIES = 0,
    CLASS_VEHICLE,
    CLASS_WEAPON,
    CLASS_UNKNOWN
};

ClassType GetClassID(const std::string &cls){
    if(cls == "supplies")
        return CLASS_SUPPLIES;
    else if(cls == "vehicle")
        return CLASS_VEHICLE;
    else if(cls == "weapon")
        return CLASS_WEAPON;
    return CLASS_UNKNOWN;
}
ClassType id;

/***************************************************图像处理******************************************************/
void ImageDeal()
{   
    lq_frame = cam.get_frame_raw();
    if(lq_frame.empty()){
        return; 
    }
      cv::resize(lq_frame,resizedFrame,cv::Size(LCDW_1,LCDH_1),0,0,cv::INTER_AREA);



      cv::cvtColor(resizedFrame,grayFrame,cv::COLOR_BGR2GRAY);

 

//  将OpenCV图像数据复制到图像数组 采用memcpy函数加快处理速度
    for (int i = 0; i < LCDH_1; i++)
    {
        uint8_t *p = grayFrame.ptr<uint8_t>(i);
        for(int w = 0; w < LCDW_1; w++)
        {
            Image_Use[i][w] = p[w];

        }
    }

    //             if(FindTargetRoiByFixedIpm(lq_frame,target_roi,target_pts,nullptr, nullptr)){
    //                 printf("函数执行\n");
    // //camera_server.update_frame_mat(target_roi);
    //     }
        
         Get01change_dajin();

        // my_sobel(Image_Zip,Image_Use); //压缩后的图像数组,

        // my_sobel_dajin(Image_Zip,Image_Use);


        imgInfoInit();

        Get_ImageTop();

        Draw_BlackSideline(Image_Use);//画边线

        Find_Sideline(imgInfo.bottom-1,imgInfo.top+ 1);//找边线
        //图片相关
        bool roi_ok = FindTargetRoiByFixedIpm(lq_frame,
                                      target_roi,
                                      &red_point,
                                      target_pts,
                                      red_pts,
                                      &valid_ratio,
                                    &erase_pts_ready);

        // if(roi_ok){
        // //         for (int i = 0; i < 4; i++)
        // //         {
        // // cv::line(lq_frame,
        // //          target_pts[i],
        // //          target_pts[(i + 1) % 4],
        // //          cv::Scalar(0, 255, 0),
        // //          2,
        // //          cv::LINE_AA);
        // //     }
        // // cv::Point red(red_point.column,red_point.row);
        // cv::circle(resizedFrame, red, 1, cv::Scalar(0, 255, 0), -1);
        // }

bool erase_done = false;

//当前帧已经算出了红块和图片四点
if (erase_pts_ready &&
    AreEraseQuadsValid(red_pts, target_pts, lq_frame.size()))
{
    SaveEraseCache(red_pts, target_pts);//保存上一帧的坐标点

    EraseTargetAndRedOnBinary(Image_Use,
                              red_pts,
                              target_pts,
                              lq_frame.size());

    erase_done = true;//完成抹除 需要重新找边线
}

else if (g_erase_cache.has_last && g_erase_cache.lost_cnt < 3)
{
    EraseTargetAndRedOnBinary(Image_Use,
                              g_erase_cache.red,
                              g_erase_cache.target,
                              lq_frame.size());

    g_erase_cache.lost_cnt++;
    erase_done = true;
}
else
{
    g_erase_cache.has_last = false;
    g_erase_cache.lost_cnt = 0;
}

// 只要执行过抹除，就重新找边线
if (erase_done)
{
    imgInfoInit();
    Get_ImageTop();
    Draw_BlackSideline(Image_Use);
    Find_Sideline(imgInfo.bottom - 1, imgInfo.top + 1);
}
    // camera_server.update_frame_mat(resizedFrame);
    // auto start_time = std::chrono::high_resolution_clock::now();
    // auto end_time = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> elapsed_ms = end_time - start_time;
    // printf("函数耗时: %.2f ms\n",elapsed_ms.count());
// camera_server.update_frame_mat(lq_frame);

// if (FindTargetRoiByFixedIpm(lq_frame,
//                             target_roi,
//                             target_pts,
//                             red_pts,
//                             &valid_ratio))
// {

//     // 画红色定位条，蓝色
//     for (int i = 0; i < 4; i++)
//     {
//         cv::line(lq_frame,
//                  red_pts[i],
//                  red_pts[(i + 1) % 4],
//                  cv::Scalar(255, 0, 0),
//                  2,
//                  cv::LINE_AA);
//     }

//     // 画目标裁剪框，绿色
//     for (int i = 0; i < 4; i++)
//     {
//         cv::line(lq_frame,
//                  target_pts[i],
//                  target_pts[(i + 1) % 4],
//                  cv::Scalar(0, 255, 0),
//                  2,
//                  cv::LINE_AA);
//     }

//     // 左上角显示透视后的 ROI，方便调试
//     if (!target_roi.empty() &&
//         lq_frame.cols >= 80 &&
//         lq_frame.rows >= 80)
//     {
//         cv::Mat show_roi;

//         cv::resize(target_roi,
//                    show_roi,
//                    cv::Size(80, 80),
//                    0,
//                    0,
//                    cv::INTER_NEAREST);

//         show_roi.copyTo(lq_frame(cv::Rect(0, 0, 80, 80)));
//     }

//     // 显示采样有效比例
//     char text[64];
//     std::snprintf(text,
//                   sizeof(text),
//                   "ratio=%.2f",
//                   static_cast<double>(valid_ratio));

//     cv::putText(lq_frame,
//                 text,
//                 cv::Point(5, 100),
//                 cv::FONT_HERSHEY_SIMPLEX,
//                 0.45,
//                 cv::Scalar(0, 255, 0),
//                 1,
//                 cv::LINE_AA);

//     // 这里 target_roi 就是固定 60x60 的透视裁剪结果
//     // 后面可以直接送入你的 NCNN 分类模型
//     //
//     // classify_result = TargetClassify(target_roi);
// }


        if(Flag.Huandao_L>0||Flag.Huandao_R>0)
        Find_Guaidian();  //找拐点
        else
        Find_Guaidian1();  //找拐点

        if(L_h_guai.flag||R_h_guai.flag)
        {
           
            //DetectRedBlock(resizedFrame,0,(int)MAX(imgInfo.top+1,real_distance_to_row(120)),93,59-MAX(imgInfo.top+1,real_distance_to_row(120)));
                                    
                    Find_right_Sideline(imgInfo.bottom-5,imgInfo.top+1);
                    Find_left_Sideline(imgInfo.bottom-5,imgInfo.top+1); 
        }
        straight_judge();

                            //                         Flag.Redblock=0;   
            // if(!red_objects.empty())
            //    {
            //     // Flag.Redblock=1;
            //     red_x_mid=resize_cx;
            //     red_y_mid=resize_cy;
            //     // err_picture=fabs(real_distance[red_y_mid]-real_distance[L_h_guai.row]);
            //     x_err_red=abs((Right_Sideline[red_y_mid]+Left_Sideline[red_y_mid])/2-red_x_mid);
            //     // if(err_picture<15)//&&abs(center.x-L_h_guai.column)<20
            //     Flag.Redblock=1;
            //    }

    
                        // Find_Sideline(imgInfo.bottom-1,imgInfo.top+ 1);//找边线


        zebra_corssing();

        // if(Flag.Zebra_cross==3)
        // {
        // ramp();
        // }



        // if(Flag.Huandao_L==0&&Flag.Huandao_R==0)
        // small_rock();
        // // }

        // if(Flag.Huandao_L!=1&&Flag.Huandao_R!=1)


       Huandao_R_imu();
       Huandao_L_imu();


       if(Flag.Huandao_L!=1&&Flag.Huandao_R!=1&&Flag.Huandao_L!=2&&Flag.Huandao_R!=2&&Flag.Huandao_L!=3&&Flag.Huandao_R!=3
       &&Flag.Huandao_L!=4&&Flag.Huandao_R!=4&&Flag.Huandao_L!=5&&Flag.Huandao_R!=5&&Flag.Huandao_L!=6&&Flag.Huandao_R!=6)
        Buxian();

        // if(Flag.Huandao_L!=1&&Flag.Huandao_R!=1&&Flag.Huandao_L!=2&&Flag.Huandao_R!=2)
        // picture();
        
        Find_Midline();


        Err_Sum();

        protect();
}