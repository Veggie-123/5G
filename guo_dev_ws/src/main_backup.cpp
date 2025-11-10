// Last update: 2024/12/5 
// 安徽芜湖----国赛----------

#include <iostream> // 标准输入输出流库
#include <cstdlib> // 标准库
#include <unistd.h> // Unix标准库

#include <opencv2/opencv.hpp> // OpenCV主头文件
#include <opencv4/opencv2/core/core.hpp> // OpenCV核心功能
#include <opencv4/opencv2/highgui.hpp> // OpenCV高层GUI功能
#include <opencv4/opencv2/imgproc/imgproc_c.h> // OpenCV图像处理功能

#include <string> // 字符串库
#include <pigpio.h> // GPIO控制库
#include <thread> // 线程库
#include <vector> // 向量容器库
#include <chrono> // 时间库

#include "Yolo.h" // Yolo库

using namespace std; // 使用标准命名空间
using namespace cv; // 使用OpenCV命名空间

//------------有关的全局变量定义------------------------------------------------------------------------------------------

std::vector<BoxInfo> result; // 存储Yolo检测结果
std::vector<BoxInfo_v5lite> result_ab; // 存储Yolo检测结果

std::string model_name_obs = "/home/pi/model/obs_fp32.mnn";
int num_classes_obs = 1;
std::vector<std::string> labels_obs{"zhuitong"};
yolo_fv2_mnn yolo_obs(0.6, 0.4, model_name_obs, num_classes_obs, labels_obs);

std::string model_name_lr = "/home/pi/model/lr_fp32.mnn";
int num_classes_lr = 2;
std::vector<std::string> labels_lr{"left", "right"};
yolo_fv2_mnn yolo_lr(0.5, 0.5, model_name_lr, num_classes_lr, labels_lr);

std::string model_name_ab = "/home/pi/model/ab_fp32_fv2.mnn";
int num_classes_ab = 2;
std::vector<std::string> labels_ab{"A", "B"};
yolo_fv2_mnn yolo_ab(0.5, 0.5, model_name_ab, num_classes_ab, labels_ab);

yolo_fv2_mnn yolo_ab_lite(0.5);

//-----------------图像相关----------------------------------------------
Mat frame; // 存储视频帧
Mat frame_a; // 存储视频帧
Mat bin_image; // 存储二值化图像--Sobel检测后图像

//-----------------巡线相关-----------------------------------------------
std::vector<cv::Point> mid; // 存储中线
std::vector<cv::Point> left_line; // 存储左线条
std::vector<cv::Point> right_line; // 存储右线条

//---------------舵机和电机相关---------------------------------------------
int error_first; // 存储第一次误差
int last_error; // 存储上一次误差
float servo_pwm_diff; // 存储舵机PWM差值
float servo_pwm; // 存储舵机PWM值

//---------------发车信号定义-----------------------------------------------
int find_first = 0; // 标记是否第一次找到蓝色挡板
int fache_sign = 0; // 标记发车信号


int bz_heighest = 0; // 避障高度
int bz_xcenter = 0; // 存储避障中心点
int bz_get = 0;
int bz_bottom = 320; // 存储避障底部点
int bz_area = 0; // 存储避障面积

std::vector<cv::Point> mid_bz; // 存储中线
std::vector<cv::Point> left_line_bz; // 存储左线条
std::vector<cv::Point> right_line_bz; // 存储右线条

int park_mid = 160; // 停车车库中线检测结果

int flag_gohead = 0; // 前进标志

int changeroad = 1; // 变道检测结果
int last_bz = 0; // 避障计数器

// 定义舵机和电机引脚号、PWM范围、PWM频率、PWM占空比解锁值
const int servo_pin = 12; // 存储舵机引脚号
const float servo_pwm_range = 10000.0; // 存储舵机PWM范围
const float servo_pwm_frequency = 50.0; // 存储舵机PWM频率
const float servo_pwm_duty_cycle_unlock = 690.0; // 存储舵机PWM占空比解锁值

//---------------------------------------------------------------------------------------------------
float servo_pwm_mid = servo_pwm_duty_cycle_unlock; // 存储舵机中值
//---------------------------------------------------------------------------------------------------

const int motor_pin = 13; // 存储电机引脚号
const float motor_pwm_range = 40000.0; // 存储电机PWM范围
const float motor_pwm_frequency = 200.0; // 存储电机PWM频率
const float motor_pwm_duty_cycle_unlock = 11400.0; // 存储电机PWM占空比解锁值

//--------------------------------------------------------------------------------------------------
float motor_pwm_mid = motor_pwm_duty_cycle_unlock; // 存储电机PWM初始化值
//--------------------------------------------------------------------------------------------------

const int yuntai_LR_pin = 22; // 存储云台引脚号
const float yuntai_LR_pwm_range = 1000.0; // 存储云台PWM范围
const float yuntai_LR_pwm_frequency = 50.0; // 存储云台PWM频率
const float yuntai_LR_pwm_duty_cycle_unlock = 66.0; //大左小右 

const int yuntai_UD_pin = 23; // 存储云台引脚号
const float yuntai_UD_pwm_range = 1000.0; // 存储云台PWM范围
const float yuntai_UD_pwm_frequency = 50.0; // 存储云台PWM频率
const float yuntai_UD_pwm_duty_cycle_unlock = 70.0; //大上下小

int first_bz_get = 0;

//---------------斑马线相关-------------------------------------------------
int banma = 0; // 斑马线检测结果
int flag_banma = 0; // 斑马线标志

//----------------变道相关---------------------------------------------------

int flag_changeroad = 0;// 变道标志

//----------------避障相关---------------------------------------------------

int count_bz = 0; // 避障计数器

//----------------停车相关---------------------------------------------------

int park_find = 0; // 停车检测结果
int flag_park_find = 0; // 停车标志

int parkchose = 0; // 停车车库检测结果
int flag_parkchose = 0; // 停车车库标志

//--------------------------------------------------------------------------

int number = 0;
int number1 = 0;
int numbera = 0;
int numberb = 0;

int bz_y2 = 170;
int number_w = 550; //3000 630  4000 600  5000 550
int number_ten_bz = 10;
int number_ten_park = 30;

// 定义舵机和电机PWM初始化函数
void servo_motor_pwmInit(void) 
{
    if (gpioInitialise() < 0) // 初始化GPIO，如果失败则返回
    {
        std::cout << "GPIO failed ! Please use sudo !" << std::endl; // 输出失败信息
        return; // 返回
    }
    else
        std::cout << "GPIO ok. Good !!" << std::endl; // 输出成功信息

    gpioSetMode(servo_pin, PI_OUTPUT); // 设置舵机引脚为输出模式
    gpioSetPWMfrequency(servo_pin, servo_pwm_frequency); // 设置舵机PWM频率
    gpioSetPWMrange(servo_pin, servo_pwm_range); // 设置舵机PWM范围
    gpioPWM(servo_pin, servo_pwm_duty_cycle_unlock); // 设置舵机PWM占空比解锁值

    gpioSetMode(motor_pin, PI_OUTPUT); // 设置电机引脚为输出模式
    gpioSetPWMfrequency(motor_pin, motor_pwm_frequency); // 设置电机PWM频率
    gpioSetPWMrange(motor_pin, motor_pwm_range); // 设置电机PWM范围
    gpioPWM(motor_pin, motor_pwm_duty_cycle_unlock); // 设置电机PWM占空比解锁值

    gpioSetMode(yuntai_LR_pin, PI_OUTPUT); // 设置云台引脚为输出模式
    gpioSetPWMfrequency(yuntai_LR_pin, yuntai_LR_pwm_frequency); // 设置云台PWM频率
    gpioSetPWMrange(yuntai_LR_pin, yuntai_LR_pwm_range); // 设置云台PWM范围
    gpioPWM(yuntai_LR_pin, yuntai_LR_pwm_duty_cycle_unlock); // 设置云台PWM占空比解锁值

    gpioSetMode(yuntai_UD_pin, PI_OUTPUT); // 设置云台引脚为输出模式
    gpioSetPWMfrequency(yuntai_UD_pin, yuntai_UD_pwm_frequency); // 设置云台PWM频率
    gpioSetPWMrange(yuntai_UD_pin, yuntai_UD_pwm_range); // 设置云台PWM范围
    gpioPWM(yuntai_UD_pin, yuntai_UD_pwm_duty_cycle_unlock); // 设置云台PWM占空比解锁值

}

//------------------------------------------------------------------------------------------------------------
cv::Mat undistort(const cv::Mat &frame) 
{
    double k1 = 0.0439656098483248; // 畸变系数k1
    double k2 = -0.0420991522460257; // 畸变系数k2
    double p1 = 0.0; // 畸变系数p1
    double p2 = 0.0; // 畸变系数p2
    double k3 = 0.0; // 畸变系数k3

    // 相机内参矩阵
    cv::Mat K = (cv::Mat_<double>(3, 3) << 176.842468665091, 0.0, 159.705914860981,
                 0.0, 176.990910857055, 120.557953465790,
                 0.0, 0.0, 1.0);

    // 畸变系数矩阵
    cv::Mat D = (cv::Mat_<double>(1, 5) << k1, k2, p1, p2, k3);
    cv::Mat mapx, mapy; // 映射矩阵
    cv::Mat undistortedFrame; // 去畸变后的图像帧

    // 初始化去畸变映射
    cv::initUndistortRectifyMap(K, D, cv::Mat(), K, frame.size(), CV_32FC1, mapx, mapy);
    // 应用映射，得到去畸变后的图像
    cv::remap(frame, undistortedFrame, mapx, mapy, cv::INTER_LINEAR);

    return undistortedFrame; // 返回去畸变后的图像
}

// 定义自定义直方图均衡化函数，输入为图像和alpha值   在ImagePreprocessing函数中调用
Mat customEqualizeHist(const Mat &inputImage, float alpha) 
{
    Mat enhancedImage; // 定义增强后的图像
    equalizeHist(inputImage, enhancedImage); // 对输入图像进行直方图均衡化

    // 减弱对比度增强的效果
    return alpha * enhancedImage + (1 - alpha) * inputImage; // 返回调整后的图像
}

cv::Mat drawWhiteLine(cv::Mat binaryImage, cv::Point start, cv::Point end, int lineWidth)
{
    cv::Mat resultImage = binaryImage.clone(); // 克隆输入的二值图像

    int x1 = start.x, y1 = start.y; // 获取起点的x和y坐标
    int x2 = end.x, y2 = end.y; // 获取终点的x和y坐标

    if (x1 == x2) // 如果起点和终点的x坐标相同，说明是垂直线
    {
        for (int y = std::min(y1, y2); y <= std::max(y1, y2); ++y) // 遍历y坐标
        {
            for (int i = -lineWidth / 2; i <= lineWidth / 2; ++i) // 遍历线宽
            {
                resultImage.at<uchar>(cv::Point(x1 + i, y)) = 255; // 将线宽范围内的像素值设为255（白色）
            }
        }
    }
    else // 如果起点和终点的x坐标不同，说明是斜线
    {
        double slope = static_cast<double>(y2 - y1) / (x2 - x1); // 计算斜率
        double intercept = y1 - slope * x1; // 计算截距

        for (int x = std::min(x1, x2); x <= std::max(x1, x2); ++x) // 遍历x坐标
        {
            int y = static_cast<int>(slope * x + intercept); // 计算对应的y坐标
            for (int i = -lineWidth / 2; i <= lineWidth / 2; ++i) // 遍历线宽
            {
                int newY = std::max(0, std::min(y + i, resultImage.rows - 1)); // 确保y坐标在图像范围内
                resultImage.at<uchar>(cv::Point(x, newY)) = 255; // 将线宽范围内的像素值设为255（白色）
            }
        }
    }

    return resultImage; // 返回绘制了白线的图像
}

cv::Mat ImageSobel(cv::Mat &frame) 
{
    // 定义图像宽度和高度
    const int width = 320;
    const int height = 240;

    // 初始化二值输出图像
    Mat binaryImage = Mat::zeros(height, width, CV_8U);
    Mat binaryImage_1 = Mat::zeros(height, width, CV_8U);

    // 转换输入图像为灰度图像
    Mat grayImage;
    cvtColor(frame, grayImage, cv::COLOR_BGR2GRAY);

    int kernelSize = 5;
    double sigma = 1.0;
    cv::Mat blurredImage;
    cv::GaussianBlur(grayImage, blurredImage, cv::Size(kernelSize, kernelSize), sigma);

    // Sobel 边缘检测
    // Mat sobelX;
    Mat sobelY;
    // Sobel(blurredImage, sobelX, CV_64F, 1, 0, 3); // x方向梯度
    Sobel(blurredImage, sobelY, CV_64F, 0, 1, 3); // y方向梯度

    // 计算梯度幅值并转换为 8 位图像
    // Mat gradientMagnitude = abs(sobelX) + abs(sobelY);
    Mat gradientMagnitude = abs(sobelY);
    convertScaleAbs(gradientMagnitude, gradientMagnitude);

    // 阈值分割并膨胀操作
    cv::threshold(gradientMagnitude, binaryImage, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(binaryImage, binaryImage, kernel,cv::Point(-1, -1), 1);

    // 定义感兴趣区域 (ROI)
    const int x_roi = 1, y_roi = 109, width_roi = 318, height_roi = 46;
    Rect roi(x_roi, y_roi, width_roi, height_roi);
    Mat croppedImage = binaryImage(roi);

    // 使用概率霍夫变换检测直线
    vector<Vec4i> lines;
    HoughLinesP(croppedImage, lines, 1, CV_PI / 180, 25, 15, 10);

    // 遍历直线并筛选有效线段
    for (const auto &l : lines) 
    {
        // 计算直线角度和长度
        double angle = atan2(l[3] - l[1], l[2] - l[0]) * 180.0 / CV_PI;
        double length = hypot(l[3] - l[1], l[2] - l[0]);

        // 筛选条件：角度范围、最小长度
        if (abs(angle) > 15) 
        {
            // 调整坐标以适应全图
            Vec4i adjustedLine = l;
            adjustedLine[0] += x_roi;
            adjustedLine[1] += y_roi;
            adjustedLine[2] += x_roi;
            adjustedLine[3] += y_roi;

            // 绘制白线
            line(binaryImage_1, Point(adjustedLine[0], adjustedLine[1]),
                Point(adjustedLine[2], adjustedLine[3]), Scalar(255), 2, LINE_AA);
        }
    }

    // 返回最终的处理图像
    return binaryImage_1;
}

void Tracking(cv::Mat &dilated_image) 
{
    // 参数检查
    if (dilated_image.empty() || dilated_image.type() != CV_8U) 
    {
        std::cerr << "Invalid input image for Tracking!" << std::endl;
        return;
    }

    int begin = 160; // 初始化起始位置
    left_line.clear(); // 清空左线条
    right_line.clear(); // 清空右线条
    mid.clear(); // 清空中线

    // 逐行搜索，从第153行到第110行
    for (int i = 153; i >= 110; --i) 
    {
        int left = begin;  // 左侧搜索起点
        int right = begin; // 右侧搜索起点
        bool left_found = false; // 标记是否找到左线
        bool right_found = false; // 标记是否找到右线

        // 搜索左线
        while (left > 1) 
        {
            if (dilated_image.at<uchar>(i, left) == 255 &&
                dilated_image.at<uchar>(i, left + 1) == 255) 
            {
                left_found = true;
                left_line.emplace_back(left, i); // 记录左线点
                break;
            }
            --left;
        }
        if (!left_found) 
        {
            left_line.emplace_back(1, i); // 左线未找到，默认记录最左侧点
        }

        // 搜索右线
        while (right < 318) 
        {
            if (dilated_image.at<uchar>(i, right) == 255 &&
                dilated_image.at<uchar>(i, right - 2) == 255) 
            {
                right_found = true;
                right_line.emplace_back(right, i); // 记录右线点
                break;
            }
            ++right;
        }
        if (!right_found) 
        {
            right_line.emplace_back(318, i); // 右线未找到，默认记录最右侧点
        }

        // 计算中点
        const cv::Point &left_point = left_line.back();
        const cv::Point &right_point = right_line.back();
        int mid_x = (left_point.x + right_point.x) / 2;
        mid.emplace_back(mid_x, i); // 记录中点

        // 更新下一行的搜索起点
        begin = mid_x;
    }
}

void Tracking_bz(cv::Mat &dilated_image) 
{
    int begin = 160; // 初始化起始位置
    left_line_bz.clear(); // 清空蓝色左线条向量
    right_line_bz.clear(); // 清空蓝色右线条向量
    mid_bz.clear(); // 清空蓝色中线向量
    
    for (int i = 153; i >= bz_heighest ; i--) // 从153行遍历到最高点行
    {
        int find_l = 0; // 初始化左侧找到标志
        int find_r = 0; // 初始化右侧找到标志
        int to_left = begin; // 初始化左侧搜索位置
        int to_right = begin; // 初始化右侧搜索位置

        while (to_left != 1) // 当左侧搜索位置不为1时
        {
            if (dilated_image.at<uchar>(i, to_left) == 255 && dilated_image.at<uchar>(i, to_left + 1) == 255) // 如果找到白色像素
            {
                find_l = 1; // 设置左侧找到标志
                left_line_bz.push_back(cv::Point(to_left, i)); // 将左侧点加入蓝色左线条向量
                break; // 跳出循环
            }
            else
            {
                to_left--; // 否则左移
            }
        }

        if (to_left == 1) // 如果左侧搜索位置为1
        {
            left_line_bz.push_back(cv::Point(1, i)); // 将(1, i)加入蓝色左线条向量
        }

        while (to_right != 318) // 当右侧搜索位置不为318时
        {
            if (dilated_image.at<uchar>(i, to_right) == 255 && dilated_image.at<uchar>(i, to_right - 2) == 255) // 如果找到白色像素
            {
                find_r = 1; // 设置右侧找到标志
                right_line_bz.push_back(cv::Point(to_right, i)); // 将右侧点加入蓝色右线条向量
                break; // 跳出循环
            }
            else
            {
                to_right++; // 否则右移
            }
        }

        if (to_right == 318) // 如果右侧搜索位置为318
        {
            right_line_bz.push_back(cv::Point(318, i)); // 将(318, i)加入蓝色右线条向量
        }
        cv::Point midx1 = left_line_bz.back(); // 获取蓝色左线条最后一个点
        cv::Point midx2 = right_line_bz.back(); // 获取蓝色右线条最后一个点
        mid_bz.push_back(cv::Point(int((midx1.x + midx2.x) / 2), i)); // 计算中点并加入蓝色中线向量
        begin = (to_right + to_left) / 2; // 更新起始位置
    }
}

// 比较两个轮廓的面积
bool Contour_Area(vector<Point> contour1, vector<Point> contour2)
{
    return contourArea(contour1) > contourArea(contour2); // 返回轮廓1是否大于轮廓2
}

// 定义蓝色挡板 寻找函数
void blue_card_find(void)  // 输入为mask图像
{   
    cout << "进入 蓝色挡板寻找 进程！" << endl;

    Mat change_frame; // 存储颜色空间转换后的图像
    cvtColor(frame, change_frame, COLOR_BGR2HSV); // 转换颜色空间

    Mat mask; // 存储掩码图像

    // 定义HSV范围 hsv颜色空间特点：色调H、饱和度S、亮度V
    Scalar scalarl = Scalar(100, 43, 46); // HSV的低值
    Scalar scalarH = Scalar(124, 255, 255); // HSV的高值 
    inRange(change_frame, scalarl, scalarH, mask); // 创建掩码

    vector<vector<Point>> contours; // 存储轮廓的向量
    vector<Vec4i> hierarcy; // 存储层次结构的向量
    findContours(mask, contours, hierarcy, RETR_EXTERNAL, CHAIN_APPROX_NONE); // 查找轮廓
    if (contours.size() > 0) // 如果找到轮廓
    {
        sort(contours.begin(), contours.end(), Contour_Area); // 按轮廓面积排序
        vector<vector<Point>> newContours; // 存储新的轮廓向量
        for (const vector<Point> &contour : contours) // 遍历每个轮廓
        {
            Point2f center; // 存储中心点
            float radius; // 存储半径
            minEnclosingCircle(contour, center, radius); // 找到最小包围圆
            if (center.y > 90 && center.y < 160) // 如果中心点在指定范围内
            {
                newContours.push_back(contour); // 添加到新的轮廓向量中
            }
        }

        contours = newContours; // 更新轮廓向量

        if (contours.size() > 0) // 如果新的轮廓向量不为空
        {
            if (contourArea(contours[0]) > 500) // 如果最大的轮廓面积大于500
            {
                cout << "找到蓝色挡板 达到面积！" << endl; // 输出找到最大的蓝色物体
                // Point2f center; // 存储中心点
                // float radius; // 存储半径
                // minEnclosingCircle(contours[0], center, radius); // 找到最小包围圆
                // circle(frame, center, static_cast<int>(radius), Scalar(0, 255, 0), 2); // 在图像上画圆
                find_first = 1; // 更新标志位
            }
            else
            {
                cout << "找到蓝色挡板 未达到面积！" << endl; // 输出未找到蓝色物体
            }
        }
    }
    else
    {
        cout << "未找到蓝色物体" << endl; // 输出未找到蓝色物体
    }
}

// 检测蓝色挡板是否移开
void blue_card_remove(void) // 输入为mask图像
{
    cout << "进入 蓝色挡板移开 进程！" << endl; // 输出进入移除蓝色挡板的过程

    Mat change_frame; // 存储颜色空间转换后的图像
    cvtColor(frame, change_frame, COLOR_BGR2HSV); // 转换颜色空间

    Mat mask; // 存储掩码图像

    // 定义HSV范围 hsv颜色空间特点：色调H、饱和度S、亮度V
    Scalar scalarl = Scalar(100, 43, 46); // HSV的低值
    Scalar scalarH = Scalar(124, 255, 255); // HSV的高值 
    inRange(change_frame, scalarl, scalarH, mask); // 创建掩码

    vector<vector<Point>> contours; // 定义轮廓向量
    vector<Vec4i> hierarcy; // 定义层次结构向量
    findContours(mask, contours, hierarcy, RETR_EXTERNAL, CHAIN_APPROX_NONE); // 查找轮廓
    if (contours.size() > 0) // 如果找到轮廓
    {
        sort(contours.begin(), contours.end(), Contour_Area); // 按面积排序轮廓
        vector<vector<Point>> newContours; // 定义新的轮廓向量
        for (const vector<Point> &contour : contours) // 遍历每个轮廓
        {
            Point2f center; // 定义中心点
            float radius; // 定义半径
            minEnclosingCircle(contour, center, radius); // 找到最小包围圆
            if (center.y > 90 && center.y < 160) // 如果中心点在指定范围内
            {
                newContours.push_back(contour); // 添加到新的轮廓向量
            }
        }

        contours = newContours; // 更新轮廓向量

        if (contours.size() == 0) // 如果没有轮廓
        {
            fache_sign = 0; // 设置开始标志为1
            cout << "前进！" << endl; // 输出移动信息
            sleep(2); // 睡眠2秒
        }
    }
    else // 如果没有找到轮廓
    {
        fache_sign = 1; // 设置开始标志为1
        cout << "蓝色挡板移开！" << endl; // 输出蓝色挡板移开信息
        sleep(2); // 睡眠2秒
    }
}

int banma_get(cv::Mat &frame) {
    // 将输入图像转换为HSV颜色空间
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    // 定义白色的下界和上界
    cv::Scalar lower_white(0, 0, 200);
    cv::Scalar upper_white(180, 30, 255);

    // 创建白色掩码
    cv::Mat mask1;
    cv::inRange(hsv, lower_white, upper_white, mask1);

    // 创建一个3x3的矩形结构元素
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    // 对掩码进行膨胀和腐蚀操作
    cv::dilate(mask1, mask1, kernel);
    cv::erode(mask1, mask1, kernel);

    // 裁剪ROI区域
    cv::Rect roi(2, 110, std::min(318 - 2, mask1.cols - 2), std::min(200, mask1.rows - 110));
    cv::Mat src = mask1(roi);
    // cv::imshow("src", src);  // 显示ROI区域

    // 查找图像中的轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(src, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 创建一个副本以便绘制轮廓
    cv::Mat contour_img = src.clone();

    int count_BMX = 0;  // 斑马线计数器
    int min_w = 10;  // 最小宽度
    int max_w = 50;  // 最大宽度
    int min_h = 10;  // 最小高度
    int max_h = 50;  // 最大高度

    // 遍历每个找到的轮廓
    for (const auto& contour : contours) {
        cv::Rect rect = cv::boundingRect(contour);  // 获取当前轮廓的外接矩形 rect
        if (min_h <= rect.height && rect.height < max_h && min_w <= rect.width && rect.width < max_w) {
            // 过滤赛道外的轮廓
            cv::rectangle(contour_img, rect, cv::Scalar(255), 2);
            count_BMX++;
        }
    }
    // 最终返回值
    if (count_BMX >= 4) {
        cout << "检测到斑马线" << endl;
        return 1;
    }
    else {
        return 0;
    }
}

int find_parking(cv::Mat frame) {
    // 将图像转换为HSV颜色空间
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    // 定义红色的HSV范围
    cv::Scalar lower_red1(0, 43, 46);
    cv::Scalar upper_red1(10, 255, 255);

    cv::Scalar lower_red2(156, 43, 46);
    cv::Scalar upper_red2(180, 255, 255);

    // 创建两个红色掩码并合并
    cv::Mat mask1, mask2, mask;
    cv::inRange(hsv, lower_red1, upper_red1, mask1);
    cv::inRange(hsv, lower_red2, upper_red2, mask2);
    cv::bitwise_or(mask1, mask2, mask);

    // 执行形态学操作，可以根据实际情况调整参数
    cv::Mat kernel = cv::Mat::ones(5, 5, CV_8UC1); // 5x5的卷积核
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);  // 开运算 去除外部小噪点
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel); // 闭运算 去除内部小洞

    // 寻找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 仅保留y轴在100-218范围内的轮廓  
    std::vector<std::vector<cv::Point>> filtered_contours;
    for (size_t i = 0; i < contours.size(); i++) {
        cv::Rect bounding_box = cv::boundingRect(contours[i]);
        if (bounding_box.y >= 100 && bounding_box.y <= 218) {
            filtered_contours.push_back(contours[i]);
        }
    }

    // 计算大于4000面积的轮廓数量
    int large_contours_count = 0;
    for (size_t i = 0; i < filtered_contours.size(); i++) {
        double area = cv::contourArea(filtered_contours[i]);
        if (area > 1000) {
            large_contours_count++;
        }
    }

    // 如果有两个以上的轮廓的面积大于4000，返回1
    if (large_contours_count >= 1) {
        return 1;
    } else {
        return 0;
    }
}

float servo_pd(int target) { // 赛道巡线控制

    int pidx = int((mid[23].x + mid[25].x) / 2); // 计算中线中点的x坐标

    cout << " PIDX: " << pidx << endl;  

    float kp = 1.0; // 比例系数
    float kd = 2.0; // 微分系数

    error_first = target - pidx; // 计算误差

    servo_pwm_diff = kp * error_first + kd * (error_first - last_error); // 计算舵机PWM差值

    last_error = error_first; // 更新上一次误差

    servo_pwm = servo_pwm_mid + servo_pwm_diff; // 计算舵机PWM值

    if (servo_pwm > 1000) // 如果PWM值大于900
    {
        servo_pwm = 1000; // 限制PWM值为900
    }
    else if (servo_pwm < 580) // 如果PWM值小于600
    {
        servo_pwm = 580; // 限制PWM值为600
    }
    return servo_pwm; // 返回舵机PWM值
}

float servo_pd_l(int target) { // 赛道巡线控制

    int pidx = int((mid[23].x + mid[20].x + mid[25].x) / 3); // 计算中线中点的x坐标

    cout << " PIDX: " << pidx << endl;  

    float kp = 1.0; // 比例系数
    float kd = 2.0; // 微分系数

    error_first = target - pidx; // 计算误差

    servo_pwm_diff = kp * error_first + kd * (error_first - last_error); // 计算舵机PWM差值

    last_error = error_first; // 更新上一次误差

    servo_pwm = servo_pwm_mid + servo_pwm_diff; // 计算舵机PWM值

    if (servo_pwm > 760) // 如果PWM值大于900
    {
        servo_pwm = 760; // 限制PWM值为900
    }
    else if (servo_pwm < 650) // 如果PWM值小于600
    {
        servo_pwm = 650; // 限制PWM值为600
    }
    return servo_pwm; // 返回舵机PWM值
}

float servo_pd_bz1(int target) { // 赛道巡线控制

    int pidx = int((mid[23].x + mid[20].x + mid[25].x) / 3); // 计算中线中点的x坐标

    cout << " PIDX: " << pidx << endl;  

    float kp = 1.5; // 比例系数
    float kd = 3.0; // 微分系数

    error_first = target - pidx; // 计算误差

    servo_pwm_diff = kp * error_first + kd * (error_first - last_error); // 计算舵机PWM差值

    last_error = error_first; // 更新上一次误差

    servo_pwm = servo_pwm_mid + servo_pwm_diff; // 计算舵机PWM值

    if (servo_pwm > 1000) // 如果PWM值大于900
    {
        servo_pwm = 1000; // 限制PWM值为900
    }
    else if (servo_pwm < 500) // 如果PWM值小于600
    {
        servo_pwm = 500; // 限制PWM值为600
    }
    return servo_pwm; // 返回舵机PWM值
}

float servo_pd_bz(int target) { // 避障巡线控制

    int pidx = mid_bz[(int)(mid_bz.size() / 2)].x;

    if(pidx < 158)
        pidx = pidx - 5;

    cout << " PIDX: " << pidx << endl;    

    // float kp = 1.5; // 比例系数
    float kp = 3.0; // 比例系数
    float kd = 3.0; // 微分系数

    error_first = target - pidx; // 计算误差

    servo_pwm_diff = kp * error_first + kd * (error_first - last_error); // 计算舵机PWM差值
    last_error = error_first; // 更新上一次误差

    servo_pwm = servo_pwm_mid + servo_pwm_diff; // 计算舵机PWM值
    if (servo_pwm > 1000) // 如果PWM值大于900
    {
        servo_pwm = 1000; // 限制PWM值为900
    }
    else if (servo_pwm < 600) // 如果PWM值小于600
    {
        servo_pwm = 600; // 限制PWM值为600
    }
    return servo_pwm; // 返回舵机PWM值
}

float servo_pd_AB(int target) { // 避障巡线控制

    int pidx = park_mid; // 计算中点的x坐标

    cout << "-------------------------PIDX FOR PARK: " << pidx << endl;                   

    float kp = 3.0; // 比例系数
    float kd = 3.0; // 微分系数

    error_first = target - pidx; // 计算误差

    servo_pwm_diff = kp * error_first + kd * (error_first - last_error); // 计算舵机PWM差值
    last_error = error_first; // 更新上一次误差

    servo_pwm = servo_pwm_mid + servo_pwm_diff; // 计算舵机PWM值
    
    if (servo_pwm > 1000) // 如果PWM值大于900
    {
        servo_pwm = 1000; // 限制PWM值为900
    }
    else if (servo_pwm < 600) // 如果PWM值小于600
    {
        servo_pwm = 600; // 限制PWM值为600
    }
    return servo_pwm; // 返回舵机PWM值
}
void motor_park(){
    gpioPWM(13, motor_pwm_mid + 400);
    gpioPWM(13, motor_pwm_mid); // 设置电机PWM
    gpioPWM(13, motor_pwm_mid - 700); // 设置电机PWM
    gpioPWM(13, motor_pwm_mid - 900);
    usleep(800000); // 延时300毫秒
    gpioPWM(12, servo_pwm_mid); // 设置舵机PWM
    gpioPWM(13, motor_pwm_mid); // 设置电机PWM
    // sleep(100);
}

void gohead(int parkchose){
    if(parkchose == 1 ){ //try to find park A
        std::cout << "gohead--------------------------------------------------------------Try To Find Park AAAAAAAAAAAAAAA" << std::endl;
        gpioPWM(13, motor_pwm_mid + 2800);
        gpioPWM(13, motor_pwm_mid + 800); // 设置电机PWM
        gpioPWM(12, 690); // 设置舵机PW0M
        sleep(1);
        gpioPWM(13, motor_pwm_mid + 800); // 设置电机PWM
        gpioPWM(12, 780); // 设置舵机PW0M
        // sleep(2);
        usleep(2200000);
        gpioPWM(13, motor_pwm_mid);
        flag_park_find == 1;
        flag_parkchose == 1;
        sleep(100);
    }
    else if(parkchose == 2){ //try to find park B
        cout << "gohead--------------------------------------------------------------Try To Find Park BBBBBBBBBBBBBBB" << endl;
        gpioPWM(13, motor_pwm_mid + 2800);
        gpioPWM(13, motor_pwm_mid + 800); // 设置电机PWM
        gpioPWM(12, 690); // 设置舵机PWM
        sleep(1);
        gpioPWM(13, motor_pwm_mid + 800); // 设置电机PWM
        gpioPWM(12, 650); // 设置舵机PW0M
        // usleep(1800000);
        sleep(2);
        gpioPWM(13, motor_pwm_mid);
        flag_park_find == 1;
        flag_parkchose == 1;
        sleep(100);
    }
}

void banma_stop(){
    gpioPWM(13, motor_pwm_mid + 800);
    // gpioPWM(13, motor_pwm_mid + 400);
    gpioPWM(13, motor_pwm_mid); // 设置电机PWM
    gpioPWM(13, motor_pwm_mid - 800); // 设置电机PWM
    usleep(500000); // 延时300毫秒
    gpioPWM(12, servo_pwm_mid); // 设置舵机PWM
    gpioPWM(13, motor_pwm_mid); // 设置电机PWM
}

void motor_changeroad(){
    if(changeroad == 1){ // 向左变道----------------------------------------------------------------
        gpioPWM(12, 825); // 设置舵机PWM
        gpioPWM(13, motor_pwm_mid + 1300); // 设置电机PWM
        usleep(1200000); // RIGHT弯道
        // usleep(1000000); // LEFT弯道
        gpioPWM(12, 610); // 设置舵机PWM
        gpioPWM(13, motor_pwm_mid + 1300); // 设置电机PWM
        usleep(800000); // 延时550毫秒
        gpioPWM(12, servo_pwm_mid); // 设置舵机PWM
        gpioPWM(13, motor_pwm_mid + 1400); // 设置电机PWM
    }
    // else if(changeroad == 2){ //向右变道----------------------------------------------------------------
    //     gpioPWM(12, 620); // 设置舵机PWM
    //     gpioPWM(13, motor_pwm_mid + 1300); // 设置电机PWM
    //     usleep(1400000);
    //     gpioPWM(12, 820); // 设置舵机PWM
    //     gpioPWM(13, motor_pwm_mid + 1300); // 设置电机PWM
    //     usleep(500000); // 延时550毫秒
    //     gpioPWM(12, servo_pwm_mid); // 设置舵机PWM
    //     gpioPWM(13, motor_pwm_mid + 1400); // 设置电机PWM
    // }
}


// 控制舵机电机
void motor_servo_contral()
{   
    float servo_pwm_now; // 存储当前舵机PWM值
    if (banma == 0 && flag_banma == 0 ){
        if(number < 50){
            gpioPWM(13, motor_pwm_mid + 1500); // 设置电机PWM
            servo_pwm_now = servo_pd(160); // 计算舵机PWM
        }
        else if (number < number_w ){
            gpioPWM(13, motor_pwm_mid + 5000); // 设置电机PWM
            servo_pwm_now = servo_pd(160); // 计算舵机PWM
            cout << "巡线-----------------------弯道1 PWM:  " << servo_pwm_now << endl;
        }
        else if (number >= number_w ){
            gpioPWM(13, motor_pwm_mid + 1400); // 设置电机PWM
            servo_pwm_now = servo_pd_l(160); // 计算舵机PWM
            cout << "巡线-----------------------弯道2 PWM:  " << servo_pwm_now << endl;
        }
        // if(number < 50){
        //     gpioPWM(13, motor_pwm_mid + 1500); // 设置电机PWM
        //     servo_pwm_now = servo_pd(160); // 计算舵机PWM
        // }
        // else if (number < number_w ){
        //     gpioPWM(13, motor_pwm_mid + 3000); // 设置电机PWM
        //     servo_pwm_now = servo_pd(160); // 计算舵机PWM
        //     cout << "巡线-----------------------弯道1 PWM:  " << servo_pwm_now << endl;
        // }
        // else if (number >= number_w ){
        //     gpioPWM(13, motor_pwm_mid + 1500); // 设置电机PWM
        //     servo_pwm_now = servo_pd_l(160); // 计算舵机PWM
        //     cout << "巡线-----------------------弯道2 PWM:  " << servo_pwm_now << endl;
        // }
        gpioPWM(servo_pin, servo_pwm_now);
    }
    else if(banma == 1 && flag_banma == 0){ // 如果检测到斑马线 且斑马线flag未完成{
        flag_banma = 1;
        banma_stop();
        system("sudo -u pi /home/pi/.nvm/versions/node/v12.22.12/bin/node /home/pi/network-rc/we2hdu.js"); // 播放音频文件
        number = 0;
        // sleep(1);
    }
    else if(flag_banma == 1 && flag_changeroad == 0){
        if(changeroad == 1){ // 向左变道----------------------------------------------------------------
            flag_changeroad = 1;
            motor_changeroad();
            number = 0;
        }else if(changeroad == 2){ //向右变道----------------------------------------------------------------
            // flag_changeroad = 1;
            // motor_changeroad();
            // number = 0;
        }
    }
    else if(flag_changeroad == 1 && count_bz < 3 ){
        // if(number < number_ten){
        //     servo_pwm_now = servo_pd(160); // 计算舵机PWM
        //     cout << "变完道-------未避障------PWM:  " << servo_pwm_now << endl;

        //     gpioPWM(motor_pin, motor_pwm_mid + 2000); 
        //     gpioPWM(servo_pin, servo_pwm_now);
        // }
        // else
         if ( bz_get == 1 ){
            servo_pwm_now = servo_pd_bz(160); // 计算舵机PWM
            cout << "INNNN避障------PWM:  " << servo_pwm_now << endl;

            gpioPWM(motor_pin, motor_pwm_mid + 1300); 
            gpioPWM(servo_pin, servo_pwm_now);
        }else{
            servo_pwm_now = servo_pd_bz1(160); // 计算舵机PWM
            cout << " 避障------未检测到------PWM:  " << servo_pwm_now << endl;

            gpioPWM(motor_pin, motor_pwm_mid + 1300); 
            gpioPWM(servo_pin, servo_pwm_now);
        }
    }
    else if(count_bz >= 3 && park_find == 0){
        servo_pwm_now = servo_pd_bz1(160); // 计算舵机PWM
        cout << " 避障结束，寻找车库-------------------PWM: "<< servo_pwm_now << endl;
        if(number < number_ten_park){
            gpioPWM(motor_pin, motor_pwm_mid + 1700); 
        }else{
            gpioPWM(motor_pin, motor_pwm_mid + 1200); 
        }
        gpioPWM(servo_pin, servo_pwm_now);
    }
    else if (park_find == 1 && flag_park_find == 0){
        flag_park_find = 1;
        motor_park();
        cout << "-----------------STOP-------------------" << endl;
        number = 0;
    }
    // else if (flag_park_find == 1 && flag_parkchose == 0 && flag_gohead == 1 && number > 7 && (numbera + numberb >= 5)){
    //     servo_pwm_now = servo_pd_AB(160); // 计算舵机PWM
    //     cout << "停车时------PWM:  " << servo_pwm_now << endl;

    //     gpioPWM(motor_pin, motor_pwm_mid + 650);
    //     gpioPWM(servo_pin, servo_pwm_now);
    // }
    // else if (flag_parkchose == 1){
    //     gpioPWM(13, motor_pwm_mid - 500); // 设置电机PWM
    //     usleep(200000); // 延时300毫秒
    //     gpioPWM(13, motor_pwm_mid); // 设置电机PWM
    //     gpioPWM(12, servo_pwm_mid); // 设置舵机PWM
    //     sleep(100);
    //     exit(0);
    // }
}

//-----------------------------------------------------------------------------------主函数-----------------------------------------------
int main(void)
{
    gpioTerminate();           // 终止GPIO操作
    servo_motor_pwmInit();     // 初始化舵机PWM

//----------------打开摄像头------------------------------------------------
    VideoCapture capture;       // 视频捕获对象
    capture.open(0);           // 打开默认摄像头

    // 设置视频属性
    capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    capture.set(cv::CAP_PROP_FPS, 30); // 设置帧率
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 320); // 设置帧宽
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 240); // 设置帧高

    if (!capture.isOpened())   // 检查摄像头是否成功打开
    {
        cout << "Can not open camera!" << endl;
        cout << "please enter any key to exit" << endl;
        cin.ignore();// 等待用户输入
        return -1;            // 返回错误代码
    }

    // 输出摄像头的属性
    cout << "FPS: " << capture.get(cv::CAP_PROP_FPS) << endl;
    cout << "Frame Width: " << capture.get(cv::CAP_PROP_FRAME_WIDTH) << endl;
    cout << "Frame Height: " << capture.get(cv::CAP_PROP_FRAME_HEIGHT) << endl;
    //---------------------------------------------------

    while (capture.read(frame)){

        frame = undistort(frame); // 对帧进行去畸变处理

        // 记录开始时间
        auto start = std::chrono::high_resolution_clock::now();
        
        // 处理发车逻辑

        // fache_sign == 0 && find_first == 0 时blue_card_find待机寻找蓝卡；
        // 找到后find_first = 1,blue_card_remove判断是否移开挡板;
        // 移开挡板fache_sign = 1,开始巡线；
        if (fache_sign == 0) // 如果开始标志为0
        {
            // 根据条件调用不同的函数
            if (find_first == 0) //   find_first = 0; 标记是否找到第一个目标  1为找到 0为未找到 默认值为0 找到后进入检测是否移开挡板
            {
                blue_card_find(); // 查找蓝卡
            }
            else
            {
                blue_card_remove(); // 移除蓝卡
            }

        }
        else // 如果开始标志不为0
        {


            number++; // 计数器加1

            if ( banma == 0 ){

                bin_image = ImageSobel(frame); // 图像预处理
                Tracking(bin_image); // 进行巡线识别


                if(number > number_w + 70){
                    banma = banma_get(frame);
                    cout << "斑马线检测---------------------------:   " << banma << endl;
                }

            }
            else if ( flag_changeroad == 1 && count_bz < 3 ){

                number1++;

                bin_image = ImageSobel(frame); // 图像预处理
                Tracking(bin_image); // 进行巡线识别

                if(number > number_ten_bz && number1 > 10){

                    bz_get = 0;

                    result.clear();
                    result = yolo_obs.detect(frame); // 进行Yolo检测

                    if (result.size() > 0){

                        BoxInfo box = result.at(0);

                        // bz_bottom = box.y2; // 计算避障底部点

                        if( box.y2 < bz_y2) {
                            bz_get = 1;
                            bz_xcenter = (box.x1 + box.x2) / 2; // 计算避障中心点
                            bz_bottom = box.y2; // 计算避障底部点
                            bz_heighest = box.y1; // 计算避障高度
                            bz_area = (box.x2 - box.x1) * (box.y2 - box.y1); // 计算避障面积
                        }

                        if(last_bz == 0 && bz_get == 0 && number1 > 5) {
                            count_bz++;
                            cout << "-------------------------切换避障方案---------------------------" << endl;
                            number1 = 0;
                            if(count_bz == 3) {
                                number = 0;
                            }
                            
                        }

                        last_bz = (box.y2 >= bz_y2) ? 1 : 0;

                    }
                    
                    ////// mark
                    if(bz_get == 1 && count_bz % 2 == 1){
                        bin_image = drawWhiteLine(bin_image, cv::Point(bz_xcenter, bz_bottom), cv::Point(int((right_line[0].x + right_line[1].x + right_line[2].x) / 3), 155), 8); // 绘制避障中心线
                        Tracking_bz(bin_image); // 进行补线后---避障巡线识别
                    }
                    else if(bz_get == 1 && count_bz % 2 == 0){
                        bin_image = drawWhiteLine(bin_image, cv::Point(bz_xcenter, bz_bottom), cv::Point(int((left_line[0].x + left_line[1].x + left_line[2].x) / 3), 155), 8); // 绘制避障中心线
                        Tracking_bz(bin_image); // 进行补线后---避障巡线识别
                    }
                    //////
                }
            }
            else if(count_bz >= 3 && park_find == 0 ){
                bin_image = ImageSobel(frame); // 图像预处理
                Tracking(bin_image); // 进行巡线识别

                if( number > number_ten_park){
                    // result.clear();
                    // result = yolo_ab.detect(frame); // 进行Yolo检测
                    // for(auto box : result) { // 遍历所有的box
                    //     if((box.label == 0 || box.label == 1) && box.x1 > 60 && box.x2 < 260){ {
                    //         park_find = 1;
                    //         break; // 退出for循环
                    //     }
                    // }
                    park_find = find_parking(frame);
                }
            }
            else if(park_find == 1 && flag_parkchose == 0 && number > 3){ //停车检测----------------------------------------------------------------------------------------------------

                result_ab.clear();
                result_ab = yolo_ab_lite.decode_v5lite(frame); // 进行Yolo检测
                int park_mid_get = 0 ;
                for(auto box : result_ab){//遍历所有的box

                    // if(box.label == 0 && parkchose == 0){
                    //     parkchose = 1;
                    //     break; // 退出for循环
                    // }else if(box.label == 1 && parkchose == 0){
                    //     parkchose = 2;
                    //     break; // 退出for循环
                    // }
                    if (box.label == 0 && parkchose == 0 ) {
                        numbera++;
                    } else if (box.label == 1 && parkchose == 0 ) {
                        numberb++;
                    }

                    if (numbera + numberb >= 5) {
                        if (numbera > numberb) {
                            parkchose = 1;
                            cout << "parkchose: 1" << endl;
                            // break; // 退出for循环
                        } else {
                            parkchose = 2;
                            cout << "parkchose: 2" << endl;
                            // break; // 退出for循环
                        }
                        // numbera = 0;
                        // numberb = 0;
                    }

                    if(flag_gohead == 0 && numbera + numberb >= 5){
                        gohead(parkchose);
                        flag_gohead = 1;
                        // numbera = 0;
                        // numberb = 0;
                        // break; // 退出for循环
                        
                    }

                    // if(parkchose == 1 && box.label == 0){
                    //     park_mid_get = 320 ;
                    //     cout << "Heeee of A: " << box.y1 << endl;  
                    //     if( box.y1 > 208){
                    //         flag_parkchose = 1;
                    //     }
                    //     if((box.x1 + box.x2)/2 < park_mid_get ){
                    //         park_mid_get = (box.x1 + box.x2) / 2;
                    //     }
                    // }
                    // else if(parkchose == 2 && box.label == 1){
                    //     park_mid_get = 0;
                    //     cout << "Heeee of B: " << box.y1 << endl;
                    //     if( box.y1 > 208){
                    //         flag_parkchose = 1;
                    //     }
                    //     if((box.x1 + box.x2)/2 > park_mid_get ){
                    //         park_mid_get = (box.x1 + box.x2) / 2;
                    //     }
                    // } 
                }
                if(park_mid_get != 0 && park_mid_get != 320){
                    park_mid = park_mid_get;
                }else{
                    park_mid = 160;
                }
            }
        }

        motor_servo_contral(); // 控制舵机电机

        // 记录结束时间
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        // 输出处理一帧所需的时间和帧率
        // std::cout << "Time per frame: " << elapsed.count() << " seconds" << std::endl;
        std::cout << "FPS: " << 1.0 / elapsed.count() << "   Number: " << number << "     Number1:  " << number1 << std::endl;

    }
}
