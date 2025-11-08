// ImageSobel 测试Demo - 简化版
// 按步骤显示每张图片

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <string>
#include <dirent.h>

using namespace std;
using namespace cv;

const int FAST_MODE = 0;
const int MIN_COMPONENT_AREA = 400;

int main(int argc, char** argv) {
    string imgPath;
    
    // 获取图片路径
    if (argc > 1) {
        imgPath = argv[1];
    } else {
        cout << "用法: ./test_sobel <图片路径>" << endl;
        return -1;
    }
    
    // 读取图片
    Mat frame = imread(imgPath, IMREAD_COLOR);
    if (frame.empty()) {
        cout << "错误: 无法读取图片 " << imgPath << endl;
        return -1;
    }
    
    cout << "原始尺寸: " << frame.cols << "x" << frame.rows << endl;
    
    // 缩放到摄像头分辨率
    Mat resizedFrame;
    resize(frame, resizedFrame, Size(320, 240));
    imshow("1. Resized 320x240", resizedFrame);
    cout << "按任意键继续..." << endl;
    waitKey(0);
    
    Rect roiRect(1, 109, 318, 46);
    Mat roiFrame = resizedFrame(roiRect).clone();
    imshow("2. ROI", roiFrame);
    cout << "按任意键继续..." << endl;
    waitKey(0);
    
    // 灰度图
    Mat grayImage;
    cvtColor(roiFrame, grayImage, COLOR_BGR2GRAY);
    imshow("3. Gray", grayImage);
    cout << "按任意键继续..." << endl;
    waitKey(0);
    
    // 均值滤波
    Mat blurredImage;
    blur(grayImage, blurredImage, Size(5, 5));
    imshow("4. Blurred", blurredImage);
    cout << "按任意键继续..." << endl;
    waitKey(0);
    
    // Sobel边缘检测
    Mat sobelX, sobelY;
    Sobel(blurredImage, sobelX, CV_64F, 1, 0, 3); // x方向梯度
    Sobel(blurredImage, sobelY, CV_64F, 0, 1, 3); // y方向梯度
    Mat gradientMagnitude = cv::abs(sobelY) + 0.5 * cv::abs(sobelX); // x方向权重减半
    Mat gradientMagnitude_8u;
    convertScaleAbs(gradientMagnitude, gradientMagnitude_8u);
    imshow("5. Sobel Gradient", gradientMagnitude_8u);
    cout << "按任意键继续..." << endl;
    waitKey(0);
    
    // 颜色空间转换 + 自适应阈值，增强不同场景下的白色跑道线
    Mat hsvImage;
    cvtColor(roiFrame, hsvImage, COLOR_BGR2HSV);
    vector<Mat> hsvChannels;
    split(hsvImage, hsvChannels); // H, S, V

    Mat claheOutput;
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(4, 4));
    clahe->apply(hsvChannels[2], claheOutput);
    GaussianBlur(claheOutput, claheOutput, Size(5, 5), 0);
    imshow("6. CLAHE V Channel", claheOutput);
    cout << "按任意键继续..." << endl;
    waitKey(0);  

    Mat adaptiveMask;
    adaptiveThreshold(claheOutput, adaptiveMask, 255,
                        ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY,
                      31, -10);
    imshow("7. Adaptive Mask", adaptiveMask);
    cout << "按任意键继续..." << endl;
    waitKey(0);

    Mat binaryImage = adaptiveMask.clone();

    medianBlur(binaryImage, binaryImage, 3);

    Mat noiseKernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(binaryImage, binaryImage, MORPH_OPEN, noiseKernel);

    imshow("8. Binary", binaryImage);
    cout << "按任意键继续..." << endl;
    waitKey(0);
    
    // 形态学操作
    Mat morphImage = binaryImage.clone();
    Mat kernel_close = getStructuringElement(MORPH_RECT, Size(9, 5));
    morphologyEx(morphImage, morphImage, MORPH_CLOSE, kernel_close);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    dilate(morphImage, morphImage, kernel, Point(-1, -1), 1);

    Mat labels, stats, centroids;
    int numLabels = connectedComponentsWithStats(morphImage, labels, stats, centroids, 8, CV_32S);
    Mat filteredMorph = Mat::zeros(morphImage.size(), CV_8U);
    for (int i = 1; i < numLabels; ++i) {
        if (stats.at<int>(i, CC_STAT_AREA) >= MIN_COMPONENT_AREA) {
            filteredMorph.setTo(255, labels == i);
        }
    }
    morphImage = filteredMorph;

    imshow("9. Morphed", morphImage);
    cout << "按任意键继续..." << endl;
    waitKey(0);
    
    // Hough直线检测
    vector<Vec4i> lines;
    HoughLinesP(morphImage, lines, 1, CV_PI / 180, 20, 15, 8);
    cout << "检测到 " << lines.size() << " 条直线" << endl;
    
    // 在原图上绘制结果
    Mat houghResult = resizedFrame.clone();
    rectangle(houghResult, Rect(1, 109, 318, 46), Scalar(0, 255, 0), 1);
    
    Mat finalImage = Mat::zeros(240, 320, CV_8U);
    
    for (const auto &l : lines) {
        double angle = atan2(l[3] - l[1], l[2] - l[0]) * 180.0 / CV_PI;
        double length = hypot(l[3] - l[1], l[2] - l[0]);
        
        if (abs(angle) > 15 && length > 8) {
            Vec4i adjustedLine = l;
            adjustedLine[0] += roiRect.x; adjustedLine[1] += roiRect.y;
            adjustedLine[2] += roiRect.x; adjustedLine[3] += roiRect.y;
            
            line(finalImage, Point(adjustedLine[0], adjustedLine[1]),
                 Point(adjustedLine[2], adjustedLine[3]), Scalar(255), 3, LINE_AA);
            
            line(houghResult, Point(adjustedLine[0], adjustedLine[1]),
                 Point(adjustedLine[2], adjustedLine[3]), Scalar(0, 0, 255), 2, LINE_AA);
        }
    }
    
    imshow("10. Hough Lines", houghResult);
    cout << "按任意键继续..." << endl;
    waitKey(0);
    
    imshow("11. Final Result", finalImage);
    cout << "按ESC退出" << endl;
    waitKey(0);
    
    destroyAllWindows();
    cout << "测试完成!" << endl;
    return 0;
}
