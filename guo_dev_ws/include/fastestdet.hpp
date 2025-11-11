#ifndef FASTESTDET_HPP
#define FASTESTDET_HPP

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "net.h"

// 检测对象结构体
struct DetectObject
{
    cv::Rect_<float> rect;      // 边界框
    int label;                  // 类别ID
    float prob;                 // 置信度
};

class FastestDet
{
public:
    /**
     * @brief 构造函数
     * @param param_path 模型参数文件路径
     * @param bin_path 模型权重文件路径
     * @param num_classes 类别数量
     * @param class_names 类别名称列表
     * @param target_size 输入图像尺寸 (默认352)
     * @param prob_threshold 置信度阈值 (默认0.65)
     * @param nms_threshold NMS阈值 (默认0.45)
     * @param num_threads 线程数 (默认4)
     * @param use_gpu 是否使用GPU (默认false)
     */
    FastestDet(const std::string& param_path,
               const std::string& bin_path,
               int num_classes,
               const std::vector<std::string>& class_names,
               int target_size = 352,
               float prob_threshold = 0.65f,
               float nms_threshold = 0.45f,
               int num_threads = 4,
               bool use_gpu = false);

    /**
     * @brief 析构函数
     */
    ~FastestDet();

    /**
     * @brief 对单帧图像进行目标检测
     * @param frame 输入图像
     * @return 检测到的目标列表
     */
    std::vector<DetectObject> detect(const cv::Mat& frame);

    /**
     * @brief 在图像上绘制检测结果
     * @param image 输入输出图像
     * @param objects 检测到的目标列表
     * @param show_label 是否显示标签 (默认true)
     * @param show_conf 是否显示置信度 (默认true)
     */
    void draw_objects(cv::Mat& image, 
                     const std::vector<DetectObject>& objects,
                     bool show_label = true,
                     bool show_conf = true);

    /**
     * @brief 设置置信度阈值
     * @param threshold 新的置信度阈值
     */
    void set_prob_threshold(float threshold);

    /**
     * @brief 设置NMS阈值
     * @param threshold 新的NMS阈值
     */
    void set_nms_threshold(float threshold);

    /**
     * @brief 获取类别名称
     * @param label 类别ID
     * @return 类别名称
     */
    std::string get_class_name(int label) const;

private:
    /**
     * @brief 生成检测候选框
     */
    void generate_proposals(const ncnn::Mat& out, 
                          int img_w, 
                          int img_h,
                          std::vector<DetectObject>& objects);

    /**
     * @brief 对候选框进行NMS处理
     */
    void nms_sorted_bboxes(const std::vector<DetectObject>& objects,
                          std::vector<int>& picked);

    /**
     * @brief 快速排序
     */
    void qsort_descent_inplace(std::vector<DetectObject>& objects, 
                              int left, 
                              int right);

    /**
     * @brief 计算两个边界框的交集面积
     */
    static inline float intersection_area(const DetectObject& a, 
                                        const DetectObject& b);

    /**
     * @brief Sigmoid函数
     */
    static inline float sigmoid(float x);

private:
    ncnn::Net net_;                             // NCNN网络
    int target_size_;                           // 输入尺寸
    float prob_threshold_;                      // 置信度阈值
    float nms_threshold_;                       // NMS阈值
    int num_classes_;                           // 类别数量
    std::vector<std::string> class_names_;      // 类别名称
    bool initialized_;                          // 是否初始化成功
};

#endif // FASTESTDET_HPP
