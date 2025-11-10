#include "fastestdet.hpp"
#include <float.h>
#include <cmath>
#include <algorithm>
#include <chrono>

// 性能计时宏
#ifdef ENABLE_PERF_TIMING
    #define PERF_TIMER_START(name) auto timer_start_##name = std::chrono::high_resolution_clock::now()
    #define PERF_TIMER_END(name) \
        do { \
            auto timer_end_##name = std::chrono::high_resolution_clock::now(); \
            auto duration_##name = std::chrono::duration_cast<std::chrono::microseconds>(timer_end_##name - timer_start_##name).count(); \
            std::cout << "[PERF] " << #name << ": " << duration_##name / 1000.0 << " ms" << std::endl; \
        } while(0)
#else
    #define PERF_TIMER_START(name)
    #define PERF_TIMER_END(name)
#endif

// 构造函数
FastestDet::FastestDet(const std::string& param_path,
                       const std::string& bin_path,
                       int num_classes,
                       const std::vector<std::string>& class_names,
                       int target_size,
                       float prob_threshold,
                       float nms_threshold,
                       int num_threads,
                       bool use_gpu)
    : target_size_(target_size)
    , prob_threshold_(prob_threshold)
    , nms_threshold_(nms_threshold)
    , num_classes_(num_classes)
    , class_names_(class_names)
    , initialized_(false)
{
    // 设置NCNN基础选项
    net_.opt.use_vulkan_compute = use_gpu;
    net_.opt.num_threads = num_threads;
    
    // 根据平台架构启用不同的优化选项
#if defined(__ARM_NEON) || defined(__aarch64__)
    // ARM平台优化
    #ifdef __APPLE__
        // Apple Silicon
        std::cout << "Detected Apple Silicon, enabling conservative optimizations..." << std::endl;
        net_.opt.use_winograd_convolution = false;     // 小模型不需要
        net_.opt.use_sgemm_convolution = true;         // SGEMM优化
        net_.opt.use_packing_layout = true;            // 布局优化
        net_.opt.use_fp16_packed = false;              // Apple Silicon不需要
        net_.opt.use_fp16_storage = false;             // 保持FP32避免转换开销
    #else
        // 嵌入式ARM平台
        std::cout << "Detected embedded ARM, enabling full ARM optimizations..." << std::endl;
        net_.opt.use_winograd_convolution = true;      // Winograd卷积优化
        net_.opt.use_sgemm_convolution = true;         // SGEMM优化
        net_.opt.use_packing_layout = true;            // 布局优化
        net_.opt.use_fp16_packed = true;               // ARM NEON FP16
        net_.opt.use_fp16_storage = true;              // FP16存储
    #endif
    net_.opt.use_int8_inference = false;               // int8推理
    net_.opt.use_fp16_arithmetic = false;              // 保持精度
    net_.opt.use_bf16_storage = false;                 // BF16存储
#else
    // x86/x64平台优化
    std::cout << "Detected x86/x64 platform, enabling x86 optimizations..." << std::endl;
    net_.opt.use_winograd_convolution = false;         // 小模型不需要
    net_.opt.use_sgemm_convolution = true;             // SGEMM优化
    net_.opt.use_int8_inference = false;               // int8推理
    net_.opt.use_packing_layout = true;                // 布局优化
    net_.opt.use_fp16_packed = false;                  // x86不支持
    net_.opt.use_fp16_storage = false;                 // 保持FP32
    net_.opt.use_fp16_arithmetic = false;              // 保持精度
    net_.opt.use_bf16_storage = false;                 // BF16存储
#endif
    
    // 内存池优化
    net_.opt.blob_allocator = nullptr;                 // 使用默认分配器
    net_.opt.workspace_allocator = nullptr;            // 使用默认工作区分配器

    // 加载模型参数
    if (net_.load_param(param_path.c_str()) != 0)
    {
        std::cerr << "Failed to load param: " << param_path << std::endl;
        return;
    }

    // 加载模型权重
    if (net_.load_model(bin_path.c_str()) != 0)
    {
        std::cerr << "Failed to load model: " << bin_path << std::endl;
        return;
    }

    initialized_ = true;
    std::cout << "FastestDet initialized successfully!" << std::endl;
    std::cout << "  Input size: " << target_size_ << std::endl;
    std::cout << "  Classes: " << num_classes_ << std::endl;
    std::cout << "  Threads: " << num_threads << std::endl;
    std::cout << "  GPU: " << (use_gpu ? "Enabled" : "Disabled") << std::endl;

#if defined(__ARM_NEON) || defined(__aarch64__)
    #ifdef __APPLE__
        std::cout << "  Platform: Apple Silicon" << std::endl;
    #else
        std::cout << "  Platform: ARM" << std::endl;
    #endif
#else
    std::cout << "  Platform: x86/x64" << std::endl;
#endif
    std::cout << "  Optimizations: Winograd=" << net_.opt.use_winograd_convolution
              << " SGEMM=" << net_.opt.use_sgemm_convolution
              << " FP16=" << net_.opt.use_fp16_storage << std::endl;
}

// 析构函数
FastestDet::~FastestDet()
{
    net_.clear();
}

// Sigmoid函数
inline float FastestDet::sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

// 计算两个边界框的交集面积
inline float FastestDet::intersection_area(const DetectObject& a, const DetectObject& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

// 快速排序
void FastestDet::qsort_descent_inplace(std::vector<DetectObject>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p)
            i++;

        while (objects[j].prob < p)
            j--;

        if (i <= j)
        {
            std::swap(objects[i], objects[j]);
            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

// NMS处理
void FastestDet::nms_sorted_bboxes(const std::vector<DetectObject>& objects,
                                   std::vector<int>& picked)
{
    picked.clear();

    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = objects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const DetectObject& a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const DetectObject& b = objects[picked[j]];

            // 类别内NMS
            if (a.label != b.label)
                continue;

            // 计算IoU
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > nms_threshold_)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

// 生成检测候选框
void FastestDet::generate_proposals(const ncnn::Mat& out, 
                                    int img_w, 
                                    int img_h,
                                    std::vector<DetectObject>& objects)
{
    PERF_TIMER_START(decode_bbox);
    const int channels = out.c;
    const int height = out.h;
    const int width = out.w;
    
    const int num_class = channels - 5; // 类别数量
    
    // 遍历特征图的每个位置
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int base_idx = i * width + j;
            
            // 从通道0获取前背景分数
            const float* pobj_ptr = out.channel(0);
            float obj_score = pobj_ptr[base_idx];
            
            // 提前退出
            if (obj_score < prob_threshold_)
                continue;
            
            // 从通道1-4获取边界框回归值
            const float* preg_ptr0 = out.channel(1);
            const float* preg_ptr1 = out.channel(2);
            const float* preg_ptr2 = out.channel(3);
            const float* preg_ptr3 = out.channel(4);
            
            float tx = preg_ptr0[base_idx];
            float ty = preg_ptr1[base_idx];
            float tw = preg_ptr2[base_idx];
            float th = preg_ptr3[base_idx];
            
            // 从通道5+找到最大类别分数和对应索引
            int class_index = 0;
            float class_score = -FLT_MAX;
            for (int k = 0; k < num_class; k++)
            {
                const float* pcls_ptr = out.channel(5 + k);
                float score = pcls_ptr[base_idx];
                if (score > class_score)
                {
                    class_score = score;
                    class_index = k;
                }
            }
            
            // 计算置信度: obj^0.6 * cls^0.4
            float confidence = pow(obj_score, 0.6f) * pow(class_score, 0.4f);
            
            if (confidence < prob_threshold_)
                continue;
            
            // 解码边界框
            float bcx = (tanh(tx) + j) / width;
            float bcy = (tanh(ty) + i) / height;
            float bw = sigmoid(tw);
            float bh = sigmoid(th);
            
            // 转换为 x1, y1, x2, y2
            float x1 = bcx - 0.5f * bw;
            float y1 = bcy - 0.5f * bh;
            float x2 = bcx + 0.5f * bw;
            float y2 = bcy + 0.5f * bh;
            
            // 裁剪到 [0, 1]
            x1 = std::max(0.0f, std::min(1.0f, x1));
            y1 = std::max(0.0f, std::min(1.0f, y1));
            x2 = std::max(0.0f, std::min(1.0f, x2));
            y2 = std::max(0.0f, std::min(1.0f, y2));
            
            // 转换为像素坐标
            x1 *= img_w;
            y1 *= img_h;
            x2 *= img_w;
            y2 *= img_h;
            
            DetectObject obj;
            obj.rect.x = x1;
            obj.rect.y = y1;
            obj.rect.width = x2 - x1;
            obj.rect.height = y2 - y1;
            obj.label = class_index;
            obj.prob = confidence;
            
            objects.push_back(obj);
        }
    }
    PERF_TIMER_END(decode_bbox);
    
#ifdef ENABLE_PERF_TIMING
    std::cout << "[PERF]   - Proposals found: " << objects.size() << std::endl;
#endif
}

// 对单帧图像进行目标检测
std::vector<DetectObject> FastestDet::detect(const cv::Mat& frame)
{
    PERF_TIMER_START(total);
    std::vector<DetectObject> objects;

    if (!initialized_)
    {
        std::cerr << "FastestDet not initialized!" << std::endl;
        return objects;
    }

    if (frame.empty())
    {
        std::cerr << "Input frame is empty!" << std::endl;
        return objects;
    }

    int img_w = frame.cols;
    int img_h = frame.rows;

    // 图像预处理
    PERF_TIMER_START(preprocess);
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(frame.data, 
                                                  ncnn::Mat::PIXEL_BGR,
                                                  img_w, img_h, 
                                                  target_size_, target_size_);

    // 归一化到 [0, 1]
    const float norm_vals[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
    in.substract_mean_normalize(0, norm_vals);
    PERF_TIMER_END(preprocess);

    // 创建提取器
    ncnn::Extractor ex = net_.create_extractor();

    // 输入数据
    ex.input("in0", in);

    // 执行推理
    PERF_TIMER_START(inference);
    ncnn::Mat out;
    ex.extract("out0", out);
    PERF_TIMER_END(inference);

    // 生成候选框
    PERF_TIMER_START(generate_proposals);
    std::vector<DetectObject> proposals;
    generate_proposals(out, img_w, img_h, proposals);
    PERF_TIMER_END(generate_proposals);

    // 按置信度排序
    PERF_TIMER_START(sort);
    if (!proposals.empty())
    {
        qsort_descent_inplace(proposals, 0, proposals.size() - 1);
    }
    PERF_TIMER_END(sort);

    // 应用NMS
    PERF_TIMER_START(nms);
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked);
    PERF_TIMER_END(nms);

    // 输出结果
    PERF_TIMER_START(copy_results);
    int count = picked.size();
    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];
    }
    PERF_TIMER_END(copy_results);

    PERF_TIMER_END(total);
    return objects;
}

// 在图像上绘制检测结果
void FastestDet::draw_objects(cv::Mat& image, 
                              const std::vector<DetectObject>& objects,
                              bool show_label,
                              bool show_conf)
{
    for (size_t i = 0; i < objects.size(); i++)
    {
        const DetectObject& obj = objects[i];
        
        int x1 = static_cast<int>(obj.rect.x);
        int y1 = static_cast<int>(obj.rect.y);
        int x2 = static_cast<int>(obj.rect.x + obj.rect.width);
        int y2 = static_cast<int>(obj.rect.y + obj.rect.height);
        
        // 绘制黄色边界框
        cv::Scalar color = cv::Scalar(0, 255, 255);  // BGR: 黄色
        cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

        // 准备标签文本
        if (show_label || show_conf)
        {
            std::string text;
            if (show_label && show_conf)
            {
                text = get_class_name(obj.label) + ": " + 
                       std::to_string(obj.prob).substr(0, 4);
            }
            else if (show_label)
            {
                text = get_class_name(obj.label);
            }
            else
            {
                text = std::to_string(obj.prob).substr(0, 4);
            }

            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, 
                                                  cv::FONT_HERSHEY_SIMPLEX, 
                                                  0.5, 1, &baseLine);

            int x = x1;
            int y = y1 - 5;
            if (y < 0)
                y = 0;
            if (x + label_size.width > image.cols)
                x = image.cols - label_size.width;

            // 绘制文本背景
            cv::rectangle(image, 
                         cv::Point(x, y - label_size.height - baseLine),
                         cv::Point(x + label_size.width, y + baseLine),
                         cv::Scalar(0, 0, 0), -1);

            // 绘制文本
            cv::putText(image, text, cv::Point(x, y),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                       cv::Scalar(0, 255, 0), 1);
        }
    }
}

// 设置置信度阈值
void FastestDet::set_prob_threshold(float threshold)
{
    prob_threshold_ = threshold;
}

// 设置NMS阈值
void FastestDet::set_nms_threshold(float threshold)
{
    nms_threshold_ = threshold;
}

// 获取类别名称
std::string FastestDet::get_class_name(int label) const
{
    if (label >= 0 && label < (int)class_names_.size())
    {
        return class_names_[label];
    }
    return "Unknown_" + std::to_string(label);
}
