## 快速开始

### 简单使用

```cpp
#include "fastestdet.hpp"

// 1. 初始化
FastestDet detector("model.param", "model.bin", 2, {"A", "B"});

// 2. 检测
std::vector<DetectObject> objects = detector.detect(frame);

// 3. 绘制
detector.draw_objects(frame, objects);
```

### 完整示例

```cpp
#include "fastestdet.hpp"
#include <opencv2/opencv.hpp>

int main()
{
    // 初始化检测器
    std::vector<std::string> class_names = {"A", "B"};
    FastestDet detector(
        "FastestDet.ncnn.param",  // 模型参数文件
        "FastestDet.ncnn.bin",    // 模型权重文件
        2,                         // 类别数量
        class_names,               // 类别名称
        352,                       // 输入尺寸
        0.65f,                     // 置信度阈值
        0.45f,                     // NMS阈值
        4,                         // 线程数
        false                      // 使用GPU (false=CPU)
    );
    
    // 单张图像检测
    cv::Mat image = cv::imread("test.jpg");
    auto objects = detector.detect(image);
    detector.draw_objects(image, objects);
    cv::imwrite("output.jpg", image);
    
    return 0;
}
```

## 类设计说明

### 数据结构

```cpp
// 检测对象结构体
struct DetectObject
{
    cv::Rect_<float> rect;      // 边界框
    int label;                  // 类别ID
    float prob;                 // 置信度
};
```

### 主要接口

#### 1. 构造函数

```cpp
FastestDet(const std::string& param_path,      // 模型参数文件路径
           const std::string& bin_path,        // 模型权重文件路径
           int num_classes,                    // 类别数量
           const std::vector<std::string>& class_names,  // 类别名称
           int target_size = 352,              // 输入尺寸
           float prob_threshold = 0.65f,       // 置信度阈值
           float nms_threshold = 0.45f,        // NMS阈值
           int num_threads = 4,                // 线程数
           bool use_gpu = false);              // 是否使用GPU
```

#### 2. 检测函数

```cpp
// 对单帧图像进行检测
std::vector<DetectObject> detect(const cv::Mat& frame);
```

#### 3. 绘制函数

```cpp
// 在图像上绘制检测结果
void draw_objects(cv::Mat& image, 
                 const std::vector<DetectObject>& objects,
                 bool show_label = true,   // 是否显示标签
                 bool show_conf = true);   // 是否显示置信度
```

#### 4. 其他功能函数

```cpp
// 设置置信度阈值
void set_prob_threshold(float threshold);

// 设置NMS阈值
void set_nms_threshold(float threshold);

// 获取类别名称
std::string get_class_name(int label) const;
```

## 使用示例

### 基本用法（单张图像检测）

```cpp
#include "fastestdet.hpp"
#include <opencv2/opencv.hpp>

int main()
{
    // 1. 设置模型路径
    std::string param_path = "FastestDet.ncnn.param";
    std::string bin_path = "FastestDet.ncnn.bin";
    
    // 2. 设置类别信息
    int num_classes = 2;
    std::vector<std::string> class_names = {"A", "B"};
    
    // 3. 初始化检测器
    FastestDet detector(param_path, 
                       bin_path,
                       num_classes,
                       class_names,
                       352,     // 输入尺寸
                       0.65f,   // 置信度阈值
                       0.45f,   // NMS阈值
                       4,       // 线程数
                       false);  // 不使用GPU
    
    // 4. 读取图像
    cv::Mat frame = cv::imread("test.jpg");
    
    // 5. 执行检测（单帧）
    std::vector<DetectObject> objects = detector.detect(frame);
    
    // 6. 处理检测结果
    for (const auto& obj : objects)
    {
        int center_x = obj.rect.x + obj.rect.width / 2;
        int center_y = obj.rect.y + obj.rect.height / 2;
        std::cout << "Class: " << detector.get_class_name(obj.label)
                  << " Score: " << obj.prob
                  << " Center: (" << center_x << ", " << center_y << ")"
                  << std::endl;
    }
    
    // 7. 绘制结果
    detector.draw_objects(frame, objects);
    cv::imwrite("output.jpg", frame);
    
    return 0;
}
```

### 视频流检测（逐帧处理）

```cpp
#include "fastestdet.hpp"
#include <opencv2/opencv.hpp>

int main()
{
    // 初始化检测器
    std::vector<std::string> class_names = {"A", "B"};
    FastestDet detector("model.param", "model.bin", 2, class_names);
    
    // 打开摄像头或视频文件
    cv::VideoCapture cap(0);  // 默认摄像头
    // cv::VideoCapture cap("video.mp4");  // 打开视频文件
    
    cv::Mat frame;
    while (true)
    {
        // 读取帧
        cap >> frame;
        if (frame.empty())
            break;
        
        // 单帧检测
        std::vector<DetectObject> objects = detector.detect(frame);
        
        // 绘制结果
        detector.draw_objects(frame, objects);
        
        // 显示结果
        cv::imshow("Detection", frame);
        if (cv::waitKey(1) == 'q')
            break;
    }
    
    return 0;
}
```

## 编译说明

### 使用CMake编译

#### 1. 基本编译

```bash
cd demo/fastestdet_example
mkdir build && cd build
cmake ..
make
./fastestdet_example
```

#### 2. macOS OpenMP支持说明

在macOS系统上，需要安装 `libomp` 以支持OpenMP并行加速：

```bash
# 使用Homebrew安装libomp
brew install libomp
```

CMakeLists.txt已经配置好了macOS的OpenMP支持，会自动链接到：
- 头文件路径: `/opt/homebrew/opt/libomp/include`
- 库文件路径: `/opt/homebrew/opt/libomp/lib`

如果你的libomp安装在其他位置，需要修改 `CMakeLists.txt` 中的相应路径。

#### 3. 指定NCNN路径（可选）

如果NCNN不在默认路径，可以通过命令行参数指定：

```bash
cmake .. -DNCNN_INCLUDE_DIR=/path/to/ncnn/include -DNCNN_LIB_DIR=/path/to/ncnn/lib
make
```

#### 4. 在项目中使用

在项目 `CMakeLists.txt` 中添加：

```cmake
cmake_minimum_required(VERSION 3.10)
project(YourProject)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

# macOS OpenMP支持
if(APPLE)
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
    include_directories("/opt/homebrew/opt/libomp/include")
    link_directories("/opt/homebrew/opt/libomp/lib")
    set(OpenMP_C_FOUND TRUE)
    set(OpenMP_CXX_FOUND TRUE)
endif()

# 查找依赖
find_package(OpenCV REQUIRED)
find_package(ncnn REQUIRED)

# 包含FastestDet头文件
include_directories(${PROJECT_SOURCE_DIR}/inc)

# 添加可执行文件
add_executable(your_app
    your_main.cpp
    src/fastestdet.cpp
)

# 链接库
target_link_libraries(your_app
    ${OpenCV_LIBS}
    ncnn
    pthread
)

# 添加OpenMP支持
if(APPLE)
    target_link_libraries(your_app omp)
else()
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(your_app OpenMP::OpenMP_CXX)
    endif()
endif()
```
