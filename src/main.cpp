#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>


int main()
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);//将opencv日志输出级别降低

    cv::Mat img = cv::imread("data/CE7DEM_1km_16bit.png");   // 相对可执行文件路径
    if (img.empty())
    {
        std::cout << "图片没读到，确认 lunar_test.png 放在 exe 同级目录\n";
        return 0;
    }
    cv::imshow("OpenCV 测试窗口", img);
    cv::waitKey(0);
    return 0;
}