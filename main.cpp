#include "torchsegmentation.h"

int main(int argc, char *argv[])
{
    cv::Mat img1 = cv::imread("/home/perticon/6.jpg");
    cv::Mat img2 = cv::imread("/home/perticon/5.jpg");

    TorchSegmentation ts;
    std::string model_path = "/home/perticon/test1/best.torchscript";
    ts.task = Task::Segment;
    ts.use_gpumat = true;
    ts.load_model(model_path);

    std::vector<cv::Mat> imgs;

    for (int var = 0; var < 5; ++var) {
        imgs.clear();
        imgs.push_back(img1);
        imgs.push_back(img2);
        // imgs.push_back(img3);
        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<Detection>> dets = ts.detect(imgs);
        auto lap = std::chrono::high_resolution_clock::now();

        std::cout <<"--------- Process Time : " << std::chrono::duration_cast<std::chrono::milliseconds>(lap-t0).count() << std::endl;
    }

    std::cout << "done ..." << std::endl;

    return 0;
}
