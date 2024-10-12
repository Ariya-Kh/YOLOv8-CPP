#ifndef OPENCVSEGMENTATION_H
#define OPENCVSEGMENTATION_H

#include "yolosegmentation.h"

class OpenCVSegmentation: public YoloSegmentation<cv::Mat>
{
public:
    OpenCVSegmentation();
    bool load_model(std::string &model_path) override;
    std::vector<std::vector<Detection>> detect(std::vector<cv::Mat> &src_imgs) override;
    cv::Mat preprocess(std::vector<cv::Mat> &src_imgs) override;
    cv::Mat preprocess(std::vector<cv::cuda::GpuMat> &src_imgs) override;
    std::vector<cv::Mat> inference(cv::Mat &model_input) override;
    FilteredOutputs calculate_bboxes(const cv::Mat &pred) override;
    cv::Mat process_mask(const cv::Mat &pred, const cv::Mat &proto, int nms_index, int batch_num) override;
    cv::Mat crop_mask(const cv::Mat &pred, const cv::Mat &det_vec, int nms_index, int batch_num) override;
    cv::Mat up_sampling(const cv::Mat &detection_mask, int batch_num) override;
private:
    cv::dnn::Net net;
};

#endif // OPENCVSEGMENTATION_H
