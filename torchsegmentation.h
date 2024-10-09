#ifndef TORCHSEGMENTATION_H
#define TORCHSEGMENTATION_H

#include "yolosegmentation.h"
#include <torch/torch.h>
#include <torch/cuda.h>
#include <torch/script.h>

class TorchSegmentation : public YoloSegmentation<torch::Tensor> {
public:
    TorchSegmentation();
    bool                       load_model(std::string &model_path) override;
    std::vector<std::vector<Detection>>   detect(std::vector<cv::Mat> &src_imgs) override;
    torch::Tensor              preprocess(std::vector<cv::cuda::GpuMat> &src_imgs) override;
    torch::Tensor              preprocess(std::vector<cv::Mat> &src_imgs) override;
    std::vector<torch::Tensor> inference(torch::Tensor &model_input) override;
    FilteredOutputs            calculate_bboxes(const torch::Tensor &pred) override;
    torch::Tensor              process_mask(const torch::Tensor &pred, const torch::Tensor &proto, int nms_index,
                                            int batch_num) override;
    torch::Tensor              crop_mask(const torch::Tensor &pred, const torch::Tensor &det_vec, int nms_index,
                                         int batch_num) override;
    cv::Mat                    up_sampling(const torch::Tensor &detection_mask, int batch_num) override;
    bool use_gpumat{true};

private:
    std::vector<cv::cuda::GpuMat> g_images;
    torch::DeviceType             device_type;
    torch::jit::script::Module    module;
};

#endif // TORCHSEGMENTATION_H
