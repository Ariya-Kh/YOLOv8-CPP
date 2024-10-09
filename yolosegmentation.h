#ifndef YOLOSEGMENTATION_H
#define YOLOSEGMENTATION_H

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>

struct PreParam {
    float ratio  = 1.0f;
    float dw     = 0.0f;
    float dh     = 0.0f;
    float height = 0;
    float width  = 0;
};

struct FilteredOutputs {
    std::vector<int>      class_ids;
    std::vector<float>    confidences;
    std::vector<cv::Rect> boxes;
    std::vector<int>      rows;
};

struct Detection{
    int class_id;
    float confidence;
    cv::Rect box;
    cv::Mat mask;
};

enum Task {Detect=0, Segment=1};


template <typename Type> class YoloSegmentation {
public:
    YoloSegmentation()                                                                              = default;
    virtual bool                     load_model(std::string &model_path)                            = 0;
    virtual std::vector<std::vector<Detection>> detect(std::vector<cv::Mat> &src_imgs)              = 0;
    virtual Type                     preprocess(std::vector<cv::Mat> &src_imgs)                     = 0;
    virtual Type                     preprocess(std::vector<cv::cuda::GpuMat> &src_imgs)            = 0;
    virtual std::vector<Type>        inference(Type &model_input)                                   = 0;
    virtual FilteredOutputs          calculate_bboxes(const Type &pred)                             = 0;
    virtual Type    process_mask(const Type &pred, const Type &proto, int nms_index, int batch_num) = 0;
    virtual Type    crop_mask(const Type &pred, const Type &det_vec, int nms_index, int batch_num)  = 0;
    virtual cv::Mat                  up_sampling(const Type &detection_mask, int batch_num) = 0;
    Task task;
protected:
    int                          newCol;
    int                          newRow;
    bool                         add_pad = false; // do not change !!!
    int                          num_of_classes;
    std::vector<FilteredOutputs> filtered_outputs;
    float general_threshold{0.5}; // for finding bboxes with high confidences and eliminating boxes with low confidences
                                  // to reduce NMS elapsed time
    std::vector<float>            conf_threshold;                  // confidence of each class for NMS
    float                         iou_threshold{0.7};              // IOU for NMS
    int                           mask_threshold{170};             // mask threshold for classes masks
    cv::Size                      input_size = cv::Size(960, 960); // input size of the network(model)
    int                           mh         = input_size.height / 4;
    int                           mw         = input_size.width / 4;
    std::vector<std::vector<int>> orig_size; // original dimensions of each image in batch
    int                           top_pad;
    int                           left_pad;
    PreParam                      pparam;
};

#endif // YOLOSEGMENTATION_H
