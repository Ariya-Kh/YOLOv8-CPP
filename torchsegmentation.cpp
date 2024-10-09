#include "torchsegmentation.h"

namespace F = torch::nn::functional;

TorchSegmentation::TorchSegmentation() : YoloSegmentation<torch::Tensor>() {}

bool TorchSegmentation::load_model(std::string &modelPath) {
    if (torch::cuda::is_available()) {
        std::cout << " -----> Inference device: CUDA" << std::endl;
        device_type = torch::kCUDA;
    } else {
        std::cout << " -----> Inference device: CPU" << std::endl;
        device_type = torch::kCPU;
    }

    try {
        module = torch::jit::load(modelPath);
        module.to(device_type);

    } catch (...) {
        std::cout << "------ Error in loading the model!" << std::endl;
        return false;
    }

    std::cout << " ----------> Model is loaded." << std::endl;

    return true;
}

std::vector<std::vector<Detection>> TorchSegmentation::detect(std::vector<cv::Mat> &src_imgs) {
    add_pad = false;
    filtered_outputs.clear();
    orig_size.clear();
    auto batch_size  = static_cast<int>(src_imgs.size());
    torch::Tensor model_input;

    if(use_gpumat){
        std::vector<cv::cuda::GpuMat> g_images;
        for (auto &img : src_imgs) {
            cv::cuda::GpuMat gImg;
            gImg.upload(img);
            g_images.push_back(gImg);
        }
        model_input = preprocess(g_images);
    }
    else{
        model_input = preprocess(src_imgs);
    }



    std::vector<torch::Tensor> outputs     = inference(model_input);

    torch::Tensor              pred        = outputs[0];
    torch::Tensor proto; //for segmentation
    int                        dimensions  = pred[0].size(0);

    if(task == Task::Segment){
        proto  = outputs[1];
        num_of_classes = dimensions - 32 - 4;
    }
    else{
        num_of_classes = dimensions - 4;
    }

    pred = pred.transpose(1, 2);

    conf_threshold.resize(num_of_classes);
    conf_threshold.clear();

    for (int i = 0; i < num_of_classes; i++) {
        conf_threshold.push_back(0.5);
    }

    std::vector<std::vector<Detection>> dets;

    for (int i = 0; i < batch_size; i++) {
        std::vector<Detection> batch_det;

        filtered_outputs.push_back(calculate_bboxes(pred[i]));
        std::vector<int>      nms_result;
        cv::dnn::NMSBoxesBatched(filtered_outputs[i].boxes, filtered_outputs[i].confidences,
                                 filtered_outputs[i].class_ids, general_threshold, iou_threshold, nms_result);

        for (int j = 0; j < nms_result.size(); j++) {
            int   index      = nms_result[j];
            int   cls_idx    = filtered_outputs[i].class_ids[index];
            float confidence = filtered_outputs[i].confidences[index];
            if (confidence < conf_threshold[cls_idx])
                continue;

            cv::Rect box = filtered_outputs[i].boxes[index];

            Detection single_det;
            single_det.class_id = cls_idx;
            single_det.confidence = confidence;
            single_det.box = box;

            if(task == Task::Segment){
                torch::Tensor process_output    = process_mask(pred[i], proto[i], index, i);
                torch::Tensor crop_output       = crop_mask(pred[i], process_output, index, i);
                cv::Mat       upSampling_output = up_sampling(crop_output, i);
                single_det.mask = upSampling_output;
            }

            batch_det.push_back(single_det);
        }

        dets.push_back(batch_det);
    }

    return dets;
}

torch::Tensor TorchSegmentation::preprocess(std::vector<cv::cuda::GpuMat> &src_imgs) {

    for (cv::cuda::GpuMat &image : src_imgs) {
        orig_size.push_back({image.rows, image.cols});

        float height = image.rows;
        float width  = image.cols;

        float r      = std::min(input_size.height / height, input_size.width / width);
        int   padw   = std::round(width * r);
        int   padh   = std::round(height * r);

        if ((int)width != padw || (int)height != padh) {
            cv::cuda::resize(image, image, cv::Size(padw, padh));
        }

        float dw    = input_size.width - padw;
        float dh    = input_size.height - padh;

        dw         /= 2.0f;
        dh         /= 2.0f;
        int top     = int(std::round(dh - 0.1f));
        int bottom  = int(std::round(dh + 0.1f));
        int left    = int(std::round(dw - 0.1f));
        int right   = int(std::round(dw + 0.1f));

        cv::cuda::copyMakeBorder(image, image, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});
        image.convertTo(image, CV_32FC3, 1.0f / 255.0f);

        top_pad       = top;
        left_pad      = left;
        add_pad       = true;

        pparam.ratio  = 1 / r;
        pparam.dw     = dw;
        pparam.dh     = dh;
        pparam.height = height;
        pparam.width  = width;
    }

    auto cols  = input_size.width;
    int  type  = src_imgs[0].type();

    int  bRows = 0;
    for (auto &gimg : src_imgs) {
        bRows += gimg.rows;
    }

    cv::cuda::GpuMat batch_image(bRows, cols, type);
    int              offset = 0;
    for (int i = 0; i < src_imgs.size(); ++i) {
        auto gimg = src_imgs[i];
        gimg.copyTo(batch_image(cv::Rect(0, offset, cols, gimg.rows)));
        offset += gimg.rows;
    }

    auto                 batch_size = static_cast<int>(src_imgs.size());

    std::vector<int64_t> sizes      = {batch_size, src_imgs[0].rows, src_imgs[0].cols, batch_image.channels()};

    int                  height     = src_imgs[0].rows;
    int                  channels   = batch_image.channels();
    long long            step       = src_imgs[0].step / sizeof(float);
    std::vector<int64_t> strides    = {height * step, step, channels, 1};
    torch::Tensor input_tensor      = torch::from_blob(batch_image.data, sizes, strides, torch::kCUDA).to(device_type);
    input_tensor                    = input_tensor.permute({0, 3, 1, 2}).contiguous();
    return input_tensor;
}

torch::Tensor TorchSegmentation::preprocess(std::vector<cv::Mat> &src_imgs) {

    for (cv::Mat &image : src_imgs) {
        orig_size.push_back({image.rows, image.cols});

        float height = image.rows;
        float width  = image.cols;

        float r      = std::min(input_size.height / height, input_size.width / width);
        int   padw   = std::round(width * r);
        int   padh   = std::round(height * r);

        if ((int)width != padw || (int)height != padh) {
            cv::resize(image, image, cv::Size(padw, padh));
        }

        float dw    = input_size.width - padw;
        float dh    = input_size.height - padh;

        dw         /= 2.0f;
        dh         /= 2.0f;
        int top     = int(std::round(dh - 0.1f));
        int bottom  = int(std::round(dh + 0.1f));
        int left    = int(std::round(dw - 0.1f));
        int right   = int(std::round(dw + 0.1f));

        cv::copyMakeBorder(image, image, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});

        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        image.convertTo(image, CV_32FC3, 1.0f / 255.0f); // normalization

        top_pad       = top;
        left_pad      = left;
        add_pad       = true;

        pparam.ratio  = 1 / r;
        pparam.dw     = dw;
        pparam.dh     = dh;
        pparam.height = height;
        pparam.width  = width;
    }

    cv::Mat batch_image;
    cv::vconcat(src_imgs, batch_image);
    auto          batchSize    = static_cast<int>(src_imgs.size());
    auto          rows         = input_size.height;
    auto          cols         = input_size.width;
    auto          channels     = batch_image.channels();
    torch::Tensor input_tensor = torch::from_blob(batch_image.data, {batchSize, rows, cols, channels}).to(device_type);
    input_tensor               = input_tensor.permute({0, 3, 1, 2}).contiguous();
    return input_tensor;
}

std::vector<torch::Tensor> TorchSegmentation::inference(torch::Tensor &model_input) {

    std::vector<torch::Tensor>      outputs;
    std::vector<torch::jit::IValue> iValue;
    iValue.emplace_back(model_input);
    torch::jit::IValue output  = module.forward(iValue);

    auto               output0 = output.toTuple()->elements()[0].toTensor();
    outputs.push_back(output0);

    if(task == Task::Segment){
        auto               output1 = output.toTuple()->elements()[1].toTensor();
        outputs.push_back(output1);
    }


    return outputs;
}

inline static float clamp(float val, float min, float max) { return val > min ? (val < max ? val : max) : min; }

FilteredOutputs     TorchSegmentation::calculate_bboxes(const torch::Tensor &pred) {

    FilteredOutputs       outputs;
    std::vector<cv::Rect> bboxes;
    auto                  pred_cpu        = pred.cpu();

    auto                 &dw              = pparam.dw;
    auto                 &dh              = pparam.dh;
    auto                 &width           = pparam.width;
    auto                 &height          = pparam.height;
    auto                 &ratio           = pparam.ratio;
    // Extract class scores starting from index 4
    torch::Tensor         class_scores    = pred_cpu.slice(1, 4, 4 + num_of_classes);

    // Find bounding boxes with at least one class score greater than 0.5
    torch::Tensor         mask            = class_scores >= general_threshold;
    torch::Tensor         indices         = mask.any(1).nonzero().squeeze(1);
    auto                  indices_data    = indices.data_ptr<int64_t>();

    // Extract the bounding boxes and their corresponding scores
    torch::Tensor         filtered_bboxes = pred_cpu.index_select(0, indices).slice(1, 0, 4);
    torch::Tensor         filtered_scores = class_scores.index_select(0, indices);

    for (int j = 0; j < indices.size(0); ++j) {
        auto data = filtered_bboxes[j];

        outputs.rows.push_back((int)indices_data[j]);

        float            x  = data[0].item<float>() - dw;
        float            y  = data[1].item<float>() - dh;
        float            w  = data[2].item<float>();
        float            h  = data[3].item<float>();

        float            x0 = clamp((x - 0.5f * w) * ratio, 0.f, width);
        float            y0 = clamp((y - 0.5f * h) * ratio, 0.f, height);
        float            x1 = clamp((x + 0.5f * w) * ratio, 0.f, width);
        float            y1 = clamp((y + 0.5f * h) * ratio, 0.f, height);

        cv::Rect_<float> bbox;
        bbox.x      = x0;
        bbox.y      = y0;
        bbox.width  = x1 - x0;
        bbox.height = y1 - y0;

        outputs.boxes.push_back(bbox);
        // auto max_value = torch::max(filtered_scores[j]);
        auto          result    = torch::max(filtered_scores[j], 0);
        torch::Tensor max_value = std::get<0>(result);
        torch::Tensor max_index = std::get<1>(result);
        outputs.confidences.push_back(max_value.item<float>());

        outputs.class_ids.push_back((int)max_index.item<float>());
    }

    return outputs;
}

torch::Tensor TorchSegmentation::process_mask(const torch::Tensor &pred, const torch::Tensor &proto, int nms_index,
                                              int batch_num) {

    auto lastIndex           = pred[nms_index].sizes()[0];
    // 32 -> number of masks
    auto firstIndex          = lastIndex - 32;
    auto mask                = pred[filtered_outputs[batch_num].rows[nms_index]].slice(0, firstIndex, lastIndex);
    mask                     = mask.view({1, 32});

    auto          b          = proto.view({32, -1});

    auto          dotProduct = torch::inner(mask, b.t());
    torch::Tensor sigmoid    = dotProduct.sigmoid();

    return sigmoid;
}

torch::Tensor TorchSegmentation::crop_mask(const torch::Tensor &pred, const torch::Tensor &det_vec, int nms_index,
                                           int batch_num) {

    auto  img           = det_vec.view({1, mh, mw});

    float x_center      = pred[filtered_outputs[batch_num].rows[nms_index]].select(0, 0).item<float>();
    float y_center      = pred[filtered_outputs[batch_num].rows[nms_index]].select(0, 1).item<float>();
    float width         = pred[filtered_outputs[batch_num].rows[nms_index]].select(0, 2).item<float>();
    float height        = pred[filtered_outputs[batch_num].rows[nms_index]].select(0, 3).item<float>();

    float x_min         = x_center - width / 2.0;
    float y_min         = y_center - height / 2.0;
    float x_max         = x_center + width / 2.0;
    float y_max         = y_center + height / 2.0;

    int   tl_x          = (int)(x_min / 4);
    int   tl_y          = (int)(y_min / 4);
    int   br_x          = (int)(x_max / 4);
    int   br_y          = (int)(y_max / 4);

    auto  row_mask      = torch::arange(0, mh).view({1, -1, 1}).expand({1, mh, mw});
    auto  col_mask      = torch::arange(0, mw).view({1, 1, -1}).expand({1, mh, mw});

    // Create masks for within bounding box and outside bounding box
    auto  inside_mask   = (col_mask >= tl_x) & (col_mask <= br_x) & (row_mask >= tl_y) & (row_mask <= br_y);
    auto  outside_mask  = 1 - inside_mask.to(torch::kFloat); // Logical NOT operation

    inside_mask         = inside_mask.to(device_type);
    img                *= inside_mask.to(torch::kFloat);

    return img;
}

cv::Mat TorchSegmentation::up_sampling(const torch::Tensor &detection_mask, int batch_num) {

    static int           y = 0;
    std::vector<cv::Mat> result;
    torch::Tensor        image              = detection_mask.clone().view({1, 1, mh, mw}) * 255;
    std::vector<int64_t> target_size        = {input_size.height, input_size.width};
    torch::Tensor        interpolated_image = F::interpolate(
        image, F::InterpolateFuncOptions().size(target_size).mode(torch::kBilinear).align_corners(true))[0];
    //    kNearest, kLinear, kBilinear, kBicubic, kTrilinear, kArea, kNearestExact

    auto               interpolated_image_cpu = interpolated_image.cpu();
    std::vector<float> tensor_data(interpolated_image_cpu.data_ptr<float>(),
                                   interpolated_image_cpu.data_ptr<float>() + interpolated_image_cpu.numel());
    cv::Mat            output_image(input_size.height, input_size.width, CV_32FC1, tensor_data.data());

    output_image.convertTo(output_image, CV_8UC1);
    cv::threshold(output_image, output_image, mask_threshold, 255, cv::THRESH_BINARY);
    // cv::imwrite("/home/perticon/upsamplee_" + std::to_string(y) + ".jpg", output_image);

    if (add_pad) {
        int size = 0;
        if (orig_size[batch_num][0] >= orig_size[batch_num][1])
            size = orig_size[batch_num][0];
        else
            size = orig_size[batch_num][1];

        newCol = (output_image.cols * orig_size[batch_num][1]) / size;
        newRow = (output_image.rows * orig_size[batch_num][0]) / size;

        cv::Rect roi(left_pad, top_pad, newCol, newRow);

        output_image(roi).copyTo(output_image);
    }

    cv::resize(output_image, output_image, cv::Size(orig_size[batch_num][1], orig_size[batch_num][0]));

    return output_image;
}
