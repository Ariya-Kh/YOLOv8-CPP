#include "opencvsegmentation.h"
#include "qdebug.h"

OpenCVSegmentation::OpenCVSegmentation() : YoloSegmentation<cv::Mat>() {}


bool OpenCVSegmentation::load_model(std::string &modelPath)
{
    try {
        net = cv::dnn::readNetFromONNX(modelPath);

        if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            qDebug() << " -----> Inference device: CUDA";
        } else {
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            qDebug() << " -----> Inference device: CPU";

        }

    } catch (...) {
        qDebug() << "------ Error in loading the model!";
        return false;
    }

    qDebug() << " ----------> Model is loaded.";

    return true;
}

std::vector<std::vector<Detection>> OpenCVSegmentation::detect(std::vector<cv::Mat> &src_imgs)
{
    add_pad = false;
    orig_size.clear();
    filtered_outputs.clear();

    auto batch_size = static_cast<int>(src_imgs.size());
    cv::Mat model_input = preprocess(src_imgs);

    std::vector<cv::Mat> outputs = inference(model_input);

    cv::Mat pred = outputs[0];
    cv::Mat proto;

    int dimensions = pred.size[1];

    if(task == Task::Segment){
        proto  = outputs[1];
        num_of_classes = dimensions - 32 - 4;
    }
    else{
        num_of_classes = dimensions - 4;
    }

    std::vector<cv::Mat> batch_preds;

    for (int i = 0; i < batch_size; ++i) {
        cv::Mat batch_pred = pred.row(i).reshape(1, dimensions);
        cv::transpose(batch_pred, batch_pred);
        batch_preds.push_back(batch_pred);
    }

    conf_threshold.resize(num_of_classes);
    conf_threshold.clear();

    for (int i = 0; i < num_of_classes; i++) {
        conf_threshold.push_back(0.5);
    }

    std::vector<std::vector<Detection>> dets;

    for (int i = 0; i < batch_size; i++) {
        std::vector<Detection> batch_det;

        filtered_outputs.push_back(calculate_bboxes(batch_preds[i]));
        std::vector<int> nms_result;
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
                cv::Mat process_output = process_mask(batch_preds[i], proto.row(i), index, i);
                cv::Mat crop_output = crop_mask(batch_preds[i], process_output, index, i);
                cv::Mat upSampling_output = up_sampling(crop_output, i);
                single_det.mask = upSampling_output;
            }

            batch_det.push_back(single_det);
        }

        dets.push_back(batch_det);
    }

    return dets;
}

cv::Mat OpenCVSegmentation::preprocess(std::vector<cv::Mat> &src_imgs){

    for (cv::Mat &image : src_imgs){
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

        top_pad       = top;
        left_pad      = left;
        add_pad       = true;

        pparam.ratio  = 1 / r;
        pparam.dw     = dw;
        pparam.dh     = dh;
        pparam.height = height;
        pparam.width  = width;
    }

    cv::Mat blob = cv::dnn::blobFromImages(src_imgs, 1.0/255.0, input_size, cv::Scalar(), true);
    return blob;
}

cv::Mat OpenCVSegmentation::preprocess(std::vector<cv::cuda::GpuMat> &src_imgs)
{
    // to do
}

std::vector<cv::Mat> OpenCVSegmentation::inference(cv::Mat &model_input){

    net.setInput(model_input);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    return outputs;
}

inline static float clamp(float val, float min, float max) { return val > min ? (val < max ? val : max) : min; }


FilteredOutputs OpenCVSegmentation::calculate_bboxes(const cv::Mat &pred){

    FilteredOutputs outputs;
    int rows = pred.size[0];
    int dimensions = pred.size[1];
    float *data = (float *)pred.data;

    auto &dw = pparam.dw;
    auto &dh = pparam.dh;
    auto &width = pparam.width;
    auto &height = pparam.height;
    auto &ratio = pparam.ratio;

    for (int r = 0; r < rows; ++r) {
        cv::Mat scores(1, num_of_classes, CV_32FC1, data + 4);
        cv::Point classIdPoint;
        double max_class_score;
        minMaxLoc(scores, 0, &max_class_score, 0, &classIdPoint);
        max_class_score = (float)max_class_score;
        if (max_class_score >= general_threshold) {

            outputs.rows.push_back(r);
            outputs.confidences.push_back(max_class_score);

            float            x  = data[0] - dw;
            float            y  = data[1] - dh;
            float            w  = data[2];
            float            h  = data[3];

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

            outputs.class_ids.push_back(classIdPoint.x);

        }

        data += dimensions;
    }

    return outputs;
}

cv::Mat OpenCVSegmentation::process_mask(const cv::Mat &pred, const cv::Mat &proto, int nms_index, int batch_num){

    int lastIndex = pred.cols;  // Assuming the last dimension is the number of features
    int firstIndex = lastIndex - 32;

    // Extract the mask slice from the prediction
    cv::Mat mask = pred.row(filtered_outputs[batch_num].rows[nms_index]).colRange(firstIndex, lastIndex);

    // Reshape proto and compute dot product
    cv::Mat b = proto.reshape(1, {32, proto.size[2]*proto.size[3]});  // Reshape proto to 32x(H*W)
    cv::Mat dotProduct = mask*b;

    // Apply sigmoid
    cv::Mat sigmoidOutput;
    cv::exp(-dotProduct, sigmoidOutput);
    sigmoidOutput = 1.0 / (1.0 + sigmoidOutput);

    return sigmoidOutput;
}

cv::Mat OpenCVSegmentation::crop_mask(const cv::Mat &pred, const cv::Mat &det_vec, int nms_index, int batch_num){

    float x_center = pred.at<float>(filtered_outputs[batch_num].rows[nms_index], 0);
    float y_center = pred.at<float>(filtered_outputs[batch_num].rows[nms_index], 1);
    float width = pred.at<float>(filtered_outputs[batch_num].rows[nms_index], 2);
    float height = pred.at<float>(filtered_outputs[batch_num].rows[nms_index], 3);

    float x_min = x_center - width / 2.0;
    float y_min = y_center - height / 2.0;
    float x_max = x_center + width / 2.0;
    float y_max = y_center + height / 2.0;


    int tl_x = (int)(x_min / 4);
    int tl_y = (int)(y_min / 4);
    int br_x = (int)(x_max / 4);
    int br_y = (int)(y_max / 4);

    // Extract the image corresponding to the current index
    cv::Mat img(mh, mw, CV_32FC1, det_vec.data);
    // Create masks for within and outside the bounding box
    cv::Mat row_mask = cv::Mat::zeros(mh, mw, CV_32F);
    cv::Mat col_mask = cv::Mat::zeros(mh, mw, CV_32F);

    // Fill row_mask
    for (int r = 0; r < mh; ++r) {
        row_mask.row(r) = cv::Scalar(r);
    }

    // Fill col_mask
    for (int c = 0; c < mw; ++c) {
        col_mask.col(c) = cv::Scalar(c);
    }

    // Mask for inside the bounding box
    cv::Mat inside_mask = (col_mask >= tl_x) & (col_mask <= br_x) & (row_mask >= tl_y) & (row_mask <= br_y);

    // Convert boolean mask to float
    inside_mask.convertTo(inside_mask, CV_32F);

    // Apply the mask to the image
    img = img.mul(inside_mask);

    // Push the result to the detection_mask vector
    return img;
}

cv::Mat OpenCVSegmentation::up_sampling(const cv::Mat &detection_mask, int batch_num){

    cv::Mat image = detection_mask.clone();

    cv::normalize(image, image, 0, 255, cv::NORM_MINMAX);
    // Find the min and max values and their locations

    cv::Mat interpolated_image;
    cv::resize(image, interpolated_image, input_size, 0, 0, cv::INTER_LINEAR);

    // Output the results
    cv::Mat output_image(input_size, CV_32FC1, interpolated_image.data);

    output_image.convertTo(output_image, CV_8UC1);

    cv::threshold(output_image, output_image, mask_threshold, 255, cv::THRESH_BINARY);

    if(add_pad){
        int size = 0;
        if(orig_size[batch_num][0] >= orig_size[batch_num][1])
            size = orig_size[batch_num][0];
        else
            size = orig_size[batch_num][1];

        int newCol = (output_image.cols*orig_size[batch_num][1])/size;
        int newRow = (output_image.rows*orig_size[batch_num][0])/size;

        cv::Rect roi(left_pad, top_pad, newCol, newRow);

        output_image(roi).copyTo(output_image);
    }

    cv::resize(output_image, output_image, cv::Size(orig_size[batch_num][1], orig_size[batch_num][0]));

    // cv::threshold(interpolated_image, interpolated_image, 200, 255, cv::THRESH_BINARY);
    return output_image;
}

