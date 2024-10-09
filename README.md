# YOLOv8-CPP
use your custom trained model for your task(detect/segment) for inference in cpp using libtorch or opencv.  
the output of the code is a struct for each detected object:  
```
struct Detection{  
    int class_id; // id of detected object  
    float confidence; // confidence of detected object  
    cv::Rect box; // bounding box of detected object  
    cv::Mat mask; // binary mask of detected object (if you are using segment task)  
}
```
You can use either `cv::Mat` or `cv::cuda::GpuMat` for libtorch code.  
