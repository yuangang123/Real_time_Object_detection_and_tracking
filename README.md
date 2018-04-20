# Real-time Object Tracking and detection for Video-Streams


*Implementation under progress*


##Pre-req:
```
1. OpenCV2
2. imutils
```


*Download weights here and place them in model_data/*
- [yolov2.weights](https://www.dropbox.com/s/57zhd75mmmc5olf/yolov2.weights?dl=0)
- [MobileNetSSD_deploy.caffemodel](https://www.dropbox.com/s/d7pxo7kw67zb0e1/MobileNetSSD_deploy.caffemodel?dl=0)


##Arguments:
```
$python3 src/main.py -h
usage: main.py [-h] [--input INPUT] [--output OUTPUT] --model MODEL
               [--config CONFIG] [--classes CLASSES] [--thr THR]

Object Detection and Tracking on Video Streams

optional arguments:
  -h, --help         show this help message and exit
  --input INPUT      Path to input image or video file. Skip this argument to
                     capture frames from a camera.
  --output OUTPUT    Path to save output as video file. Skip this argument if
  					 you don't want the output to be saved. 
  --model MODEL      Path to a binary file of model that contains trained weights.
                     It could be a file with extensions .caffemodel (Caffe) or
                     .weights (Darknet)
  --config CONFIG    Path to a text file of model that contains network
                     configuration. It could be a file with extensions
                     .prototxt (Caffe) or .cfg (Darknet)
  --classes CLASSES  Optional path to a text file with names of classes to
                     label detected objects.
  --thr THR          Confidence threshold for detection. Default: 0.35
```


Execute code from root directory. Example: 
```
python3 src/main.py --model model_data/yolov2.weights --config model_data/yolov2.cfg --classes model_data/coco_classes.txt --input media/sample_video.mp4 --output out/sample_output.avi
```


or 


```
python3 src/main.py --model model_data/MobileNetSSD_deploy.caffemodel --config model_data/MobileNetSSD_deploy.prototxt --classes model_data/MobileNet_classes.txt --input media/sample_video.mp4 --output out/sample_output.avi
```


*Note: --input can be ommitted, which will activate stream from webcam. New objects are detected when current objects being tracked are lost, or when 'q' is pressed*


## MobileNet_SSD with KCF tracker

[![MobileNet_SSD with KCF tracker](https://raw.githubusercontent.com/apoorvavinod/Real_time_Object_detection_and_tracking/master/misc/MobileNet_SSD_KCF.gif)](https://www.youtube.com/watch?v=levZEJKcPjM&feature=youtu.be "MobileNet_SSD with KCF tracker")


## YOLOv2 with KCF tracker

[![YOLOv2 with KCF tracker](https://raw.githubusercontent.com/apoorvavinod/Real_time_Object_detection_and_tracking/master/misc/YOLOv2_with_KCF.gif)](https://www.youtube.com/watch?v=KmyrSarmvhg&feature=youtu.be "YOLOv2 with KCF tracker")


