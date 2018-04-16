# Real-time Object Tracking and detection for Video-Streams


*Implementation under progress*


*Download weights here and place them in model_data/*
- [yolov2.weights](https://www.dropbox.com/s/57zhd75mmmc5olf/yolov2.weights?dl=0)
- [MobileNetSSD_deploy.caffemodel](https://www.dropbox.com/s/d7pxo7kw67zb0e1/MobileNetSSD_deploy.caffemodel?dl=0)


Execute code from root directory. Example: 
```
python3 src/main.py --model model_data/yolov2.weights --config model_data/yolov2.cfg --classes model_data/coco_classes.txt --input media/sample_video.mp4
```


or 


```
python3 src/main.py --model model_data/MobileNetSSD_deploy.caffemodel --config model_data/MobileNetSSD_deploy.prototxt --classes model_data/MobileNet_classes.txt --input media/sample_video.mp4
```


[![MobileNet_SSD with KCF tracker](https://imgur.com/a/kdXFn)](https://www.youtube.com/watch?v=levZEJKcPjM&feature=youtu.be "MobileNet_SSD with KCF tracker")


*Note: --input can be ommitted, which will activate stream from webcam. New objects are detected when current objects being tracked are lost, or when 'q' is pressed*