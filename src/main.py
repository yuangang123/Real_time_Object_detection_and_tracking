import cv2 as cv
import argparse
import sys
import numpy as np
import time
import imutils
from object_detection import object_detector



def drawPred(frame, classes, classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))

    label = '%.2f' % conf

    # Print a label of class.
    if classes:
        assert(classId < len(classes))
        label = '%s: %s' % (classes[classId], label)
    #print(label)

    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

def postprocess(frame, out, threshold, classes, framework):

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    objects_detected = dict()

    if framework == 'Caffe':
        # Network produces output blob with a shape 1x1xNx7 where N is a number of
        # detections and an every detection is a vector of values
        # [batchId, classId, confidence, left, top, right, bottom]
        for detection in out[0, 0]:
            confidence = detection[2]
            if confidence > threshold:
                left = int(detection[3] * frameWidth)
                top = int(detection[4] * frameHeight)
                right = int(detection[5] * frameWidth)
                bottom = int(detection[6] * frameHeight)
                classId = int(detection[1]) - 1  # Skip background label
                drawPred(frame, classes, classId, confidence, int(left), int(top), int(right), int(bottom))
                i = 0
                label = classes[classId]
                label_with_num = str(label) + '_' + str(i)
                while(True):
                    if label_with_num not in objects_detected.keys():
                        break
                    label_with_num = str(label) + '_' + str(i)
                    i = i+1
                objects_detected[label_with_num] = (int(left),int(top),int(right - left), int(bottom-top)) 
                print(label_with_num + ' at co-ordinates '+ str(objects_detected[label_with_num]))

    else:
        # Network produces output blob with a shape NxC where N is a number of
        # detected objects and C is a number of classes + 4 where the first 4
        # numbers are [center_x, center_y, width, height]
        for detection in out:
            confidences = detection[5:]
            classId = np.argmax(confidences)
            confidence = confidences[classId]
            if confidence > threshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = center_x - (width / 2)
                top = center_y - (height / 2)
                drawPred(frame, classes, classId, confidence, int(left), int(top), int(left + width), int(top + height))
                i = 0
                label = classes[classId]
                label_with_num = str(label) + '_' + str(i)
                while(True):
                    if label_with_num not in objects_detected.keys():
                        break
                    label_with_num = str(label) + '_' + str(i)
                    i = i+1
                objects_detected[label_with_num] = (int(left),int(top),int(width),int(height))  
                print(label_with_num + ' at co-ordinates '+ str(objects_detected[label_with_num]))

    return objects_detected

def intermediate_detections(stream, predictor, multi_tracker, tracker, threshold, classes):
    while True:
        _,frame = stream.read()
        predictions = predictor.predict(frame)

        objects_detected = postprocess(frame, predictions, threshold, classes, predictor.framework)

        #Forcing the video to play till more than one objects are detected
        if len(objects_detected) > 1:
            break

    objects_list = list(objects_detected.keys())
    print('Tracking the following objects', objects_list)

    multi_tracker = cv.MultiTracker_create()
    for items in objects_detected.items():
        ok = multi_tracker.add(cv.TrackerKCF_create(), frame, items[1])
        #ok = multi_tracker.add(cv.TrackerMedianFlow_create(), frame, items[1])  
        
    return stream, objects_detected, objects_list, multi_tracker 

def process(args):


    objects_detected = dict()


    #ToDo: Put this in intermediate_detection
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[2]

    if tracker_type == 'BOOSTING':
        tracker = cv.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv.TrackerGOTURN_create()

    predictor = object_detector(args.model, args.config)
    multi_tracker = cv.MultiTracker_create()
    stream = cv.VideoCapture(args.input if args.input else 0)
    
    if args.classes:
        with open(args.classes, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')
    else:
        classes = list(np.arange(0,100))

    stream, objects_detected, objects_list, multi_tracker = intermediate_detections(stream, predictor, multi_tracker, tracker, args.thr, classes)    

    while True:
        grabbed, frame = stream.read()

        if not grabbed:
            break

        timer = cv.getTickCount()

        ok, bboxes = multi_tracker.update(frame)

        fps = cv.getTickFrequency() / (cv.getTickCount() - timer)

        if ok:
            for i,boxes in enumerate(bboxes): 
                label = objects_list[i]
                p1 = (int(boxes[0]), int(boxes[1]))
                p2 = (int(boxes[0] + boxes[2]), int(boxes[1] + boxes[3]))
                cv.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                #cv.putText(frame, label, p1, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 10))
                left = p1[0]
                top = p1[1]
                labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                top = max(top, labelSize[1])
                cv.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv.FILLED)
                cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))


        else:
            cv.putText(frame, 'Tracking Failure. Trying to detect more objects', (100,80), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            stream, objects_detected, objects_list, multi_tracker = intermediate_detections(stream, predictor, multi_tracker, tracker, args.thr, classes)   

        # Display FPS on frame
        cv.putText(frame, "FPS : " + str(int(fps)), (100,50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
        
        #Resize
        frame = imutils.resize(frame, width=500)

        # Display result
        cv.imshow("Tracking", frame)
 
        
        k = cv.waitKey(1) & 0xff

        #Force detect new objects
        if k == ord('q'):
            print('Refreshing. Detecting New objects')
            cv.putText(frame, 'Refreshing. Detecting New objects', (100,80), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            stream, objects_detected, objects_list, multi_tracker = intermediate_detections(stream, predictor, multi_tracker, tracker, args.thr, classes)  
            
        # Exit if ESC pressed    
        if k == 27 : break 


def main():
    
    parser = argparse.ArgumentParser(description='Object Detection and Tracking on Video Streams')
    
    parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')

    parser.add_argument('--model', required=True,
                        help='Path to a binary file of model contains trained weights. '
                             'It could be a file with extensions .caffemodel (Caffe), '
                             '.weights (Darknet)')
    parser.add_argument('--config',
                        help='Path to a text file of model contains network configuration. '
                             'It could be a file with extensions .prototxt (Caffe), .cfg (Darknet)')
    parser.add_argument('--framework', choices=['caffe', 'darknet'],
                        help='Optional name of an origin framework of the model. '
                             'Detect it automatically if it does not set.')
    parser.add_argument('--classes', help='Optional path to a text file with names of classes to label detected objects.')
    
    parser.add_argument('--thr', type=float, default=0.5, help='Confidence threshold')
    
    args = parser.parse_args()


    process(args)

if __name__ == '__main__':
    main()