
import cv2
from threading import Thread
import sys
from queue import Queue

class WebcamVideoStream:
    def __init__(self, src=0):
        #initialize the video camera stream and read the first frame
        #from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
 
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False
        
    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self
 
    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
 
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
 
    def read(self):
        # return the frame most recently read
        return True, self.frame
 
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


class FastStreamer:
    def __init__(self, src=0, queueSize = 1024):
        #initialize the video camera stream and read the first frame
        #from the stream
        self.stream = cv2.VideoCapture(src)

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)

        
    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self
 
    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            if not self.Q.full():
                # otherwise, read the next frame from the stream
                (grabbed, frame) = self.stream.read()
                
                if not grabbed:
                    self.stop()
                    return

                self.Q.put(frame)

    def read(self):
        # return the frame most recently read
        return True, self.Q.get()

    def more(self):
        #Return true if there are still frames in the Q
        return self.Q.qsize() > 0
 
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


