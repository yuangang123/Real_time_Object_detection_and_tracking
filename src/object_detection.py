import cv2 as cv
import sys
import numpy as np


class object_detector:

    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.framework = None
        self.load_model()

    def load_model(self):
        if self.model.endswith('weights') and self.cfg.endswith('cfg'):
            self.net = cv.dnn.readNetFromDarknet(self.cfg, self.model)
            self.framework = 'Darknet'
        elif self.model.endswith('caffemodel') and self.cfg.endswith('prototxt'):
            self.net = cv.dnn.readNetFromCaffe(self.cfg, self.model)
            self.framework = 'Caffe'
        else:
            sys.exit('Wrong input for model weights and cfg')

        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


    def predict(self,frame):

        # Create a 4D blob from a frame.
        if self.framework == 'Darknet':
            blob = cv.dnn.blobFromImage(frame, 0.007843, (416, 416), 127.5, crop = False)
        else:
            blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)

        # Run a model
        self.net.setInput(blob)
        out = self.net.forward()
        
        return out

    