#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 21:04:10 2018

@author: liniu
"""

import cv2
import numpy as np
import skimage.io
from utils import resize_image
from face_detector import YoloFace

def draw_rect(img, dboxes, save_file):
    cv_img = np.copy(img)
    tmp_channel = np.copy(cv_img[:,:,0])
    cv_img[:,:,0] = cv_img[:,:,2]
    cv_img[:,:,2] = tmp_channel  
    for i, dbox in enumerate(dboxes):
        cv2.rectangle(cv_img, (int(dbox["left"]), int(dbox["top"])), (int(dbox["right"]), int(dbox["bottom"])), (0,0,255), 3)              
    cv2.imwrite(save_file, cv_img)


# load model
yolo_model = YoloFace('YOLO_detector')    

# read and resize image
img = skimage.io.imread('input.jpg')
img, _ = resize_image(img, yolo_model)

# detect faces and extract features with one feature vector corresponding to one face
dboxes, feats = yolo_model.yolo_detect_face(img, thresh=0.5)
       
# draw bounding box 
draw_rect(img, dboxes, 'output.jpg')



