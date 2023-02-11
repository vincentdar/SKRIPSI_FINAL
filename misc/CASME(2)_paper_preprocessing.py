import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import math
import os
import dlib
from typing import List, Mapping, Optional, Tuple, Union

negative_path = r"D:\CASME\CASME(2)\Spotting_paper_cropped\Negative"
positive_path = r"D:\CASME\CASME(2)\Spotting_paper_cropped\Positive"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")

def crop_face(image):
    # detect the faces
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray)
    for rect in rects:
        
        # apply the shape predictor to the face ROI
        shape = predictor(gray, rect)  

        h, w, _ = image.shape
        x_left = w
        y_top = h
        x_right = 0
        y_bot = 0
        for index in range(0, 68):
            x_left = min(shape.part(index).x, x_left)    
            y_top = min(shape.part(index).y, y_top)    
            x_right = max(shape.part(index).x, x_right)    
            y_bot = max(shape.part(index).y, y_bot)     
        

        cropped = image[y_top:y_bot, x_left:x_right] 
        return cropped


# Iterate through subject
for subject in os.listdir(positive_path):
    pos_subject_dir = os.path.join(positive_path, subject)
    neg_subject_dir = os.path.join(negative_path, subject)

    # Iterate through video
    for video in os.listdir(pos_subject_dir):
        pos_video_dir = os.path.join(pos_subject_dir, video)
        neg_video_dir = os.path.join(neg_subject_dir, video)        

        # Positive
        for images in os.listdir(pos_video_dir):
            image_dir = os.path.join(pos_video_dir, images)            

            
            image = cv2.imread(image_dir)
            crop_1 = crop_face(image)
            h, w, _ = crop_1.shape
            
            # detect the faces
            # apply the shape predictor to the face ROI
            cropped_rect = dlib.rectangle(left=0, top=0, right=w, bottom=h)
            shape = predictor(cv2.cvtColor(crop_1, cv2.COLOR_BGR2GRAY), cropped_rect)  
            
            x_left = w
            y_top = h
            x_right = 0
            y_bot = 0
            for index in range(0, 68):                
                x_left = min(shape.part(index).x, x_left)    
                y_top = min(shape.part(index).y, y_top)    
                x_right = max(shape.part(index).x, x_right)    
                y_bot = max(shape.part(index).y, y_bot)     
                

            crop_2 = crop_1[y_top:y_bot, x_left:x_right] 

            # cv2.imwrite(image_dir, crop_2)
            print("Written", image_dir)
            break

        # Negative
        for images in os.listdir(neg_video_dir):
            image_dir = os.path.join(neg_video_dir, images)            

            image = cv2.imread(image_dir)
            crop_1 = crop_face(image)
            h, w, _ = crop_1.shape
            
            # detect the faces
            # apply the shape predictor to the face ROI
            cropped_rect = dlib.rectangle(left=0, top=0, right=w, bottom=h)
            shape = predictor(cv2.cvtColor(crop_1, cv2.COLOR_BGR2GRAY), cropped_rect)  
            
            x_left = w
            y_top = h
            x_right = 0
            y_bot = 0
            for index in range(0, 68):                
                x_left = min(shape.part(index).x, x_left)    
                y_top = min(shape.part(index).y, y_top)    
                x_right = max(shape.part(index).x, x_right)    
                y_bot = max(shape.part(index).y, y_bot)     
                

            crop_2 = crop_1[y_top:y_bot, x_left:x_right] 

            # cv2.imwrite(image_dir, crop_2)
            print("Written", image_dir)

            break
        break
    break

        
        

