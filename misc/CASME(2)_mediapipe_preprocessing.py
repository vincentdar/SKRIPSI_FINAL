import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import math
import os
import dlib
import pandas as pd
from typing import List, Mapping, Optional, Tuple, Union

negative_path = r"D:\CASME\CASME(2)\Spotting_Version2_Localize\Negative"
positive_path = r"D:\CASME\CASME(2)\Spotting_Version2_Localize\Positive"


df = pd.read_csv("bounding_box.csv")

# Iterate through subject
for subject in os.listdir(positive_path):
    pos_subject_dir = os.path.join(positive_path, subject)
    neg_subject_dir = os.path.join(negative_path, subject)

    # Iterate through video positive
    for video in os.listdir(pos_subject_dir):
        pos_video_dir = os.path.join(pos_subject_dir, video)        

        category = video[:-2]
        print(category)
        

        find_in_df = df[(df['video'] == category) & (df['subject'] == subject)].index
        data = df.iloc[find_in_df] 

        if data.empty:
            category = video[:-3]

            find_in_df = df[(df['video'] == category) & (df['subject'] == subject)].index
            data = df.iloc[find_in_df] 

        print(data)

        try:
            xleft = int(data['xleft'])
            ytop = int(data['ytop'])
            xright = int(data['xright'])
            ybot = int(data['ybot'])
        except Exception:
            print("NO BOUNDING BOX AVAILABLE")
            break
        

        # Positive
        for images in os.listdir(pos_video_dir):
            image_dir = os.path.join(pos_video_dir, images)            
            
            image = cv2.imread(image_dir)
                     
            try:
                cropped = image[ytop:ybot, xleft:xright]
                cropped = cv2.resize(cropped, (224, 224), cv2.INTER_AREA)
                cv2.imwrite(image_dir, cropped)
                print("Written", image_dir)  
            except:
                pass
            
                              

    # Iterate through video negative
    for video in os.listdir(neg_subject_dir):        
        neg_video_dir = os.path.join(neg_subject_dir, video) 

        category = video[:-2]
        print(category)
        

        find_in_df = df[(df['video'] == category) & (df['subject'] == subject)].index
        data = df.iloc[find_in_df] 

        if data.empty:
            category = video[:-3]

            find_in_df = df[(df['video'] == category) & (df['subject'] == subject)].index
            data = df.iloc[find_in_df] 

        print(data)

        try:
            xleft = int(data['xleft'])
            ytop = int(data['ytop'])
            xright = int(data['xright'])
            ybot = int(data['ybot'])
        except Exception:
            print("NO BOUNDING BOX AVAILABLE")
            break
        

        # Negative
        for images in os.listdir(neg_video_dir):
            image_dir = os.path.join(neg_video_dir, images)                        
            image = cv2.imread(image_dir)
                     
            try:
                cropped = image[ytop:ybot, xleft:xright]
                cropped = cv2.resize(cropped, (224, 224), cv2.INTER_AREA)
                cv2.imwrite(image_dir, cropped)
                print("Written", image_dir)  
            except:
                pass
                                

        
        

