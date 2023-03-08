import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import math
import os
import dlib
import pandas as pd
from localize import Localize
from cnnlstm import CNNLSTM
from evaluation import MyEvaluation
from typing import List, Mapping, Optional, Tuple, Union

df = pd.read_csv("CAS(ME)^2evaluation.csv")

test_video = [
            r"D:\CASME\CASME(2)\rawvideo\rawvideo\s15\15_0102disgustingteeth.avi"
            #   r"D:\CASME\CASME(2)\rawvideo\rawvideo\s36\36_0401girlcrashing.avi",
            #   r"D:\CASME\CASME(2)\rawvideo\rawvideo\s36\36_0505funnyinnovations.avi",

            #   r"D:\CASME\CASME(2)\rawvideo\rawvideo\s37\37_0101disgustingteeth.avi",
            #   r"D:\CASME\CASME(2)\rawvideo\rawvideo\s37\37_0402beatingpregnantwoman.avi",
            #   r"D:\CASME\CASME(2)\rawvideo\rawvideo\s37\37_0502funnyerrors.avi",
            #   r"D:\CASME\CASME(2)\rawvideo\rawvideo\s37\37_0505funnyinnovations.avi",
            #   r"D:\CASME\CASME(2)\rawvideo\rawvideo\s37\37_0507climbingthewall.avi",
            #   r"D:\CASME\CASME(2)\rawvideo\rawvideo\s37\37_0508funnydunkey.avi",

            #   r"D:\CASME\CASME(2)\rawvideo\rawvideo\s38\38_0502funnyerrors.avi",
            #   r"D:\CASME\CASME(2)\rawvideo\rawvideo\s38\38_0507climbingthewall.avi",

            #   r"D:\CASME\CASME(2)\rawvideo\rawvideo\s40\40_0401girlcrashing.avi",
            #   r"D:\CASME\CASME(2)\rawvideo\rawvideo\s40\40_0502funnyerrors.avi",
            #   r"D:\CASME\CASME(2)\rawvideo\rawvideo\s40\40_0503unnyfarting.avi"
            ]



localization_algorithm = Localize()
# Initialize Spotting Algorithm
prediction_model = CNNLSTM()        
# self.prediction_model.mobilenet("core/transfer_mobilenet_cnnlstm_tfrecord_2/cp.ckpt")
prediction_model.mobilenet("D:\CodeProject2\SKRIPSI_FINAL\core\Weights\Transfer_mobilenet_cnnlstm_localize_tfrecord_pyramid_1/cp.ckpt")

# Pyramid
# "D:\CodeProject2\SKRIPSI_FINAL\core\Transfer_mobilenet_cnnlstm_localize_tfrecord_pyramid_1/cp.ckpt"
# "D:\CodeProject2\SKRIPSI_FINAL\core\Transfer_mobilenet_localize_cnnlstm_tfrecord/cp.ckpt"

eval = MyEvaluation()


for video in test_video:
    # find the detection
    raw_filename = video.split('\\')[-1][:-4]
    find_in_df = df[(df['raw_path'] == raw_filename)]

    detection = []
    for index, item in find_in_df.iterrows():
        onset = int(item['onset'])
        apex = int(item['apex'])
        offset = int(item['offset'])

        if offset == 0:
            offset = apex

        detection.append((onset, offset))


    print("Filename:", raw_filename)
    print("Detection:", detection)
    # Evaluation Loop
    cap = cv2.VideoCapture(video)

    arm_the_localization = True
    detected = True
    xleft = 0
    ytop = 0
    xright = 0
    ybot = 0


    # Read the file
    sliding_window = []
    itr = 0
    while True:    
        ret, frame = cap.read()        
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            # Process
            break

        itr += 1
        # Our operations on the frame come here
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if arm_the_localization:
            arm_the_localization = False
            detected, xleft, ytop, xright, ybot = localization_algorithm.mp_face_mesh_crop_preprocessing(frame)

        cropped = frame[ytop:ybot, xleft:xright]
        # Display the resulting frame
        cv2.imshow('frame', cropped)

        # Process the frame using CNN-LSTM
        frame = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255

        sliding_window.append(rgb)
        if len(sliding_window) > 12:                
            sliding_window.pop(0)

        if len(sliding_window) == 12:
            np_sliding_window = np.expand_dims(np.array(sliding_window), axis=0) 
            conf, label = prediction_model.process(np_sliding_window, conf=0.7)
            start_frame = itr - 11
            end_frame = itr
                
            sliding_window = []                         
            
            print("Label of frame", start_frame, "to", end_frame, "Label", label, "Confidence Level:", conf)

            eval.count_casme2(start_frame, end_frame, label, detection)


        if cv2.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
    eval.to_csv(raw_filename)
    eval.print_total()
    eval.reset_count()
    print("==============================")

print("Evaluation Done")


