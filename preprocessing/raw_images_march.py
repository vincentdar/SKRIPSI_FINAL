# Raw Images preprocessing
# Convert videos to images
# Note: only raw videos not subject splitted
import cv2
import numpy as np
import mediapipe as mp
import math
import os
from localize import Localize
from typing import List, Mapping, Optional, Tuple, Union


file_ls = [
    # "D:\Dataset Skripsi Batch Final\\25 FPS\S50.mov",
    # "D:\Dataset Skripsi Batch Final\\25 FPS\S51.mov",
    # "D:\Dataset Skripsi Batch Final\\25 FPS\S52.mov",
    # "D:\Dataset Skripsi Batch Final\\25 FPS\S53.mov",
    # "D:\Dataset Skripsi Batch Final\\25 FPS\S54.mov",

    # "D:\Dataset Skripsi Batch Final\\25 FPS\S57.mov",
    # "D:\Dataset Skripsi Batch Final\\25 FPS\S10.mov",
    # "D:\Dataset Skripsi Batch Final\\25 FPS\S118.mov",
    # "D:\Dataset Skripsi Batch Final\\25 FPS\S98.mov",

    # "D:\Dataset Skripsi Batch Final\\25 FPS\S67.mov",
    # "D:\Dataset Skripsi Batch Final\\25 FPS\S85.mov",
    # "D:\Dataset Skripsi Batch Final\\25 FPS\S101.mov",
    # "D:\Dataset Skripsi Batch Final\\25 FPS\S130.mov",

    # "D:\Dataset Skripsi Batch Final\\25 FPS\S124.mov",
    # "D:\Dataset Skripsi Batch Final\\25 FPS\S125.mov",
    # "D:\Dataset Skripsi Batch Final\\25 FPS\S117.mov",
    # "D:\Dataset Skripsi Batch Final\\25 FPS\S160.mov",

#     "D:\Dataset Skripsi Batch Final\\25 FPS\S170.mov",
#     "D:\Dataset Skripsi Batch Final\\25 FPS\S168.mov",
    # "D:\Dataset Skripsi Batch Final\\25 FPS\S174.mov",
#     "D:\Dataset Skripsi Batch Final\\25 FPS\S22.mov",

    "D:\Dataset Skripsi Batch Final\\25 FPS\S1.mp4",
    "D:\Dataset Skripsi Batch Final\\25 FPS\S2.mp4",
    "D:\Dataset Skripsi Batch Final\\25 FPS\S3.mp4",
    "D:\Dataset Skripsi Batch Final\\25 FPS\S4.mp4",
]

target_ls = [
    "D:\Dataset Skripsi Batch Final Image",
    ]


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
localization_algorithm = Localize() 

def read_video(filename):        
    frame_count = 0
    cap = cv2.VideoCapture(filename)   

    # Manipulate capture the folder name and remove .mp4 text
    target_folder_name = filename.split('\\')[-1][:-4]         
    target_full_path = os.path.join(target_ls[0], target_folder_name)    
    if not os.path.exists(target_full_path):
        os.mkdir(os.path.join(target_ls[0], target_folder_name))
       
    # Set up file to tracked failed (not detected) frame
    target_failed_frame = target_folder_name + ".txt"
    failed_full_path = os.path.join(target_ls[0], target_failed_frame)     

    if os.path.exists(failed_full_path):
        os.remove(failed_full_path)
        failed_file_tracker = open(failed_full_path, 'w')
    else:
        failed_file_tracker = open(failed_full_path, 'w')
        
    blank_frame = np.zeros((224, 224, 3))

    # Check if stream opened successfully
    if (cap.isOpened() == False): 
        print("Error opening video stream or file")
        
    while(cap.isOpened()):   
        ret, frame = cap.read()
        faceROI = np.zeros((224, 224, 3))
        if ret == True:                                  
            try:  
                          
                # detected, xleft, ytop, xright, ybot = localization_algorithm.mp_face_mesh_crop_preprocessing(frame)                                
                # faceROI = frame[ytop:ybot, xleft:xright]
                # faceROI = cv2.resize(faceROI, (224, 224), interpolation=cv2.INTER_AREA)
                # print("Face detected : Frame", frame_count)
                faceROI = localization_algorithm.mp_face_mesh_crop(frame)        
                cv2.imshow('Localized Frame', faceROI)                 

                # write frame to folder 
                written_filename = "img" + str(frame_count).zfill(5) + ".jpg"
                final_written_filename = os.path.join(target_full_path, written_filename)            
                cv2.imwrite(final_written_filename, faceROI)     # save frame as JPEG file

            except Exception as e:      
                print("Exception Occured")          
                # print("Face NOT detected : Frame", frame_count)
                cv2.imshow('Localized Frame', blank_frame)                
                  
                # write frame to folder 
                written_filename = "img" + str(frame_count).zfill(5) + ".jpg"
                final_written_filename = os.path.join(target_full_path, written_filename)            
                cv2.imwrite(final_written_filename, blank_frame)     # save frame as JPEG file  
                failed_file_tracker.write(written_filename + "\n")            
            frame_count += 1
            
            # Press Q on keyboard to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break                
        else: 
            break
        
    cap.release()        
    cv2.destroyAllWindows()
    # Close the file
    # failed_file_tracker.close()
    print("File:", filename, "Frame Count:", frame_count)

if __name__ == "__main__":
    for filename in file_ls:
        read_video(filename)
    
    print("Preprocessing DONE")
    print("Result on Path:", target_ls[0])