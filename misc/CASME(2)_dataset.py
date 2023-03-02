import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from typing import List, Mapping, Optional, Tuple, Union

def generate_spotting(target_path, positive_path, negative_path, subjects):
    negative_dir = negative_path
    positive_dir = positive_path
        
    WINDOW_SIZE = 12        
   
    total_records = 0
    count_le_sw = 0
    count_12_sw = 0
    count_24_sw = 0
    count_48_sw = 0
    count_96_sw = 0
    with open(target_path, 'w') as writer:    
        for subject in subjects:        
            print("PROCESSING SUBJECT", subject)
            pos_subject_dir = os.path.join(positive_dir, subject)
            neg_subject_dir = os.path.join(negative_dir, subject)

            # Positive
            print("Positive Dataset")
            videos= [f.name for f in os.scandir(pos_subject_dir) if f.is_dir()]
            for video in videos:
                pos_video_dir = os.path.join(pos_subject_dir, video)  
                print(pos_video_dir)             

                # Positive Image
                images = sorted([int(f[3:-4])for f in os.listdir(pos_video_dir) if f != "Thumbs.db"])        
                videos = []
                for file in images:
                    loaded_file = "img" + str(file) + ".jpg"
                    if os.path.isfile(os.path.join(pos_video_dir, loaded_file)):                                      
                        modified_data = subject + "/" + video + "/" + loaded_file
                        videos.append(modified_data)

                        if len(videos) == WINDOW_SIZE:
                            count_12_sw += 1

                            for x in videos:                                                 
                                writer.write(x)
                                writer.write(",")

                            writer.write("1\n")
                            total_records += 1
                            videos = [] 
                if len(videos) > 0 and len(videos) < WINDOW_SIZE:
                    count_le_sw += 1

                    itr = 0
                    for x in videos:   
                        last_file = x                                              
                        writer.write(x)
                        writer.write(",")
                        itr += 1
                    
                    for i in range(0, WINDOW_SIZE - itr):
                        writer.write(last_file)
                        writer.write(",")

                    writer.write("1\n")
                    total_records += 1
                    videos = []   
                                         

            print("Negative Dataset")
            videos= [f.name for f in os.scandir(neg_subject_dir) if f.is_dir()] 
            for video in videos:                    
                neg_video_dir = os.path.join(neg_subject_dir, video)
                print(neg_video_dir)

                images = sorted([int(f[3:-4])for f in os.listdir(neg_video_dir) if f != "Thumbs.db"])        
                videos = []
                for file in images:
                    loaded_file = "img" + str(file) + ".jpg"
                    if os.path.isfile(os.path.join(neg_video_dir, loaded_file)):                                      
                        modified_data = subject + "/" + video + "/" + loaded_file
                        videos.append(modified_data)

                        if len(videos) == WINDOW_SIZE:
                            count_12_sw += 1

                            for x in videos:                                                 
                                writer.write(x)
                                writer.write(",")

                            writer.write("0\n")
                            total_records += 1
                            videos = []   
                if len(videos) > 0 and len(videos) < WINDOW_SIZE:
                    count_le_sw += 1

                    itr = 0
                    for x in videos:  
                        last_file = x                                               
                        writer.write(x)
                        writer.write(",")
                        itr += 1

                    for i in range(0, WINDOW_SIZE - itr):
                        writer.write(last_file)
                        writer.write(",")

                    writer.write("0\n")
                    total_records += 1
                    videos = []    
                                    
        writer.close()
    print("TFRECORD DATASET GENERATED: TOTAL RECORDS", total_records)
    print("TOTAL OF LESS THAN 12 SLIDING WINDOW", count_le_sw)
    print("TOTAL OF 12 SLIDING WINDOW", count_12_sw)

def holdout_spotting_TFRecord():
    subjects = sorted([f.name for f in os.scandir(r"D:\CASME\CASME(2)\Spotting_Version2_Localize\Positive") if f.is_dir()]) 

    negative_dir = r"D:\CASME\CASME(2)\Spotting_Version2_Localize\Negative"
    positive_dir = r"D:\CASME\CASME(2)\Spotting_Version2_Localize\Positive"       
    generate_spotting('CAS(ME)2_dataset_spotting_version2_localize.csv', positive_dir, negative_dir, subjects)    



holdout_spotting_TFRecord()
    