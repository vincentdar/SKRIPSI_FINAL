import os
import pandas as pd
import numpy as np
import shutil

raw_images_path = "D:\Dataset Skripsi Batch Final Image"
target_path = r"D:\Dataset Skripsi Batch Final Label\Negative"

df = pd.read_csv("label_4sub.csv")

subjects = df['Subject'].unique()

for subject in subjects:
    raw_subject_path = os.path.join(raw_images_path, subject)    
    find_in_df = df[df['Subject'] == subject]

    detections = []
    for index, item in find_in_df.iterrows():
        start = int(item['Start'])        
        end = int(item['End'])

        detections.append((start, end))
            
    sorted_file = sorted([int(f[3:-4]) for f in os.listdir(raw_subject_path)])
    length_file = len(sorted_file)    
    
    mask_array = np.zeros((length_file))
    for detection in detections:
        mask_array[detection[0]:detection[1]+1] = 1

    # Create Subject
    try:
        os.mkdir(os.path.join(target_path, subject))
    except Exception as e:
        pass
    
    # Create Directory Counter
    directory_counter = 1
    try:
        os.mkdir(os.path.join(target_path, subject, str(directory_counter)))
    except Exception as e:
        pass

    former_flag = False
    current_flag = False
    counter = 0
    for file in sorted_file:
        if mask_array[counter] == 0:
            current_flag = True
            filename = "img" + str(file).zfill(5) + ".jpg"
            if os.path.isfile(os.path.join(raw_subject_path, filename)):
                shutil.copy2(os.path.join(raw_subject_path, filename),
                            os.path.join(target_path, subject, str(directory_counter)))
            former_flag = True
        else:
            current_flag = False
            if current_flag != former_flag:
                directory_counter += 1
                try:
                    os.mkdir(os.path.join(target_path, subject, str(directory_counter)))
                except Exception as e:
                    pass
            former_flag = False
        counter += 1
    