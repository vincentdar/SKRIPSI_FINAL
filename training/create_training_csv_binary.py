import os
import pandas as pd
import cv2
import numpy as np


def image_label(number, label):      
    y = 0 # Unknown
    for index, row in label.iterrows():
        start = row['Start']
        end = row['End']
        if number >= start and number <= end:
            y = row["Label"]
            return y
    return y

def create_training_csv_multiclass(data_path, label_path, sheet_name, writer):
    # Policy: Discard frames that don't meet the 12 sliding window requirement    
    df = pd.read_excel(label_path, sheet_name=sheet_name)
    subjects = list(df['Subject'].unique())
    print("Subjects:", subjects)
    
    for subject in subjects:
        print("Processing Subject", subject)

        subject_data_path = os.path.join(data_path, subject)        
        subject_df = df[df['Subject'] == subject]

        numbers = sorted([int(f[3:-4]) for f in os.listdir(subject_data_path)])        
        list_filename = []
        current_label = -1
        past_label = -1
        for number in numbers: 
            filename = "img" + str(number).zfill(5) + ".jpg"
            path_filename = os.path.join(subject_data_path, filename)
        
            # Label
            current_label = image_label(number, subject_df)  

            # Discard if the image is in excepted image
            # RISK: if we allow the model to train on blank image, it could disrupt training process
            img = cv2.imread(path_filename)
            all_zeros = not np.any(img)
            if all_zeros:
                current_label = 0

            # Write to CSV
            if current_label == past_label:
                list_filename.append(path_filename)
                if len(list_filename) == 12:                    
                    for filename in list_filename:
                        writer.write(filename)
                        writer.write(',')
                    writer.write(str(past_label))
                    writer.write('\n')
                    list_filename = []
            else:
                # POLICY DISCARD
                list_filename = []
                list_filename.append(path_filename)
            past_label = current_label   
                                                                                        

if __name__ == "__main__":    
    # Training
    filewriter = open('training_pubspeak_25042023_face_detection.csv',  'w')
    filewriter.write("1,2,3,4,5,6,7,8,9,10,11,12,Label\n")
    create_training_csv_multiclass( "D:\Dataset Skripsi Batch Final Image Face Detection",
                                    "D:\CodeProject2\SKRIPSI_FINAL\pubspeak_label_25042023.xlsx",
                                    "Training",
                                    filewriter)
    filewriter.close()

    # Testing
    filewriter = open('testing_pubspeak_25042023_face_detection.csv',  'w')
    filewriter.write("1,2,3,4,5,6,7,8,9,10,11,12,Label\n")
    create_training_csv_multiclass( "D:\Dataset Skripsi Batch Final Image Face Detection",
                                    "D:\CodeProject2\SKRIPSI_FINAL\pubspeak_label_25042023.xlsx",
                                    "Testing",
                                    filewriter)
    filewriter.close()
    

