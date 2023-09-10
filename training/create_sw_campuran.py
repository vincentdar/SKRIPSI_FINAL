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

def create_training_csv_binary_sw_campuran(data_path, label_path, sheet_name, writer):
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
        list_label = []
        min_number_0 = 6
        min_number_1 = 6
        counter = 0
        for number in numbers: 
            filename = "img" + str(number).zfill(5) + ".jpg"
            path_filename = os.path.join(subject_data_path, filename)
        
            # Label
            current_label = image_label(number, subject_df)  
            list_label.append(current_label)
            list_filename.append(path_filename) 
            # print(current_label)

            if len(list_label) == 12:                
                label_0 = 0
                label_1 = 0
                for label in list_label:                    
                    if label == 0:
                        label_0 += 1
                    elif label == 1:
                        label_1 += 1
                
                if label_0 == min_number_0 and label_1 == min_number_1:
                    counter += 1
                    for filename in list_filename:
                        writer.write(filename)
                        writer.write(',')
                    writer.write(str(min_number_0))
                    writer.write(',')
                    writer.write(str(min_number_1))
                    writer.write('\n')

                list_label.pop(0)
                list_filename.pop(0) 
        print("Subject count:", counter)
        
 
                                                                   

if __name__ == "__main__":    

    filewriter = open('sw_campuran_6_6.csv',  'w')
    filewriter.write("1,2,3,4,5,6,7,8,9,10,11,12,Negative,Positive\n")
    create_training_csv_binary_sw_campuran( "D:\Dataset Skripsi Batch Final Image Face Detection",
                                            "D:\CodeProject2\SKRIPSI_FINAL\pubspeak_label_BARU.xlsx",
                                            "Testing",
                                            filewriter)
    filewriter.close()
    

