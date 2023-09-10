import os
import pandas as pd
import cv2
import numpy as np


def image_label(number, label, class_names):
    merged_class_names = ["Unknown", "Showing Emotions", "Blank Face", "Reading",
                   "Head Tilt", "Occlusion"]        
    y = 0 # Unknown
    for index, row in label.iterrows():
        start = row['Start']
        end = row['End']
        if number >= start and number <= end:
            justification = row['Justification']
            if justification == "Eye Contact":
                justification = "Showing Emotions"
            elif justification == "Sudden Eye Change":
                justification = "Showing Emotions"
            elif justification == "Smiling":
                justification = "Showing Emotions"
            elif justification == "Not Looking":
                justification = "Showing Emotions"
            y = merged_class_names.index(justification)
            return y
    return y

def create_training_csv_multiclass_sw_campuran(data_path, label_path, sheet_name, writer, class_names):
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
        counter = 0
        for number in numbers: 
            filename = "img" + str(number).zfill(5) + ".jpg"
            path_filename = os.path.join(subject_data_path, filename)
        
            # Label
            current_label = image_label(number, subject_df, class_names)  
            list_label.append(current_label)
            list_filename.append(path_filename) 
            # print(current_label)

            if len(list_label) == 12:                
                labels = [0,0,0,0,0,0]
                for label in list_label:                    
                    if label == 0:
                        labels[0] += 1
                    elif label == 1:
                        labels[1] += 1
                    elif label == 2:
                        labels[2] += 1
                    elif label == 3:
                        labels[3] += 1
                    elif label == 4:
                        labels[4] += 1
                    elif label == 5:
                        labels[5] += 1  

                write = True        
                for index, label in enumerate(labels):                    
                    if label >= 12:
                        write = False
                    
                if write == True:
                    counter += 1
                    for filename in list_filename:
                        writer.write(filename)
                        writer.write(',')
                    for label in list_label:
                        writer.write(str(label))
                        writer.write(',')
                    writer.write('\n')

                list_label.pop(0)
                list_filename.pop(0) 
        print("Subject count:", counter)

  

            


if __name__ == "__main__":
    class_names = ["Unknown", "Eye Contact", "Blank Face", "Showing Emotions", "Reading",
                   "Sudden Eye Change", "Smiling", "Not Looking", "Head Tilt", "Occlusion"]
            
    

    filewriter = open('sw_campuran_multiclass_detail.csv',  'w')
    filewriter.write("1,2,3,4,5,6,7,8,9,10,11,12,label_1,label_2,label_3,label_4,label_5,label_6,label_7,label_8,label_9,label_10,label_11,label_12,\n")
    create_training_csv_multiclass_sw_campuran( "D:\Dataset Skripsi Batch Final Image Face Detection",
                                                "D:\CodeProject2\SKRIPSI_FINAL\pubspeak_label_BARU.xlsx",
                                                "Testing",
                                                filewriter,
                                                class_names)
    filewriter.close()

    
    

