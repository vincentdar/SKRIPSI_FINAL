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

def create_training_csv_multiclass(data_path, label_path, sheet_name, writer, class_names):
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
            current_label = image_label(number, subject_df, class_names)  

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

def create_training_csv_multiclass_augmented(data_path, label_path, sheet_name, writer, class_names):
    # Policy: Discard frames that don't meet the 12 sliding window requirement    
    df = pd.read_excel(label_path, sheet_name=sheet_name)
    subjects = list(df['Subject'].unique())
    print("Subjects:", subjects)

    augmentation = ["None", "Flip"]
    
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
            current_label = image_label(number, subject_df, class_names)  

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
                    writer.write(',')                    
                    writer.write(str(0))
                    writer.write('\n')     

                    if past_label == 4 or past_label == 5:     
                        for filename in list_filename:
                            writer.write(filename)
                            writer.write(',')
                        writer.write(str(past_label))
                        writer.write(',')                    
                        writer.write(str(1))
                        writer.write('\n') 

                    if past_label == 4 or past_label == 5:
                        list_filename.pop(0)
                    else:                    
                        list_filename = []
            else:
                # POLICY DISCARD
                list_filename = []
                list_filename.append(path_filename)
            past_label = current_label   

def create_training_csv_multiclass_WSL(data_path, label_path, sheet_name, writer, class_names):
    WSL_PATH = "/mnt/d/Dataset Skripsi Batch Final Image Face Detection"
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
            wsl_path = WSL_PATH + "/" + subject + "/" + filename
        
            # Label
            current_label = image_label(number, subject_df, class_names)
              
            # Discard if the image is in excepted image
            # RISK: if we allow the model to train on blank image, it could disrupt training process
            img = cv2.imread(path_filename)
            all_zeros = not np.any(img)
            if all_zeros:
                current_label = 0

            # Write to CSV
            if current_label == past_label:
                list_filename.append(wsl_path)
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
                list_filename.append(wsl_path)
            past_label = current_label   
                                                                                        

if __name__ == "__main__":
    class_names = ["Unknown", "Eye Contact", "Blank Face", "Showing Emotions", "Reading",
                   "Sudden Eye Change", "Smiling", "Not Looking", "Head Tilt", "Occlusion"]
            
    # Training
    filewriter = open('training_pubspeak_multiclass_21032023_face_detection_augmented.csv',  'w')
    filewriter.write("1,2,3,4,5,6,7,8,9,10,11,12,Label,Augmentation\n")
    create_training_csv_multiclass_augmented( "D:\Dataset Skripsi Batch Final Image Face Detection",
                                              "D:\CodeProject2\SKRIPSI_FINAL\pubspeak_label_21032023.xlsx",
                                              "Training",
                                              filewriter,
                                              class_names)
    filewriter.close()

    # Testing
    # filewriter = open('testing_pubspeak_multiclass_21032023_face_detection.csv',  'w')
    # filewriter.write("1,2,3,4,5,6,7,8,9,10,11,12,Label\n")
    # create_training_csv_multiclass( "D:\Dataset Skripsi Batch Final Image Face Detection",
    #                                 "D:\CodeProject2\SKRIPSI_FINAL\pubspeak_label_21032023.xlsx",
    #                                 "Testing",
    #                                 filewriter,
    #                                 class_names)
    # filewriter.close()

    # WSL
    # Training
    # filewriter = open('training_pubspeak_multiclass_21032023_face_detection_WSL.csv',  'w')
    # filewriter.write("1,2,3,4,5,6,7,8,9,10,11,12,Label\n")
    # create_training_csv_multiclass_WSL( "D:\Dataset Skripsi Batch Final Image Face Detection",
    #                                 "D:\CodeProject2\SKRIPSI_FINAL\pubspeak_label_21032023.xlsx",
    #                                 "Training",
    #                                 filewriter,
    #                                 class_names)
    # filewriter.close()

    # # Testing
    # filewriter = open('testing_pubspeak_multiclass_21032023_face_detection_WSL.csv',  'w')
    # filewriter.write("1,2,3,4,5,6,7,8,9,10,11,12,Label\n")
    # create_training_csv_multiclass_WSL( "D:\Dataset Skripsi Batch Final Image Face Detection",
    #                                 "D:\CodeProject2\SKRIPSI_FINAL\pubspeak_label_21032023.xlsx",
    #                                 "Testing",
    #                                 filewriter,
    #                                 class_names)
    # filewriter.close()
    
    

