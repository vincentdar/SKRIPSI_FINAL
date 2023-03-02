import pandas as pd
import numpy as np
import os
import shutil

label = "D:\CASME\CASMEII\CASMEII_preprocess.csv"

data_path =  "D:\CASME\CASMEII\CASME2-RAW-Localize\CASME2-RAW"

df = pd.read_csv(label)

sw_le = 0
sw_12 = 0
sw_pos = 0
sw_neg = 0

with open('CASMEII_dataset.csv', 'w') as writer: 
    for index, row in df.iterrows():
        subject = row['Subject']
        folder = row['Folder']

        onset = row['Onset']
        offset = row['Offset']
        raw_dir = os.path.join(data_path, subject, folder)
        
        files = sorted([int(f[3:-4]) for f in os.listdir(raw_dir)])
        video = []
        old_filename_path = ""  
        old_saved_path = ""  

        flag = False    
        last_number = 0
        last_label = 0

        positive_length = offset - onset
        get_left = max(0, onset - positive_length)
        get_right = offset + positive_length

        for number in files:
            filename = "img" + str(number) + ".jpg"
            filename_path = os.path.join(raw_dir, filename)

            saved_path = os.path.join(subject, folder, filename)
            if os.path.isfile(filename_path):
                if number >= get_left and number <= get_right:                    
                    flag =  ((number >= onset and number <= offset) and (last_number < onset or last_number > offset) or 
                            (number < onset or number > offset) and (last_number >= onset and last_number <= offset))  

                    label = int(number >= onset and number <= offset)                               
                    if flag:                
                        if len(video) > 0:
                            amount_to_copy = 12 - len(video)                                                        
                            for i in range(0, amount_to_copy):
                                video.append(old_saved_path)  
                            print("Less Than 12 Sliding Window with label", last_label) 
                            print(video)                       
                            for item in video:
                                writer.write(subject +"/" + folder + "/" + str(item.split('\\')[-1]))
                                writer.write(',')
                            writer.write(str(last_label) + "\n")
                            sw_le += 1
                            # Stat
                            if last_label == 1:
                                sw_pos += 1
                            else:
                                sw_neg += 1

                        if last_label == 0:
                            last_label = 1
                        else:
                            last_label = 0
                        
                        video = []
                        video.append(saved_path) 
                    else:
                        if len(video) == 12:
                            print("12 Sliding Window with label", label)     
                            print(video)                   
                            for item in video:
                                writer.write(subject +"/" + folder + "/" + str(item.split('\\')[-1]))
                                writer.write(',')
                            sw_12 += 1
                            # Stat
                            if label == 1:
                                sw_pos += 1
                            else:
                                sw_neg += 1
                            writer.write(str(label) + "\n")
                            video = []
                        video.append(saved_path)  

                last_number = number 
                old_saved_path = saved_path          
    

            
print("sw_le", sw_le)
print("sw_12", sw_12)
print("Positive Label", sw_pos)
print("Negative Label", sw_neg)


                