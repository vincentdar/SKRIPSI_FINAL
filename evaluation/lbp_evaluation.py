import cv2
import lbplib
import numpy as np
import os
import statistics
import pandas as pd


def get_lbp_histogram(img):
    resized = cv2.resize(img, (216, 216), cv2.INTER_AREA)
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)  

    stride = resized.shape[0] // 6   

    concatenated = [] 
    for i in range(0, 6):        
        for j in range(0, 6):
            # Sliding window from top left to bottom right            
            block = resized[stride * i:stride * (i + 1), stride * j:stride * (j + 1)]            
            lbp = lbplib.spatialLBP(block)  

            # Histogram Equalization / Normalization
            equ = cv2.equalizeHist(lbp)                        
            histogram, bin_edges = np.histogram(equ, bins=256, range=(0, 255))            

            concatenated.append(histogram) 

    return concatenated    


def compare_histogram(head, main, tail):    
    x2_dist_list = []    
    for index in range(0, len(head)):
        aff = (head[index] + tail[index]) // 2
        x2_dist = chiSquareDistance(aff, main[index])         
        x2_dist_list.append(x2_dist)  

    # Calculate Difference Vector
    m = 12
    # Sort descending
    x2_dist_list.sort(reverse=True)

    difference_vector = sum(x2_dist_list[:12]) / m
    return difference_vector

def chiSquareDistance(x, y):     
    distance = 0
    for i in range(len(x)):
        top = (x[i] - y[i]) ** 2
        bottom = (x[i] + y[i])  
        if bottom == 0:
            value = 0   
        else:
            value = top / bottom

        distance += value
    return distance


file = open('lbp_pubspeak21032022_ver2.csv', 'w')
data = "Subject,CDF,MaxCDF,MeanCDF,Total\n"
file.write(data)
if __name__ == "__main__":
    # label = pd.read_csv("D:\CASME\CASMEII\CASMEII_preprocess.csv")
    # label= pd.read_csv()
    # for index, row in label.iterrows():
        # subject = row['Subject']
        # folder = row['Folder']
        # onset = row['Onset']
        # offset = row['Offset']
    for i in range(0, 2):
        if i == 0:
            subject = "S4"
        else:
            subject = "S50"        


        # stride = (65 - 1) // 2 # 200 FPS
        stride = (9 - 1) // 2 # 25 FPS

        # data_path = "D:\CASME\CASMEII\CASME2-RAW-Localize\CASME2-RAW"
        # path = os.path.join(data_path, subject, folder)
        data_path = "D:\Dataset Skripsi Batch Final Image"
        path = os.path.join(data_path, subject)
        print("Path:", path)
        
        file_number = sorted([int(f[3:-4]) for f in os.listdir(path)])
        total_images = len(file_number)

        print("Total Frames", total_images)
        # data = subject + "," + folder + "," + str(offset-onset) + "," + str(total_images-(offset-onset))+"," + str(total_images) + "\n"        
        # file.write(data)   

        filenames = []          
        images = []
        iteration = 0

        # Get all file
        for number in file_number:            
            img_filename = "img" + str(number).zfill(5) + ".jpg"
            img_path = os.path.join(path, img_filename)
            if os.path.isfile(img_path):
                if number == iteration:                    
                    img = cv2.imread(img_path)                
                    images.append(img)
                    filenames.append(number)
                    iteration += stride
                
        # Calculate the Feature Difference
        difference_vector_list = []
        for index in range(1, len(images) - 1):    
            fv_head = get_lbp_histogram(images[index - 1])
            fv_main = get_lbp_histogram(images[index])
            fv_tail = get_lbp_histogram(images[index + 1])

            difference_vector = compare_histogram(fv_head, fv_main, fv_tail) 
            difference_vector_list.append(difference_vector)
            print("Index", filenames[index], difference_vector)   

        # Contrasted Difference Vector
        contrasted_vector_list = []
        constrasted_filenames_list = []
        for index in range(1, len(difference_vector_list) - 1):
            contrasted_vector = max(0, difference_vector_list[index] - 0.5 * (difference_vector_list[index - 1] 
                                                                    + difference_vector_list[index + 1]))
                        
            constrasted_filenames_list.append(filenames[index + 1])
            contrasted_vector_list.append((contrasted_vector))
            print("Contrasted Vector", filenames[index + 1], contrasted_vector)

        try:
            max_contrasted_vector = max(contrasted_vector_list)
            mean_contrasted_vector = statistics.mean(contrasted_vector_list)
            print("Max", max_contrasted_vector)
            print("Mean", mean_contrasted_vector)
        except Exception:
            print("Mean and Max Calculation encountered empty sequence")
            continue

        tau = 0.05
        threshold = mean_contrasted_vector + tau * (max_contrasted_vector - mean_contrasted_vector)   

        for index in range(0, len(constrasted_filenames_list)):            
            data = subject + "," + str(constrasted_filenames_list[index]) + "," + str(contrasted_vector_list[index]) + "," + str(max_contrasted_vector) + "," + str(mean_contrasted_vector) + "," + str(total_images) + "\n"
            print(data)
            file.write(data)        

file.close()          