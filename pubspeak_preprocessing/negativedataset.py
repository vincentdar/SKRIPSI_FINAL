# Script for deleting positive images in rawpic dataset
import os

count = 15

positive_path = "D:\Dataset Skripsi Batch 1 Self Label\Positive\\10 September\S4"
raw_negative_path = "D:\Dataset Skripsi Batch 1 Images Negative\\10 September\Clea Alvina"

for item in os.listdir(positive_path):
    print("Folder", item)
    detected = os.path.join(positive_path, item)
    for images in os.listdir(detected):        
        if os.path.isfile(os.path.join(detected, images)):            
            image_to_delete = os.path.join(raw_negative_path, images)
            print(image_to_delete)
            if os.path.isfile(image_to_delete):                
                print("DELETING")              
                os.remove(image_to_delete)    
