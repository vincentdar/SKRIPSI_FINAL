import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import cv2

def parse_image(image):  
  """
  Parse image and resized it to (224, 224)
  Args: filename (str)
  Return: tf.tensor
  """
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, [224, 224])
  image = np.array(image)
  return image

def check_if_label(subject_df, image_itr):
  for index, row in subject_df.iterrows():
    start = row['Start']
    end = row['End']
    
    if image_itr >= start and image_itr <= end:
        return 1

  return 0
  
def generate_data(subject_list, df):    
    for subject in subject_list:    
        subject_df = df[df['Subject'] == subject]
        iterator = 0
        current_label = 0    

        try:
            cap = cv2.VideoCapture("D:\Dataset Skripsi Batch Final\\25 FPS\\" + subject + ".mp4")
        except Exception as e:        
            cap = cv2.VideoCapture("D:\Dataset Skripsi Batch Final\\25 FPS\\" + subject + ".mov")

        images = []
        while cap.isOpened():      
            success, image = cap.read()
            image = parse_image(image)
            if not success:            
                break
            
            # Policy, Discard frame that cuts through the label
            label = check_if_label(subject_df, iterator)
            if label != current_label:
                images = []
                images.append(image)
            else:
                images.append(image)    

            current_label = label
            

            if len(images) == 12:            
                yield tf.convert_to_tensor(images, dtype=tf.float32), tf.convert_to_tensor(label, dtype=tf.int64)

            iterator += 1
            

  

df = pd.read_csv("label_4sub.csv")
subject_list = list(df['Subject'].unique())

# TF.DATA.API
train_ds = tf.data.Dataset.from_generator(generate_data, args=[subject_list, df], 
                                            output_types=(tf.float32, tf.int64),
                                            output_shapes = ((12, 224, 224, 3), ()))

for videos, labels in train_ds.take(1):        
    plt.figure(figsize=(24, 16))
    for i in range(12):
        plt.subplot(3, 4, i + 1)
        plt.imshow(videos[i])
        plt.title("Image " + str(i) + " Label " + str(labels))
    plt.show()  
    break


