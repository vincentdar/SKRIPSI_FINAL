import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import cv2
import math
from keras import backend as K

def parse_image(filename):  
  image = tf.io.read_file(filename)
  image = tf.io.decode_image(image)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, [224, 224])  
  return image

def create_dataset(features, label):
  length = label.shape[0]    
  for i in range(0, length):
    feature = features[i]
    images = []
    for j in range(0, feature.shape[0]):
      filename = feature[j]
      image = parse_image(filename)
      images.append(image)      

    video = tf.convert_to_tensor(images, dtype=tf.float32)    
    yield video, tf.one_hot(label[i], depth=6)      

    # idk why images variable automatically empties itself


def plot_video(ds, num_of_takes):
  for videos, labels in ds.take(num_of_takes):        
    plt.figure(figsize=(24, 18))
    for i in range(12):
      plt.subplot(3, 4, i + 1)
      plt.imshow(videos[i])
      plt.title("Image " + str(i) + " Label " + str(labels))
    plt.show() 

def read_dataframe(filename):
  df = pd.read_csv(filename)  
  df = df.sample(frac=1, random_state=42)  
  features = df.drop('Label', axis=1)
  labels = df['Label']
  return features, labels

if __name__ == "__main__":   
  test_features, test_labels = read_dataframe("testing_pubspeak_multiclass_21032023_face_detection.csv")
  test_ds = tf.data.Dataset.from_generator(create_dataset,
                                        args=(tf.convert_to_tensor(test_features), tf.convert_to_tensor(test_labels)),
                                        output_types=(tf.float32, tf.int64),
                                        output_shapes=((12, 224, 224, 3), (6)))  
  
  plot_video(test_ds, 8)