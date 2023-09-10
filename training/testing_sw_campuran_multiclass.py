import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import cv2
import math

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score
import seaborn as sns

def parse_image(filename):  
  image = tf.io.read_file(filename)
  image = tf.io.decode_image(image)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, [224, 224])  

  return image
         

def create_dataset(features):    
  length = features.shape[0]      
  for i in range(0, length):
    feature = features[i]
    images = []    
    for j in range(0, feature.shape[0]):
      filename = feature[j]      
      image = parse_image(filename)
      images.append(image) 
    video = tf.convert_to_tensor(images, dtype=tf.float32)       
    yield video

def plot_video(ds, num_of_takes):
  for videos, labels in ds.take(num_of_takes):        
    plt.figure(figsize=(24, 18))
    for i in range(12):
      plt.subplot(3, 4, i + 1)
      plt.imshow(videos[i])
      plt.title("Image " + str(i) + " Label " + str(labels))
    plt.show() 

def create_mobilenet():  
  mobilenet = tf.keras.applications.MobileNet(input_shape=(224, 224, 3), include_top=False)
  # mobilenet.trainable = False

  # If you wanted to modify the mobilenet layer
  # Freeze all layer except for the last 20
  mobilenet.trainable = True
  for layer in mobilenet.layers[:-20]:
      layer.trainable = False

  # CNN Model
  cnn = tf.keras.models.Sequential()  
  cnn.add(mobilenet)  
  cnn.add(tf.keras.layers.GlobalAveragePooling2D())  

  # RNN Model
  rnn = tf.keras.models.Sequential()
  rnn.add(tf.keras.layers.TimeDistributed(cnn))
  rnn.add(tf.keras.layers.LSTM(32))

  rnn.add(tf.keras.layers.Dense(6, activation="softmax"))

  rnn.build(input_shape=(None, 12, 224, 224, 3)) 
  print(rnn.summary())
  
  return rnn

def compile_model(model):
  model.compile(loss='categorical_crossentropy',
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                metrics=['accuracy'])
  
  return model

def read_dataframe(filename):
  df = pd.read_excel(filename)    
  features = df.drop(['label_1','label_2','label_3','label_4','label_5','label_6','label_7','label_8','label_9','label_10','label_11', 'label_12'], axis=1)    
  return df, features


def multiclass_report(gt, pred, video):
    print("ACCURACY REPORT")
    print("Accuracy:", accuracy_score(np.array(gt), np.array(pred)))
    print("CLASSIFICATION REPORT")
    print(classification_report(np.array(gt), np.array(pred), 
                                labels=[0, 1, 2, 3, 4, 5],
                                zero_division=0))
    confusionHeatmapCategorical(gt, pred, video)

def multiclass_report_10(gt, pred, video):
    print("ACCURACY REPORT")
    print("Accuracy:", accuracy_score(np.array(gt), np.array(pred)))
    print("CLASSIFICATION REPORT")
    print(classification_report(np.array(gt), np.array(pred), 
                                labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                zero_division=0))
    confusionHeatmapCategorical(gt, pred, video)
    

def binary_report(gt, pred, video):        
    print("ACCURACY REPORT")
    print("Accuracy:", accuracy_score(np.array(gt), np.array(pred)))
    print("CLASSIFICATION REPORT")
    print(classification_report(np.array(gt), np.array(pred), 
                                labels=[0, 1],
                                zero_division=0))
    confusionHeatmapBinary(gt, pred, video)
   

def confusionHeatmapBinary(gt, pred, video):    
    conf_matrix = confusion_matrix(np.array(gt), np.array(pred), labels=[0, 1])
    plt.figure(figsize=(10,8), dpi=75)
    # Scale up the size of all text
    sns.set(font_scale = 1)

    ax = sns.heatmap(conf_matrix, annot=True, fmt='d', )
    ax.set_xlabel("Predicted", fontsize=14, labelpad=20)
    ax.xaxis.set_ticklabels(['0', '1'])

    ax.set_ylabel("Actual", fontsize=14, labelpad=20)
    ax.yaxis.set_ticklabels(['0', '1'])

    ax.set_title(video + " Binary Confusion Matrix", fontsize=14, pad=20)

    plt.show()

def confusionHeatmapCategorical(gt, pred, video):    
    conf_matrix = confusion_matrix(np.array(gt), np.array(pred), labels=[0, 1, 2, 3, 4, 5])
    plt.figure(figsize=(10,8), dpi=75)
    # Scale up the size of all text
    sns.set(font_scale = 1)

    ax = sns.heatmap(conf_matrix, annot=True, fmt='d', )
    ax.set_xlabel("Predicted", fontsize=14, labelpad=20)
    ax.xaxis.set_ticklabels(["Unknown", "Showing Emotions", "Blank Face", "Reading", "Head Tilt", "Occlusion"])

    ax.set_ylabel("Actual", fontsize=14, labelpad=20)
    ax.yaxis.set_ticklabels(["Unknown", "Showing Emotions", "Blank Face", "Reading", "Head Tilt", "Occlusion"])

    ax.set_title(video + " Categorical Confusion Matrix", fontsize=14, pad=20)

    plt.show()

def confusionHeatmapCategorical_10(gt, pred, video):    
    conf_matrix = confusion_matrix(np.array(gt), np.array(pred), labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.figure(figsize=(10,8), dpi=75)
    # Scale up the size of all text
    sns.set(font_scale = 1)

    ax = sns.heatmap(conf_matrix, annot=True, fmt='d', )
    ax.set_xlabel("Predicted", fontsize=12, labelpad=10)
    ax.xaxis.set_ticklabels(["Unknown", "Eye\nContact", "Blank\nFace", "Showing\nEmotions", "Reading",
                   "Sudden\nEye\nChange", "Smiling", "Not\nLooking", "Head\nTilt", "Occlusion"])

    ax.set_ylabel("Actual", fontsize=12, labelpad=10)
    ax.yaxis.set_ticklabels(["Unknown", "Eye\nContact", "Blank\nFace", "Showing\nEmotions", "Reading",
                   "Sudden\nEye\nChange", "Smiling", "Not\nLooking", "Head\nTilt", "Occlusion"])

    ax.set_title(video + " Categorical Confusion Matrix", fontsize=14, pad=20)

    plt.show()

def dynamic_threshold(input, threshold):
  return math.floor(input) if input <= threshold else math.ceil(input)

if __name__ == "__main__":   
  # vf = np.vectorize(dynamic_threshold)

  checkpoint_path = "checkpoint/local_mobilenet_cnnlstm_newpubspeak25042023_multiclass_focal_loss_10_epoch/cp.ckpt"
  model = create_mobilenet()  
  model.load_weights(checkpoint_path)
  model = compile_model(model)


  df, predict_features = read_dataframe("sw_campuran_multiclass/sw_campuran_multiclass_detail.xlsx")        
  preds = []
  confs = []
  counter = 0
  for video in create_dataset(tf.convert_to_tensor(predict_features)):      
    conf = model.predict(np.expand_dims(video, axis=0), verbose=0)
    confs.append(conf)
    pred = tf.argmax(conf, axis=1)
    preds.append(int(pred[0]))
    counter += 1

  print("NP_CONFS")  
  np_confs = np.array(confs)
  np_confs_0 = np_confs[:,:,0]
  np_confs_1 = np_confs[:,:,1]
  np_confs_2 = np_confs[:,:,2]
  np_confs_3 = np_confs[:,:,3]
  np_confs_4 = np_confs[:,:,4]
  np_confs_5 = np_confs[:,:,5]

  
  np_preds = np.array(preds)
  df['conf_0'] = np_confs_0
  df['conf_1'] = np_confs_1
  df['conf_2'] = np_confs_2
  df['conf_3'] = np_confs_3
  df['conf_4'] = np_confs_4
  df['conf_5'] = np_confs_5
  df['Label'] = np_preds

  
    
  df.to_excel("sw_campuran_multiclass_hasil/sw_campuran_multiclass_detail_hasil.xlsx")


  
  

  










