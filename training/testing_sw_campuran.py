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
  # cnn.add(tf.keras.layers.Flatten())

  cnn.add(tf.keras.layers.GlobalAveragePooling2D())
  # cnn.add(tf.keras.layers.Dense(32))

  # RNN Model
  rnn = tf.keras.models.Sequential()
  rnn.add(tf.keras.layers.TimeDistributed(cnn))
  rnn.add(tf.keras.layers.LSTM(32))

  rnn.add(tf.keras.layers.Dense(1, activation="sigmoid"))

  rnn.build(input_shape=(None, 12, 224, 224, 3)) 
  print(rnn.summary())
  
  return rnn

def compile_model(model):
  model.compile(loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                metrics=['accuracy'])
  
  return model

def read_dataframe(filename):
  df = pd.read_csv(filename)    
  features = df.drop(['Negative', 'Positive'], axis=1)    
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
  vf = np.vectorize(dynamic_threshold)

  checkpoint_path = "checkpoint/local_mobilenet_cnnlstm_unfreezelast20_newpubspeak25042023_10_epoch_mix/cp.ckpt"
  model = create_mobilenet()  
  model.load_weights(checkpoint_path)
  model = compile_model(model)

  list_csv = ["sw_campuran_1_11",
              "sw_campuran_2_10",
              "sw_campuran_3_9",
              "sw_campuran_4_8",
              "sw_campuran_5_7",
              "sw_campuran_6_6",
              "sw_campuran_7_5",
              "sw_campuran_8_4",
              "sw_campuran_9_3",
              "sw_campuran_10_2",
              "sw_campuran_11_1",]
  
  for csv in list_csv:
    df, predict_features = read_dataframe("sw_campuran_binary/" + csv + ".csv")  
    print("CSV Processed:", csv)
    preds = []
    for video in create_dataset(tf.convert_to_tensor(predict_features)):      
      pred = model.predict(np.expand_dims(video, axis=0), verbose=0)
      preds.append(pred[0][0])
    np_preds = np.array(preds)
    df['Conf'] = np_preds

    list_confidence = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for conf in list_confidence:      
      threshold = conf
      np_label = vf(np_preds, threshold)
      df[conf] = np_label
      
    df.to_excel("sw_campuran_binary_hasil/" + csv + ".xlsx")


  
  

  










