import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import cv2
import math
from keras import backend as K

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score
import seaborn as sns


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

  rnn.add(tf.keras.layers.Dense(6, activation="softmax"))

  rnn.build(input_shape=(None, 12, 224, 224, 3)) 
  print(rnn.summary())
  
  return rnn

def categorical_focal_loss(alpha, gamma=2.):
    """
    Github Link: https://github.com/umbertogriffo/focal-loss-keras
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = tf.cast(-y_true, tf.float32) * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss_fixed


def compile_model(model):
  model.compile(loss='categorical_crossentropy',
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                metrics=['accuracy'])
  
  return model

def compile_model_focal_loss(model):
  model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25, .25, .25, .25]], gamma=2)],
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                metrics=['accuracy'])
  
  return model

def read_dataframe(filename):
  df = pd.read_csv(filename)  
  features = df.drop('Label', axis=1)
  labels = df['Label']
  return features, labels

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


if __name__ == "__main__": 
  # class_names = ["Unknown", "Eye Contact", "Blank Face", "Showing Emotions", "Reading",
  #                  "Sudden Eye Change", "Smiling", "Not Looking", "Head Tilt", "Occlusion"]                  
  class_names = ["Unknown", "Showing Emotions", "Blank Face", "Reading",
                   "Head Tilt", "Occlusion"]                
  
  test_features, test_labels = read_dataframe("categorical/testing_S168_multiclass.csv")
  # test_features, test_labels = read_dataframe("categorical/coba.csv")
  
  test_ds = tf.data.Dataset.from_generator(create_dataset,
                                        args=(tf.convert_to_tensor(test_features), tf.convert_to_tensor(test_labels)),
                                        output_types=(tf.float32, tf.int64),
                                        output_shapes=((12, 224, 224, 3), (6)))  
  


  checkpoint_path = "checkpoint/local_mobilenet_cnnlstm_newpubspeak25042023_multiclass_focal_loss_10_epoch_mix/cp.ckpt"

  test_ds = test_ds.prefetch(tf.data.AUTOTUNE).batch(4)

  model = create_mobilenet()  
  model.load_weights(checkpoint_path)
  model = compile_model(model)
  pred = model.predict(test_ds)
  pred = tf.argmax(pred, axis=1)
  pred = np.array(pred)
  gt = np.array(test_labels)
  multiclass_report(gt, pred, "testdata.csv")  
  



       









         

