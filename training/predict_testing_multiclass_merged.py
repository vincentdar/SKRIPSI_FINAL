import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import cv2
from keras import backend as K


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

  # rnn.add(tf.keras.layers.Dense(10, activation="softmax")
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

if __name__ == "__main__":
   checkpoint_path = "checkpoint/local_mobilenet_cnnlstm_unfreezelast20_newpubspeak15032023_multiclass_merged_10_epoch/cp.ckpt" 
   
