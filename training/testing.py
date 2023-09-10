import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import cv2

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score
import seaborn as sns

def parse_image(filename, augment_code):  
  image = tf.io.read_file(filename)
  image = tf.io.decode_image(image)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, [224, 224])  

  if augment_code == 0:
    pass
  elif augment_code == 1:
    # Flip 
    image = tf.image.flip_left_right(image)
  return image
         

def create_dataset(features, label):    
  length = label.shape[0]    
  for i in range(0, length):
    feature = features[i]
    images = []
    for j in range(0, feature.shape[0]):
      filename = feature[j]
      image = parse_image(filename, 0)
      images.append(image)      

    video = tf.convert_to_tensor(images, dtype=tf.float32)    
    yield video, label[i]    

    # idk why images variable automatically empties itself

# def create_dataset(features, label):
#   length = label.shape[0]    
#   for i in range(0, length):
#     feature = features[i]
#     images = []
#     for j in range(0, feature.shape[0]):
#       filename = feature[j]
#       image = parse_image(filename)
#       images.append(image)      

#     video = tf.convert_to_tensor(images, dtype=tf.float32)    
#     yield video, label[i]  

#     # idk why images variable automatically empties itself

def create_dataset_augmented(features, label, augment):    
  length = label.shape[0]    
  for i in range(0, length):
    feature = features[i]
    
    # Augmentation Code    
    augment_code = augment[i]

    images = []
    for j in range(0, feature.shape[0]):
      filename = feature[j]
      image = parse_image(filename, augment_code)
      images.append(image)      

    video = tf.convert_to_tensor(images, dtype=tf.float32)    
    yield video, label[i]      

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

  rnn.add(tf.keras.layers.Dense(1, activation="sigmoid"))

  rnn.build(input_shape=(None, 12, 224, 224, 3)) 
  print(rnn.summary())
  
  return rnn



# def create_mobilenet_bilstm():  
#   mobilenet = tf.keras.applications.MobileNet(input_shape=(224, 224, 3), include_top=False)
#   # mobilenet.trainable = False

#   # If you wanted to modify the mobilenet layer
#   # Freeze all layer except for the last 20
#   mobilenet.trainable = True
#   for layer in mobilenet.layers[:-20]:
#       layer.trainable = False

#   # CNN Model
#   cnn = tf.keras.models.Sequential()  
#   cnn.add(mobilenet)
#   # cnn.add(tf.keras.layers.Flatten())

#   cnn.add(tf.keras.layers.GlobalAveragePooling2D())
#   # cnn.add(tf.keras.layers.Dense(32))

#   # RNN Model
#   rnn = tf.keras.models.Sequential()
#   rnn.add(tf.keras.layers.TimeDistributed(cnn))
#   rnn.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)))
#   rnn.add(tf.keras.layers.Dense(1, activation="sigmoid"))

#   rnn.build(input_shape=(None, 12, 224, 224, 3)) 
#   print(rnn.summary())
  
#   return rnn


def create_C3D():
  # Inspiration is Multi Segment CNN: https://openaccess.thecvf.com/content_cvpr_2016/papers/Shou_Temporal_Action_Localization_CVPR_2016_paper.pdf
  inputs = tf.keras.Input(shape=(12, 224, 224, 3))
  x = tf.keras.layers.Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation="relu")(inputs)
  x = tf.keras.layers.MaxPool3D(pool_size=(1,2,2))(x)
  x = tf.keras.layers.Conv3D(128, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation="relu")(x)  
  x = tf.keras.layers.MaxPool3D(pool_size=(2,2,2))(x)
  x = tf.keras.layers.Conv3D(256, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation="relu")(x)    
  x = tf.keras.layers.MaxPool3D(pool_size=(2,2,2))(x)   
  # x = tf.keras.layers.Flatten()(x) 
  x = tf.keras.layers.GlobalAveragePooling3D()(x)  
  outputs = tf.keras.layers.Dense(1)(x)

  model = tf.keras.Model(inputs=inputs, outputs=outputs, name="multisegmentcnn")  
  return model



def compile_model(model):
  model.compile(loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                metrics=['accuracy'])
  
  return model

def read_dataframe(filename):
  df = pd.read_csv(filename)  
  try:
    features = df.drop(['Label', 'Augmentation'], axis=1)
    augments = df['Augmentation']
  except:
    features = df.drop('Label', axis=1)
    augments = None
  labels = df['Label']
  
  
  return features, labels, augments


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
  test_features, test_labels, _ = read_dataframe("binary/testing_S168.csv")
  
  test_ds = tf.data.Dataset.from_generator(create_dataset,
                                        args=(tf.convert_to_tensor(test_features), tf.convert_to_tensor(test_labels)),
                                        output_types=(tf.float32, tf.int64),
                                        output_shapes=((12, 224, 224, 3), ()))  
    
  
  checkpoint_path = "checkpoint/local_mobilenet_cnnlstm_unfreezelast20_newpubspeak25042023_10_epoch_mix/cp.ckpt"

  test_ds = test_ds.prefetch(tf.data.AUTOTUNE).batch(4)

  model = create_mobilenet()  
  model.load_weights(checkpoint_path)
  model = compile_model(model)
  pred = model.predict(test_ds)
  pred = tf.round(pred)
  pred = np.array(pred)
  gt = np.array(test_labels)
  binary_report(gt, pred, "S168")  

  










