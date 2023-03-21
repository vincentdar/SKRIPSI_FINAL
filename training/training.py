import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import cv2

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

def create_mobilenet_bilstm():  
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
  rnn.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)))
  rnn.add(tf.keras.layers.Dense(1, activation="sigmoid"))

  rnn.build(input_shape=(None, 12, 224, 224, 3)) 
  print(rnn.summary())
  
  return rnn


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
  df = df.sample(frac=1, random_state=42)  
  features = df.drop('Label', axis=1)
  labels = df['Label']
  return features, labels


if __name__ == "__main__":                 
  train_features, train_labels = read_dataframe("binary/training_pubspeak_21032023_face_detection.csv")
  test_features, test_labels = read_dataframe("binary/testing_pubspeak_21032023_face_detection.csv")

  train_ds = tf.data.Dataset.from_generator(create_dataset,
                                          args=(tf.convert_to_tensor(train_features), tf.convert_to_tensor(train_labels)),
                                          output_types=(tf.float32, tf.int64),
                                          output_shapes=((12, 224, 224, 3), ()))  
  
  test_ds = tf.data.Dataset.from_generator(create_dataset,
                                        args=(tf.convert_to_tensor(test_features), tf.convert_to_tensor(test_labels)),
                                        output_types=(tf.float32, tf.int64),
                                        output_shapes=((12, 224, 224, 3), ()))  
  
  

  checkpoint_path = "checkpoint/local_mobilenet_cnnlstm_unfreezelast20_newpubspeak21032023_10_epoch/cp.ckpt"
  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                          monitor='val_accuracy',
                                                          save_weights_only=True,
                                                          save_best_only=True,
                                                          verbose=1)
    

  train_ds = train_ds.prefetch(tf.data.AUTOTUNE).batch(4)
  test_ds = test_ds.prefetch(tf.data.AUTOTUNE).batch(4)
  
  # Mobilenet
  model = create_mobilenet()
  # model = create_C3D()
  # model = create_mobilenet_bilstm()
  model = compile_model(model)

  history = model.fit(train_ds,
                      validation_data=test_ds,
                      epochs=10,
                      callbacks=[checkpoint_callback])








