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
  mobilenet.trainable = False

  # CNN Model
  cnn = tf.keras.models.Sequential()  
  cnn.add(mobilenet)
  cnn.add(tf.keras.layers.Flatten())

#   cnn.add(tf.keras.layers.GlobalAveragePooling2D())
#   cnn.add(tf.keras.layers.Dense(32))

  # RNN Model
  rnn = tf.keras.models.Sequential()
  rnn.add(tf.keras.layers.TimeDistributed(cnn))
  rnn.add(tf.keras.layers.LSTM(32))
  rnn.add(tf.keras.layers.Dense(1, activation="sigmoid"))

  rnn.build(input_shape=(None, 12, 224, 224, 3)) 
  
  return rnn

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
  test_features, test_labels = read_dataframe("testing_pubspeak.csv")
  
  test_ds = tf.data.Dataset.from_generator(create_dataset,
                                        args=(tf.convert_to_tensor(test_features), tf.convert_to_tensor(test_labels)),
                                        output_types=(tf.float32, tf.int64),
                                        output_shapes=((12, 224, 224, 3), ()))  
    

  checkpoint_path = "checkpoint/local_mobilenet_cnnlstm_flatten_newpubspeak_10_epoch_val_s4/cp.ckpt"  
    
  
  test_ds = test_ds.prefetch(tf.data.AUTOTUNE).batch(4)
  
  model = create_mobilenet()  
  model.load_weights(checkpoint_path)
  model = compile_model(model)
  model.evaluate(test_ds)
