import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import cv2
import math
from keras import backend as K

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
    yield video, tf.one_hot(label[i], depth=6)      

    # idk why images variable automatically empties itself

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

def create_efficientnetb0():  
  tf.config.run_functions_eagerly(True)
  efficientnet = tf.keras.applications.efficientnet.EfficientNetB0(input_shape=(224, 224, 3),
                                                                   include_top=False,
                                                                   pooling="avg")
  efficientnet.trainable = False


  # net = tf.keras.applications.EfficientNetB0(include_top = False)
  # net.trainable = False

  # rnn = tf.keras.Sequential([
  #     tf.keras.layers.Rescaling(scale=255),
  #     tf.keras.layers.TimeDistributed(net),
  #     tf.keras.layers.Dense(10),
  #     tf.keras.layers.GlobalAveragePooling3D()
  # ])
  
  # rnn.build(input_shape=(None, 12, 224, 224, 3))
  # print(rnn.summary())

  inputs = tf.keras.Input(shape=(224, 224, 3))
  x = tf.keras.applications.efficientnet.preprocess_input(inputs)
  x = efficientnet(x)
  # outputs = tf.keras.layers.GlobalAveragePooling2D()(x)
  cnn = tf.keras.Model(inputs=inputs, outputs=x, name="efficientnetb0")
  print(cnn.summary())

  # RNN Model
  rnn_inputs = tf.keras.Input(shape=(12, 224, 224, 3))
  rnn_x = tf.keras.layers.TimeDistributed(cnn)(rnn_inputs)
  rnn_x = tf.keras.layers.LSTM(32)(rnn_x)
  
  rnn_outputs = tf.keras.layers.Dense(6, activation="softmax")(rnn_x)

  rnn = tf.keras.model(inputs=[rnn_inputs], outputs=[rnn_outputs], name="cnnlstm")  
  # rnn.build(input_shape=(None, 12, 224, 224, 3)) 
  # print(rnn.summary())
  
  return rnn


def create_mobilenet():  
  mobilenet = tf.keras.applications.MobileNet(input_shape=(224, 224, 3), include_top=False)
  mobilenet.trainable = False

  # If you wanted to modify the mobilenet layer
  # Freeze all layer except for the last 20
  # mobilenet.trainable = True
  # for layer in mobilenet.layers[:-20]:
  #     layer.trainable = False

      

  # CNN Model
  cnn = tf.keras.models.Sequential()  
  cnn.add(mobilenet)
  # cnn.add(tf.keras.layers.Flatten())

  cnn.add(tf.keras.layers.GlobalAveragePooling2D())
  # cnn.add(tf.keras.layers.Dense(32))
  print(cnn.summary())

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


# def create_C3D():
#   # Inspiration is Multi Segment CNN: https://openaccess.thecvf.com/content_cvpr_2016/papers/Shou_Temporal_Action_Localization_CVPR_2016_paper.pdf
#   inputs = tf.keras.Input(shape=(12, 224, 224, 3))
#   x = tf.keras.layers.Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation="relu")(inputs)
#   x = tf.keras.layers.MaxPool3D(pool_size=(1,2,2))(x)
#   x = tf.keras.layers.Conv3D(128, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation="relu")(x)  
#   x = tf.keras.layers.MaxPool3D(pool_size=(2,2,2))(x)
#   x = tf.keras.layers.Conv3D(256, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation="relu")(x)    
#   x = tf.keras.layers.MaxPool3D(pool_size=(2,2,2))(x)   
#   # x = tf.keras.layers.Flatten()(x) 
#   x = tf.keras.layers.GlobalAveragePooling3D()(x)  
#   outputs = tf.keras.layers.Dense(1)(x)

#   model = tf.keras.Model(inputs=inputs, outputs=outputs, name="multisegmentcnn")  
#   return model



def compile_model(model):
  model.run_eagerly = True
  model.compile(loss='categorical_crossentropy',
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                metrics=['accuracy'])
  
  return model

def compile_model_focal_loss(model):
  model.compile(loss=[categorical_focal_loss(alpha=[[0.25, 0.25, 0.25, 0.25, 0.25, 0.25]], gamma=2)],
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                metrics=['accuracy'])
  
  return model

def read_dataframe(filename):
  df = pd.read_csv(filename)  
  df = df.sample(frac=1, random_state=42)
  # df = df.sample(frac=1)  
  
  try:
    features = df.drop(['Label', 'Augmentation'], axis=1)
    augments = df['Augmentation']
  except:
    features = df.drop('Label', axis=1)
    augments = None
  labels = df['Label']
  
  
  return features, labels, augments


if __name__ == "__main__":   
  # tf.keras.backend.clear_session()
  # tf.config.optimizer.set_jit(True) # Enable XLA.
  class_names = ["Unknown", "Showing Emotions", "Blank Face", "Reading",
                   "Head Tilt", "Occlusion"]                
  train_features, train_labels, train_augments = read_dataframe("categorical/training_pubspeak_multiclass_25042023_face_detection.csv")
  test_features, test_labels, _ = read_dataframe("categorical/testing_pubspeak_multiclass_25042023_face_detection.csv")
  
  # print(tf.convert_to_tensor(train_augments))

  # train_ds = tf.data.Dataset.from_generator(create_dataset_augmented,
  #                                         args=(tf.convert_to_tensor(train_features), tf.convert_to_tensor(train_labels), tf.convert_to_tensor(train_augments)),
  #                                         output_types=(tf.float32, tf.int64),
  #                                         output_shapes=((12, 224, 224, 3), (6)))  
  
  train_ds = tf.data.Dataset.from_generator(create_dataset,
                                          args=(tf.convert_to_tensor(train_features), tf.convert_to_tensor(train_labels)),
                                          output_types=(tf.float32, tf.int64),
                                          output_shapes=((12, 224, 224, 3), (6))) 
  
  test_ds = tf.data.Dataset.from_generator(create_dataset,
                                        args=(tf.convert_to_tensor(test_features), tf.convert_to_tensor(test_labels)),
                                        output_types=(tf.float32, tf.int64),
                                        output_shapes=((12, 224, 224, 3), (6)))  
  
  # train_ds = train_ds.filter(lambda x, y: tf.argmax(y) == 4)
  # plot_video(train_ds, 8)  
  # plt.figure()
  # plt.subplot(1, 2, 1)
  # img = parse_image("C:/Users/vince/Pictures/349249.jpg", 0)
  # plt.imshow(img)
  # flip_img = parse_image("C:/Users/vince/Pictures/349249.jpg", 1)
  # plt.subplot(1, 2, 2)
  # plt.imshow(flip_img)
  # plt.show()

  checkpoint_path = "checkpoint/local_mobilenet_cnnlstm_newpubspeak25042023_multiclass_focal_loss_10_epoch/cp.ckpt"
  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                          monitor='val_accuracy',
                                                          save_weights_only=True,
                                                          save_best_only=True,
                                                          verbose=1)
  

    

  train_ds = train_ds.prefetch(tf.data.AUTOTUNE).batch(4)
  test_ds = test_ds.prefetch(tf.data.AUTOTUNE).batch(4)
  
  # Mobilenet
  model = create_mobilenet()
  # model = create_efficientnetb0()  
  # model = compile_model(model)  
  model = compile_model_focal_loss(model)

  history = model.fit(train_ds,
                      validation_data=test_ds,
                      epochs=10,
                      callbacks=[checkpoint_callback])              
  
  # convert the history.history dict to a pandas DataFrame:     
  hist_df = pd.DataFrame(history.history) 

  # or save to csv: 
  hist_csv_file = 'history/history_local_mobilenet_cnnlstm_newpubspeak25042023_multiclass_focal_loss_10_epoch.csv'
  with open(hist_csv_file, mode='w') as f:
      hist_df.to_csv(f)









         

