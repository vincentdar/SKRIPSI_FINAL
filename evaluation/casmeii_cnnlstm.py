import cv2
import numpy as np
import os
import statistics
import pandas as pd
import tensorflow as tf
from keras import backend as K

class CNNLSTM:
    def __init__(self, parent=None):
        print("CNNLSTM Module Loaded")
        print("Number of GPU Available:", len(tf.config.list_physical_devices('GPU')))
        self.model = None

    def test_cudnn(self):
        x = tf.zeros((3, 3), dtype=tf.float32)
        y = tf.ones((3, 3), dtype=tf.float32)
        z = tf.matmul(x, y)
        print(z)

    
    def mobilenet_binary(self, weights_path=None):
        mobilenet = tf.keras.applications.MobileNet(input_shape=(224, 224, 3), include_top=False)
        mobilenet.trainable = False

        # CNN Model
        cnn = tf.keras.models.Sequential()
        cnn.add(mobilenet)
        # cnn.add(tf.keras.layers.Flatten())
        cnn.add(tf.keras.layers.GlobalAveragePooling2D())        
        # cnn.add(tf.keras.layers.Dense(32)) # Just for the sake of it (Local Training)

        # RNN Model
        rnn = tf.keras.models.Sequential()
        rnn.add(tf.keras.layers.TimeDistributed(cnn))
        rnn.add(tf.keras.layers.LSTM(32))
        rnn.add(tf.keras.layers.Dense(1, activation="sigmoid"))


        rnn.build(input_shape=(None, 12, 224, 224, 3)) 

        rnn.compile(loss='binary_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    metrics=['accuracy'])
        
        rnn.load_weights(weights_path).expect_partial()    
        self.model = rnn        

    def mobilenet_categorical(self, weights_path=None):
        # class_names = ["Unknown", "Showing Emotions", "Blank Face", "Reading", "Head Tilt", "Occlusion"]     
        mobilenet = tf.keras.applications.MobileNet(input_shape=(224, 224, 3), include_top=False)
        mobilenet.trainable = False

        # CNN Model
        cnn = tf.keras.models.Sequential()
        cnn.add(mobilenet)
        # cnn.add(tf.keras.layers.Flatten())
        cnn.add(tf.keras.layers.GlobalAveragePooling2D())        
        # cnn.add(tf.keras.layers.Dense(32)) # Just for the sake of it (Local Training)

        # RNN Model
        rnn = tf.keras.models.Sequential()
        rnn.add(tf.keras.layers.TimeDistributed(cnn))
        rnn.add(tf.keras.layers.LSTM(32))
        rnn.add(tf.keras.layers.Dense(6, activation="softmax"))


        rnn.build(input_shape=(None, 12, 224, 224, 3)) 

        # rnn.compile(loss='categorical_crossentropy',
        #             optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        #             metrics=['accuracy'])
        
        rnn.load_weights(weights_path).expect_partial()    
        self.model = rnn          
        # self.model = self.compile_model_categorical(self.model)
        self.model = self.compile_model_categorical_focal_loss(self.model)

    def categorical_focal_loss(self, alpha, gamma=2.):
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
    
    def compile_model_categorical(self, model):
        model.compile(loss='categorical_crossentropy',
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                        metrics=['accuracy'])
        
        return model


    def compile_model_categorical_focal_loss(self, model):
        model.compile(loss=[self.categorical_focal_loss(alpha=[[.25, .25, .25, .25, .25, .25]], gamma=2)],
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                        metrics=['accuracy'])
  
        return model          

    def process(self, data, conf=0.5):                
        pred = np.squeeze(self.model.predict(data, verbose=0))
        if pred >= conf:
            label = 1
        else:
            label = 0
        return pred, label
    
    def process_categorical(self, data):
        pred = np.squeeze(self.model.predict(data, verbose=0))
        conf = np.max(pred, axis=0)
        label = np.argmax(pred, axis=0)
        return conf, label
            

if __name__ == "__main__":
    file = open('cnnlstm_casmeii.csv', 'w')
    data = "Subject,Folder,Number,Conf,Label,GT\n"
    file.write(data)
    label = pd.read_csv("D:\CASME\CASMEII\CASMEII_preprocess.csv")
    prediction_model = CNNLSTM()
    prediction_model.mobilenet_binary("D:/CodeProject2/SKRIPSI_FINAL/core/Weights/Transfer_mobilenet_unfreezelast20_100_cnnlstm_localize_casmeii_ver2/cp.ckpt")                                   
    for index, row in label.iterrows():
        subject = row['Subject']
        folder = row['Folder']
        onset = row['Onset']
        offset = row['Offset']

        data_path = "D:\CASME\CASMEII\CASME2-RAW-Localize\CASME2-RAW"
        path = os.path.join(data_path, subject, folder)
        print("Processing Path:", path)
        
        file_number = sorted([int(f[3:-4]) for f in os.listdir(path)])
        total_images = len(file_number)

        # Get all file
        iterator = 0
        norm_sliding_window = []
        for number in file_number:
            img_filename = "img" + str(number) + ".jpg"
            img_path = os.path.join(path, img_filename)
            img = cv2.imread(img_path)              
            normalized = img / 255 
            norm_sliding_window.append(normalized)
            if len(norm_sliding_window) == 12:
                np_sliding_window = np.expand_dims(np.array(norm_sliding_window), axis=0) 
                conf, label = prediction_model.process(np_sliding_window, conf=0.5)
                 
                for frame in norm_sliding_window:
                    if iterator >= onset and iterator <= offset:
                        gt = "1"
                    else:
                        gt = "0"
                    file.write("{},{},{},{},{},{}\n".format(subject,folder,iterator,conf,label,gt))
                    iterator += 1   

                norm_sliding_window = []  

            

        if len(norm_sliding_window) > 0:
            for frame in norm_sliding_window:
                if iterator >= onset and iterator <= offset:
                    gt = "1"
                else:
                    gt = "0"
                file.write("{},{},{},{},{},{}\n".format(subject,folder,iterator,str(0),str(0),gt))
                iterator += 1  

        print("Total Iterator", iterator) 
        

    file.close()
                
          
    