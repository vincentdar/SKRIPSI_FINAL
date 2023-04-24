import tensorflow as tf
import cv2
import numpy as np
from keras import backend as K

class CNNLSTM:
    def __init__(self, parent=None):        
        print("Number of GPU Available:", len(tf.config.list_physical_devices('GPU')))
        self.model = None
        self.is_categorical = False

    def test_cudnn(self):
        with tf.device('/GPU:0'):
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
        self.is_categorical = True

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
        conf = pred
        label = np.argmax(pred, axis=0)
        return conf, label

                


if __name__ == '__main__':    
    model = CNNLSTM()
    model.mobilenet()

    sliding_window = []
    itr = 0
    cap = cv2.VideoCapture("D:/CASME/CASME(2)/rawvideo/rawvideo/s16/16_0101disgustingteeth.avi")    
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:         
            itr += 1   
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255
            cv2.imshow("RGB", rgb)
            # print(rgb)            
            

            sliding_window.append(rgb)
            if len(sliding_window) > 12:                
                sliding_window.pop(0)

            if len(sliding_window) == 12:
                np_sliding_window = np.expand_dims(np.array(sliding_window), axis=0)                
                print("Label of frame", itr - 12, "to", itr, model.process(np_sliding_window))                
            

            
            
            # Press Q on keyboard to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break            
        else: 
            break

