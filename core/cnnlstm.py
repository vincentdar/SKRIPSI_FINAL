import tensorflow as tf
import cv2
import numpy as np

class CNNLSTM:
    def __init__(self, parent=None):
        print("CNNLSTM Module Loaded")
        self.model = None
    
    def mobilenet(self, weights_path=None):
        mobilenet = tf.keras.applications.MobileNet(input_shape=(224, 224, 3), include_top=False)
        mobilenet.trainable = False

        # CNN Model
        cnn = tf.keras.models.Sequential()
        cnn.add(mobilenet)
        # cnn.add(tf.keras.layers.Flatten())
        cnn.add(tf.keras.layers.GlobalAveragePooling2D())

        # RNN Model
        rnn = tf.keras.models.Sequential()
        rnn.add(tf.keras.layers.TimeDistributed(cnn))
        rnn.add(tf.keras.layers.LSTM(32))
        rnn.add(tf.keras.layers.Dense(1, activation="sigmoid"))


        rnn.build(input_shape=(None, 12, 224, 224, 3)) 

        rnn.compile(loss='binary_crossentropy',
                    optimizer=tf.keras.optimizers.legacy.Adam(),
                    metrics=['accuracy'])

        # rnn.load_weights("transfer_mobilenet_cnnlstm_tfrecord/cp.ckpt").expect_partial() 
        rnn.load_weights(weights_path).expect_partial()    
        self.model = rnn        
        # self.model.load_weights("transfer_mobilenet_cnnlstm_tfrecord/cp.ckpt").expect_partial()    

    def process(self, data, conf=0.5):                
        pred = np.squeeze(self.model.predict(data))
        if pred >= conf:
            label = 1
        else:
            label = 0
        return pred, label
                


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

