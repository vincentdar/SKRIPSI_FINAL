import tensorflow as tf
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np

class CASMESequence(tf.keras.utils.Sequence):
    def __init__(self, x_df, y_df, batch_size):
        self.x, self.y = x_df, y_df
        self.batch_size = batch_size


    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __parse_image(self, filename, label):    
        if label == 1:
            image = tf.io.read_file("D:/CASME/CASME(2)/Spotting_Version2_Localize" + "/Positive/" + filename)
            image = tf.io.decode_jpeg(image, channels=3)
            image = tf.image.convert_image_dtype(image, tf.float32) 
            image = np.array(image)
        else:
            image = tf.io.read_file("D:/CASME/CASME(2)/Spotting_Version2_Localize" + "/Negative/" + filename)
            image = tf.io.decode_jpeg(image, channels=3)
            image = tf.image.convert_image_dtype(image, tf.float32)  
            image = np.array(image)
        return image


    def __getitem__(self, idx):
        batch_x = self.x.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]

        # Process the batch and load the image  
        
        video_batch = []     
        for index, row in batch_x.iterrows():
            video = [] 
            for item in row:
               image = self.__parse_image(item, batch_y.iloc[index])       
               video.append(image) 
                                
            video = np.array(video)
            video_batch.append(np.array(video))
            video = []   
        
        return np.array(video_batch), batch_y
    

# Read DF
df = pd.read_csv("CAS(ME)2_dataset_trial.csv")
sliding_window = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
numeric_features = df[sliding_window]
target = df.pop('Label')
x, y = CASMESequence(numeric_features, target, 6)[0]
print(x.shape)
print(y)



# for videos, labels in train_ds.take(1):        
#     plt.figure(figsize=(24, 16))
#     for i in range(12):
#         plt.subplot(3, 4, i + 1)
#         plt.imshow(videos[0][i])
#         plt.title("Image " + str(i) + " Label " + str(labels[0].numpy()))

#     plt.show()
#     break