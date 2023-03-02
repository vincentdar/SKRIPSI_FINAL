from cnnlstm import CNNLSTM
from evaluation import MyEvaluation
import pandas as pd
import os
import cv2
import numpy as np

# Initialize Spotting Algorithm
prediction_model = CNNLSTM()                    
prediction_model.mobilenet("D:\CodeProject2\SKRIPSI_FINAL\core\Transfer_mobilenet_unfreezelast20_100_cnnlstm_localize_casmeii/cp.ckpt")

label = "D:\CASME\CASMEII\CASMEII_preprocess.csv"
data_path =  "D:\CASME\CASMEII\CASME2-RAW-Localize\CASME2-RAW"

eval = MyEvaluation()

df = pd.read_csv(label)
for index, row in df.iterrows():
    subject = row['Subject']
    folder = row['Folder']
    onset = row['Onset']
    offset = row['Offset']

    images_path = os.path.join(data_path, subject, folder)
    
    image_numbers = sorted([int(f[3:-4]) for f in os.listdir(images_path)])

    images = []
    for number in image_numbers:
        image_filename = "img" + str(number) + ".jpg"
        image_path = os.path.join(images_path, image_filename)
        
        if os.path.isfile(image_path):
            img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)            
            images.append(img)

            if len(images) == 12:
                np_sliding_window = np.expand_dims(np.array(images), axis=0)                                
                conf, label = prediction_model.process(np_sliding_window, conf=0.7)
                # print("Label of frame", number - 11, "to", number, "Label", label, "Confidence Level:", conf)
                eval.count_casme2(number - 11, number, label, [(onset, offset)])
                images = []     

    eval.print_total()
    eval.reset_count()   

    
