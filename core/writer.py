import os
import cv2
import numpy as np

class Writer:
    def __init__(self):
        self.destination_folder = "results"  
        self.detection = 0
        self.current_label = -1
        self.past_label = -1              

    def createDirectory(self, path):
        try:
            os.mkdir(path)
        except Exception as e:                            
            pass  

    def saveImages(self, image, frame_number, subject):              
        write_filename = "img" + str(frame_number).zfill(5) + ".jpg"
        write_path = os.path.join(self.destination_folder, subject, str(self.detection) + "_" + str(self.current_label), write_filename)         
        cv2.imwrite(write_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))                

    def writeToImages(self, image, frame_number, subject, label):
        self.current_label = label
        if self.current_label != self.past_label:
            self.detection += 1            
            self.createDirectory(os.path.join(self.destination_folder, subject, str(self.detection) + "_" + str(self.current_label)))
        self.past_label = label

        self.saveImages(image, frame_number, subject)

    def reset_state(self):
        self.detection = 0
        self.current_label = -1
        self.past_label = -1     
