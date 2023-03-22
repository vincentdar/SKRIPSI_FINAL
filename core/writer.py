import os
import cv2
import numpy as np

class Writer:
    def __init__(self):
        # Image
        self.destination_folder = "results" 
        self.categorical_destination_folder = "results_categorical" 
        self.detection = 0
        self.current_label = -1
        self.past_label = -1     

        # Video
        self.video_writer = None   
        self.video_destination_folder = "results_video" 
        
             

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

    def writeToImagesCategorical(self, image, frame_number, subject, label):
        self.current_label = label
        if self.current_label != self.past_label:
            self.detection += 1            
            self.createDirectory(os.path.join(self.categorical_destination_folder, subject, str(self.detection) + "_" + str(self.current_label)))
        self.past_label = label
        self.saveImages(image, frame_number, subject)

    def reset_state(self):
        self.detection = 0
        self.current_label = -1
        self.past_label = -1    

    def set_videowriter(self, subject, image, class_names):
        if self.video_writer == None:
            h, w, _ = image.shape
            filename = os.path.join(self.video_destination_folder, subject + ".mp4")
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            self.video_writer = cv2.VideoWriter(filename,
                                                fourcc,
                                                25.0,
                                                (w, h))
            
    def close_videowriter(self):
        if self.set_videowriter != None:
            self.video_writer.release()
        
    def writeToVideo(self, image, subject, label, class_names):
        self.set_videowriter(subject, image, class_names)        
        
        cv2.putText(image, class_names[label], (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2)
        self.video_writer.write(image)
