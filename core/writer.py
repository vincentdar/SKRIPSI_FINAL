import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime

class Writer:
    def __init__(self, report):
        # Reporting
        self.report = report
        # Image
        self.destination_folder = "results_binary" 
        self.categorical_destination_folder = "results_categorical" 
        self.detection = 0
        self.current_label = -1
        self.past_label = -1     

        # Video
        self.video_writer = None   
        self.video_destination_folder = "results_video" 

        # Clip
        self.clip_destination_folder = "results_clip"
        self.detection_folder = ""


        self.width = None
        self.height = None                

    def createDirectory(self, path):
        try:
            os.mkdir(path)
        except Exception as e: 
            print(e)                           
            pass  

    def saveImages(self, image, frame_number, label, subject, class_names):      

        write_filename = "img" + str(frame_number).zfill(5) + ".jpg"
        write_path = os.path.join(self.destination_folder, subject, str(self.detection) + "_" + str(self.current_label), write_filename)         
        if len(class_names) > 2:
            gt_label = self.report.getGTLabel(subject, frame_number)
        else:
            gt_label = self.report.getGTLabelBinary(subject, frame_number) 

        # Predicted
        h, w, _ = image.shape
        cv2.putText(image, "Pred: " + class_names[label], (int(0.01 * w), int(0.05 * h)),
                     cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)     
        # GT
        if gt_label != -1:
            cv2.putText(image, "GT: " + class_names[gt_label], (int(0.01 * w), int(0.12 * h)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)    
        else:
            # cv2.putText(image, "GT: " + "None", (int(0.01 * self.width), int(0.12 * self.height)),
            #         cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)    
            # EMPTY 
            pass
        # cv2.imwrite(write_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(write_path, image) 

    def saveImagesCategorical(self, image, frame_number, label, subject, class_names):      

        write_filename = "img" + str(frame_number).zfill(5) + ".jpg"
        write_path = os.path.join(self.categorical_destination_folder, subject, str(self.detection) + "_" + str(self.current_label), write_filename)         
        if len(class_names) > 2:
            gt_label = self.report.getGTLabel(subject, frame_number)
        else:
            gt_label = self.report.getGTLabelBinary(subject, frame_number) 

        print("GT", gt_label)
        print("Label", label)
        
        # Predicted
        h, w, _ = image.shape
        cv2.putText(image, "Pred: " + class_names[label], (int(0.01 * w), int(0.05 * h)),
                     cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)     
        # GT
        if gt_label != -1:
            cv2.putText(image, "GT: " + class_names[gt_label], (int(0.01 * w), int(0.12 * h)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)    
        else:
            # cv2.putText(image, "GT: " + "None", (int(0.01 * self.width), int(0.12 * self.height)),
            #         cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)    
            # EMPTY 
            pass
        # cv2.imwrite(write_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(write_path, image)                   

    def writeToImages(self, image, frame_number, subject, label, class_names):
        self.current_label = label
        if self.current_label != self.past_label:
            self.detection += 1            
            self.createDirectory(os.path.join(self.destination_folder, subject, str(self.detection) + "_" + str(self.current_label)))
        self.past_label = label
        self.saveImages(image, frame_number, label, subject, class_names)    

    def writeToImagesCategorical(self, image, frame_number, subject, label, class_names):
        self.current_label = label
        if self.current_label != self.past_label:
            self.detection += 1            
            self.createDirectory(os.path.join(self.categorical_destination_folder, subject, str(self.detection) + "_" + str(self.current_label)))
        self.past_label = label
        self.saveImagesCategorical(image, frame_number, label, subject, class_names)    

    def reset_state(self):
        self.detection = 0
        self.current_label = -1
        self.past_label = -1   
        
    #   Video     
    def set_videowriter(self, subject, image, class_names):
        if self.video_writer == None:
            h, w, _ = image.shape
            filename = os.path.join(self.video_destination_folder, subject + ".mp4")
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            self.video_writer = cv2.VideoWriter(filename,
                                                fourcc,
                                                25.0,
                                                (w, h))
            self.width = w
            self.height = h    
            
    def close_videowriter(self):
        if self.video_writer != None:
            self.video_writer.release()
            self.video_writer = None
        
    def writeToVideo(self, image, subject, label, number, class_names):
        self.set_videowriter(subject, image, class_names)    
        if len(class_names) > 2:
            gt_label = self.report.getGTLabel(subject, number)
        else:
            gt_label = self.report.getGTLabelBinary(subject, number) 
                      
        # Predicted
        new_class_names = ["Mimik Wajah masih salah", "Mimik Wajah sudah benar"]
        cv2.putText(image, "Prediksi: " + new_class_names[label], (int(0.01 * self.width), int(0.05 * self.height)),
                     cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)     
        # GT
        # if gt_label != -1:
        #     cv2.putText(image, "GT: " + class_names[gt_label], (int(0.01 * self.width), int(0.12 * self.height)),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)    
        # else:
        #     # cv2.putText(image, "GT: " + "None", (int(0.01 * self.width), int(0.12 * self.height)),
        #     #         cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)    
        #     # EMPTY 
        #     pass
        self.video_writer.write(image)

    def set_videowriter_raw(self, image, output_destination_folder):        
        if self.video_writer == None:
            h, w, _ = image.shape
            filename = os.path.join(output_destination_folder + ".mp4")
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            self.video_writer = cv2.VideoWriter(filename,
                                                fourcc,
                                                25.0,
                                                (w, h))
            self.width = w
            self.height = h

    def writeToVideo_raw(self, image, destination_folder):
        self.set_videowriter_raw(image, destination_folder)
        self.video_writer.write(image)

    def writeToVideo_annotated(self, image, destination_folder, subject, label, number, class_names):
        self.set_videowriter_raw(image, destination_folder)    
        if len(class_names) > 2:
            gt_label = self.report.getGTLabel(subject, number)
        else:
            gt_label = self.report.getGTLabelBinary(subject, number) 
                  
        # Predicted
        cv2.putText(image, "Pred: " + class_names[label], (int(0.01 * self.width), int(0.05 * self.height)),
                     cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)     
        # GT
        if gt_label != -1:
            cv2.putText(image, "GT: " + class_names[gt_label], (int(0.01 * self.width), int(0.12 * self.height)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)    
        else:
            # cv2.putText(image, "GT: " + "None", (int(0.01 * self.width), int(0.12 * self.height)),
            #         cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)    
            # EMPTY 
            pass
        self.video_writer.write(image)


    def writeToVideoHPE(self, image, subject):
        self.set_videowriter(subject, image, "")           
        self.video_writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR)) 

    #   Clipped Video
    def writeToClip(self, image, subject, label, itr, class_names):        
        self.current_label = label
        if self.current_label != self.past_label:
            self.close_videowriter()
            self.detection += 1  
            current_class_names = class_names[label]   
            self.detection_folder = os.path.join(self.clip_destination_folder, subject, str(self.detection) + "_" + str(current_class_names))       
        self.past_label = label
        self.writeToVideo_annotated(image, self.detection_folder, subject, label, itr, class_names)


