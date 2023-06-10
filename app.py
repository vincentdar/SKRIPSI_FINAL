# PyQt5 Video player
#!/usr/bin/env python

from PyQt5.QtCore import QDir, Qt, QUrl, pyqtSignal, pyqtSlot, QThread
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
        QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget, QProgressBar)
from PyQt5.QtWidgets import QMainWindow,QWidget, QPushButton, QAction, QTextEdit
from PyQt5.QtGui import QIcon, QPixmap, QColor, QImage, QTextCursor
from core.localize import Localize
from core.cnnlstm import CNNLSTM
from core.headPoseEstimation import HeadPoseEstimation
from core.settings import Settings
from core.writer import Writer
from core.report import Report
from evaluation.evaluation import MyEvaluation
import pandas as pd
import sys
import cv2
import numpy as np
import time
import os
from datetime import datetime
from functools import wraps


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)    
    append_output_log = pyqtSignal(str)
    update_progress_bar = pyqtSignal(int)

    def __init__(self):
        QThread.__init__(self)
        self.settings = Settings()              
        self.settings.load()
        self.settings_dict = self.settings.settings_dict
        print(self.settings_dict)
        # Initialize Localization Algorithm
        self.localization_algorithm = Localize() 

        # Initialize Spotting Algorithm
        self.prediction_model = CNNLSTM()  
        if self.settings_dict['classification'] == "binary":            
            self.prediction_model.mobilenet_binary("training\checkpoint\local_mobilenet_cnnlstm_unfreezelast20_newpubspeak25042023_10_epoch/cp.ckpt")
        elif self.settings_dict['classification'] == "multiclass":
            self.prediction_model.mobilenet_categorical("training\checkpoint\local_mobilenet_cnnlstm_newpubspeak25042023_multiclass_focal_loss_10_epoch/cp.ckpt")

        # DO NOT CHANGE UNLESS PERMITTED
        # self.prediction_model.test_cudnn()          
        # self.prediction_model.mobilenet_binary("core\weights/transfer_mobilenet_pubspeak_cnnlstm_tfrecord/cp.ckpt") #Benedict Hebert
        # self.prediction_model.mobilenet_binary("core\weights/transfer_mobilenet_unfreezelast20_pubspeak_cnnlstm_tfrecord_pyramid_1_loso_S1/cp.ckpt") #Model 9
        # self.prediction_model.mobilenet_binary("core\weights/transfer_mobilenet_unfreezelast20_pubspeak_cnnlstm_tfrecord_pyramid_1_loso_S2/cp.ckpt") #Model 10
        # self.prediction_model.mobilenet_binary("core\weights/transfer_mobilenet_unfreezelast20_pubspeak_cnnlstm_tfrecord_pyramid_1_loso_S3/cp.ckpt") #Model 11
        
        # self.prediction_model.mobilenet_binary("training\checkpoint\local_mobilenet_cnnlstm_unfreezelast20_newpubspeak_10_epoch_val_s4/cp.ckpt")
        # self.prediction_model.mobilenet_binary("training\checkpoint\local_mobilenet_cnnlstm_newpubspeak21032023_10_epoch/cp.ckpt")
        # self.prediction_model.mobilenet_binary("training\checkpoint\local_mobilenet_cnnlstm_unfreezelast20_newpubspeak21032023_10_epoch/cp.ckpt")
        # self.prediction_model.mobilenet_binary("training\checkpoint\local_mobilenet_cnnlstm_unfreezelast20_newpubspeak21032023_augmented_10_epoch/cp.ckpt")
        # self.prediction_model.mobilenet_binary("training\checkpoint\local_mobilenet_cnnlstm_unfreezelast20_newpubspeak25042023_10_epoch/cp.ckpt")

        # self.prediction_model.mobilenet_binary("training\checkpoint\local_mobilenet_cnnlstm_unfreezelast20_newpubspeak21032023_augmented_10_epoch/cp.ckpt")
        # self.prediction_model.mobilenet_categorical("training\checkpoint\local_mobilenet_cnnlstm_unfreezelast20_newpubspeak15032023_multiclass_10_epoch/cp.ckpt") #Model 22
        
        # self.prediction_model.mobilenet_categorical("training\checkpoint\local_mobilenet_cnnlstm_newpubspeak21032023_multiclass_merged_10_epoch/cp.ckpt")
        # self.prediction_model.mobilenet_categorical("training\checkpoint\local_mobilenet_cnnlstm_unfreezelast20_newpubspeak21032023_multiclass_merged_10_epoch/cp.ckpt") 
        # self.prediction_model.mobilenet_categorical("training\checkpoint\local_mobilenet_cnnlstm_newpubspeak21032023_multiclass_focal_loss_merged_10_epoch/cp.ckpt")        
        # self.prediction_model.mobilenet_categorical("training\checkpoint\local_mobilenet_cnnlstm_unfreezelast20_newpubspeak21032023_multiclass_merged_augmented_10_epoch/cp.ckpt")                                   
        # self.prediction_model.mobilenet_categorical("training\checkpoint\local_mobilenet_cnnlstm_newpubspeak25042023_multiclass_focal_loss_10_epoch/cp.ckpt")                                   
            
        # Recording Utility
        self.startRecord = False

        # Report
        self.report = Report(self.prediction_model.is_categorical)
        self.report_folder = "reports"

        # Writer
        self.writer = Writer(self.report)
        # ['Images', 'Clip', 'Video']
        if self.settings_dict['output_type'] == "binary":            
            self.output_video_type = "Images"
        elif self.settings_dict['output_type'] == "multiclass":
            self.output_video_type = "Clip"
        elif self.settings_dict['output_type'] == "multiclass":
            self.output_video_type = "Video"         
       
        # HPE
        self.headPoseEstimation = HeadPoseEstimation()
        self.use_hpe = False

        # Class Names
        self.categorical_class_names = ["Unknown", "Showing Emotions", "Blank Face", "Reading", "Head Tilt", "Occlusion"]
        # self.categorical_class_names = ["Unknown", "Eye Contact", "Blank Face", "Showing Emotions", "Reading",
        #            "Sudden Eye Change", "Smiling", "Not Looking", "Head Tilt", "Occlusion"]
        # self.binary_class_names = ["Unfocused", "Focused"]
        self.binary_class_names = ["Negative", "Positive"]
        
        # Control the thread
        self.filename = ""  
        self.ourPause = False
        self.threadRunning = True
        self.interruptCurrentProcess = False
        self.startProcess = False       
        
    def set_filename(self, filename):
        self.filename = filename                 

    def setPause(self, pause):
        self.ourPause = pause

    def getPause(self):
        return self.ourPause

    def setThreadRunning(self, threadRunning):
        self.threadRunning = threadRunning

    def getThreadRunning(self):
        return self.threadRunning

    def setInterruptCurrentProcess(self, interruptCurrentProcess):
        if self.startProcess:
            self.interruptCurrentProcess = interruptCurrentProcess            
        else:
            self.interruptCurrentProcess = False

        if self.startRecord:
            self.interruptCurrentProcess = interruptCurrentProcess            
        else:
            self.interruptCurrentProcess = False

    def getInterruptCurrentProcess(self):
        return self.interruptCurrentProcess

    def setStartProcess(self, startProcess):
        self.startProcess = startProcess

    def getStartProcess(self):
        return self.startProcess 

    def setStartRecord(self, startRecord):
        self.startRecord = startRecord 

    def getStartRecord(self):
        return self.startRecord       

    def run(self):
        while self.threadRunning: 
            if self.startProcess:         
                self.process_video()                       
                # self.process_video_pyramid()    
            if self.startRecord:
                self.record_video()             
                  

    def timeit(func):
        @wraps(func)
        def timeit_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            total_time = end_time - start_time
            print(f'Function {func.__name__} Took {total_time:.4f} seconds')
            return result
        return timeit_wrapper
    
    def record_video(self):
        self.success_frame = np.full((224, 224, 3), (0, 255, 0), dtype=np.int8) 
        self.interrupt_frame =  np.full((224, 224, 3), (255, 0, 0), dtype=np.int8)   

        now = datetime.now()
        output_destination_folder = "output/" + str(now).replace(":","_") 
                 
        cap = cv2.VideoCapture(0)
     
        if (cap.isOpened()== False): 
            print("Error recording video")
            return
        
        print("Start recording...")
        self.append_output_log.emit("Start recording...")
        while True:
            ret, frame = cap.read()
            if ret:                
                # Handle Interruption i.e change file to be processed
                if self.startRecord == False:                                       
                    self.append_output_log.emit("Record Stopped")
                    self.change_pixmap_signal.emit(self.success_frame)                                                                                 
                    break                 

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     
                self.writer.writeToVideo_raw(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), output_destination_folder)                                                          
                try:
                    self.change_pixmap_signal.emit(frame)                                                          
                except Exception as e:                    
                    self.change_pixmap_signal.emit(frame)  
                              
                            
        self.filename = output_destination_folder+".mp4"                                             
        self.append_output_log.emit("Record Finished") 
        self.change_pixmap_signal.emit(self.success_frame)  
        self.writer.close_videowriter()      
    
    @timeit
    def process_video(self):  
        # Check if filename is empty
        if self.filename == '':
            self.startProcess = False
            return 
        cap = cv2.VideoCapture(self.filename)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))                       

        # Get the minimum n (12) window
        sliding_window = []
        norm_sliding_window = []
        full_frame_sliding_window = []
        itr = 0
        write_itr = 0
        msg_pause_emit = True
        self.blank_frame = np.zeros((224, 224, 3))

        stride = 1
        # Writing detection variables
        subject = self.filename.split('/')[-1]
        subject = subject[:-4]

        self.success_frame = np.full((224, 224, 3), (0, 255, 0), dtype=np.int8) 
        self.interrupt_frame =  np.full((224, 224, 3), (255, 0, 0), dtype=np.int8)    
        
        # Create Target Directory    
        if self.prediction_model.is_categorical:    
            self.writer.createDirectory(os.path.join("results_categorical", subject))
        else:
            self.writer.createDirectory(os.path.join("results_binary", subject))

        # Create Target "Clip" Directory
        if self.output_video_type == "Clip":
            self.writer.createDirectory(os.path.join("results_clip", subject))

        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
            return
        
        self.append_output_log.emit("Processing Video")
        while True:
            # Pausing
            while self.ourPause:
                if self.interruptCurrentProcess:
                    break                
                time.sleep(0.2)
                if msg_pause_emit:
                    self.append_output_log.emit("Processing Paused")
                    msg_pause_emit = False
            # Re arm the message to re emit
            msg_pause_emit = True
            
            # Handle Interruption i.e change file to be processed
            if self.interruptCurrentProcess:  
                self.writer.close_videowriter()   
                if self.use_hpe:
                    self.report.writeToAngles(subject)
                    self.report.clearPredictions()   
                else:
                    self.report.writeToCSV(subject, stride) 
                    self.report.clearPredictions()                                         
                self.append_output_log.emit("Processing Interrupted")
                self.change_pixmap_signal.emit(self.interrupt_frame)  
                self.startProcess = False
                self.interruptCurrentProcess = False
                self.update_progress_bar.emit(100) 
                break
            
            # Read Frame from video
            ret, frame = cap.read()
            if ret:     
                full_frame = frame.copy()                      
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                
                
                if self.use_hpe:
                    success, x, y, z, bb_frame = self.headPoseEstimation.process(frame.copy())
                    self.report.addAngles(x, y, z)
                    self.writer.writeToVideoHPE(bb_frame, subject)

                # Send Bounding box frame
                # bb_frame = self.localization_algorithm.mp_localize_bounding_box(frame)
                # bb_frame = self.localization_algorithm.mp_face_mesh_crop_fixed_bb_centroid(frame.copy())
                (detection, xleft, ytop, xright, ybot) = self.localization_algorithm.mp_face_mesh_crop_preprocessing(frame) # Semua menggunakan face mesh
                try:
                    temp_frame = frame[ytop:ybot,
                                    xleft:xright]
                    bb_frame = cv2.resize(temp_frame, (224, 224), interpolation=cv2.INTER_AREA)                    
                except Exception:
                    bb_frame = self.blank_frame
                    pass

                cv2.rectangle(full_frame, 
                            (xleft, ytop), 
                            (xright, ybot),
                            (0, 0, 255), 2)
                # bb_frame = self.localization_algorithm.mp_face_mesh_crop_fixed_bb_nose_tip(frame) 
                # bb_frame = self.localization_algorithm.mp_localize_crop_scale(frame) # Ide Ko Hans
                # bb_frame = self.localization_algorithm.mp_localize_crop(frame)
                # bb_frame = self.localization_algorithm.localizeFace_mediapipe(frame)
                               
                try:
                    self.change_pixmap_signal.emit(bb_frame)                                                          
                except Exception as e:                    
                    self.change_pixmap_signal.emit(frame)                                                      

                # # Dlib Correlation Tracker
                # bb_frame = self.localization_algorithm.dlib_correlation_tracker(frame)
                # try:
                #     self.change_pixmap_signal.emit(bb_frame)
                # except Exception as e:
                #     self.change_pixmap_signal.emit(frame)

                # Process the frame using CNN-LSTM
                frame = cv2.resize(bb_frame, (224, 224), interpolation=cv2.INTER_AREA)
                normalized = frame / 255

                try:
                    self.change_pixmap_signal.emit(frame)                                                          
                except Exception as e:                    
                    self.change_pixmap_signal.emit(frame)   
                

                sliding_window.append(frame)
                norm_sliding_window.append(normalized)
                full_frame_sliding_window.append(full_frame)
                if len(norm_sliding_window) > 12:
                    sliding_window.pop(0)  
                    norm_sliding_window.pop(0)
                    full_frame_sliding_window.pop(0)
                
                if len(norm_sliding_window) == 12:
                    np_sliding_window = np.expand_dims(np.array(norm_sliding_window), axis=0) 
                    if self.prediction_model.is_categorical:
                        conf, label = self.prediction_model.process_categorical(np_sliding_window)
                    else:
                        conf, label = self.prediction_model.process(np_sliding_window, conf=0.7)  

                    # Progress Bar
                    update = int((itr / length) * 100)                    
                    self.update_progress_bar.emit(update)                   
                                       
                    start_frame = itr - 11
                    end_frame = itr
                    if stride == 12:
                        # 12 Strides
                        if self.prediction_model.is_categorical:
                            self.report.addPredictions_Categorical(self.categorical_class_names, start_frame, end_frame, conf, label)
                        else:
                            self.report.addPredictions_Binary(self.binary_class_names, start_frame, end_frame, conf, label)
                        print("Label of frame", start_frame, "to", end_frame, "Label", label)
                        
                        if self.output_video_type == "Images":                                                             
                            for image in sliding_window:                                                
                                if self.prediction_model.is_categorical:
                                    # Categorical Writing
                                    self.writer.writeToImagesCategorical(image, start_frame, subject, label)  
                                else:
                                    # Binary Writing
                                    self.writer.writeToImages(image, start_frame, subject, label)                                                                        
                                start_frame += 1   
                        elif self.output_video_type == "Video":
                            # Video
                            for image in full_frame_sliding_window:  
                                if self.prediction_model.is_categorical:                      
                                    self.writer.writeToVideo(image, subject, label, itr, self.categorical_class_names)
                                else:
                                    self.writer.writeToVideo(image, subject, label, itr, self.binary_class_names)
                                start_frame += 1  
                        elif self.output_video_type == "Clip":
                            # Clip
                            for image in full_frame_sliding_window:  
                                if self.prediction_model.is_categorical:                      
                                    self.writer.writeToClip(image, subject, label, itr, self.categorical_class_names)
                                else:
                                    self.writer.writeToClip(image, subject, label, itr, self.binary_class_names)
                                start_frame += 1  
                                                                                                            
                        sliding_window = []
                        norm_sliding_window = []  
                        full_frame_sliding_window = []                                                
                    else:
                        # 1 Strides
                        if write_itr != itr:                        
                            for image in full_frame_sliding_window:                                                            
                                if self.prediction_model.is_categorical:                      
                                    self.writer.writeToVideo(image, subject, label, itr, self.categorical_class_names)
                                    self.report.addPredictions_Categorical(self.categorical_class_names, start_frame, end_frame, conf, label)
                                    pass
                                else:
                                    self.writer.writeToVideo(image, subject, label, write_itr, self.binary_class_names) 
                                    self.report.addPredictions_Binary(self.binary_class_names, start_frame, end_frame, conf, label)

                                print("Label of frame number", itr, "and written in", write_itr ,"Label", label)
                                write_itr += 1  
                        else:
                            image = full_frame_sliding_window[-1]
                            self.writer.writeToVideo(image, subject, label, write_itr, self.binary_class_names)          
                                                    
                            if self.prediction_model.is_categorical:
                                # self.report.addPredictions_Categorical(self.categorical_class_names, start_frame, end_frame, conf, label)
                                pass
                            else:                                                                        
                                self.report.addPredictions_Binary(self.binary_class_names, start_frame, end_frame, conf, label)
                            
                            print("Label of frame number", itr, "and written in", write_itr ,"Label", label)
                            write_itr += 1                                                    
                itr += 1
            else:        
                self.writer.close_videowriter()         
                if self.use_hpe:
                    self.report.writeToAngles(subject)
                    self.report.clearPredictions()   
                else:
                    self.report.writeToCSV(subject, stride) 
                    self.report.clearPredictions()                        
                self.append_output_log.emit("Video Finished") 
                self.change_pixmap_signal.emit(self.success_frame)                    
                self.startProcess = False 
                self.filename = ""  
                self.update_progress_bar.emit(100)                                                                    
                break

    @timeit
    def process_video_pyramid(self):  
        # Check if filename is empty
        if self.filename == '':
            self.startProcess = False
            return                

        cap = cv2.VideoCapture(self.filename)

        # Get the minimum n (12) window
        norm_sliding_window_12 = []
        norm_sliding_window_24 = []
        norm_sliding_window_48 = []
        norm_sliding_window_96 = []                
        itr = 0        
        msg_pause_emit = True

        stride = 12
        # Writing detection variables
        subject = self.filename.split('/')[-1]
        subject = subject[:-4]

        self.success_frame = np.full((224, 224, 3), (0, 255, 0), dtype=np.int8) 
        self.interrupt_frame =  np.full((224, 224, 3), (255, 0, 0), dtype=np.int8)  
            
        
        # Create Target Directory    
        if self.prediction_model.is_categorical:    
            self.writer.createDirectory(os.path.join("results_categorical", subject))
        else:
            self.writer.createDirectory(os.path.join("results_binary", subject))


        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
            return
        
        self.append_output_log.emit("Processing Video")
        while True:
            # Pausing
            while self.ourPause:
                if self.interruptCurrentProcess:
                    break                
                time.sleep(0.2)
                if msg_pause_emit:
                    self.append_output_log.emit("Processing Paused")
                    msg_pause_emit = False
            # Re arm the message to re emit
            msg_pause_emit = True
            
            # Handle Interruption i.e change file to be processed
            if self.interruptCurrentProcess:  
                self.writer.close_videowriter()   
                if self.use_hpe:
                    self.report.writeToAngles(subject)
                    self.report.clearPredictions()   
                else:
                    self.report.writeToCSVBinary_pyramid(subject, stride, itr) 
                    self.report.clearPredictions()                                         
                self.append_output_log.emit("Processing Interrupted")
                self.change_pixmap_signal.emit(self.interrupt_frame)  
                self.startProcess = False
                self.interruptCurrentProcess = False
                break
            
            # Read Frame from video
            ret, frame = cap.read()
            if ret:     
                full_frame = frame.copy()                      
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                
                
                # Send Bounding box frame                
                bb_frame = self.localization_algorithm.mp_face_mesh_crop(frame) # Semua menggunakan face mesh
          
                try:
                    self.change_pixmap_signal.emit(bb_frame)                                                          
                except Exception as e:                    
                    self.change_pixmap_signal.emit(frame) 
                                                    
                # Process the frame using CNN-LSTM
                frame = cv2.resize(bb_frame, (224, 224), interpolation=cv2.INTER_AREA)
                normalized = frame / 255

                # Pyramid                                 
                norm_sliding_window_12.append(normalized)
                if itr % 2 == 0:
                    norm_sliding_window_24.append(normalized)
                if itr % 4 == 0:
                    norm_sliding_window_48.append(normalized)
                if itr % 8 == 0:
                    norm_sliding_window_96.append(normalized)
                
                
                if len(norm_sliding_window_12) == 12:
                    np_sliding_window = np.expand_dims(np.array(norm_sliding_window_12), axis=0) 
                    if self.prediction_model.is_categorical:
                        conf, label = self.prediction_model.process_categorical(np_sliding_window)
                    else:
                        conf, label = self.prediction_model.process(np_sliding_window, conf=0.7)                    
                    
                   
                    start_frame = itr - 11
                    end_frame = itr
                    self.report.addPredictions_Binary_pyramid(self.binary_class_names, start_frame, end_frame, conf, label, 12)
                    
                    print("Pyramid", 12,"Label of frame", start_frame, "to", end_frame, "Label", label)  

                    norm_sliding_window_12 = []
                if len(norm_sliding_window_24) == 12:
                    np_sliding_window = np.expand_dims(np.array(norm_sliding_window_24), axis=0) 
                    if self.prediction_model.is_categorical:
                        conf, label = self.prediction_model.process_categorical(np_sliding_window)
                    else:
                        conf, label = self.prediction_model.process(np_sliding_window, conf=0.7)                    
                    
                   
                    start_frame = itr - 22
                    end_frame = itr
                    self.report.addPredictions_Binary_pyramid(self.binary_class_names, start_frame, end_frame, conf, label, 24)
                    
                    print("Pyramid", 24,"Label of frame", start_frame, "to", end_frame, "Label", label)  

                    norm_sliding_window_24 = []
                if len(norm_sliding_window_48) == 12:
                    np_sliding_window = np.expand_dims(np.array(norm_sliding_window_48), axis=0) 
                    if self.prediction_model.is_categorical:
                        conf, label = self.prediction_model.process_categorical(np_sliding_window)
                    else:
                        conf, label = self.prediction_model.process(np_sliding_window, conf=0.7)                    
                    
                   
                    start_frame = itr - 44
                    end_frame = itr           
                    self.report.addPredictions_Binary_pyramid(self.binary_class_names, start_frame, end_frame, conf, label, 48)         
                    print("Pyramid", 48,"Label of frame", start_frame, "to", end_frame, "Label", label)   

                    norm_sliding_window_48 = [] 
                if len(norm_sliding_window_96) == 12:
                    np_sliding_window = np.expand_dims(np.array(norm_sliding_window_96), axis=0) 
                    if self.prediction_model.is_categorical:
                        conf, label = self.prediction_model.process_categorical(np_sliding_window)
                    else:
                        conf, label = self.prediction_model.process(np_sliding_window, conf=0.7)                    
                    
                   
                    start_frame = itr - 88
                    end_frame = itr                    
                    self.report.addPredictions_Binary_pyramid(self.binary_class_names, start_frame, end_frame, conf, label, 96)
                    print("Pyramid", 96,"Label of frame", start_frame, "to", end_frame, "Label", label)                     

                    norm_sliding_window_96 = []                                                                                                                                                                                                                        
                itr += 1
            else:        
                self.writer.close_videowriter()         
                if self.use_hpe:
                    self.report.writeToAngles(subject)
                    self.report.clearPredictions()   
                else:
                    self.report.writeToCSVBinary_pyramid(subject, stride, itr) 
                    self.report.clearPredictions()                        
                self.append_output_log.emit("Video Finished") 
                self.change_pixmap_signal.emit(self.success_frame)                    
                self.startProcess = False 
                self.filename = ""                                                                   
                break
                    

class VideoWindow(QMainWindow):
    def __init__(self, parent=None):
        super(VideoWindow, self).__init__(parent)        
        self.setWindowTitle("ISpeak Public Speaking expressions Detection")
        self.disply_width = 640
        self.display_height = 480
        self.filename = '' 

        # with open('Ubuntu.qss', 'r', encoding='utf-8') as file:
        #     stylesheet = file.read()
        # self.setStyleSheet(stylesheet)
        
        # Upload Process Widget
        self.recordButton = QPushButton()
        self.recordButton.setEnabled(True)
        self.recordButton.setText("Record")    
        # self.recordButton.setMinimumSize(150, 150) 
        # self.recordButton.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)        
        self.recordButton.clicked.connect(self.record)

        self.uploadButton = QPushButton()
        self.uploadButton.setEnabled(True)
        self.uploadButton.setText("Upload")         
        self.uploadButton.clicked.connect(self.openFile)

        self.processButton = QPushButton()
        self.processButton.setEnabled(True)
        self.processButton.setText("Process")    
        self.processButton.clicked.connect(self.process)

        # OpenCV Image Label
        self.image_label = QLabel()
        self.image_label.resize(self.disply_width, self.display_height)
        self.black_frame =  np.full((224, 224, 3), (0, 0, 0), dtype=np.int8)
        self.update_image(self.black_frame)

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)  
        self.thread.append_output_log.connect(self.append_output_log)
        self.thread.update_progress_bar.connect(self.update_progressbar)
        # start the thread        
        self.thread.start()                 

        # Control Widget
        self.playButton = QPushButton()        
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)

        self.errorLabel = QLabel()
        self.errorLabel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        # Output Log Widget
        self.ProgressBarLabel = QLabel()
        self.ProgressBarLabel.setText("Progress")        

        self.pbar = QProgressBar()
        self.pbar.setValue(0)

        self.outputLogLabel = QLabel()
        self.outputLogLabel.setText("Output Log:")        

        self.outputLogText = QTextEdit()
        self.outputLogText.setText("Welcome..")         
        self.outputLogText.setMaximumHeight(72)        

        # Create new action
        openAction = QAction(QIcon('open.png'), '&Open', self)        
        openAction.setShortcut('Ctrl+O')
        openAction.setStatusTip('Open movie')
        openAction.triggered.connect(self.openFile)

        # Create exit action
        exitAction = QAction(QIcon('exit.png'), '&Exit', self)        
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.exitCall)

        # Create menu bar and add action
        menuBar = self.menuBar()
        fileMenu = menuBar.addMenu('&File')        
        fileMenu.addAction(openAction)        
        fileMenu.addAction(exitAction)        
        # settingsBar = menuBar.addMenu('&Settings')        

        # Create a widget for window contents
        wid = QWidget(self)
        self.setCentralWidget(wid)

        # Upload Process Left Button Layout
        uploadProcessLayout = QVBoxLayout()
        uploadProcessLayout.setContentsMargins(0, 0, 0, 0)
        uploadProcessLayout.addWidget(self.recordButton)
        uploadProcessLayout.addWidget(self.uploadButton)
        uploadProcessLayout.addWidget(self.processButton) 
        uploadProcessLayout.addStretch()             

        # Create layouts to place inside widget
        controlLayout = QHBoxLayout()
        controlLayout.setContentsMargins(0, 0, 0, 0)
        controlLayout.addWidget(self.playButton)        

        # VideoWidget Layout
        videoWidgetLayout = QVBoxLayout()
        videoWidgetLayout.setContentsMargins(0, 0, 0, 0)
        videoWidgetLayout.addWidget(self.image_label)        
        videoWidgetLayout.addLayout(controlLayout)

        # Theatre Layout
        theatreLayout = QHBoxLayout()
        theatreLayout.setContentsMargins(0, 0, 0, 0)        
        theatreLayout.addLayout(uploadProcessLayout)
        theatreLayout.addLayout(videoWidgetLayout)

        # Output Log Layout
        outputLogLayout = QVBoxLayout()       
        outputLogLayout.addWidget(self.ProgressBarLabel)
        outputLogLayout.addWidget(self.pbar)         
        outputLogLayout.addWidget(self.outputLogLabel)
        outputLogLayout.addWidget(self.outputLogText)
        outputLogLayout.addStretch(0)
        outputLogLayout.setContentsMargins(0, 0, 0, 0)  
        outputLogLayout.addSpacing(0)

        # Final Theatre Layout
        finalTheatreLayout = QVBoxLayout()
        finalTheatreLayout.setContentsMargins(0, -1, 0, -1) 
        finalTheatreLayout.addLayout(theatreLayout)
        finalTheatreLayout.addLayout(outputLogLayout)

        # Application Layout
        layout = QVBoxLayout()
        layout.addLayout(finalTheatreLayout)                    
        layout.addWidget(self.errorLabel)

        # Set widget to contain window contents
        wid.setLayout(layout)

    def record(self):
        if self.thread.getStartRecord() == True:
            print("Recording Stop")
            self.append_output_log("Recording Stop") 
            self.thread.setStartRecord(False)

            self.uploadButton.setEnabled(True)           
            self.processButton.setEnabled(True)           
            self.playButton.setEnabled(True)                     
        elif self.thread.getStartRecord() == False:
            print("Recording Start")
            self.append_output_log("Recording Start") 
            self.thread.setStartRecord(True) 

            self.uploadButton.setEnabled(False)           
            self.processButton.setEnabled(False)           
            self.playButton.setEnabled(False)           

    def openFile(self):        
        filename, _ = QFileDialog.getOpenFileName(self, "Open Video", QDir.homePath())        
        if self.filename == '':     
            if filename != '': 
                self.filename = filename
                self.thread.set_filename(self.filename)
                self.append_output_log("Image Upload Succesful") 
            else:
                self.append_output_log("Filename empty") 
        else:
            self.filename = filename  
            self.thread.set_filename(self.filename)
            self.thread.setInterruptCurrentProcess(True)  
                          

    def exitCall(self):
        self.thread.setThreadRunning(False)
        sys.exit(app.exec_())

    def play(self):                
        if self.thread.getPause() == True:
            self.thread.setPause(False)
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.thread.setPause(True)            
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    # def handleError(self):
    #     self.playButton.setEnabled(False)
    #     self.errorLabel.setText("Error: " + self.mediaPlayer.errorString())    

    # All the function below are to control and receive from class VideoThread    
    def terminate_thread(self):
        self.thread.setThreadRunning(False)

    @pyqtSlot(int)
    def update_progressbar(self, msg):        
        self.pbar.setValue(int(msg))
        # if self.pbar.value() >= 99:
        #     self.pbar.setValue(0)            

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        # rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)        
        self.image_label.setPixmap(qt_img)

    @pyqtSlot(str)
    def append_output_log(self, msg):
        """Append the message to output log"""
        self.outputLogText.append(msg) 
        self.outputLogText.verticalScrollBar().setValue(self.outputLogText.verticalScrollBar().maximum())

    def process(self):
        """Convert from an opencv image to QPixmap"""                            
        self.thread.setStartProcess(True)        
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))      
        

if __name__ == '__main__':    
    app = QApplication(sys.argv)       
    player = VideoWindow()
    player.resize(640, 480)
    player.show()
    sys.exit(app.exec_())