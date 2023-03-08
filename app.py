# PyQt5 Video player
#!/usr/bin/env python

from PyQt5.QtCore import QDir, Qt, QUrl, pyqtSignal, pyqtSlot, QThread
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
        QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget)
from PyQt5.QtWidgets import QMainWindow,QWidget, QPushButton, QAction, QTextEdit
from PyQt5.QtGui import QIcon, QPixmap, QColor, QImage, QTextCursor
from core.localize import Localize
from core.cnnlstm import CNNLSTM
from core.headPoseEstimation import HeadPoseEstimation
from core.settings import Settings
from core.writer import Writer
from evaluation.evaluation import MyEvaluation
import sys
import cv2
import numpy as np
import time
import os


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)    
    append_output_log = pyqtSignal(str)

    def __init__(self):
        QThread.__init__(self)
        # self.settings = Settings()        
        # self.settings.load()
        # self.settings_dict = self.settings.settings_dict
        # Initialize Localization Algorithm
        self.localization_algorithm = Localize() 

        # Initialize Spotting Algorithm
        self.prediction_model = CNNLSTM()        
        # # self.prediction_model.mobilenet("core/transfer_mobilenet_cnnlstm_tfrecord_2/cp.ckpt")
        # self.prediction_model.mobilenet("D:\CodeProject2\SKRIPSI_FINAL\core\Weights\Transfer_mobilenet_cnnlstm_localize_tfrecord_pyramid_1/cp.ckpt")
        self.prediction_model.mobilenet("training\checkpoint\local_mobilenet_cnnlstm_newpubspeak_10_epoch_val_s4/cp.ckpt")
        
        # Writer
        self.writer = Writer()
        self.destination_folder = "results"
        
        self.headPoseEstimation = HeadPoseEstimation()

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
        self.interruptCurrentProcess = interruptCurrentProcess

    def getInterruptCurrentProcess(self):
        return self.interruptCurrentProcess

    def setStartProcess(self, startProcess):
        self.startProcess = startProcess

    def getStartProcess(self):
        return self.startProcess          

    def run(self):
        while self.threadRunning: 
            if self.startProcess:           
                try:            
                    self.process_video() 
                except Exception as e:
                    print(e)
                    pass              

    def process_video(self):  
        # Check if filename is empty
        if self.filename == '':
            return                

        cap = cv2.VideoCapture(self.filename)

        # Get the minimum n (12) window
        sliding_window = []
        norm_sliding_window = []
        itr = 0
        msg_pause_emit = True

        # Writing detection variables
        subject = self.filename.split('/')[-1]
        subject = subject[:-4]
        
        # Create Target Directory        
        self.writer.createDirectory(os.path.join(self.destination_folder, subject))                

        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
            return
        
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
                self.append_output_log.emit("Processing Interrupted")
                self.startProcess = False
                self.interruptCurrentProcess = False
                break
            
            # Read Frame from video
            ret, frame = cap.read()
            if ret:                           
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # success, hpe_frame = self.headPoseEstimation.process(frame.copy())
                # Send Bounding box frame
                # bb_frame = self.localization_algorithm.mp_localize_bounding_box(frame)
                # bb_frame = self.localization_algorithm.mp_face_mesh_crop_fixed_bb_centroid(frame.copy())
                bb_frame = self.localization_algorithm.mp_face_mesh_crop(frame)
                # bb_frame = self.localization_algorithm.mp_face_mesh_crop_fixed_bb_nose_tip(frame)

                # bb_frame = self.localization_algorithm.mp_localize_crop_scale(frame) # Ide Ko Hans
                # bb_frame = self.localization_algorithm.mp_localize_crop(frame)                
                try:
                    self.change_pixmap_signal.emit(bb_frame)                    
                except Exception as e:                    
                    self.change_pixmap_signal.emit(frame)                                         

                # Dlib Correlation Tracker
                # bb_frame = self.localization_algorithm.dlib_correlation_tracker(frame)
                # try:
                #     self.change_pixmap_signal.emit(bb_frame)
                # except Exception as e:
                #     self.change_pixmap_signal.emit(frame)
                
                # Process the frame using CNN-LSTM
                frame = cv2.resize(bb_frame, (224, 224), interpolation=cv2.INTER_AREA)
                normalized = frame / 255

                sliding_window.append(frame)
                norm_sliding_window.append(normalized)
                if len(norm_sliding_window) > 12:
                    sliding_window.pop(0)            
                    norm_sliding_window.pop(0)
                
                if len(norm_sliding_window) == 12:
                    np_sliding_window = np.expand_dims(np.array(norm_sliding_window), axis=0) 
                    conf, label = self.prediction_model.process(np_sliding_window, conf=0.7)
                    start_frame = itr - 11
                    end_frame = itr
                                        
                    for image in sliding_window:                        
                        self.writer.writeToImages(image, start_frame, subject, label) 
                        start_frame += 1                        
                        
                    sliding_window = []
                    norm_sliding_window = []                         
                    print("Label of frame", start_frame, "to", end_frame, "Label", label, "Confidence Level:", conf)                    
                                                       
                itr += 1
            else:                              
                self.append_output_log.emit("Video Finished")                    
                self.startProcess = False 
                self.filename = ""                                                                   
                break
                    



class VideoWindow(QMainWindow):
    def __init__(self, parent=None):
        super(VideoWindow, self).__init__(parent)        
        self.setWindowTitle("ISpeak Micro-expressions Detection")
        self.disply_width = 640
        self.display_height = 480
        self.filename = '' 
        
        # Upload Process Widget
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

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)  
        self.thread.append_output_log.connect(self.append_output_log)
        # start the thread        
        self.thread.start()                 

        # Control Widget
        self.playButton = QPushButton()        
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)

        self.errorLabel = QLabel()
        self.errorLabel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        # Output Log Widget
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

        # Create a widget for window contents
        wid = QWidget(self)
        self.setCentralWidget(wid)

        # Upload Process Left Button Layout
        uploadProcessLayout = QVBoxLayout()
        uploadProcessLayout.setContentsMargins(0, 0, 0, 0)
        uploadProcessLayout.addWidget(self.uploadButton)
        uploadProcessLayout.addWidget(self.processButton)        

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

    def openFile(self):        
        filename, _ = QFileDialog.getOpenFileName(self, "Open Video", QDir.homePath())        
        if self.filename == '':      
            self.filename = filename
            self.thread.set_filename(self.filename)
            self.append_output_log("Image Upload Succesful") 
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

    def handleError(self):
        self.playButton.setEnabled(False)
        self.errorLabel.setText("Error: " + self.mediaPlayer.errorString())

    # All the function below are to control and receive from class VideoThread
    def terminate_thread(self):
        self.thread.setThreadRunning(False)

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
        self.append_output_log("Processing Video")
        self.thread.setStartProcess(True)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))      
        

if __name__ == '__main__':    
    app = QApplication(sys.argv)       
    player = VideoWindow()
    player.resize(640, 480)
    player.show()
    sys.exit(app.exec_())