# PyQt5 Video player
#!/usr/bin/env python

from PyQt5.QtCore import QDir, Qt, QUrl, pyqtSignal, pyqtSlot, QThread
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
        QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget)
from PyQt5.QtWidgets import QMainWindow,QWidget, QPushButton, QAction, QTextEdit
from PyQt5.QtGui import QIcon, QPixmap, QColor, QImage
from core.localize import Localize
from core.cnnlstm import CNNLSTM
import sys
import cv2
import numpy as np
import time


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)    

    def __init__(self):
        QThread.__init__(self)
        self.filename = ""  
        self.ourPause = False
        # Initialize Localization Algorithm
        self.localization_algorithm = Localize() 

        # Initialize Spotting Algorithm
        self.prediction_model = CNNLSTM()        
        self.prediction_model.mobilenet("core/transfer_mobilenet_cnnlstm_tfrecord_2/cp.ckpt")

        # Get the minimum 12 window
        self.sliding_window = []
    
    def set_filename(self, filename):
        self.filename = filename                 

    def setPause(self, pause):
        self.ourPause = pause

    def getPause(self):
        return self.ourPause

    def run(self):        
        cap = cv2.VideoCapture(self.filename)
        itr = 0
        while (cap.isOpened()):
            while self.ourPause:
                time.sleep(0.2)

            ret, frame = cap.read()
            if ret:
                itr += 1
                # Send Bounding box frame
                # bb_frame = self.localization_algorithm.mp_localize_bounding_box(frame)
                # bb_frame = self.localization_algorithm.mp_localize_crop(frame)
                # try:
                #     self.change_pixmap_signal.emit(bb_frame)
                # except Exception as e:
                #     self.change_pixmap_signal.emit(frame)
                

                # Send normal frame
                self.change_pixmap_signal.emit(frame)


                # Dlib Correlation Tracker
                # bb_frame = self.localization_algorithm.dlib_correlation_tracker(frame)
                # try:
                #     self.change_pixmap_signal.emit(bb_frame)
                # except Exception as e:
                #     self.change_pixmap_signal.emit(frame)
                
                # Process the frame using CNN-LSTM
                frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255

                self.sliding_window.append(rgb)
                if len(self.sliding_window) > 12:                
                    self.sliding_window.pop(0)

                if len(self.sliding_window) == 12:
                    np_sliding_window = np.expand_dims(np.array(self.sliding_window), axis=0) 
                    conf, label = self.prediction_model.process(np_sliding_window, conf=0.7)
                    print("Label of frame", itr - 12, "to", itr, "Label", label, "Confidence Level:", conf)                   


class VideoWindow(QMainWindow):
    def __init__(self, parent=None):
        super(VideoWindow, self).__init__(parent)
        self.setWindowTitle("ISpeak Micro-expressions Detection")
        self.disply_width = 640
        self.display_height = 480 
        
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
        self.filename, _ = QFileDialog.getOpenFileName(self, "Open Video", QDir.homePath())        
        if self.filename != '':            
            self.outputLogText.append("Image Upload Succesful")         

    def exitCall(self):
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

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)        
        self.image_label.setPixmap(qt_img)

    def process(self):
        """Convert from an opencv image to QPixmap"""            
        self.outputLogText.append("Processing Video")
        # start the thread
        self.thread.set_filename(self.filename)
        self.thread.start()  
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))      
        

if __name__ == '__main__':    
    app = QApplication(sys.argv)    
    player = VideoWindow()
    player.resize(640, 480)
    player.show()
    sys.exit(app.exec_())