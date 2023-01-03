# PyQt5 Video player
#!/usr/bin/env python

from PyQt5.QtCore import QDir, Qt, QUrl
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
        QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget)
from PyQt5.QtWidgets import QMainWindow,QWidget, QPushButton, QAction, QTextEdit
from PyQt5.QtGui import QIcon
import sys
import cv2

class VideoWindow(QMainWindow):

    def __init__(self, parent=None):
        super(VideoWindow, self).__init__(parent)
        self.setWindowTitle("ISpeak Micro-expressions Detection") 

        # Upload Process Widget
        self.uploadButton = QPushButton()
        self.uploadButton.setEnabled(True)
        self.uploadButton.setText("Upload")        
        self.uploadButton.clicked.connect(self.openFile)

        self.processButton = QPushButton()
        self.processButton.setEnabled(True)
        self.processButton.setText("Process")    
        self.processButton.clicked.connect(self.exitCall)

        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        videoWidget = QVideoWidget()        
        videoWidget.setMinimumSize(1280, 720)

        # Control Widget
        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)

        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.setPosition)

        self.errorLabel = QLabel()
        self.errorLabel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        # Output Log Widget
        self.outputLogLabel = QLabel()
        self.outputLogLabel.setText("Output Log:")        

        self.outputLogText = QTextEdit()
        self.outputLogText.setText("This is QTextEdit")         
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
        controlLayout.addWidget(self.positionSlider)

        # VideoWidget Layout
        videoWidgetLayout = QVBoxLayout()
        videoWidgetLayout.setContentsMargins(0, 0, 0, 0)
        videoWidgetLayout.addWidget(videoWidget)
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

        self.mediaPlayer.setVideoOutput(videoWidget)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.error.connect(self.handleError)

    def openFile(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Video", QDir.homePath())

        if fileName != '':
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(fileName))) 
            self.mediaPlayer.play()
            self.mediaPlayer.pause()  
            self.mediaPlayer.setPosition(0)                    
            self.playButton.setEnabled(True)

    def exitCall(self):
        sys.exit(app.exec_())

    def play(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def mediaStateChanged(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def positionChanged(self, position):
        self.positionSlider.setValue(position)

    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)

    def handleError(self):
        self.playButton.setEnabled(False)
        self.errorLabel.setText("Error: " + self.mediaPlayer.errorString())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = VideoWindow()
    player.resize(640, 480)
    player.show()
    sys.exit(app.exec_())