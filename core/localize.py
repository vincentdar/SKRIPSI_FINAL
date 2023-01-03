import cv2
import numpy as np
import mediapipe as mp
import math
import os
import dlib
from typing import List, Mapping, Optional, Tuple, Union


class Localize:
    def __init__(self, parent=None):
        # print("Face Localization Module Loaded")
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.frame_count = 0

        # Tracking / Temporal Smoothing Algorithm
        self.tracked_point = []
        self.blank_frame = np.zeros((224, 224, 3))
        self.corr_tracker = dlib.correlation_tracker()
        self.corr_reset_iterator = 0

    def _normalized_to_pixel_coordinates(self,
        normalized_x: float, normalized_y: float, image_width: int,
        image_height: int) -> Union[None, Tuple[int, int]]:
        """Converts normalized value pair to pixel coordinates."""

        # Checks if the float value is between 0 and 1.
        def is_valid_normalized_value(value: float) -> bool:
            return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                            math.isclose(1, value))

        if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):
            # TODO: Draw coordinates even if it's outside of the image bounds.
            return None

        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return x_px, y_px

    def localizeFace_mediapipe(self, image, return_image=True):
        height, width, _ = image.shape    
        faceROI = np.zeros((224, 224, 3))
        xleft, ytop, xright, ybot = 0, 0, 0, 0
        detected = False

        with self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection: 
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)

            # Draw the face detection annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.detections:
                detected = True
                for detection in results.detections:
                    # mp_drawing.draw_detection(image, detection)
                    location = detection.location_data
                    relative_bounding_box = location.relative_bounding_box
                    relative_bounding_box = location.relative_bounding_box
                    rect_start_point = self._normalized_to_pixel_coordinates(
                        relative_bounding_box.xmin, relative_bounding_box.ymin, width,
                        height)
                    rect_end_point = self._normalized_to_pixel_coordinates(
                        relative_bounding_box.xmin + relative_bounding_box.width,
                        relative_bounding_box.ymin + relative_bounding_box.height, width,
                        height)

                    xleft, ytop=rect_start_point
                    xright, ybot=rect_end_point
                    faceROI = image[ytop:ybot,xleft:xright]
                    faceROI = cv2.resize(faceROI, (224, 224), interpolation=cv2.INTER_CUBIC)
                    break
        if return_image:   
            return faceROI
        else:
            return (detected, xleft, ytop, xright, ybot)
    
    # Tracking the bounding box via average to "smooth out"
    def average_tracked_point(self, tracked_point):
        length = len(tracked_point)        
        
        if length == 0:
            return None

        xleft = 0
        ytop = 0
        xright = 0
        ybot = 0

        for element in tracked_point:        
            xleft += element[1]
            ytop += element[2]
            xright += element[3]
            ybot += element[4]
        
        # print("Tracked Length", length ,"Avg Tracking", int(xleft / length), int(ytop / length),
        # int(xright / length) -  int(xleft / length), int(ybot / length) -  int(ytop / length))
        return int(xleft / length), int(ytop / length), int(xright / length), int(ybot / length)

    def mp_localize_bounding_box(self, frame):
        tracker = self.localizeFace_mediapipe(frame, False)

        if tracker[0] == True:
            self.tracked_point.append(tracker)            
            if len(self.tracked_point) > 4:
                self.tracked_point.pop(0)    

        avg_bounding_box = self.average_tracked_point(self.tracked_point)
        if avg_bounding_box == None:
            return frame
        else:            
            # Showing Bounding Box
            cv2.rectangle(frame, (avg_bounding_box[0], avg_bounding_box[1]),
             (avg_bounding_box[2], avg_bounding_box[3]), (0, 255, 0), 2)
     
    def mp_localize_crop(self, frame):
        tracker = self.localizeFace_mediapipe(frame, False)

        if tracker[0] == True:
            self.tracked_point.append(tracker)            
            if len(self.tracked_point) > 4:
                self.tracked_point.pop(0)    

        avg_bounding_box = self.average_tracked_point(self.tracked_point)

        try:                
            faceROI = frame[avg_bounding_box[1]:avg_bounding_box[3], avg_bounding_box[0]:avg_bounding_box[2]]
            faceROI = cv2.resize(faceROI, (224, 224), interpolation=cv2.INTER_AREA)
            return faceROI
        except Exception as e:
            return self.blank_frame      

    def dlib_correlation_tracker(self, frame):        
        try:
            self.corr_tracker.update(frame)
            pos = self.corr_tracker.get_position()
            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # Showing Bounding Box
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            print("Tracking..")
            self.corr_reset_iterator += 1
            if self.corr_reset_iterator > 5:
                self.corr_reset_iterator = 0
                raise Exception()

        except Exception as e:
            detected, startX, startY, endX, endY = self.localizeFace_mediapipe(frame, False)
            if detected:
                rect = dlib.rectangle(startX, startY, endX, endY)
                self.corr_tracker.start_track(frame, rect)

                # Showing Bounding Box
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                print("Restart Tracking")
        
        return frame
            

