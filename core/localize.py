import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import math
import os
import dlib
from typing import List, Mapping, Optional, Tuple, Union


class Localize:
    def __init__(self, parent=None):
        # print("Face Localization Module Loaded")
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.frame_count = 0

        # Tracking / Temporal Smoothing Algorithm
        self.tracked_point = []
        self.blank_frame = np.zeros((224, 224, 3))
        self.corr_tracker = dlib.correlation_tracker()
        self.corr_reset_iterator = 0

        # Scaling Algorithm (Ide Ko Hans)
        self.former_bb_area = 1
        self.init_area = False

        # Face Mesh Support Algorithm
        self.centroid_tracker = []
        self.bb_length = 0        
        self.init_length_bb = False

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)



    def plot_centroid_length(self, filename):
        dist_ls = []
        start_x, start_y = self.centroid_tracker[0]
        for x, y in self.centroid_tracker[1:]:
            dist = math.sqrt((x - start_x) ** 2 + (y - start_y) ** 2)
            start_x = x
            start_y = y
            dist_ls.append(dist)

        plt.plot(np.array(dist_ls), ls='-')
        plt.savefig(filename)
        self.centroid_tracker = []
        
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

    def mp_localize_crop_scale(self, frame):    
        tracker = self.localizeFace_mediapipe(frame, False)

        if tracker[0] == True:
            self.tracked_point.append(tracker)            
            if len(self.tracked_point) > 1:
                self.tracked_point.pop(0)    

        avg_bounding_box = self.average_tracked_point(self.tracked_point)

        # Scale the image
        deltaX = avg_bounding_box[2] - avg_bounding_box[0]
        deltaY = avg_bounding_box[3] - avg_bounding_box[1]        


        # Centroid generation to check the translation
        x_center = int(round((avg_bounding_box[2] + avg_bounding_box[0]) / 2))
        y_center = int(round((avg_bounding_box[3] + avg_bounding_box[1]) / 2))

        w, h, _ = frame.shape
        ratio_x = x_center /  w     
        ratio_y = y_center /  h     

        area = deltaX * deltaY        

        # Init Area
        if self.init_area == False:
            self.init_area = True
            self.former_bb_area = area
        
        # Calc diff and rescale the frame
        diff = area / self.former_bb_area
                
        frame = cv2.resize(frame, (int(round(h * diff)), int(round(w * diff))),
                            interpolation = cv2.INTER_LINEAR)  

        new_w, new_h, _ = frame.shape

        trans_x = int(ratio_x * new_w)
        trans_y = int(ratio_y * new_h)

        movement_x = trans_x - x_center
        movement_y = trans_y - y_center

        xleft = avg_bounding_box[0] + movement_x
        ytop = avg_bounding_box[1] + movement_y
        xright = avg_bounding_box[2] + movement_x
        ybot = avg_bounding_box[3] + movement_y

        try:                
            faceROI = frame[ytop:ybot,
                            xleft:xright]
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
    
    def mp_face_mesh_bounding_box(self, frame):
        with self.mp_face_mesh.FaceMesh(max_num_faces=1,
                                        refine_landmarks=True,
                                        min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5) as face_mesh:
            results = face_mesh.process(frame)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            frame.flags.writeable = False
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:                    
                    # CODE FROM https://github.com/google/mediapipe/issues/1737
                    # mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
                    h, w, c = frame.shape
                    cx_min=  w
                    cy_min = h
                    cx_max = cy_max= 0
                    for id, lm in enumerate(face_landmarks.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        if cx<cx_min:
                            cx_min=cx
                        if cy<cy_min:
                            cy_min=cy
                        if cx>cx_max:
                            cx_max=cx
                        if cy>cy_max:
                            cy_max=cy

                    padding_x = int((cx_max - cx_min) * 0.05)
                    padding_y = int((cy_max - cy_min) * 0.05)
                    
                    cv2.rectangle(frame, 
                                (cx_min - padding_x, cy_min - padding_y), 
                                (cx_max + padding_x, cy_max + padding_y),
                                (0, 0, 255), 2)
                    # cv2.rectangle(frame, 
                    #             (cx_min, cy_min), 
                    #             (cx_max, cy_max),
                    #             (0, 0, 255), 2)
                    # try:
                    #     resized = frame[cy_min - padding_y:cy_max + padding_y,
                    #                     cx_min - padding_x:cx_max + padding_x].copy()
                    #     resized = cv2.resize(resized, (224, 224))
                    # except Exception:
                    #     resized = np.zeros((224, 224))
                    # END CODE
                    return frame
                
    def mp_face_mesh_draw(self, frame):
        print("A")
        with self.mp_face_mesh.FaceMesh(max_num_faces=1,
                                        refine_landmarks=True,
                                        min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5) as face_mesh:
            results = face_mesh.process(frame)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            frame.flags.writeable = True
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks: 
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_contours_style())
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style())
            return frame
                
    # Tracking the bounding box via median to "smooth out"
    def median_tracked_point(self, tracked_point):        
        length = len(tracked_point)        
        
        if length == 0:
            return None

        xleft = []
        ytop = []
        xright = []
        ybot = []

        for element in tracked_point:        
            xleft.append(element[1])
            ytop.append(element[2])
            xright.append(element[3])
            ybot.append(element[4])

        # Sort to get the median
        xleft.sort()
        ytop.sort()
        xright.sort()
        ybot.sort()

        # Assume odd length
        median_index = int(length / 2) + 1 

        return xleft[median_index], ytop[median_index], xright[median_index], ybot[median_index]
                
    def mp_face_mesh_crop(self, frame):
        with self.mp_face_mesh.FaceMesh(max_num_faces=1,
                                        refine_landmarks=True,
                                        min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5) as face_mesh:
            results = face_mesh.process(frame)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            frame.flags.writeable = False
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:                    
                    # CODE FROM https://github.com/google/mediapipe/issues/1737
                    # mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
                    h, w, c = frame.shape
                    cx_min=  w
                    cy_min = h
                    cx_max = cy_max= 0
                    for id, lm in enumerate(face_landmarks.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        if cx<cx_min:
                            cx_min=cx
                        if cy<cy_min:
                            cy_min=cy
                        if cx>cx_max:
                            cx_max=cx
                        if cy>cy_max:
                            cy_max=cy

                    padding_x = int((cx_max - cx_min) * 0.05)
                    padding_y = int((cy_max - cy_min) * 0.05)   

                    xleft = cx_min - padding_x
                    ytop = cy_min - padding_y
                    xright = cx_max + padding_x
                    ybot = cy_max + padding_y

                    # cv2.rectangle(frame, 
                    #             (xleft, ytop), 
                    #             (xright, ybot),
                    #             (255, 0, 0), 2)
                    
                    # Tracking Algorithm: MEDIAN
                    # Tuple (detected, xleft, ytop, xright, ybot)
                    # tracker = ( True,
                    #             cx_min - padding_x,
                    #             cy_min - padding_y,
                    #             cx_max + padding_x,
                    #             cy_max + padding_y )
                    
                    # if tracker[0] == True:
                    #     self.tracked_point.append(tracker)            
                    #     if len(self.tracked_point) > 11:
                    #         self.tracked_point.pop(0)    
                    
                    
                    # (xleft, ytop, xright, ybot) = self.median_tracked_point(self.tracked_point) 
                    # END MEDIAN                    

                    # 1:1 Aspect ratio Bounding Box Without Centroid
                    # x_delta = xright - xleft
                    # y_delta = ybot - ytop

                    # # Use y_delta as the longest
                    # diff = round((y_delta - x_delta) / 2)

                    # xleft = xleft - diff
                    # xright = xright + diff


                    # cv2.rectangle(frame, 
                    #             (xleft, ytop), 
                    #             (xright, ybot),
                    #             (255, 255, 0), 2)
                    # return frame
                    
                    # END 1:1 Aspect ratio Bounding Box Without Centroid 
                            

                    # 1:1 Aspect Ratio Bounding Box

                    # Find the distance and take the longest
                    x_delta = xright - xleft
                    y_delta = ybot - ytop

                    bb_length = max(x_delta, y_delta)

                    # Centroid generation
                    x_center = int(round((xright + xleft) / 2))
                    y_center = int(round((ytop + ybot) / 2))

                    # Centroid Tracker
                    self.centroid_tracker.append((x_center, y_center))

                    xleft = x_center - int(round(0.5 * bb_length))
                    ytop = y_center - int(round(0.5 * bb_length))
                    xright = x_center + int(round(0.5 * bb_length))
                    ybot = y_center + int(round(0.5 * bb_length))
                    
                    # cv2.rectangle(frame, 
                    #             (xleft, ytop), 
                    #             (xright, ybot),
                    #             (0, 0, 255), 2)
                    # return frame
                
                    # END 1:1 Aspect Ratio Bounding Box

                    
                    try:
                        resized = frame[ytop:ybot,
                                        xleft:xright]
                        resized = cv2.resize(resized, (224, 224), interpolation=cv2.INTER_AREA)
                        return resized
                    except Exception:
                        pass
                    # END CODE
            return self.blank_frame
        
    def mp_face_mesh_crop_preprocessing(self, frame, padding=True):
        with self.mp_face_mesh.FaceMesh(max_num_faces=1,
                                        refine_landmarks=True,
                                        min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5) as face_mesh:
            results = face_mesh.process(frame)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            frame.flags.writeable = False
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:                    
                    # CODE FROM https://github.com/google/mediapipe/issues/1737
                    # mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
                    h, w, c = frame.shape
                    cx_min=  w
                    cy_min = h
                    cx_max = cy_max= 0
                    for id, lm in enumerate(face_landmarks.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        if cx<cx_min:
                            cx_min=cx
                        if cy<cy_min:
                            cy_min=cy
                        if cx>cx_max:
                            cx_max=cx
                        if cy>cy_max:
                            cy_max=cy

                    if padding:
                        padding_x = int((cx_max - cx_min) * 0.05)
                        padding_y = int((cy_max - cy_min) * 0.05)  
                    else:
                        padding_x = 0
                        padding_y = 0 
                     

                    xleft = cx_min - padding_x
                    ytop = cy_min - padding_y
                    xright = cx_max + padding_x
                    ybot = cy_max + padding_y

                    # cv2.rectangle(frame, 
                    #             (xleft, ytop), 
                    #             (xright, ybot),
                    #             (255, 0, 0), 2)
                    
                    # Tracking Algorithm: MEDIAN
                    # Tuple (detected, xleft, ytop, xright, ybot)
                    # tracker = ( True,
                    #             cx_min - padding_x,
                    #             cy_min - padding_y,
                    #             cx_max + padding_x,
                    #             cy_max + padding_y )
                    
                    # if tracker[0] == True:
                    #     self.tracked_point.append(tracker)            
                    #     if len(self.tracked_point) > 11:
                    #         self.tracked_point.pop(0)    
                    
                    
                    # (xleft, ytop, xright, ybot) = self.median_tracked_point(self.tracked_point) 
                    # END MEDIAN                    

                    # 1:1 Aspect ratio Bounding Box Without Centroid
                    # x_delta = xright - xleft
                    # y_delta = ybot - ytop

                    # # Use y_delta as the longest
                    # diff = round((y_delta - x_delta) / 2)

                    # xleft = xleft - diff
                    # xright = xright + diff


                    # cv2.rectangle(frame, 
                    #             (xleft, ytop), 
                    #             (xright, ybot),
                    #             (255, 255, 0), 2)
                    # return frame
                    
                    # END 1:1 Aspect ratio Bounding Box Without Centroid 
                            

                    # 1:1 Aspect Ratio Bounding Box

                    # Find the distance and take the longest
                    x_delta = xright - xleft
                    y_delta = ybot - ytop

                    bb_length = max(x_delta, y_delta)

                    # Centroid generation
                    x_center = int(round((xright + xleft) / 2))
                    y_center = int(round((ytop + ybot) / 2))

                    # Centroid Tracker
                    self.centroid_tracker.append((x_center, y_center))

                    xleft = x_center - int(round(0.5 * bb_length))
                    ytop = y_center - int(round(0.5 * bb_length))
                    xright = x_center + int(round(0.5 * bb_length))
                    ybot = y_center + int(round(0.5 * bb_length))
                    
                    # cv2.rectangle(frame, 
                    #             (xleft, ytop), 
                    #             (xright, ybot),
                    #             (0, 0, 255), 2)
                    # return frame
                
                    # END 1:1 Aspect Ratio Bounding Box                    
                    return (True, xleft, ytop, xright, ybot)
                    # END CODE
            return (False, xleft, ytop, xright, ybot)
                
    def mp_face_mesh_crop_fixed_bb_nose_tip(self, frame):
        with self.mp_face_mesh.FaceMesh(max_num_faces=1,
                                        refine_landmarks=True,
                                        min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5) as face_mesh:
            results = face_mesh.process(frame)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            frame.flags.writeable = False
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:                                      
                    # CODE FROM https://github.com/google/mediapipe/issues/1737
                    # mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
                    h, w, c = frame.shape
                    cx_min=  w
                    cy_min = h
                    cx_max = cy_max= 0

                    itr = 0
                    nose_tip_x = 0
                    nose_tip_y = 0
                    for id, lm in enumerate(face_landmarks.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        if cx<cx_min:
                            cx_min=cx
                        if cy<cy_min:
                            cy_min=cy
                        if cx>cx_max:
                            cx_max=cx
                        if cy>cy_max:
                            cy_max=cy

                        # Find the nose tip: 
                        if itr == 1:
                            # cv2.circle(frame, (cx, cy), 2, (255, 0, 0), 1)
                            nose_tip_x = cx
                            nose_tip_y = cy
                        itr += 1

                    padding_x = int((cx_max - cx_min) * 0.05)
                    padding_y = int((cy_max - cy_min) * 0.05)   

                    xleft = cx_min - padding_x
                    ytop = cy_min - padding_y
                    xright = cx_max + padding_x
                    ybot = cy_max + padding_y

                    if self.init_length_bb == False:
                        self.init_length_bb = True
                        x_delta = xright - xleft
                        y_delta = ybot - ytop

                        # 1:1 Aspect Ratio BB
                        self.bb_length = max(x_delta, y_delta)                        
                    
                    xleft = nose_tip_x - int(round(0.5 * self.bb_length))
                    ytop = nose_tip_y - int(round(0.5 * self.bb_length))
                    xright = nose_tip_x + int(round(0.5 * self.bb_length))
                    ybot = nose_tip_y + int(round(0.5 * self.bb_length))
                    


                    cv2.rectangle(frame, 
                                (xleft, ytop), 
                                (xright, ybot),
                                (0, 0, 255), 2)
                    # return frame

                    try:
                        resized = frame[ytop:ybot,
                                        xleft:xright]
                        resized = cv2.resize(resized, (224, 224), interpolation=cv2.INTER_AREA)
                        return resized
                    except Exception:
                        pass
                    # END CODE
            return self.blank_frame
                
    def mp_face_mesh_crop_fixed_bb_centroid(self, frame):
        with self.mp_face_mesh.FaceMesh(max_num_faces=1,
                                        refine_landmarks=True,
                                        min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5) as face_mesh:
            results = face_mesh.process(frame)            
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            frame.flags.writeable = False
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:                                      
                    # CODE FROM https://github.com/google/mediapipe/issues/1737
                    # mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
                    h, w, c = frame.shape
                    cx_min=  w
                    cy_min = h
                    cx_max = cy_max= 0
                    
                    for id, lm in enumerate(face_landmarks.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        if cx<cx_min:
                            cx_min=cx
                        if cy<cy_min:
                            cy_min=cy
                        if cx>cx_max:
                            cx_max=cx
                        if cy>cy_max:
                            cy_max=cy

                    padding_x = int((cx_max - cx_min) * 0.05)
                    padding_y = int((cy_max - cy_min) * 0.05)   

                    xleft = cx_min - padding_x
                    ytop = cy_min - padding_y
                    xright = cx_max + padding_x
                    ybot = cy_max + padding_y

                    # 1:1 Aspect Ratio Bounding Box

                    # Find the distance and take the longest AND FIXED IT
                    if self.init_length_bb == False:
                        self.init_length_bb = True
                        x_delta = xright - xleft
                        y_delta = ybot - ytop

                        # 1:1 Aspect Ratio BB
                        self.bb_length = max(x_delta, y_delta)   

                    # Centroid generation
                    x_center = int(round((xright + xleft) / 2))
                    y_center = int(round((ytop + ybot) / 2))

                    # Centroid Tracker
                    self.centroid_tracker.append((x_center, y_center))
 
                    xleft = x_center - int(round(0.5 * self.bb_length))
                    ytop = y_center - int(round(0.5 * self.bb_length))
                    xright = x_center + int(round(0.5 * self.bb_length))
                    ybot = y_center + int(round(0.5 * self.bb_length))
                    # END 1:1 Aspect Ratio Bounding Box
    
                    # cv2.rectangle(frame, 
                    #             (xleft, ytop), 
                    #             (xright, ybot),
                    #             (0, 0, 255), 2)
                    # return frame

                    try:
                        resized = frame[ytop:ybot,
                                        xleft:xright]
                        resized = cv2.resize(resized, (224, 224), interpolation=cv2.INTER_AREA)
                        return resized
                    except Exception:
                        pass
                    # END CODE
            return self.blank_frame
        
    def mp_face_mesh_crop_fixed_bb_centroid_preprocessing(self, frame):
        with self.mp_face_mesh.FaceMesh(max_num_faces=1,
                                        refine_landmarks=True,
                                        min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5) as face_mesh:
            results = face_mesh.process(frame)            
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            frame.flags.writeable = False
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:                                      
                    # CODE FROM https://github.com/google/mediapipe/issues/1737
                    # mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
                    h, w, c = frame.shape
                    cx_min=  w
                    cy_min = h
                    cx_max = cy_max= 0
                    
                    for id, lm in enumerate(face_landmarks.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        if cx<cx_min:
                            cx_min=cx
                        if cy<cy_min:
                            cy_min=cy
                        if cx>cx_max:
                            cx_max=cx
                        if cy>cy_max:
                            cy_max=cy

                    padding_x = int((cx_max - cx_min) * 0.05)
                    padding_y = int((cy_max - cy_min) * 0.05)   

                    xleft = cx_min - padding_x
                    ytop = cy_min - padding_y
                    xright = cx_max + padding_x
                    ybot = cy_max + padding_y

                    # 1:1 Aspect Ratio Bounding Box

                    # Find the distance and take the longest AND FIXED IT
                    if self.init_length_bb == False:
                        self.init_length_bb = True
                        x_delta = xright - xleft
                        y_delta = ybot - ytop

                        # 1:1 Aspect Ratio BB
                        self.bb_length = max(x_delta, y_delta)   

                    # Centroid generation
                    x_center = int(round((xright + xleft) / 2))
                    y_center = int(round((ytop + ybot) / 2))

                    # Centroid Tracker
                    self.centroid_tracker.append((x_center, y_center))
 
                    xleft = x_center - int(round(0.5 * self.bb_length))
                    ytop = y_center - int(round(0.5 * self.bb_length))
                    xright = x_center + int(round(0.5 * self.bb_length))
                    ybot = y_center + int(round(0.5 * self.bb_length))
                    # END 1:1 Aspect Ratio Bounding Box
    
                    return (True, xleft, ytop, xright, ybot)
                    # END CODE
            return (False, xleft, ytop, xright, ybot)
        

# if __name__ == "__main__":
#     localize = Localize()

#     img = cv2.imread("D:\CASME\CASMEII\CASME2-RAW-NoVideo\CASME2-RAW\sub01\EP02_01f\img1.jpg")

#     detected, xleft, ytop, xright, ybot = localize.mp_face_mesh_crop_preprocessing(img)

    
#     cropped = img[ytop:ybot, xleft:xright]

#     cv2.imshow("Padding 10", cropped)

#     detected, xleft, ytop, xright, ybot = localize.mp_face_mesh_crop_preprocessing(img, padding=False)

#     cropped = img[ytop:ybot, xleft:xright]
#     cv2.imshow("No Padding", cropped)
#     cv2.waitKey(0)

            

            

