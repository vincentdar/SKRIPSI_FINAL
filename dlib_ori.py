import cv2
import dlib
import numpy as np

# load the face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture("D:/CobaCut/Timeline 1.mov")
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (960, 540))
        output_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        
        # detect the faces
        rects = detector(gray)

        # go through the face bounding boxes
        for rect in rects:
            # extract the coordinates of the bounding box
            x1 = rect.left()
            y1 = rect.top()
            x2 = rect.right()
            y2 = rect.bottom()
            cropped_frame = output_frame[y1:y2, x1:x2]
            cropped_frame = cv2.resize(cropped_frame, (224, 224))
            cv2.imshow("Cropped Alignment", cropped_frame)
            
            break
        # Press Q on keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break            
    else: 
        break

                