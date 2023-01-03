import cv2
import dlib
import numpy as np

# load the face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture("D:/Dataset Skripsi Batch 1/data/10 September/Close Up/Kevin Christian Efendy.mp4")
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
            # apply the shape predictor to the face ROI
            shape = predictor(gray, rect)    

            
            # Get coord         
            left_eye_x = shape.part(39).x
            left_eye_y = shape.part(39).y
            cv2.circle(frame, (left_eye_x, left_eye_y), 4, (255, 0, 0), -1)
            right_eye_x = shape.part(42).x
            right_eye_y = shape.part(42).y
            cv2.circle(frame, (right_eye_x, right_eye_y), 4, (255, 0, 0), -1)
            nose_tip_x = shape.part(33).x
            nose_tip_y = shape.part(33).y
            cv2.circle(frame, (nose_tip_x, nose_tip_y), 4, (255, 0, 0), -1)
            cv2.line(frame, (left_eye_x, left_eye_y), (right_eye_x, right_eye_y), (0, 0, 255), 2)

            # Angle
            dY = right_eye_y - left_eye_y
            dX = right_eye_x - left_eye_x
            angle = np.degrees(np.arctan2(dY, dX))
            # Scaling
            dist = np.sqrt((dX ** 2) + (dY ** 2))            
            scale = dist / dist

            # Get Rotation Matrix
            eyesCenter = ((right_eye_x + left_eye_x) // 2, (right_eye_y + left_eye_y) // 2)
            M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)


            # Warp Affine Transformation
            output_frame = cv2.warpAffine(output_frame, M, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_CUBIC) 

            # Re Predict to find the new coordinate
            output_grey = cv2.cvtColor(output_frame, cv2.COLOR_BGR2GRAY) 
            try:
                rects = detector(output_grey)
                shape = predictor(output_grey, rects[0])    


                left_eye_x = shape.part(39).x
                left_eye_y = shape.part(39).y
                # cv2.circle(output_frame, (left_eye_x, left_eye_y), 4, (0, 0, 255), -1)
                right_eye_x = shape.part(42).x
                right_eye_y = shape.part(42).y
                # cv2.circle(output_frame, (right_eye_x, right_eye_y), 4, (0, 0, 255), -1)
                nose_tip_x = shape.part(33).x
                nose_tip_y = shape.part(33).y
                # cv2.circle(output_frame, (nose_tip_x, nose_tip_y), 4, (0, 0, 255), -1)
                # cv2.line(output_frame, (left_eye_x, left_eye_y), (right_eye_x, right_eye_y), (0, 0, 255), 2)   

                eyesCenter = ((right_eye_x + left_eye_x) // 2, (right_eye_y + left_eye_y) // 2)
                
                # Generate a bounding box            
                x1 = eyesCenter[0] - 120
                y1 = eyesCenter[1] - 80
                x2 = eyesCenter[0] + 120
                y2 = eyesCenter[1] + 160

                # cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  
                    
            
                cropped_frame = output_frame[y1:y2, x1:x2]
                cv2.imshow("Cropped Alignment", cropped_frame)
            except Exception as e:
                cv2.imshow("Cropped Alignment", np.zeros((224, 224)))
            break

        # Press Q on keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break            
    else: 
        break


        
# cv2.imshow("Image", image)
# cv2.waitKey(0)