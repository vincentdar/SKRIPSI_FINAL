import os
import cv2
import sys
import dlib
import argparse
import numpy as np

#import Face Recognition libraries
import mediapipe as mp

class HeadPoseEstimation:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
                                max_num_faces=1,
                                refine_landmarks=True,
                                min_detection_confidence=0.8,
                                min_tracking_confidence=0.8)

        self.face_model = self.ref3DModel()

    def process(self, img):
        frame = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
        h, w, c = frame.shape
        results = self.face_mesh.process(frame)
        hpe_success = False
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                shape = []
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:                      
                        x, y = int(lm.x * w), int(lm.y * h)
                        shape.append([x, y])

                refImgPts = self.ref2dImagePointsMediapipe(shape)
                
                self.drawFace2DMesh(frame, refImgPts, isClosed=True)  
                self.drawCircle(frame, shape)              
                
                height, width, channels = img.shape
                focalLength = 1 * width
                cameraMatrix = self.cameraMatrix(focalLength, (height / 2, width / 2))
                mdists = np.zeros((4, 1), dtype=np.float64)

                # calculate rotation and translation vector using solvePnP
                success, rotationVector, translationVector = cv2.solvePnP(
                    self.face_model, refImgPts, cameraMatrix, mdists)

                noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
                noseEndPoint2D, jacobian = cv2.projectPoints(
                    noseEndPoints3D, rotationVector, translationVector, cameraMatrix, mdists)
                
                #  draw nose line
                p1 = (int(refImgPts[0, 0]), int(refImgPts[0, 1]))
                p2 = (int(noseEndPoint2D[0, 0, 0]), int(noseEndPoint2D[0, 0, 1]))
                cv2.line(frame, p1, p2, (110, 220, 0),
                        thickness=2, lineType=cv2.LINE_AA)
                
                # calculating euler angles
                rmat, jac = cv2.Rodrigues(rotationVector)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                x = np.arctan2(Qx[2][1], Qx[2][2])
                y = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] * Qy[2][1] ) + (Qy[2][2] * Qy[2][2])))
                z = np.arctan2(Qz[0][0], Qz[1][0])

                # print("Angles[1]", angles[1])
                if angles[1] < -45:
                    GAZE = "Reject"
                elif angles[1] > 45:
                    GAZE = "Reject"
                else:
                    GAZE = "Accept"
                    hpe_success = True
            
            cv2.putText(frame, GAZE, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)
            cv2.putText(frame, "x:" + str(np.round(angles[0], 2)), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "y:" + str(np.round(angles[1], 2)), (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "z:" + str(np.round(angles[2], 2)), (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if hpe_success:            
            return True, frame
        else:
            return False, frame
        

    def drawCircle(self, img, shapes):
        for shape in shapes:
            cv2.circle(img, (int(shape[0]), int(shape[1])), 3, (0, 0, 255), 3)   

    def drawFace2DMesh(self, img, shapes, isClosed=False):
        # Left Eye to Right Eye
        cv2.line(img, (int(shapes[2][0]), int(shapes[2][1])), (int(shapes[3][0]), int(shapes[3][1])), (176, 176, 255), 2)
        # Left Eye to Nose tip
        cv2.line(img, (int(shapes[2][0]), int(shapes[2][1])), (int(shapes[0][0]), int(shapes[0][1])), (176, 176, 255), 2)
        # Right Eye to Nose tip
        cv2.line(img, (int(shapes[3][0]), int(shapes[3][1])), (int(shapes[0][0]), int(shapes[0][1])), (176, 176, 255), 2)
        # Left Eye to Left Mouth
        cv2.line(img, (int(shapes[2][0]), int(shapes[2][1])), (int(shapes[4][0]), int(shapes[4][1])), (176, 176, 255), 2)
        # Right Eye to Right Mouth
        cv2.line(img, (int(shapes[3][0]), int(shapes[3][1])), (int(shapes[5][0]), int(shapes[5][1])), (176, 176, 255), 2)
        # Nose tip to Left Mouth
        cv2.line(img, (int(shapes[0][0]), int(shapes[0][1])), (int(shapes[4][0]), int(shapes[4][1])), (176, 176, 255), 2)
        # Nose tip to Right Mouth
        cv2.line(img, (int(shapes[0][0]), int(shapes[0][1])), (int(shapes[5][0]), int(shapes[5][1])), (176, 176, 255), 2)
        # Chin to Left Mouth
        cv2.line(img, (int(shapes[1][0]), int(shapes[1][1])), (int(shapes[4][0]), int(shapes[4][1])), (176, 176, 255), 2)
        # Chin Eye to Right Mouth
        cv2.line(img, (int(shapes[1][0]), int(shapes[1][1])), (int(shapes[5][0]), int(shapes[5][1])), (176, 176, 255), 2)

    def ref3DModel(self):
        modelPoints = [[0.0, 0.0, 0.0],             # Nose Tip
                    [0.0, -330.0, -65.0],        # Chin
                    [-225.0, 170.0, -135.0],     # Left Eye Corner
                    [225.0, 170.0, -135.0],      # Right Eye Corner
                    [-150.0, -150.0, -125.0],    # Mouth Left Corner
                    [150.0, -150.0, -125.0]]     # Mouth Right Corner
        return np.array(modelPoints, dtype=np.float64)


    def ref2dImagePoints(self, shape):
        imagePoints = [[shape.part(30).x, shape.part(30).y],
                    [shape.part(8).x, shape.part(8).y],
                    [shape.part(36).x, shape.part(36).y],
                    [shape.part(45).x, shape.part(45).y],
                    [shape.part(48).x, shape.part(48).y],
                    [shape.part(54).x, shape.part(54).y]]
        return np.array(imagePoints, dtype=np.float64)

    def ref2dImagePointsMediapipe(self, shape):
        imagePoints = [[shape[0][0], shape[0][1]],
                    [shape[3][0], shape[3][1]],
                    [shape[1][0], shape[1][1]],
                    [shape[4][0], shape[4][1]],
                    [shape[2][0], shape[2][1]],
                    [shape[5][0], shape[5][1]]]
        return np.array(imagePoints, dtype=np.float64)


    def cameraMatrix(self, fl, center):
        mat = [[fl, 1, center[0]],
                        [0, fl, center[1]],
                        [0, 0, 1]]
        return np.array(mat, dtype=np.float)