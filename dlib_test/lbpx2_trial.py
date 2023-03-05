import core.lbplib
import core.localize
import cv2

cap = cv2.VideoCapture("D:/CASME/CASME(2)/rawvideo/rawvideo/s15/15_0101disgustingteeth.avi")
localizationAlgorithm = core.localize.Localize()
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        localized_frame = localizationAlgorithm.mp_localize_crop(frame)   
        cv2.imshow("224 x 224 Localized Frame", localized_frame) 

        lbp_frame = core.lbplib.spatialLBP(cv2.cvtColor(localized_frame, cv2.COLOR_RGB2GRAY), False)        
        cv2.imshow("222 x 222 LBP Frame", lbp_frame) 
        

        
        # Press Q on keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break            
    else: 
        break