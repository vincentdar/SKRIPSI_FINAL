# Raw Images preprocessing
# Convert videos to images
# Note: only raw videos not subject splitted
import cv2
import numpy as np
import mediapipe as mp
import math
import os
from localize import Localize
from typing import List, Mapping, Optional, Tuple, Union

# file_ls = ["G:/Dataset Skripsi Batch 1/data/16 Agustus/Kelompok Evelyne Indrawati, Jonatan Waluyo, Yolanda Teresya, Alison Trixie. 16 Agustus .mp4"]
# target_ls = ["G:/Dataset Skripsi Batch 1 Images/data/16 Agustus"]

file_ls = [ #"D:/Dataset Skripsi Batch 1/data/10 September/Close Up/Benedict Hebert.mp4",
#            "D:/Dataset Skripsi Batch 1/data/10 September/Close Up/Chyntia Dewi.mp4",
#            "D:/Dataset Skripsi Batch 1/data/10 September/Close Up/Claudia Maria Kiona Doy Ximenes.mp4",
#            "D:/Dataset Skripsi Batch 1/data/10 September/Close Up/Clea Alvina.mp4",
#            "D:/Dataset Skripsi Batch 1/data/10 September/Close Up/Franklin hedwig aprilio tan.mp4",
        #    "D:/Dataset Skripsi Batch 1/data/10 September/Close Up/Genesis Sheraline Putri Indonesia.mp4",
        #    "D:/Dataset Skripsi Batch 1/data/10 September/Close Up/Jeffson Jonathan.mp4",
        #    "D:/Dataset Skripsi Batch 1/data/10 September/Close Up/Jenica Tendean.mp4",
        #    "D:/Dataset Skripsi Batch 1/data/10 September/Close Up/Jeniffer Princess.mp4",
        #    "D:/Dataset Skripsi Batch 1/data/10 September/Close Up/Jessica Natalia.mp4",
        #    "D:/Dataset Skripsi Batch 1/data/10 September/Close Up/Joice Nathania Andrey.mp4",
        #    "D:/Dataset Skripsi Batch 1/data/10 September/Close Up/Kevin Christian Efendy.mp4",
        #    "D:/Dataset Skripsi Batch 1/data/10 September/Close Up/Kevin Jonathan.mp4",
        #    "D:/Dataset Skripsi Batch 1/data/10 September/Close Up/Kezia Ivana.mp4",
        #    "D:/Dataset Skripsi Batch 1/data/10 September/Close Up/Meliana Cristine Dewi Winata.mp4",
        #    "D:/Dataset Skripsi Batch 1/data/10 September/Close Up/Preysella Anggreani Parinding Gala.mp4",
        #    "D:/Dataset Skripsi Batch 1/data/10 September/Close Up/Reynard Christian.mp4",
        #    "D:/Dataset Skripsi Batch 1/data/10 September/Close Up/Ryan Christian Han.mp4"
           ]

file_ls = [
    # "D:/Dataset Skripsi Batch 1/data/6 September/AUDRIANA FELICIA_CHRISTY NAOMI PANGGABEAN_ADHE PUTRI ANGGRAENI Close Up.mp4",
    # "D:/Dataset Skripsi Batch 1/data/6 September/GENESIS SHERALINE PUTRI SUSANTO_ELIKA JESSLYN MAGDALENA_VIVALDI LANDE KIDINGALLO_ANGELA TIFFANY Close Up.mp4",
    # "D:/Dataset Skripsi Batch 1/data/6 September/RYAN CHRISTIAN HAN_MICHELLE MARCELIA UMBAS_JESSICA RACHEL SUNARTA_VANESSA FRANZELINE Close Up.mp4",
    # "D:/Dataset Skripsi Batch 1/data/6 September/SELENA THEANA ALFIANDY_REYNARD CHRISTIAN_JOICE NATHANIA ANDREY_YESHUA IMANUEL Close up.mp4",
]

# file_ls = [
#     "D:/Dataset Skripsi Batch 1/data/30 Agustus/YEFTA SONY_EMANUELLA EUGENIA_ASHLEY ANGELICA_BRANDON IMANUEL_Close Up.mp4"    
# ]

# file_ls = [
#     "D:/Dataset Skripsi Batch 1/data/23 Agustus/CHYNTIA DEWI__JENICA TENDEAN_KEVIN JONATHAN_JEFFSON JONATHAN_Close Up_.mp4",
#     "D:/Dataset Skripsi Batch 1/data/23 Agustus/VABITHA CHRISTABELLE_MELIANA CRISTINE_NICHOLAS ALEXANDER_Close Up.mp4",
# ]

file_ls = [
    "D:/Dataset Skripsi Batch 3 25 fps/S30.mov",
    "D:/Dataset Skripsi Batch 3 25 fps/S31.mov",
    "D:/Dataset Skripsi Batch 3 25 fps/S32.mov",
    "D:/Dataset Skripsi Batch 3 25 fps/S33.mov",
    "D:/Dataset Skripsi Batch 3 25 fps/S34.mov",
    "D:/Dataset Skripsi Batch 3 25 fps/S35.mov",
    "D:/Dataset Skripsi Batch 3 25 fps/S36.mov",
    "D:/Dataset Skripsi Batch 3 25 fps/S37.mov",

    # "D:/Dataset Skripsi Batch 3 25 fps/S38.mov",
    # "D:/Dataset Skripsi Batch 3 25 fps/S39.mov",
    # "D:/Dataset Skripsi Batch 3 25 fps/S40.mov",
    # "D:/Dataset Skripsi Batch 3 25 fps/S41.mov",
    # "D:/Dataset Skripsi Batch 3 25 fps/S42.mov",
    # "D:/Dataset Skripsi Batch 3 25 fps/S43.mov",
    # "D:/Dataset Skripsi Batch 3 25 fps/S44.mov",
    # "D:/Dataset Skripsi Batch 3 25 fps/S45.mov"
]
target_ls = ["D:/Dataset Skripsi Batch 3 Images V2"]

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
localization_algorithm = Localize() 

def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None

  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

# Haar Cascade based Face Localization
def localizeFace_haar(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_gray) 
    faceROI = np.zeros((224, 224, 3))

    for (x,y,w,h) in faces:              
        # xywh based
        faceROI = frame[y:y+h,x:x+w]
        faceROI = cv2.resize(faceROI, (224, 224), interpolation=cv2.INTER_CUBIC)

        # 224 x 224 center based
        # center = (x + w//2, y + h//2)          
        # faceROI = frame_gray[center[1] - 112:center[1] + 112, center[0] - 112: center[0] + 112] 
               
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        break
     
    return faceROI

# Mediapipe based Face Localization
# return_image set to TRUE to return image, else the boxes coord will be returned
def localizeFace_mediapipe(image, return_image=True):
    height, width, _ = image.shape    
    faceROI = np.zeros((224, 224, 3))
    xleft, ytop, xright, ybot = 0, 0, 0, 0
    detected = False

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection: 
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections:
            detected = True
            for detection in results.detections:                
                location = detection.location_data
                relative_bounding_box = location.relative_bounding_box
                relative_bounding_box = location.relative_bounding_box
                rect_start_point = _normalized_to_pixel_coordinates(
                    relative_bounding_box.xmin, relative_bounding_box.ymin, width,
                    height)
                rect_end_point = _normalized_to_pixel_coordinates(
                    relative_bounding_box.xmin + relative_bounding_box.width,
                    relative_bounding_box.ymin + relative_bounding_box.height, width,
                    height)

                # Anticipate Broken Bounding Box i.e out of images coordinate
                if rect_start_point == None or rect_end_point == None:
                    detected = False
                    break   

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
def average_tracked_point(tracked_point):
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
    #  int(xright / length) -  int(xleft / length), int(ybot / length) -  int(ytop / length))
    return int(xleft / length), int(ytop / length), int(xright / length), int(ybot / length)



def read_video(filename):    
    frame_count = 0
    cap = cv2.VideoCapture(filename)   

    # Manipulate capture the folder name and remove .mp4 text
    target_folder_name = filename.split('/')[-1][:-4]    

    target_full_path = os.path.join(target_ls[0], target_folder_name) 
    if not os.path.exists(target_full_path):
        os.mkdir(os.path.join(target_ls[0], target_folder_name))
       
    # Set up file to tracked failed (not detected) frame
    target_failed_frame = target_folder_name + ".txt"
    failed_full_path = os.path.join(target_ls[0], target_failed_frame) 

    if os.path.exists(failed_full_path):
        os.remove(failed_full_path)
        failed_file_tracker = open(failed_full_path, 'w')
    else:
        failed_file_tracker = open(failed_full_path, 'w')
    

    # Tracking Algo
    tracked_point = []
    blank_frame = np.zeros((224, 224, 3))

    # Check if stream opened successfully
    if (cap.isOpened() == False): 
        print("Error opening video stream or file")
        
    while(cap.isOpened()):   
        ret, frame = cap.read()
        faceROI = np.zeros((224, 224, 3))
        if ret == True:          
            # With tracking OFF
            # localized_frame = localizeFace_haar(frame)
            # localized_frame = localizeFace_mediapipe(frame)
            # cv2.imshow('Localized Frame', localized_frame) 

            # With tracking ON
            # tracker = localizeFace_mediapipe(frame, False)

            # if tracker[0] == True:
            #     tracked_point.append(tracker)            
            #     if len(tracked_point) > 1:
            #         tracked_point.pop(0)
            
            # avg_bounding_box = average_tracked_point(tracked_point) 

            # Showing Bounding Box
            # cv2.rectangle(frame, (avg_bounding_box[0], avg_bounding_box[1]), (avg_bounding_box[2], avg_bounding_box[3]), (0, 255, 0), 2)
            # cv2.imshow('Localized Frame', frame)   

            
                
            try:            
                detected, xleft, ytop, xright, ybot = localization_algorithm.mp_face_mesh_crop_preprocessing(frame)                                
                faceROI = frame[ytop:ybot, xleft:xright]
                faceROI = cv2.resize(faceROI, (224, 224), interpolation=cv2.INTER_AREA)
                # print("Face detected : Frame", frame_count)
                cv2.imshow('Localized Frame', faceROI)                 

                # write frame to folder 
                written_filename = "img" + str(frame_count).zfill(5) + ".jpg"
                final_written_filename = os.path.join(target_full_path, written_filename)            
                cv2.imwrite(final_written_filename, faceROI)     # save frame as JPEG file

            except Exception as e:
                # print("Face NOT detected : Frame", frame_count)
                cv2.imshow('Localized Frame', blank_frame)                
                  
                # write frame to folder 
                written_filename = "img" + str(frame_count).zfill(5) + ".jpg"
                final_written_filename = os.path.join(target_full_path, written_filename)            
                cv2.imwrite(final_written_filename, blank_frame)     # save frame as JPEG file  
                failed_file_tracker.write(written_filename + "\n")                                               
            

            frame_count += 1
            
            # Press Q on keyboard to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break                
        else: 
            break
        
    cap.release()        
    cv2.destroyAllWindows()
    # Close the file
    failed_file_tracker.close()
    print("File:", filename, "Frame Count:", frame_count)

if __name__ == "__main__":
    for filename in file_ls:
        read_video(filename)
    
    print("Preprocessing DONE")
    print("Result on Path:", target_ls[0])