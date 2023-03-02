from localize import Localize
import os
import cv2

raw_path = r"D:\CASME\CASMEII\CASME2-RAW-Localize\CASME2-RAW"
localization_algorithm = Localize()


data = []
# Iterate through subject
for subject in os.listdir(raw_path):
    subject_dir = os.path.join(raw_path, subject)    

    # Iterate through video
    for video in os.listdir(subject_dir):
        video_dir = os.path.join(subject_dir, video)
                     
        for image in os.listdir(video_dir):
            image_dir = os.path.join(video_dir, image)
            
            image = cv2.imread(image_dir)
            detected, xleft, ytop, xright, ybot = localization_algorithm.mp_face_mesh_crop_preprocessing(image)
            data.append(subject + "," + video + "," + str(xleft) + "," + str(ytop) + "," + str(xright) + "," + str(ybot) + "\n")

            break

        



# f = open('casmeii_bounding_box.csv', 'w')
# for line in data:
#     print(line)
#     f.write(line)
# f.close()
