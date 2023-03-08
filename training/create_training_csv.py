import os

negative_data_path = r"D:\Dataset Skripsi Batch Final Label\Negative"
positive_data_path = r"D:\Dataset Skripsi Batch Final Label\Positive"


def create_training_csv(data_path, label, writer):
    # Policy: Discard frames that don't meet the 12 sliding window requirement
    subjects = os.listdir(data_path)
    for subject in subjects:
        subject_data_path = os.path.join(data_path, subject)
        detections = os.listdir(subject_data_path)
        for detection in detections:
            detection_data_path = os.path.join(subject_data_path, detection)
            numbers = sorted([int(f[3:-4]) for f in os.listdir(detection_data_path)])
            counter = 0
            list_filename = []
            for number in numbers:                                
                filename = "img" + str(number).zfill(5) + ".jpg"
                path_filename = os.path.join(detection_data_path, filename)
                list_filename.append(path_filename)

            sliding_window_12 = len(list_filename) // 12

            for index in range(0, sliding_window_12):
                for filename in list_filename[12 * index:12 * (index + 1)]:
                    writer.write(filename)
                    writer.write(',')
                writer.write(str(label))
                writer.write('\n')
                                                                                     

if __name__ == "__main__":
    filewriter = open('training_pubspeak.csv',  'w')
    filewriter.write("1,2,3,4,5,6,7,8,9,10,11,12,Label\n")
    create_training_csv(positive_data_path, 1, filewriter)
    create_training_csv(negative_data_path, 0, filewriter)
    filewriter.close()
    

