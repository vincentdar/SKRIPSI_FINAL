import pandas as pd
import os

class Report:
    def __init__(self):
        self.destination_folder = "reports"
        self.predictions = [] 

        self.categorical_class_names = ["Unknown", "Showing Emotions", "Blank Face", "Reading", "Head Tilt", "Occlusion"]  

        # Evaluation        
        training_df = pd.read_excel("D:\CodeProject2\SKRIPSI_FINAL\pubspeak_label_BARU.xlsx", sheet_name="Training") 
        testing_df = pd.read_excel("D:\CodeProject2\SKRIPSI_FINAL\pubspeak_label_BARU.xlsx", sheet_name="Testing") 
        self.df = pd.concat([training_df, testing_df], ignore_index=True)
        print(list(self.df['Subject'].unique()))

    def addPredictions_Categorical(self, class_names, start_frame, end_frame, conf, label):
        # Tuple format (class_names, Number, conf, label)

         
        sentence = (class_names[label], start_frame, end_frame, conf, label)
        self.predictions.append(sentence)

    def addPredictions_Binary(self, class_names, start_frame, end_frame, conf, label):
        # Tuple format (class_names, Number, conf, label)        
         
        sentence = (class_names[label], start_frame, end_frame, conf, label)
        self.predictions.append(sentence)

    def writeToCSV(self, subject): 
        have_evaluation = False
        if subject in list(self.df['Subject'].unique()):
            have_evaluation = True

        f = open(os.path.join(self.destination_folder, subject + "_focal.csv"), "w")
        f.write("Index" + "," + "Class Names" + "," + "Conf" + "," + "Number" + "," + "Label" + "," + "GT" + "\n")
        iterate = 0
        for index, sentence in enumerate(self.predictions):
            for i in range(0, 12):
                class_name = sentence[0]
                start_frame = sentence[1]
                end_frame = sentence[2]
                conf = sentence[3]
                label = sentence[4]

                if have_evaluation:
                    gt = self.getGTLabel(subject, iterate)
                else:
                    gt = -1
                f.write(str(index) + "," + class_name +  "," + str(conf) + "," + str(iterate) 
                        + "," + str(label) + "," + str(gt) + "\n")
                iterate += 1
        f.close()

    def getGTLabel(self, subject, number):
        subject_df = self.df[self.df['Subject'] == subject]
        
        for index, row in subject_df.iterrows():
            start = row['Start']
            end = row['End']
            if number >= start and number <= end:
                justification = row['Justification']
                y = self.categorical_class_names.index(justification)
                return y

        return -1 # Unknown


    def clearPredictions(self):
        self.predictions = []


