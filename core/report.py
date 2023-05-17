import pandas as pd
import os

class Report:
    def __init__(self, is_categorical):
        self.destination_folder = "reports"
        self.predictions = [] 

        self.is_categorical = is_categorical
        self.categorical_class_names = ["Unknown", "Showing Emotions", "Blank Face", "Reading", "Head Tilt", "Occlusion"]  

        self.binary_class_names = ["Negative", "Positive"]
        # Evaluation        
        training_df = pd.read_excel("D:\CodeProject2\SKRIPSI_FINAL\pubspeak_label_BARU.xlsx", sheet_name="Training") 
        testing_df = pd.read_excel("D:\CodeProject2\SKRIPSI_FINAL\pubspeak_label_BARU.xlsx", sheet_name="Testing") 
        

        self.df = pd.concat([training_df, testing_df], ignore_index=True)
        self.subjects = list(self.df['Subject'].unique())
        print(self.subjects)

        # self.categorical_class_names = ["Unknown", "Eye Contact", "Blank Face", "Showing Emotions", "Reading",
        #            "Sudden Eye Change", "Smiling", "Not Looking", "Head Tilt", "Occlusion"]
        # training_df = pd.read_excel("D:\CodeProject2\SKRIPSI_FINAL\pubspeak_label_15032023.xlsx", sheet_name="Training") 
        # testing_df = pd.read_excel("D:\CodeProject2\SKRIPSI_FINAL\pubspeak_label_15032023.xlsx", sheet_name="Testing") 
        # self.df = pd.concat([training_df, testing_df], ignore_index=True)
        # self.subjects = list(self.df['Subject'].unique())
        # print(list(self.df['Subject'].unique()))
        

    def addPredictions_Categorical(self, class_names, start_frame, end_frame, conf, label):
        # Tuple format (class_names, Number, "Unknown", "Showing Emotions", "Blank Face", "Reading", "Head Tilt", "Occlusion", label)
        
        sentence = (class_names[label], start_frame, end_frame,
                     conf[0], conf[1], conf[2], conf[3], conf[4], conf[5], 
                     label)
        self.predictions.append(sentence)

    def addPredictions_Binary(self, class_names, start_frame, end_frame, conf, label):
        # Tuple format (class_names, Number, conf, label)        
         
        sentence = (class_names[label], start_frame, end_frame, conf, label)
        self.predictions.append(sentence)

    def addPredictions_Binary_pyramid(self, class_names, start_frame, end_frame, conf, label, pyramid):
        # Tuple format (class_names, Number, conf, label, pyramid)        
         
        sentence = (class_names[label], start_frame, end_frame, conf, label, pyramid)
        self.predictions.append(sentence)

    def writeToCSVBinary_pyramid(self, subject, stride, total_frame): 
        have_evaluation = False
        if subject in list(self.df['Subject'].unique()):
            have_evaluation = True
        
        f = open(os.path.join(self.destination_folder, subject + "_binary.csv"), "w")
        f.write("Pyramid" + "," + "Class Names" + "," + "Pyramid" + "," + "Conf" + "," + "Number" + "," + "Label" + "," + "GT" + "\n")
        for frame in range(0, total_frame):
            best_pyramid_0 = 0
            best_pyramid_1 = 0
            conf_0 = 0.7
            conf_1 = 0.7  
            votes_0 = 0        
            votes_1 = 0        
            for sentence in self.predictions:                
                class_name = sentence[0]
                start_frame = sentence[1]
                end_frame = sentence[2]
                conf = sentence[3]
                label = sentence[4]
                pyramid = sentence[5]
                if frame >= start_frame and frame <= end_frame:
                    if label == 0 and conf < conf_0:
                        conf_0 = conf
                        best_pyramid_0 = pyramid
                        votes_0 += 1
                    if label == 1 and conf > conf_1:
                        conf_1 = conf
                        best_pyramid_1 = pyramid
                        votes_1 += 1
            # Judgment Time
            if votes_1 > votes_0:
                final_vote_label = 1
                final_conf = conf_1
                best_pyramid = best_pyramid_1
            elif votes_1 == votes_0:
                final_vote_label = 1
                final_conf = conf_1
                best_pyramid = best_pyramid_1
            else:
                final_vote_label = 0
                final_conf = conf_0
                best_pyramid = best_pyramid_0

            if have_evaluation:
                gt = self.getGTLabelBinary(subject, frame)
            else:
                gt = -1
            f.write(str(best_pyramid) + "," + class_name +  "," + str(final_conf) + "," + str(frame) 
                    + "," + str(final_vote_label) + "," + str(gt) + "\n")
        f.close()     

    def addAngles(self, x, y, z):
        sentence = (x, y, z)
        self.predictions.append(sentence)

    def writeToAngles(self, subject):
        f = open(os.path.join(self.destination_folder, subject + "_angles.csv"), "w")
        f.write("Number" + "," + "X" + "," + "Y" + "," + "Z" +"\n")
        iterate = 0
        for index, sentence in enumerate(self.predictions):
            x = sentence[0]
            y = sentence[1]
            z = sentence[2]
            f.write(str(iterate) + "," + str(x) + "," + str(y) + "," + str(z) + "\n")
            iterate += 1
        f.close()

    def writeToCSVCategorical(self, subject): 
        have_evaluation = False
        if subject in list(self.df['Subject'].unique()):
            have_evaluation = True

        f = open(os.path.join(self.destination_folder, subject + "_categorical.csv"), "w")
        f.write("Index" + "," + "Class Names" + "," 
                + "Unknown" + ","  + "Showing Emotions" + ","  + "Blank Face" + ","
                + "Reading" + ","  + "Head Tilt" + ","  + "Occlusion" + "," 
                + "Number" + "," + "Label" + "," + "GT" + "\n")
        iterate = 0
        for index, sentence in enumerate(self.predictions):
            for i in range(0, 12):
                class_name = sentence[0]
                start_frame = sentence[1]
                end_frame = sentence[2]
                conf0 = sentence[3]
                conf1 = sentence[4]
                conf2 = sentence[5]
                conf3 = sentence[6]
                conf4 = sentence[7]
                conf5 = sentence[8]
                label = sentence[9]

                if have_evaluation:
                    gt = self.getGTLabel(subject, iterate)
                else:
                    gt = -1
                f.write(str(index) + "," + class_name +  "," + 
                        str(conf0) + "," + str(conf1) + "," + str(conf2) + "," +
                        str(conf3) + "," + str(conf4) + "," + str(conf5) + "," +
                        str(iterate) + "," + str(label) + "," + str(gt) + "\n")
                iterate += 1
        f.close()

    def writeToCSVBinary(self, subject): 
        have_evaluation = False
        if subject in list(self.df['Subject'].unique()):
            have_evaluation = True

        f = open(os.path.join(self.destination_folder, subject + "_binary.csv"), "w")
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
                    gt = self.getGTLabelBinary(subject, iterate)
                else:
                    gt = -1
                f.write(str(index) + "," + class_name +  "," + str(conf) + "," + str(iterate) 
                        + "," + str(label) + "," + str(gt) + "\n")
                iterate += 1
        f.close()     
    

    def writeSummaryCategorical(self, subject):
        # Write to CSV        
        f = open(os.path.join(self.destination_folder, subject + "_categorical.csv"), "w")
        f.write("Index" + "," + "Class Names" + "," + "Start Frame" + "," + "End Frame" 
                + "Unknown" + ","  + "Showing Emotions" + ","  + "Blank Face" + ","
                + "Reading" + ","  + "Head Tilt" + ","  + "Occlusion" + "," 
                + "Label"  + "\n")        
        for index, sentence in enumerate(self.predictions):
                class_name = sentence[0]
                start_frame = sentence[1]
                end_frame = sentence[2]
                conf0 = sentence[3]
                conf1 = sentence[4]
                conf2 = sentence[5]
                conf3 = sentence[6]
                conf4 = sentence[7]
                conf5 = sentence[8]
                label = sentence[9]

                f.write(str(index) + "," + class_name +  "," + 
                        str(start_frame) + "," + str(end_frame) + "," +
                        str(conf0) + "," + str(conf1) + "," + str(conf2) + "," +
                        str(conf3) + "," + str(conf4) + "," + str(conf5) + "," +
                        str(label) + "\n")
                iterate += 1
        f.close()

    def getGTLabel(self, subject, number):
        
        
        if subject in self.subjects:
            subject_df = self.df[self.df['Subject'] == subject]
            for index, row in subject_df.iterrows():
                start = row['Start']
                end = row['End']
                if number >= start and number <= end:
                    justification = row['Justification']
                    y = self.categorical_class_names.index(justification)
                    return y
            return 0 # Unknown
        else:
            return -1

        
    
    def getGTLabelBinary(self, subject, number):                        
        if subject in self.subjects:
            subject_df = self.df[self.df['Subject'] == subject]
            for index, row in subject_df.iterrows():
                start = row['Start']
                end = row['End']
                if number >= start and number <= end:
                    label = row['Label']                
                    return label
            return 0 # Unknown
        else:
            return -1
        
    
    def addPredictions_Binary_perframe(self, class_names, number, conf, label):
        sentence = (class_names[label], number, conf, label)
        self.predictions.append(sentence)


    def writeToCSV(self, subject, stride=12):
        if stride == 1:
            if self.is_categorical:
                # self.writeToCSVCategorical(subject)
                print("Writing per frame categorical IS NOT IMPLEMENTED")
            else:
                self.writeToCSVBinary_perframe(subject)
    
        else:
            if self.is_categorical:
                self.writeToCSVCategorical(subject)                
            else:
                self.writeToCSVBinary(subject)
    

    def writeToCSVBinary_perframe(self, subject): 
        have_evaluation = False
        if subject in list(self.df['Subject'].unique()):
            have_evaluation = True

        f = open(os.path.join(self.destination_folder, subject + "_binary.csv"), "w")
        f.write("Index" + "," + "Class Names" + "," + "Conf" + "," + "Number" + "," + "Label" + "," + "GT" + "\n")
        iterate = 0
        for index, sentence in enumerate(self.predictions):
            class_name = sentence[0]
            number = sentence[1]                
            conf = sentence[3]
            label = sentence[4]

            if have_evaluation:
                gt = self.getGTLabelBinary(subject, iterate)
            else:
                gt = -1
            f.write(str(index) + "," + class_name +  "," + str(conf) + "," + str(iterate) 
                    + "," + str(label) + "," + str(gt) + "\n")
            iterate += 1
        f.close()     
    
    

    def clearPredictions(self):
        self.predictions = []

    


