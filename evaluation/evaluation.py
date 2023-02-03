import pandas as pd
import numpy as np

class MyEvaluation:
    def __init__(self):
        self.history = []
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        

    def read_label(self, label_filename, subject):
        self.label_filename = label_filename
        self.label = pd.read_csv(self.label_filename)   
        self.label = self.label.loc[self.label["Subject"] == subject].reset_index()

        
           
    def count(self, start_frame, end_frame, prediction): 
        flag_raised = False       
        for idx in self.label.index:
            label_start = self.label.iloc[idx]['Start']
            label_end = self.label.iloc[idx]['End']                         
            # Inside
            if start_frame >= label_start and end_frame <= label_end:                
                flag_raised = True
                iou = end_frame - start_frame
                if prediction == 1:
                    self.tp += 1
                    self.history.append([start_frame, end_frame, "TP", "Inside", label_start, label_end])                    
                else:
                    self.fn += 1         
                    self.history.append([start_frame, end_frame, "FN", "Inside", label_start, label_end])              
            # Left Side
            elif start_frame < label_start and end_frame > label_start:                
                flag_raised = True
                iou = end_frame - label_start                
                if prediction == 1 and iou > 7:
                    self.tp += 1
                    self.history.append([start_frame, end_frame, "TP", "Left Side", label_start, label_end])
                elif prediction == 1 and iou <= 7:
                    self.fp += 1
                    self.history.append([start_frame, end_frame, "FP", "Left Side", label_start, label_end])
                elif prediction == 0 and iou > 7:
                    self.fn += 1
                    self.history.append([start_frame, end_frame, "FN", "Left Side", label_start, label_end]) 
                elif prediction == 0 and iou < 7:
                    self.tn += 1
                    self.history.append([start_frame, end_frame, "TN", "Left Side", label_start, label_end])           
            # Right Side
            elif start_frame < label_end and end_frame > label_end:                
                flag_raised = True
                iou = label_end - start_frame                
                if prediction == 1 and iou > 7:
                    self.tp += 1
                    self.history.append([start_frame, end_frame, "TP", "Right Side", label_start, label_end]) 
                elif prediction == 1 and iou <= 7:
                    self.fp += 1
                    self.history.append([start_frame, end_frame, "FP", "Right Side", label_start, label_end]) 
                elif prediction == 0 and iou > 7:
                    self.fn += 1
                    self.history.append([start_frame, end_frame, "FN", "Right Side", label_start, label_end]) 
                elif prediction == 0 and iou < 7:
                    self.tn += 1
                    self.history.append([start_frame, end_frame, "TN", "Right Side", label_start, label_end]) 
        # Outside    
        if not flag_raised:            
            if prediction == 0:      
                self.tn += 1
                self.history.append([start_frame, end_frame, "TN", "Outside", label_start, label_end]) 
            else:
                self.fp += 1
                self.history.append([start_frame, end_frame, "FP", "Outside", label_start, label_end]) 
              
    def print_total(self):
        print("TP:", self.tp)
        print("FP:", self.fp)
        print("FN:", self.fn)
        print("TN:", self.tn)

    def reset_count(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0        
        self.tn = 0
        self.history = []

    def to_csv(self):
        np.savetxt("evaluation.csv", 
           self.history,
           delimiter =", ", 
           fmt ='% s')
            
            


    # def find_nearest(self, value):   
    #     pass     
    #     # idx = self.label.iloc[(self.label['Start'] - value).abs().argsort()[:1]].index
    #     # start_label = self.label.iloc[idx]['Start']
    #     # end_label = self.label.iloc[idx]['End']
    #     # return idx