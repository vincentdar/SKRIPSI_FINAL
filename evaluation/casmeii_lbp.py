import pandas as pd
import numpy as np


file = open('lbp_casmeii_edisi2.csv', 'w')
data = "Subject,Folder,Number,Label,GT\n"
file.write(data)
df = pd.read_excel("lbp_result_ver2.xlsx", "lbp_result_ver2")

current_subject = ""
current_folder = ""
current_gt_onset_offset = []
current_sp_onset_offset = []
current_spotted = 0
current_eval = ""
current_total = 0
for index, row in df.iterrows():
    subject = row['Subject']
    folder = row['Folder']
    onsetGT = row['OnsetGT']
    offsetGT = row['OffsetGT']
    sp_onset = row['SP_onset']
    sp_offset = row['SP_offset'] 
    spotted = row['Spotted']       
    eval = row['Evaluation']
    total = row['Total']

    
    # if eval == "TN":
    #     continue

    
    if folder != current_folder:        
        for i in range(0, current_total):   
            label = 0           
            for onset, offset in current_sp_onset_offset:         
                if i >= onset and i <= offset:
                    label = 1 

            gt = 0
            for onset, offset in current_gt_onset_offset:         
                if i >= onset and i <= offset:
                    gt = 1       
            file.write("{},{},{},{},{}\n".format(current_subject,current_folder,i,label,gt))

        current_subject = subject
        current_folder = folder
        current_sp_onset_offset = []
        current_gt_onset_offset = []
        current_eval = eval
        current_total = total

    current_sp_onset_offset.append([sp_onset, sp_offset])
    current_gt_onset_offset.append([onsetGT, offsetGT])


    
    

# file.close()

