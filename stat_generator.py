from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def multiclass_report(gt, pred, video):
    print("ACCURACY REPORT")
    print("Accuracy:", accuracy_score(np.array(gt), np.array(pred)))
    print("CLASSIFICATION REPORT")
    print(classification_report(np.array(gt), np.array(pred), 
                                labels=[0, 1, 2, 3, 4, 5],
                                zero_division=0))
    confusionHeatmapCategorical(gt, pred, video)

def multiclass_report_10(gt, pred, video):
    print("ACCURACY REPORT")
    print("Accuracy:", accuracy_score(np.array(gt), np.array(pred)))
    print("CLASSIFICATION REPORT")
    print(classification_report(np.array(gt), np.array(pred), 
                                labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                zero_division=0))
    confusionHeatmapCategorical(gt, pred, video)
    

def binary_report(gt, pred, video):        
    print("ACCURACY REPORT")
    print("Accuracy:", accuracy_score(np.array(gt), np.array(pred)))
    print("CLASSIFICATION REPORT")
    print(classification_report(np.array(gt), np.array(pred), 
                                labels=[0, 1],
                                zero_division=0))
    confusionHeatmapBinary(gt, pred, video)
   

def confusionHeatmapBinary(gt, pred, video):    
    conf_matrix = confusion_matrix(np.array(gt), np.array(pred), labels=[0, 1])
    plt.figure(figsize=(10,8), dpi=75)
    # Scale up the size of all text
    sns.set(font_scale = 1)

    ax = sns.heatmap(conf_matrix, annot=True, fmt='d', )
    ax.set_xlabel("Predicted", fontsize=14, labelpad=20)
    ax.xaxis.set_ticklabels(['0', '1'])

    ax.set_ylabel("Actual", fontsize=14, labelpad=20)
    ax.yaxis.set_ticklabels(['0', '1'])

    ax.set_title(video + " Binary Confusion Matrix", fontsize=14, pad=20)

    plt.show()

def confusionHeatmapCategorical(gt, pred, video):    
    conf_matrix = confusion_matrix(np.array(gt), np.array(pred), labels=[0, 1, 2, 3, 4, 5])
    plt.figure(figsize=(10,8), dpi=75)
    # Scale up the size of all text
    sns.set(font_scale = 1)

    ax = sns.heatmap(conf_matrix, annot=True, fmt='d', )
    ax.set_xlabel("Predicted", fontsize=14, labelpad=20)
    ax.xaxis.set_ticklabels(["Unknown", "Showing Emotions", "Blank Face", "Reading", "Head Tilt", "Occlusion"])

    ax.set_ylabel("Actual", fontsize=14, labelpad=20)
    ax.yaxis.set_ticklabels(["Unknown", "Showing Emotions", "Blank Face", "Reading", "Head Tilt", "Occlusion"])

    ax.set_title(video + " Categorical Confusion Matrix", fontsize=14, pad=20)

    plt.show()

def confusionHeatmapCategorical_10(gt, pred, video):    
    conf_matrix = confusion_matrix(np.array(gt), np.array(pred), labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.figure(figsize=(10,8), dpi=75)
    # Scale up the size of all text
    sns.set(font_scale = 1)

    ax = sns.heatmap(conf_matrix, annot=True, fmt='d', )
    ax.set_xlabel("Predicted", fontsize=12, labelpad=10)
    ax.xaxis.set_ticklabels(["Unknown", "Eye\nContact", "Blank\nFace", "Showing\nEmotions", "Reading",
                   "Sudden\nEye\nChange", "Smiling", "Not\nLooking", "Head\nTilt", "Occlusion"])

    ax.set_ylabel("Actual", fontsize=12, labelpad=10)
    ax.yaxis.set_ticklabels(["Unknown", "Eye\nContact", "Blank\nFace", "Showing\nEmotions", "Reading",
                   "Sudden\nEye\nChange", "Smiling", "Not\nLooking", "Head\nTilt", "Occlusion"])

    ax.set_title(video + " Categorical Confusion Matrix", fontsize=14, pad=20)

    plt.show()

def generateReport(df, video):
    if len(df['Class Names'].unique()) > 2:
        multiclass_report(df['GT'], df['Label'], video)
    else:
        binary_report(df['GT'], df['Label'], video)


def sliding_window_iou(df):
    if len(df['Class Names'].unique()) > 2:
        sw_df = []
        unique_index = df['Index'].unique()
        for index in unique_index:
            df_index = df[df['Index'] == index]
            diff = 0                
            for _, row, in df_index.iterrows():
                if row['Label'] == row['GT']:
                    diff += 1
                
            tIoU = diff / 12
            max_gt = df_index['GT'].max()
            max_label = df_index['Label'].max()
            if max_label == 0:
                conf = df_index[df_index['Label'] == max_label]['Unknown']
            elif max_label == 1:
                conf = df_index[df_index['Label'] == max_label]['Showing Emotions']
            elif max_label == 2:
                conf = df_index[df_index['Label'] == max_label]['Blank Face']
            elif max_label == 3:
                conf = df_index[df_index['Label'] == max_label]['Reading']
            elif max_label == 4:
                conf = df_index[df_index['Label'] == max_label]['Head Tilt']
            elif max_label == 5:
                conf = df_index[df_index['Label'] == max_label]['Occlusion']

            data = [df_index['Index'].values[0], df_index['Class Names'].values[0], 
                    df_index['Number'].values[0], conf,
                    max_label, max_gt,
                    tIoU]
            sw_df.append(data)

        return pd.DataFrame(sw_df, columns=["Index", "Class Names",
                                            "Number", "Conf",
                                            "Label", "GT",
                                            "tIoU"])
    else:
        sw_df = []
        unique_index = df['Index'].unique()
        for index in unique_index:
            df_index = df[df['Index'] == index]
            diff = 0                
            for _, row, in df_index.iterrows():
                if row['Label'] == row['GT']:
                    diff += 1
                
            tIoU = diff / 12
            max_gt = df_index['GT'].max()
            max_label = df_index['Label'].max() 
            conf = df_index[df_index['Label'] == max_label]['Conf'].max()           

            data = [df_index['Index'].values[0], df_index['Class Names'].values[0], 
                    df_index['Number'].values[0], conf,
                    max_label, max_gt,
                    tIoU]
            sw_df.append(data)

        return pd.DataFrame(sw_df, columns=["Index", "Class Names",
                                            "Number", "Conf",
                                            "Label", "GT",
                                            "tIoU"])
    


def ClassificationReport(df, tIoU_thres=0.5):  
    results = []
    if len(df['Class Names'].unique()) > 2:
        # 6 Class
        results.append([0, 0, 0, 0]) # 0
        results.append([0, 0, 0, 0]) # 1
        results.append([0, 0, 0, 0]) # 2
        results.append([0, 0, 0, 0]) # 3
        results.append([0, 0, 0, 0]) # 4
        results.append([0, 0, 0, 0]) # 5
    else:
        results.append([0, 0, 0, 0]) # 0
        results.append([0, 0, 0, 0]) # 1        
    for _, row in df.iterrows():
        if (row['Label'] == row['GT']) and row['tIoU'] > tIoU_thres:            
            for i, value in enumerate(results):
                value = results[i]    
                if i == row['Label']:
                    # TP
                    tp = value[0]
                    value[0] = tp + 1
                else:
                    tn = value[3]
                    value[3] = tn + 1
        elif (row['Label'] == row['GT']) and row['tIoU'] <= tIoU_thres:  
            for i, value in enumerate(results):                   
                if i == row['Label']:
                    # FP
                    fp = value[1]
                    value[1] = fp + 1
                    # FN
                    fn = value[2]
                    value[2] = fn + 1                                    
                else:
                    tn = value[3]
                    value[3] = tn + 1
        else:
             for i, value in enumerate(results):                   
                if i == row['Label']:
                    # FP
                    fp = value[1]
                    value[1] = fp + 1
                elif i == row['GT']:
                    # FN
                    fn = value[2]
                    value[2] = fn + 1
                else:
                    tn = value[3]
                    value[3] = tn + 1        

        
    return results

def describe_results(results):
    for i, results in enumerate(results):
        if i == 0:
            print("\tTP\tFP\tFN\tTN")
        print("{}\t{}\t{}\t{}\t{}".format(i, results[0], results[1], results[2], results[3]))

def calculate_AP(df):  
    pr = []  
    class_length = 0
    for i in np.arange(0, 1.1, 0.1):
        results = ClassificationReport(df, tIoU_thres=i) 
        # describe_results(results)
        class_length = len(results)       
        # print("tIoU Threshold:", i)        
        for i, result in enumerate(results):            
            precision_divisor = result[0] + result[1]            
            if precision_divisor == 0:
                precision = 0
            else:
                precision = result[0] / precision_divisor

            recall_divisor = result[0] + result[2]
            if recall_divisor == 0:
                recall = 0
            else:
                recall = result[0] / recall_divisor

            # print(i, " : ", precision, recall)
            pr.append([precision, recall])

    ap_scores = []
    for i in range(0, class_length):        
        ap_score = 0
        itr = 0        
        current_precision = 0
        current_recall = 0
        for item in pr[i::class_length]:
            if itr == 0:
                current_precision = item[0]
                current_recall = item[1]
            else:                
                prcsn = max(item[0], current_precision)
                rcll = abs(item[1] - current_recall)                

                ap_score += prcsn * rcll 

                current_precision = item[0]
                current_recall = item[1]
            itr += 1
        print("Class:", i, "AP SCORE:", ap_score)
        ap_scores.append(ap_score)

    return ap_scores

def calculate_mAP(df):
    ap_scores = calculate_AP(df)
    mAP = np.mean(ap_scores)
    print("Mean Average Precision (mAP) based on tIoU:", mAP)
               
                
def plot_segmentation(df):
    frames = df['Number']
    conf = df['Conf']
    label = df['Label']
    gt = df['GT']
        
    diff = []
    negative_diff = []
    for i in range(0, len(label)):
        if label[i] == gt[i]:
            diff.append(0.5)
            negative_diff.append(0)
        else:
            diff.append(0)
            negative_diff.append(0.5)       
    

    plt.figure(figsize=(10,8), dpi=75)            
    
    
    plt.fill_between(frames, gt, color='red')
    plt.fill_between(frames, conf, color='blue')
    plt.fill_between(frames, diff, color='green')
    plt.fill_between(frames, negative_diff, color='yellow')
    plt.title("Segmentation")
    plt.show()

def plot_segmentation_prediction(df):
    frames = df['Number']    
    label = df['Label']    
        
    # diff = []
    # negative_diff = []
    # for i in range(0, len(label)):
    #     if label[i] == 1:
    #         diff.append(1)
    #         negative_diff.append(0)
    #     else:
    #         diff.append(0)
    #         negative_diff.append(1)       
    

    plt.figure(figsize=(10,8), dpi=75)            
    
    
    plt.fill_between(frames, label, color='blue')
    # plt.fill_between(frames, negative_diff, color='red')
    plt.title("Segmentation")
    plt.show()

def plot_angles(df, video):
    frame = df['Number']
    x = df['X']
    y = df['Y']
    z = df['Z']
    
    

    plt.figure(figsize=(10,8), dpi=75)            
    plt.plot(frame, x, color='blue')
    plt.plot(frame, y, color='yellow')
    plt.plot(frame, z, color='green')    
    plt.title(str(video) + " angles")
    plt.legend()
    plt.show()




if __name__ == "__main__":   
    # print("LBP CASMEII")
    # filename = "evaluation\\lbp_casmeii_cut.csv"        
    # video = filename.split('\\')[-1]
    # lbp_df = pd.read_csv(filename)           
    # binary_report(lbp_df['GT'], lbp_df['Label'], video) 
     

    # print("CNN LSTM CASMEII")
    # filename = "evaluation\\cnnlstm_casmeii_cut.csv"        
    # video = filename.split('\\')[-1]
    # df = pd.read_csv(filename)          
    # binary_report(df['GT'], df['Label'], video)

    
    # filename = "reports\\binary\local_mobilenet_cnnlstm_unfreezelast20_newpubspeak25042023_10_epoch\S50_binary.csv"
    
    # # filename = "reports\\local_mobilenet_cnnlstm_unfreezelast20_newpubspeak21032023_multiclass_merged_10_epoch\S50_categorical.csv"
    # video = filename.split('\\')[-1]
    # df = pd.read_csv(filename)  
    # plot_segmentation_prediction(df)  

    # print("Sliding Window Based Evaluation (12 strides)")
    # sw_df = sliding_window_iou(df)      
    # calculate_mAP(sw_df)               
    # generateReport(sw_df, video)

    # print("Per Frame Based Evaluation")    
    # generateReport(df, video)

    # filename = "evaluation\\pubspeak21032023_lbp_summary.csv"    
    # df = pd.read_csv(filename)         
    # binary_report(df['GT'], df['Label'], "LBP-X2")    

    # Buku
    filename = r"D:\CodeProject2\SKRIPSI_FINAL\reports\PerFramePrediction\local_mobilenet_cnnlstm_unfreezelast20_newpubspeak21032023_10_epoch\testdata.csv"
    df = pd.read_csv(filename)      
    print("Per Frame Based Evaluation")    
    generateReport(df, "Testing Set (S4, S50) 1 stride")

    # filename = r"reports\\BUKU\\pyramid\\S3_binary.csv"
    # df = pd.read_csv(filename)       
    # print("Per Frame Based Evaluation")    
    # generateReport(df, "S3")

    # Independen
    # filename = r"reports\\BUKU\categorical\local_mobilenet_cnnlstm_unfreezelast20_newpubspeak21032023_multiclass_merged_10_epoch\s4_categorical.csv"
    
    # filename = "reports\\local_mobilenet_cnnlstm_unfreezelast20_newpubspeak21032023_multiclass_merged_10_epoch\S50_categorical.csv"
    # video = filename.split('\\')[-1]
    # df = pd.read_csv(filename)  
    # # plot_segmentation_prediction(df)  

    # # print("Sliding Window Based Evaluation (12 strides)")
    # # sw_df = sliding_window_iou(df)      
    # # calculate_mAP(sw_df)               
    # # generateReport(sw_df, "Testing Set (S4, S50)")

    # print("Per Frame Based Evaluation")    
    # generateReport(df, video)

    # filename = r"reports\\BUKU\categorical\local_mobilenet_cnnlstm_unfreezelast20_newpubspeak21032023_multiclass_merged_10_epoch\s50_categorical.csv"
    
    # # filename = "reports\\local_mobilenet_cnnlstm_unfreezelast20_newpubspeak21032023_multiclass_merged_10_epoch\S50_categorical.csv"
    # video = filename.split('\\')[-1]
    # df = pd.read_csv(filename)  
    # # plot_segmentation_prediction(df)  

    # # print("Sliding Window Based Evaluation (12 strides)")
    # # sw_df = sliding_window_iou(df)      
    # # calculate_mAP(sw_df)               
    # # generateReport(sw_df, "Testing Set (S4, S50)")

    # print("Per Frame Based Evaluation")    
    # generateReport(df, video)

    # # 10 Class
    # filename = r"reports\\BUKU\Model22\testdata.csv"
    # df = pd.read_csv(filename)  
    # multiclass_report_10(df['GT'], df['Label'], "testdata.csv")
    # confusionHeatmapCategorical_10(df['GT'], df['Label'], "testdata.csv")


    # Buku HPE
    # filename = r"reports\\BUKU\HPE\Radian\S4_angles.csv"
    # df = pd.read_csv(filename)  
    # video = filename.split('\\')[-1]  

    # plot_angles(df, video)   







    

    
    
