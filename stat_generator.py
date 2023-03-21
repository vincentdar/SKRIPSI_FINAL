from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # df = pd.read_csv(
    # "reports/local_mobilenet_cnnlstm_unfreezelast20_newpubspeak15032023_multiclass_merged_10_epoch/S50.csv")
    df = pd.read_csv(
    "reports/S50.csv")
    pred = df['Label']
    gt = df['GT']    
    conf_matrix = confusion_matrix(np.array(gt), np.array(pred), labels=[0, 1, 2, 3, 4, 5])
    # Change figure size and increase dpi for better resolution
    plt.figure(figsize=(8,6), dpi=100)
    # Scale up the size of all text
    sns.set(font_scale = 1.1)

    # Plot Confusion Matrix using Seaborn heatmap()
    # Parameters:
    # first param - confusion matrix in array format   
    # annot = True: show the numbers in each heatmap cell
    # fmt = 'd': show numbers as integers. 
    ax = sns.heatmap(conf_matrix, annot=True, fmt='d', )

    # set x-axis label and ticks. 
    ax.set_xlabel("Predicted", fontsize=14, labelpad=20)
    ax.xaxis.set_ticklabels(['0', '1', '2', '3', '4', '5'])

    # set y-axis label and ticks
    ax.set_ylabel("Actual", fontsize=14, labelpad=20)
    ax.yaxis.set_ticklabels(['0', '1', '2', '3', '4', '5'])

    # set plot title
    ax.set_title("Confusion Matrix", fontsize=14, pad=20)

    print("ACCURACY REPORT")
    print("Accuracy:", accuracy_score(np.array(gt), np.array(pred)))
    print("CLASSIFICATION REPORT")
    print(classification_report(np.array(gt), np.array(pred), 
                                labels=[0, 1, 2, 3, 4, 5],
                                zero_division=0))

    plt.show()
    
