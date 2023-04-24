import pandas as pd
import numpy as np

file = open('pubspeak21032023_lbp_summary_ver2.csv', 'w')
data = "Subject,Number,CDF\n"
file.write(data)

df = pd.read_csv("lbp_pubspeak21032022_ver2.csv")
unique = df['Subject'].unique()

for item in unique:
    subject_df = df[df['Subject'] == item]
    total = subject_df['Total'].iloc[0]
    number = np.zeros((total,))       
    
    print("Processing", item)
    for i in range(0, number.shape[0]):
        highest_cdf = 0
        for index, row in subject_df.iterrows():
            if i >= row["Onset"] and i <= row['Offset']:
                if row['CDF'] >= row['MeanCDF']:
                    if row['CDF'] >= highest_cdf:
                        highest_cdf = 1
            
        data = str(item) + "," + str(i) + "," + str(highest_cdf) + "\n"
        file.write(data)


file.close()
