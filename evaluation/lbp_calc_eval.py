import pandas as pd

df = pd.read_csv("lbp_result.csv")

f = open("lbp_result_summary.csv", 'w')
current_folder = None
current_image = None
current_cdf = 0
current_max_cdf = None
current_mean_cdf = None
current_onset = None
current_offset = None
f.write("{},{},{},{},{},{},{},{}\n".format("Folder", "Image", "CDF",
                                           "MaxCDF", "MeanCDF",
                                           "Onset", "Offset","Detected"))
for index, row in df.iterrows():
    folder = row['Folder']
    image = row['Image']
    cdf = row['CDF']
    max_cdf = row['MaxCDF']
    mean_cdf = row['MeanCDF']
    onset = row['Onset']
    offset = row['Offset']      

    if current_folder == None:
        current_folder = folder
        current_image = image
        current_cdf = cdf
        current_max_cdf = max_cdf
        current_mean_cdf = mean_cdf
        current_onset = onset
        current_offset = offset

    if current_folder != folder:
        detected = current_image[3:-4]
        f.write("{},{},{},{},{},{},{},{}\n".format(current_folder, current_image, current_cdf,
                                           current_max_cdf, current_mean_cdf,
                                           current_onset, current_offset, detected))
        current_folder = folder
        current_image = image
        current_cdf = cdf
        current_max_cdf = max_cdf
        current_mean_cdf = mean_cdf
        current_onset = onset
        current_offset = offset
        
    if current_cdf < cdf:
        current_cdf = cdf
        current_image = image

    

    

f.close()
