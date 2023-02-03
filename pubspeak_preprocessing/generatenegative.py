import os
import random
import shutil

positive_path = "D:\Dataset Skripsi Batch 1 Self Label\Positive\\10 September\S4"
negative_path = r"D:\Dataset Skripsi Batch 1 Self Label\Negative\S4"
rawpic_path = "D:\Dataset Skripsi Batch 1 Images Negative\\10 September\Clea Alvina"

print("Generating Negative Dataset")

 
for video in os.listdir(positive_path):     
    video_path = os.path.join(positive_path, video)
    
    clip_length = len(os.listdir(video_path))
    print(video_path, "having", clip_length, "files")
     
    # Make directory here ====================================
    video_dir = os.path.join(negative_path, video)     
    os.mkdir(video_dir)    

           
    rawpic_images = sorted([ int(f[3:-4]) for f in os.listdir(rawpic_path) if f != "Thumbs.db"])    
    rejected = True
    while rejected:
        try:
            start_random = random.randint(rawpic_images[1], rawpic_images[-1])
            end_random = start_random + clip_length
            print("start_random          :", start_random)
            print("end_random            :", end_random)
            
            index = rawpic_images.index(start_random)
            selected_raw = rawpic_images[index: index + clip_length]

            itr = start_random
            count = 0
            for selected in selected_raw:                    
                if itr == selected:                    
                    count += 1       
                itr += 1                
            if count == clip_length:
                rejected = False
        except:
            print("Wrong Random - Iterating Over")  
    
    print("GENERATING DATA DONE")
    images = [ os.path.join(rawpic_path, "img" + str(f).zfill(5) + ".jpg") for f in selected_raw]
    for image in images:
        # Copy image
        shutil.copy2(image, video_dir)    

            
print("GENERATING DATA COMPLETE")
