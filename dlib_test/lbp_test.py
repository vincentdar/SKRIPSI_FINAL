import lbplib

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# def spatialLBP(img):
    
#     lbp_image = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
#     zero_padded = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
#     zero_padded[1:-1, 1:-1] = img    
    
#     for i in range(1, zero_padded.shape[0] - 1, 1):
#         for j in range(1, zero_padded.shape[1] - 1, 1):
#             threshold_cell = zero_padded[i, j]
                     
#             # 0 1 2
#             # 3 X 4
#             # 5 6 7
#             encoding0 = zero_padded[i - 1, j - 1] >= threshold_cell
#             encoding1 = zero_padded[i - 1, j] >= threshold_cell
#             encoding2 = zero_padded[i - 1, j + 1] >= threshold_cell

#             encoding3 = zero_padded[i, j - 1] >= threshold_cell
#             encoding4 = zero_padded[i, j + 1] >= threshold_cell

#             encoding5 = zero_padded[i + 1, j - 1] >= threshold_cell
#             encoding6 = zero_padded[i + 1, j] >= threshold_cell
#             encoding7 = zero_padded[i + 1, j + 1] >= threshold_cell

#             # Clockwise Implementation
#             encoded = str(int(encoding0)) + str(int(encoding1)) + str(int(encoding2)) + str(int(encoding4)) + str(int(encoding7)) + str(int(encoding6)) + str(int(encoding5)) + str(int(encoding3))

#             binary = encoded.encode('ascii')
#             # print(encoded, int(binary, 2))
#             lbp_image[i - 1, j - 1] = int(binary, 2)

                
                

#     return lbp_image

def chiSquareDistance(x, y):     
    distance = 0
    for i in range(len(x)):
        top = (x[i] - y[i]) ** 2
        bottom = (x[i] + y[i])  
        if bottom == 0:
            value = 0   
        else:
            value = top / bottom

        distance += value
    return distance



img = cv2.imread('73927.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('brick.jpg', cv2.IMREAD_GRAYSCALE)

img = cv2.resize(img, (224, 224))
img2 = cv2.resize(img2, (224, 224))


st = time.time()
lbp = lbplib.spatialLBP(img, False)
lbp2 = lbplib.spatialLBP(img2, False)
print(lbp.shape)

# lbp = lbplib.py_lbp(img)
# lbp2 = lbplib.py_lbp(img2)
et = time.time()

elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

# Broken
# lbp = lbplib.tf_lbp(img)
# lbp2 = lbplib.tf_lbp(img2)

cv2.imshow('image', img)
cv2.imshow('lbp', lbp)

cv2.imshow('image', img2)
cv2.imshow('lbp', lbp2)


histogram, bin_edges = np.histogram(lbp, bins=256, range=(0, 255))
histogram2, bin_edges2 = np.histogram(lbp2, bins=256, range=(0, 255))

distance = chiSquareDistance(histogram, histogram2)
print("Chi-Square Distance:", distance)


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.set_title("Grayscale Histogram IMG 1")
ax1.set_xlabel("grayscale value")
ax1.set_ylabel("pixel count")
ax1.set_xlim([0, 256])  # <- named arguments do not work here
ax1.plot(bin_edges[0:-1], histogram)  # <- or here

ax2.set_title("Grayscale Histogram IMG 2")
ax2.set_xlabel("grayscale value")
ax2.set_ylabel("pixel count")
ax2.set_xlim([0, 256])  # <- named arguments do not work here
ax2.plot(bin_edges2[0:-1], histogram2)  # <- or here
plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()