#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Source: https://github.com/dwday/lbp_tensorflow_python/blob/main/lib/lbplib.py
"""
LBP implementations using Python and Tensorflow
py_lbp : lbp using Pyton 
tf_lbp : lbp using TensorFlow
for details:
AKGÜN, D. "A TensorFlow implementation of Local Binary Patterns Transform." 
MANAS Journal of Engineering 9.1: 15-21, 2021, doi.org/10.51354/mjen.822630
https
"""

"""
It has both github and my own implementation of Local Binary Pattern
"""
import tensorflow as tf
import numpy as np

# Not mine, github
def py_lbp(Im):
    # Local Binary Patterns 
    rows=Im.shape[0]
    cols=Im.shape[1]
    L=np.zeros((rows,cols),dtype='uint8')
    I=np.zeros((rows+2,cols+2),dtype='uint8')
    
    #Zero padding
    I[1:rows+1,1:cols+1]=Im
    
    #Select center pixel
    for i in range(1,rows+1):
        for j in range(1,cols+1):
            #Compute LBP transform
            L[i-1,j-1]=\
            ( I[i-1,j]  >= I[i,j] )*1+\
            ( I[i-1,j+1]>= I[i,j] )*2+\
            ( I[i,j+1]  >= I[i,j] )*4+\
            ( I[i+1,j+1]>= I[i,j] )*8+\
            ( I[i+1,j]  >= I[i,j] )*16+\
            ( I[i+1,j-1]>= I[i,j] )*32+\
            ( I[i,j-1]  >= I[i,j] )*64+\
            ( I[i-1,j-1]>= I[i,j] )*128;  
    
    return L

# Not mine, github but idk itss broken
def tf_lbp(Im):    
    paddings = tf.constant([[0,0],[1, 1], [1, 1]])
    Im=tf.pad(Im, paddings,"CONSTANT")        
    M=Im.shape [1]
    N=Im.shape [2]      
   
    # Select the pixels of masks in the form of matrices
    y00=Im[:,0:M-2, 0:N-2]
    y01=Im[:,0:M-2, 1:N-1]
    y02=Im[:,0:M-2, 2:N  ]
    #     
    y10=Im[:,1:M-1, 0:N-2]
    y11=Im[:,1:M-1, 1:N-1]
    y12=Im[:,1:M-1, 2:N  ]
    #
    y20=Im[:,2:M, 0:N-2]
    y21=Im[:,2:M, 1:N-1]
    y22=Im[:,2:M, 2:N ]  
    
    
    # y00  y01  y02
    # y10  y11  y12
    # y20  y21  y22
    
    # Comparisons 
    # 1 -----------------------------------------        
    g=tf.greater_equal(y01,y11) 
    z=tf.multiply(tf.cast(g,dtype='uint8'), 
                  tf.constant(1,dtype='uint8') )      
    # 2 -----------------------------------------
    g=tf.greater_equal(y02,y11)
    tmp=tf.multiply(tf.cast(g,dtype='uint8'), 
                    tf.constant(2,dtype='uint8') )
    z =tf.add(z,tmp)              
    # 3 -----------------------------------------
    g=tf.greater_equal(y12,y11)
    tmp=tf.multiply(tf.cast(g,dtype='uint8'), 
                    tf.constant(4,dtype='uint8') )
    z =tf.add(z,tmp)
    # 4 -----------------------------------------
    g=tf.greater_equal(y22,y11)
    tmp=tf.multiply(tf.cast(g,dtype='uint8'), 
                    tf.constant(8,dtype='uint8') )
    z =tf.add(z,tmp)  
    # 5 -----------------------------------------
    g=tf.greater_equal(y21,y11)
    tmp=tf.multiply(tf.cast(g,dtype='uint8'), 
                    tf.constant(16,dtype='uint8') )
    z =tf.add(z,tmp)  
    # 6 -----------------------------------------
    g=tf.greater_equal(y20,y11)
    tmp=tf.multiply(tf.cast(g,dtype='uint8'), 
                    tf.constant(32,dtype='uint8') )
    z =tf.add(z,tmp)  
    # 7 -----------------------------------------
    g=tf.greater_equal(y10,y11)
    tmp=tf.multiply(tf.cast(g,dtype='uint8'), 
                    tf.constant(64,dtype='uint8') )
    z =tf.add(z,tmp)  
    # 8 -----------------------------------------
    g=tf.greater_equal(y00,y11)
    tmp=tf.multiply(tf.cast(g,dtype='uint8'), 
                    tf.constant(128,dtype='uint8') )
    z =tf.add(z,tmp)  
    #--------------------------------------------
    return tf.cast(z,dtype=tf.uint8)

def spatialLBP(img, zero_padding=True):
    """
    Spatial LBP with zero padding or false (smaller output)
    """
    if zero_padding:
        lbp_image = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        zero_padded = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
        zero_padded[1:-1, 1:-1] = img    
        
        for i in range(1, zero_padded.shape[0] - 1, 1):
            for j in range(1, zero_padded.shape[1] - 1, 1):
                threshold_cell = zero_padded[i, j]
                        
                # Index visualized
                # 0 1 2
                # 3 X 4
                # 5 6 7
                encoding0 = zero_padded[i - 1, j - 1] >= threshold_cell
                encoding1 = zero_padded[i - 1, j] >= threshold_cell
                encoding2 = zero_padded[i - 1, j + 1] >= threshold_cell

                encoding3 = zero_padded[i, j - 1] >= threshold_cell
                encoding4 = zero_padded[i, j + 1] >= threshold_cell

                encoding5 = zero_padded[i + 1, j - 1] >= threshold_cell
                encoding6 = zero_padded[i + 1, j] >= threshold_cell
                encoding7 = zero_padded[i + 1, j + 1] >= threshold_cell

                # Clockwise Implementation
                encoded = str(int(encoding0)) + str(int(encoding1)) + str(int(encoding2)) + str(int(encoding4)) + str(int(encoding7)) + str(int(encoding6)) + str(int(encoding5)) + str(int(encoding3))

                binary = encoded.encode('ascii')
                # print(encoded, int(binary, 2))
                lbp_image[i - 1, j - 1] = int(binary, 2)
                                
        return lbp_image
    else:
        lbp_image = np.zeros((img.shape[0] - 2, img.shape[1] - 2), dtype=np.uint8)        
        fill_padded = img   

        for i in range(1, fill_padded.shape[0] - 1, 1):
            for j in range(1, fill_padded.shape[1] - 1, 1):
                threshold_cell = fill_padded[i, j]
                        
                # Index visualized
                # 0 1 2
                # 3 X 4
                # 5 6 7
                encoding0 = fill_padded[i - 1, j - 1] >= threshold_cell
                encoding1 = fill_padded[i - 1, j] >= threshold_cell
                encoding2 = fill_padded[i - 1, j + 1] >= threshold_cell

                encoding3 = fill_padded[i, j - 1] >= threshold_cell
                encoding4 = fill_padded[i, j + 1] >= threshold_cell

                encoding5 = fill_padded[i + 1, j - 1] >= threshold_cell
                encoding6 = fill_padded[i + 1, j] >= threshold_cell
                encoding7 = fill_padded[i + 1, j + 1] >= threshold_cell

                # Clockwise Implementation
                encoded = str(int(encoding0)) + str(int(encoding1)) + str(int(encoding2)) + str(int(encoding4)) + str(int(encoding7)) + str(int(encoding6)) + str(int(encoding5)) + str(int(encoding3))

                binary = encoded.encode('ascii')                
                lbp_image[i - 1, j - 1] = int(binary, 2)
                                

        return lbp_image 
    

def LBPonThreeChannel(img):
    blue = img[:,:,0]
    green = img[:,:,1]
    red = img[:,:,2]

    lbp_blue = spatialLBP(blue)
    lbp_green = spatialLBP(green)
    lbp_red = spatialLBP(red)

    # blank = np.zeros((224, 224, 3))
    # blank = [255, 255, 255]
    # blank_blue = blank.copy()
    # blank_blue[:,:,0] = lbp_blue
    # print(blank_blue)
    
    # blank_green = blank.copy()
    # blank_green[:,:,1] = lbp_green
    # print(blank_green)
    
    # blank_red = blank.copy()
    # blank_red[:,:,2] = lbp_red

    return lbp_blue, lbp_green, lbp_red
    


def chiSquareDistance(x, y):     
    """
    Calculate Chi Square distance between 2 histogram
    """
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

def lbp_histogram(lbp0, lbp1):
    """
    Turn Image into Histogram
    """
    histogram0, bin_edges0 = np.histogram(lbp0, bins=256, range=(0, 255))
    histogram1, bin_edges1 = np.histogram(lbp1, bins=256, range=(0, 255))
    return histogram0, histogram1
