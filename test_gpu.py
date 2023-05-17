import os
import tensorflow as tf
import cv2
import numpy as np
from keras import backend as K
os.environ['CUDA_VISIBLE_DEVICES'] ="0"
print("Number of GPU Available:", len(tf.config.list_physical_devices('GPU')))
