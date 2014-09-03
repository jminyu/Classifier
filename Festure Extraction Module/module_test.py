import numpy as np
import matplotlib.pyplot as plt
import cv2

from removing_noise import MorpologyClose

from skimage.feature import hog
from skimage import data, color, exposure
import sys
import os
import PIL
import cv2.cv as cv







cap_video = cv2.VideoCapture('D:/Data/final data/dev20val20/devel-1-20_valid-1-20/devel01/K_1.avi')

fourcc = cv2.cv.CV_FOURCC('M','J','P','G')
processed_video = cv2.VideoWriter('D:/Data/final data/dev20val20/Result/remove_outliers.avi',fourcc,10,(320,240))
HOG_stream = cv2.VideoWriter('D:/Data/final data/dev20val20/Result/HOG_stream.avi',fourcc,10,(320,240))

MorpologyClose(cap_video,processed_video,HOG_stream,'D:/Data/final data/dev20val20/Result/hog.csv')


