import numpy as np
import matplotlib.pyplot as plt
import cv2

from skimage.feature import hog
from skimage import data, color, exposure
import sys
import os
import PIL
import cv2.cv as cv


cap_video = cv2.VideoCapture('D:/Data/final data/dev20val20/devel-1-20_valid-1-20/devel01/K_1.avi')

fourcc = cv2.cv.CV_FOURCC('M','J','P','G')
processed_video = cv2.VideoWriter('D:/Data/final data/dev20val20/Result/remove_outliers.avi',fourcc,10,(320,240))
HOG_stream = cv2.VideoWriter('D:/Data/final data/dev20val20/Result/HOG_stream.avi',fourcc,10,(40,40))


frame_count = 0
kernel = np.ones((5,5),np.uint8)
while(cap_video.isOpened()):
    ret, frame = cap_video.read()
    if ret==True:
        morphology_frame = cv2.morphologyEx(frame,cv2.MORPH_CLOSE,kernel) #remove outlier using open operation
        _gray_temp = color.rgb2gray(morphology_frame)
        hog_frame = hog(_gray_temp,orientations=16,pixels_per_cell=(8,8),cells_per_block=(1,1),visualise=True)

        #HOG_stream.write(hog_frame)
        plt.imshow(hog_frame)
        plt.show()
        processed_video.write(morphology_frame)
        cv2.imshow('frame',morphology_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


HOG_stream.release()
cap_video.release()
processed_video.release()
cv2.destroyAllWindows()
