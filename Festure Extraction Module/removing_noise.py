from pyparsing import delimitedList

__author__ = 'Schmidtz'
import numpy as np
import matplotlib.pyplot as plt
import cv2

from skimage.feature import hog
from skimage import data, color, exposure
import sys
import os
import PIL
import cv2.cv as cv

import csv

def hog_extraction(frame):
    """
    HOG(Histogram of Oriented Gradients) extraction and saving data module
    :param frame: video frame of input video
    :return: hog dat(array) and hog_image(shown for people)
    """
    gray_frame = color.rgb2gray(frame)
    hog_dat,hog_image = hog(gray_frame,orientations=16,pixels_per_cell=(40,40),cells_per_block=(2,2),visualise=True)
    return hog_dat,hog_image







def MorpologyClose(cap_stream,morp_stream,HOG_stream,csv_file_name):
    """
    Morphology processing(open operation) and Extracting HOG, saving data(HOG array) instance
    :param cap_stream: captured video file which is non-processed noise removal procecdure
    :param morp_stream: file structure for morphology processing conclusion
    :return morp_stream : outliser removal video file
    """
    wfile = open(csv_file_name,"wb")
    csv_writer = csv.writer(wfile,delimiter='\t',quoting=csv.QUOTE_ALL)
    kernel = np.ones((5,5),np.uint8) #kernal function for open operation
    while(cap_stream.isOpened()):
        ret, frame = cap_stream.read()
        if ret==True:
            morp_frame = cv2.morphologyEx(frame,cv2.MORPH_CLOSE,kernel) #remove outlier using open operation
            morp_stream.write(morp_frame)
            hog_dat,hog_image = hog_extraction(morp_frame)
            print hog_dat.shape
            csv_writer.writerow(hog_dat)

            ##showing hog image session
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
            cv2.imshow('frame',hog_image_rescaled)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap_stream.release()
    morp_stream.release()
    cv2.destroyAllWindows()
    print 'MorphyProcess fin'

