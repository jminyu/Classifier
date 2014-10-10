__author__ = 'Schmidtz'

import sys
import os
import PIL
import cv2
import cv2.cv as cv
from etc_module import *
import numpy as np
from numpy import random
import matplotlib.pyplot as plt

import csv

import scipy as sp
from skimage.feature import hog
from skimage import data, color, exposure
from mayavi import mlab
from mayavi.mlab import surf




def hog_extraction(frame):
    """
    HOG(Histogram of Oriented Gradients) extraction and saving data module
    :param frame: video frame of input video
    :return: hog dat(array) and hog_image(shown for people)
    """
    gray_frame = color.rgb2gray(frame)
    hog_dat,hog_image = hog(gray_frame,orientations=16,pixels_per_cell=(40,40),cells_per_block= (2,2),visualise=True)
    return hog_dat,hog_image



def motion_trajectories_stack_feature_image(cap_stream,morp_stream):
    kernel = np.ones((5,5),np.uint8) #kernal function for open operation
    stack_feature = np.zeros((240,320),dtype=float);
    frame_t1 = np.zeros((240,320),dtype=float)
    start = 0;
    if cap_stream.isOpened():
        ret, frame = cap_stream.read()
        morp_frame = cv2.morphologyEx(frame,cv2.MORPH_CLOSE,kernel) #remove outlier using open operation
        morp_stream.write(morp_frame)
        while(True):
            frame_t1 = color.rgb2gray(morp_frame)

            if ret==True:
                #if a pixel have a different value to previous frame then it occur
                ret, frame = cap_stream.read()
                if ret==True:
                    morp_frame = cv2.morphologyEx(frame,cv2.MORPH_CLOSE,kernel) #remove outlier using open operation
                    morp_stream.write(morp_frame)
                    frame_t2 = color.rgb2gray(morp_frame)
                    stack_feature = stack_feature + (frame_t2-frame_t1)*0.1
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break

            else:
                break

    return stack_feature



def motion_trajectories_stack_feature_HOG(cap_stream,morp_stream):
    kernel = np.ones((5,5),np.uint8) #kernal function for open operation
    stack_feature = np.zeros((240,320),dtype=float);
    while(cap_stream.isOpened()):
        ret, frame = cap_stream.read()
        if ret==True:
            morp_frame = cv2.morphologyEx(frame,cv2.MORPH_CLOSE,kernel) #remove outlier using open operation
            morp_stream.write(morp_frame)
            hog_dat,hog_image = hog_extraction(morp_frame)
            stack_feature = stack_feature + hog_image

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    return stack_feature


'''
def test_surf():
    """Test surf on regularly spaced co-ordinates like MayaVi."""
    def f(x, y):
        sin, cos = np.sin, np.cos
        return sin(x + y) + sin(2 * x - y) + cos(3 * x + 4 * y)

    x, y = np.mgrid[-7.:7.05:0.1, -5.:5.05:0.05]
    s = surf(x, y, f)
    #cs = contour_surf(x, y, f, contour_z=0)
    return s

test_surf()
'''


def d3_plotting(stack_feature):

    data_array = np.array(stack_feature)
    fig = plt.figure()
    #x_data,y_data = np.meshgrid(np.arange(data_array.shape[1]),np.arange(data_array.shape[0]))

    x_data = np.linspace(0,319,320)
    y_data = np.linspace(0,239,240)
    stack_feature = 100*stack_feature

    #def z(x,y):
    #    return stack_feature[x,y]
    #X,Y = np.mgrid(x_data,y_data)
    s = surf(x_data,y_data,stack_feature)
    mlab.show()






def MorpologyClose(cap_stream,morp_stream,csv_file_name):
    """
    Morphology processing(open operation) and Extracting HOG, saving data(HOG array) instance
    :param cap_stream: captured video file which is non-processed noise removal procecdure
    :param morp_stream: file structure for morphology processing conclusion
    :return morp_stream : outliser removal video file
    """
    wfile = open(csv_file_name,"wb")
    csv_writer = csv.writer(wfile,delimiter='\t',quoting=csv.QUOTE_NONE)
    kernel = np.ones((5,5),np.uint8) #kernal function for open operation
    while(cap_stream.isOpened()):
        ret, frame = cap_stream.read()
        if ret==True:
            morp_frame = cv2.morphologyEx(frame,cv2.MORPH_CLOSE,kernel) #remove outlier using open operation
            morp_stream.write(morp_frame)
            hog_dat,hog_image = hog_extraction(morp_frame)
            #print hog_dat.shape
            csv_writer.writerow(hog_dat)
            #hog_bar(hog_dat)



            ##showing hog image session
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
            plt.figure()
            cv2.imshow('frame',hog_image_rescaled)

            #hog_image_rescaled = 240*320 image

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap_stream.release()
    morp_stream.release()
    cv2.destroyAllWindows()
    print 'MorphyProcess fin'

