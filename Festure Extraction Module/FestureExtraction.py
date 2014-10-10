"""
Feature Extraction module
:Dev data: 2014.09.02
:J.m.yu @ GIST - Ph.D Candidate

Feature list
1. HOG(Histogram of oriented gradients)
2. HOF(Histogram of Optical flow)

"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

from removing_noise import MorpologyClose
from removing_noise import motion_trajectories_stack_feature, d3_plotting

from skimage.feature import hog
from skimage import data, color, exposure
import sys
import os
import PIL
import cv2.cv as cv

from mayavi import mlab
from mayavi.mlab import surf


cap_video = cv2.VideoCapture('F:/Dataset/hand gesture/subject1_dep/K_person_1_backgroud_1_illumination_1_pose_1_actionType_9.avi')
fourcc = cv2.cv.CV_FOURCC('M','J','P','G')
processed_video = cv2.VideoWriter('F:/Dataset/hand gesture/subject1_dep/Result/K_person_1_backgroud_1_illumination_1_pose_1_actionType_9.avi',fourcc,10,(320,240))

#MorpologyClose(cap_video,processed_video,'F:/Dataset/hand gesture/subject1_dep/Result/hog.csv')
print 'starting extraction of stack feature'
stack_feature = motion_trajectories_stack_feature(cap_video,processed_video)
print 'stacking is fin'
d3_plotting(stack_feature)


