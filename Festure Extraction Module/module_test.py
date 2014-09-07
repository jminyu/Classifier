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
import csv


plt.figure(1)
count = 0
with open('D:/Data/final data/dev20val20/Result/hog.csv','rb') as f:
    reader =  csv.reader(f,delimiter='\t')
    for row in reader:
        for element in row:
            print float(element)
            plt.plot(count,float(element),'b--')
            count = count+1
        plt.show()
        count = 0






