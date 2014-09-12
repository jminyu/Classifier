__author__ = 'Schmidtz'


import numpy as np
import matplotlib.pyplot as plt

import cv2
import scipy as sp
import csv

from skimage.feature import hog
from skimage import data, color, exposure
import sys
import os
import PIL
import cv2.cv as cv


def hog_bar(hog_dat):
    bar_index = np.size(hog_dat)
    x_axis = sp.arange(bar_index)
    plt.figure()
    plt.bar(x_axis,hog_dat)
    plt.xlabel('hog vector')
    plt.ylabel('cell')
    plt.tight_layout()
    plt.show()



def bar_graph_streaming(csv_file,row_countt):
    figure_count = 5
    sub_count = 0
    with open(csv_file,'rb') as f:
        reader =  csv.reader(f,delimiter='\t')
        for i in range(0,figure_count):
            row_count=0
            plt.figure(i)
            fft, axarr = plt.subplots(5,sharex=True)
            for row in reader:
                size = np.size(row)
                x_axis = sp.arange(size)
                conv_element = np.zeros((size,1))
                print 'processing',i,'th figure',row_count,'th hog data'

                for element in row:
                    conv_element[sub_count,0] = float(element)
                    sub_count = sub_count+1

                axarr[row_count].bar(x_axis,conv_element[:,0])
                row_count = row_count +1
                sub_count = 0
                del conv_element
                if row_count==row_countt:
                    break;

        plt.show()


def _m_bar_graph_streaming(csv_file,row_countt):
    with open(csv_file,'rb') as f:
        reader =  csv.reader(f,delimiter='\t')
        row_count=0
        fft, axarr = plt.subplots(row_countt,sharex=True)
        for row in reader:
            sub_count = 0
            size = np.size(row)
            x_axis = sp.arange(size)
            conv_element = np.zeros((size,1))
            print 'processing',row_count,'th hog data'

            for element in row:
                conv_element[sub_count,0] = float(element)
                sub_count = sub_count+1

            axarr[row_count].bar(x_axis,conv_element[:,0])
            row_count = row_count +1
            if row_count==row_countt:
                break
            del conv_element
    plt.show()


if __name__ == "__main__":
    file_name = 'D:/Data/final data/dev20val20/Result/hog.csv'
    plt.close('all')
    _m_bar_graph_streaming(file_name,40)