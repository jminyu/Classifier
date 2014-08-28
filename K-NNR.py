"""
K-NNR(K-Nearest Neighbor)
:Dev data: 2014.08,28
:J.m.yu @ GIST - Ph.D Candidate

algorithm steps
#Traning phase
*According to labeled data, modeling normal distribution
*when modeling normal distribution,using Euclidean algorithm
*the number of pdf is equal to the number of  class which is we already know.

 #Classification phase
 computing pdf.(need computing eucliean distance to each point)

Paramter
:dat : input vector 2*N [dat[0,:] = x value, dat[1,:] = y value
"""

__author__ = 'Schmidtz@GIST Ph.D Candidate'

import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import matplotlib as mpl
import numpy as np
from numpy import matlib
from numpy import *
from numpy.random import *
import pylab as p
import math
from scipy import stats, mgrid, c_, reshape, random, rot90, linalg

from DataGen import genData,genData_gaussian,distance



def data_labeling(dat,label): #label is integer (identification of data)
    """
    :param dat: input data(vector)
    :param label: label number which is you want
    :return: labeled_dat (labeled data)
    """
    [dim, num_of_dat] = np.shape(dat)
    labeled_dat = np.zeros((dim+1,num_of_dat))

    labeled_dat[0,:] = dat[0,:]
    labeled_dat[1,:] = dat[1,:]
    labeled_dat[2,:] = label #labeled_dat[:,2] :labeling number
    return labeled_dat


def sort_data(dat):#insertion sort
    """
    sorting data using distance value
    :param dat:
    :return:
    """
    [dim,lenth_Dat] = np.shape(dat)
    for i in range(1, lenth_Dat):
        val = dat[3,i]
        j = i - 1
        while (j >= 0) and (dat[3,j] > val):
            dat[3,j+1] = dat[3,j]
            j = j - 1
        dat[:,j+1] = dat[:,i]

    return dat


if __name__ == "__main__":
    random.seed(12345)

    num_of_dat = 500
    dim_of_dat = 2
    K = 3

    means1 = np.mat('1;1')
    covs1 = np.mat('0.3 0 ; 0 0.2')
    dat1 = genData_gaussian(num_of_dat,means1,covs1)
    label_dat1 = data_labeling(dat1,1)

    means2 = np.mat('5;2')
    covs2 = np.mat('0.4 0 ; 0 0.4')
    dat2 = genData_gaussian(num_of_dat,means2,covs2)
    label_dat2 = data_labeling(dat2,2)

    means3 = np.mat('1;5')
    covs3 = np.mat('0.3 0 ; 0 0.2')
    dat3 = genData_gaussian(num_of_dat,means3,covs3)
    label_dat3 = data_labeling(dat3,3)

    temp_dat = zeros((dim_of_dat+2,num_of_dat*3))
    temp_dat[0:2,0:500] = label_dat1
    temp_dat[0:2,500:1000] = label_dat2
    temp_dat[0:2,1000:1500] = label_dat3

    test_point = np.mat('0.3;0.3')

    for i in range(0,num_of_dat*3):
        temp_dist = distance(temp_dist[0:1,i],test_point)
        temp_dat[3,i] = temp_dist

    sorted_dat = sort_data(temp_dat);

    print '3-NNR algorithm conclusion'

    print '1st neighbor label : ',sorted_dat[2,0]

    print '2st neighbor label : ',sorted_dat[2,1]

    print '3st neighbor label : ',sorted_dat[2,2]