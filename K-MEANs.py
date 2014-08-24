"""
k-means clustering algorithm
:Dev data: 2014.08,24
:J.m.yu @ GIST - Ph.D Candidate

algorithm steps
1. making initial cetral set(code book) using original data set
2. E step : clustering using distance formular(Euclidean)
  - data set should be divided K cluster
3. M step : renewal central of cluster
4. computing total distortion
5. repeat 2~4 steps

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

def distance(x1,x2):
    """
    :param x1: input vector 1
    :param x2: input vetor 2
    :return: distance which between x1 and x2
    """
    k = np.size(x1)
    y  = 0.0;
    for i in range(0,k):
        y = y+pow((x1[i]-x2[i]),2)
    y = math.sqrt(y)
    return y

def KMEANS(dat,num,code_book):
    """
    :param dat: input vector
    :param num: number of code(cluster)
    :param code_book: initial code book
    :return: final code book
    """
    [dim,num_dat] = np.size(dat)
    iter = 0
    while iter==num_dat:
        iter = iter+1

def Average_Disortion(dat,codebook):
    """
    :param dat: set of input vectorW
    :param codebook: set of central of cluster
    :return: average value of disortion of all data
    """
    [dim, num_dat] = size(dat)
    [dim_code,num_code] = size(codebook)
    total_disortion = 0.0
    inf = 10000.0

    for i in range(0,num_dat):
        min_key = inf
        for j in range(0,num_code):
            temp_key = distance(dat[:,i],codebook[:,j])
            if temp_key < min_key:
                min_key = temp_key
        total_disortion = total_disortion + temp_key

    return total_disortion/num_dat






def genData(Ndat):
        c1 = 0.5
        r1 = 0.4
        r2 = 0.3
        # generate enough data to filter
        N = 20*Ndat
        X = array(random_sample(N))
        Y = array(random_sample(N))
        X1 = X[(X-c1)*(X-c1) + (Y-c1)*(Y-c1) < r1*r1]
        Y1 = Y[(X-c1)*(X-c1) + (Y-c1)*(Y-c1) < r1*r1]
        X2 = X1[(X1-c1)*(X1-c1) + (Y1-c1)*(Y1-c1) > r2*r2]
        Y2 = Y1[(X1-c1)*(X1-c1) + (Y1-c1)*(Y1-c1) > r2*r2]
        X3 = X2[ abs(X2-Y2)>0.05 ]
        Y3 = Y2[ abs(X2-Y2)>0.05 ]
        #X3 = X2[ X2-Y2>0.15 ]
        #Y3 = Y2[ X2-Y2>0.15]
        X4=zeros(Ndat, dtype=float32)
        Y4=zeros(Ndat, dtype=float32)
        for i in xrange(Ndat):
            if (X3[i]-Y3[i]) >0.05:
                X4[i] = X3[i] + 0.08
                Y4[i] = Y3[i] + 0.18
            else:
                X4[i] = X3[i] - 0.08
                Y4[i] = Y3[i] - 0.18
        print "X", size(X3[0:Ndat]), "Y", size(Y3)
        return(vstack((X4[0:Ndat],Y4[0:Ndat])))







if __name__ == "__main__":
    random.seed(12345)
    dat = genData(500)
    temp_dist = distance(dat[:,0],dat[:,1])
    print temp_dist
    print 'k-means algorithm'
