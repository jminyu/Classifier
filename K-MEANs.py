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

from DataGen import genData

def KMEANs_procedure(num,dat,initial_codebook):
    """
    :param num: repeatation ammount
    :param dat: input vector set
    :param initial_codebook: initial central of clusters
    :return: NON
    """
    [Dim,num_of_codebook] = np.shape(initial_codebook)
    AvgDist = zeros(num)
    code_book = initial_codebook

    for i in range(0,num):
        AvgDist[i],code_book  = KMEANS_general(dat,code_book)#082614 debug
        print i,'th code book\n'
        for j in range(0,num_of_codebook):
            print j,code_book[:,j]

    return AvgDist


def distance(x1,x2):
    """
    calculating distance between two point(x1,x2)
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

def KMEANS_general(dat,code_book):
    """
    :param dat: input vector set
    :param code_book:input codebook -cluster list
    :param num: clustering ammount
    :return: AvgDist(Average ot total Disortion), temp_codebook(temporal codebook)
    """
    [dim,num_dat] = np.shape(dat)
    [dim_code,num_code] = np.shape(code_book)
    temp_dat = zeros((dim+1,num_dat)) #expand data
    temp_dat[0,:] = dat[0,:] # x axis value
    temp_dat[1,:] = dat[1,:] # y axis value
    #temp_dat[2,:] = clustering value
    temp_codebook = zeros(np.shape(code_book))
    AvgDist = 0.0
    inf = 10000.0
    count = 0.0

    #clustering phase
    for i in range(0,num_dat):
        min_key = inf
        for j in range(0,num_code):
            temp_distance = distance(temp_dat[0:1,i],code_book[:,j])
            if temp_distance<min_key:
                min_key = temp_distance
                temp_dat[2,i] = j #assignment cluster number

    #code_book renewal
    for j in range(0,num_code):
        for i in range(0,num_dat):
            if temp_dat[2,i]== j:
                temp_codebook[0,j] = temp_codebook[0,j] + temp_dat[0,i]
                temp_codebook[1,j] = temp_codebook[1,j] + temp_dat[1,i]
                count = count+1
        temp_codebook[0:1,j] = temp_codebook[0:1,j]/count


    #computing total disortion
    AvgDist = Average_Disortion(dat,code_book)
    Graph_clust(temp_dat,num_of_codebook)
    return AvgDist,temp_codebook
    #calculating total disortion

def KMEANS_LBG(dat,code_book):
    """
    binary-split k-means algorithm
    :param dat: input vector
    :param code_book: initial code book
    :return: final code book
    """
    [dim,num_dat] = np.shape(dat)
    [dim_code,num_code] = np.size(code_book)
    temp_dat = zeros((dim+1,num_dat),type=float)
    temp_codebook = zeros(size(code_book))
    temp_codebook = code_book
    ret_codebook = zeros(size(code_book))
    AvgDist = []; #total average distortion
    inf = 10000.0
    for i in range(0,num_dat):
        min_key = inf;
        for j in range(0,num_code):
            if distance(dat[0:1,i],code_book[:,j]) < min_key:
                min_key = distance(dat[0:1,i],code_book[:,j])
                dat[2,i]  = j

    color = 'rgbkcmy'


def Graph_clust(dat,num_of_cluster):
    """
    :param dat: input data(vector)
    :param num_of_cluster: number of code(size of codebook)
    :return:Non
    """
    plt.figure(1)
    color = ["r.","g.","b.","c.","m.","k.","y."]
    [Dim,num] = np.shape(dat)
    for j in range(0,num_of_codebook):
        for i in range(0,num):
            if dat[2,i]==j:
                plt.plot(dat[0,i],dat[1,i],color[j])
    plt.title('Simplest default with labels')
    plt.show()




def Average_Disortion(dat,codebook):
    """
    computing total disortion of dat
    :param dat: set of input vectorW
    :param codebook: set of central of cluster
    :return: average value of disortion of all data
    """
    [dim, num_dat] = np.shape(dat)
    [dim_code,num_code] = np.shape(codebook)
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


if __name__ == "__main__":
    random.seed(12345)
    dat = genData(500)
    num_of_codebook = 4
    clustering_count = 10;
    x_axis = linspace(0,9,clustering_count)
    AvgDist = zeros(10)
    code_book = genData(num_of_codebook)

    #temp_dist = distance(dat[:,0],dat[:,1])
    #print temp_dist

    print 'k-means algorithm'
    print 'initial code book'
    for i in range(0,num_of_codebook):
        print i,code_book[:,i]
    AvgDist = KMEANs_procedure(clustering_count,dat,code_book)

    plt.figure(2)
    plt.plot(x_axis, AvgDist)
    plt.title('Total Average Disortion')
    plt.show()

