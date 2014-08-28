from pandas.core.strings import _length_check

__author__ = 'Schmidtz'
import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import matplotlib as mpl
import numpy as np
import scipy as sc
from numpy import matlib
from numpy import *
from numpy.random import *
import pylab as p
import math
from scipy import stats, mgrid, c_, reshape, random, rot90, linalg


def Mtx_multipl(matrix1,matrix2):
    product = np.zeros((len(matrix1),len(matrix2[0])))
    for i in range(0,len(matrix1)):
        for j in range(0,len(matrix2[0])):
            for k in range(0,len(matrix2)):
                product[i][j] += matrix1[i][k]*matrix2[k][j]
    return product

def MTX_transpose(matrix , mode):
    mtx_length = np.size(matrix)
    if mode=='rowtocol':
        t_matrix = zeros((mtx_length,1))
        for i in range(0,mtx_length):
            t_matrix[i,:] = matrix[i]
    if mode=='coltorow':
        t_matrix = zeros((1,mtx_length))
        for i in range(0,mtx_length):
            t_matrix[:,i] = matrix[i]
    return t_matrix


def genData_gaussian(Ndat,means, covs):
    """
    generating data set which is follow to gaussian distribution
    :param Ndat: number of dat(vector)
    :return:data set
    """

    dim = np.size(means)
    dat = zeros((dim,Ndat))
    [row_cov,col_cov] = np.shape(covs)
    if [row_cov,col_cov]!=[dim,dim]:
        print 'Incompatible dimension'
        quit()


    invmtx_whiten = np.linalg.inv(whiten(covs))
    rand_field = np.random.randn(dim,Ndat)
    rand_field2 = ones((1,Ndat))
    dat = dot(invmtx_whiten,rand_field)+dot(means,rand_field2)
    return dat

    X = array(random_sample(Ndat))
    Y = array(random_sample(Ndat))
    return (vstack((X[0:Ndat],Y[0:Ndat])))


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



def whiten(covmtx,fudge =1E-18):
    """
    whitening transformation about covariance matrix
    :param covmtx:
    :return transposed matrix:
    """
    d, V = np.linalg.eigh(covmtx)
    D = np.diag(1. / np.sqrt(d+fudge))
    W = np.dot(np.dot(V, D), V.T)
    return W

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
    means = np.array([1,1])
    covs = np.array([0.3,0],[0,0.2]);
    dat = genData_gaussian(500,means,covs)