"""
Gaussian Mixture Model clustering algorithm
:Dev data: 2014.08,27
:J.m.yu @ GIST - Ph.D Candidate
"""
from runpy import _ModifiedArgv0

__author__ = 'Schmidtz@GIST Ph.D Candidate'

import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import matplotlib as mpl
import numpy as np
from numpy import matrix
from numpy import matlib
from numpy import *
from numpy.random import *
import pylab as p
import math
from scipy import stats, mgrid, c_, reshape, random, rot90, linalg

from DataGen import genData, Mtx_multipl

def initParams(Dim,num_of_code):
    """
    returing parameter from trainGMM
    :param Dim: dimension of input vector(data)
    :param num_of_code: number of cluster
    :return: means(mean value), cov(covariance matrix), mix_weight(mixture probabilty array)
    """
    means = genData(num_of_code)
    covs = ones((Dim,num_of_code))
    mix_weight = ones((1,num_of_code))/num_of_code
    return means, covs, mix_weight

def gauss_pdf(dat,means, covs):
    """
    returing multivariate gaussian pdf value using dat(input vector), means and covariance matrix
    :param dat: input vector(data)(dim : R*1)
    :param means: means value about input vector (dim : R*1)
    :param covs: covariance matrix about input vector (dim : R*R)
    :return: multivariate normal distribution value(pdf) (dim : 1*1)
    """
    [dim] = np.shape(dat)
    value_sub_means = dat - means
    det_covs = np.linalg.det(covs)
    value_sub_means = np.reshape(value_sub_means,(1,2))
    transpost_mtx = np.reshape(value_sub_means,(2,1))
    invcov = np.linalg.inv(covs)
    temp_valu1 = Mtx_multipl(value_sub_means,invcov)
    temp_value = Mtx_multipl(temp_valu1,transpost_mtx)
    nd_value = np.exp(-0.5*temp_value)/(pow(2*math.pi,dim/2)*np.sqrt(det_covs))  #normal distribution probabilty value
    return nd_value




def loglikeGMM(dat,means, covs,mix_weight):
    """
    computing loglikelihood probabilty according to diagonal matrix
    :param dat: input vector(data)
    :param means: mean value of input vectors
    :param covs: covariance matrix
    :param mix_weight: mixture weight(prior probabilty)
    :return: loglike_prob(loglikelihood probabilty function-matrix), new_mix(renewal mixture probabilty - matrix)
    """
    [dim, num_of_dat] = np.shape(dat)
    [dim_of_code,num_of_code] = np.shape(mix_weight)
    new_mix = np.zeros((num_of_code,num_of_dat))
    diagcov = np.zeros((dim,dim))
    for n in range(0,num_of_dat):
        for m in range(0,num_of_code):
            temp_diagcov = np.diag(covs)
            for k in range(0,np.size(temp_diagcov)):
                diagcov[k,k] = temp_diagcov[k];
            gausval = gauss_pdf(dat[:,n],means[:,m],diagcov)
            new_mix[m,n] =mix_weight[:,m]*gausval
    total_loglike = log10(new_mix)
    return total_loglike,new_mix


def trainGMM(dat,means,covs,mix_weight,modeltype,threshold):
    [dim,num_of_data] = np.shape(dat);
    [dim_of_mix_weight,weight_length] = np.shape(mix_weight)
    max_iteration = 100
    #calculate initial likelihood:
    [loglike_prob,renewal_mix_weight] = loglikeGMM(dat,means,covs,mix_weight)
    sum_of_loglike = sum(loglike_prob)
    xvars = zeros((1,num_of_data))
    print 'Log-likelihood probabilty at initialization : ',sum_of_loglike

    plt.figure(1)
    plt.plot(means[0,:],means[1,:],'bo') #plot initial centroid of GMM


    #interation until convergence
    for iter in range(0,max_iteration):
        print iter,'-iteration'
        #Estimation(E) step : calculate posterior under current parameter values
        for n in range(0,num_of_data):
            renewal_mix_weight[:,n] = renewal_mix_weight[:,n]/sum(renewal_mix_weight[:,n])

        #Maximization(M) step : update parameter values to maximise posterior probabilty
        for m in range(0,weight_length):
            #mean values update
            prob_sum = sum(renewal_mix_weight[m,:])
            means[:,m] = (dat*transpose(renewal_mix_weight[m,:]))/prob_sum
            #covariance values update
            if modeltype=='scalarcov':
                for n in range(0,num_of_data):
                    meansub = dat[:,n]=means[:,m]
                    xvars[:,n] = transpose(meansub)*meansub
                covs[:,m] = ones((dim,1))*(xvars*transpose(renewal_mix_weight[m,:]))/(dim*prob_sum)
            if modeltype == 'diagcov':
                for n in range(0,num_of_data):
                    meansub = dat[:,n]=means[:,m]
                    xvarmat = multiply(meansub,meansub) #memory allocation in xcarmat
                covs[:,m] = xvarmat*transpose(renewal_mix_weight[m,:])/prob_sum
            mix_weight[m] = prob_sum/num_of_data #update mixture weight

        #plot new component centroid:
        plt.plot(means[0,:],means[1,:],'bo')
        print mix_weight
        print means
        print covs

        #calculating new log-likelihood probabilty:
        [new_loglike_prob,renewal_mix_weight] = loglikeGMM(dat,means, covs, mix_weight)
        new_sum_of_loglike = sum(new_loglike_prob)
        print 'Log-likelihood probabilty : ',new_sum_of_loglike

        #test of convergence:
        if new_sum_of_loglike < sum_of_loglike:
            print 'log-likelihood decreased'
        if((new_sum_of_loglike-sum_of_loglike)/num_of_data)<threshold:
            print 'convergence'
            break
        else:
            sum_of_loglike = new_sum_of_loglike
    return means,covs, mix_weight

def classify_GMM(dat,means,covs,mix_weight):
    """
    :param dat: input vector (data which you want to classify)
    :param means: mean matrix (each cluster consis of several Gaussian distribution thus, each cluster have several means
    :param covs: covariance matrix list of each normal distribution
    :param mix_weight: mixture weight
    :return:
    """
    [dim,num_of_dat] = np.shape(dat)
    [dim_ofcluster,num_of_cluster] = np.shape(means)








if __name__ == "__main__":
    random.seed(12345)
    dat = genData(500)

    #parameter initialization
    [means, covs, weight] = initParams(2,4);

    #Training GMM about data(dat)
    [means, covs, weight] = trainGMM(dat,means,covs,weight,'diagcov',0.00001)
    print 'trainining complete'

    #ploting


