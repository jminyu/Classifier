__author__ = 'Schmidtz'
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

from sklearn import hmm


def foroward_alg(initial,transition, observation,dat):
    """
    :param initial: start probabiltuy
    :param transition: transition probabilty matrix
    :param observation: observation probabilty matrix
    :return observ_prob :observation sequence probabilty
    """
    print "Forward algorithm processing"
    temp_prob = 0.0;
    observation_prob = 0.0;

    #STEP1 initialization
    _sequence_length = np.size(dat)
    _state_scale = np.size(initial)
    forward_variable = np.zeros((_state_scale,_sequence_length))
    for i in range(0,_state_scale):
        forward_variable[i,0] = initial[0,i] * observation[i,dat[0,0]]

    #STEP2 Derivation
    for t in range(0,_sequence_length-1):
        for j in range(0,_state_scale):
            for k in range(0,_state_scale):
                temp_prob = temp_prob+forward_variable[k,t]*transition[k,j]
            forward_variable[j,t+1] = temp_prob*observation[j,dat[0,t+1]]
            temp_prob = 0.0

    #STEP3 Termination
    for i in range(0,_state_scale):
        observation_prob = observation_prob + forward_variable[i,_sequence_length-1]

    return observation_prob


def backward_alg(initial,transition, observation,dat):
    """
    :param initial: initial probabilty matrix
    :param transition: transition probabilty matrix
    :param observation: observation probabilty matrix
    :return observ_prob :  observation sequence probabilty
    """
    print 'Backward algorithm processing'
    temp_prob = 0.0;
    observation_prob = 0.0;

    #STEP1 initialization
    _sequence_length = np.size(dat)
    _state_scale = np.size(initial)
    backward_variable = np.zeros((_state_scale,_sequence_length))
    for i in range(0,_state_scale):
        backward_variable[i,_sequence_length-1] = 1.0


    #STEP2 Derivation
    for t in range(_sequence_length-2,-1,-1):
        for i in range(0,_state_scale):
            for j in range(0,_state_scale):
                temp_prob = temp_prob+transition[i,j]*observation[j,dat[0,t+1]]*backward_variable[j,t+1]
            backward_variable[i,t] = temp_prob
            temp_prob = 0.0

    #STEP3 Termination
    for i in range(0,_state_scale):
        observation_prob = observation_prob + initial[0,i]*observation[i,dat[0,0]]*backward_variable[i,0]


    return observation_prob


def initHMMs(dats,states):
    """
    initialization hmm paramters
    :param dats: data sample structure(matrix)
    :param M: number of pdfs for each state
    :return:hmm :initialized hmm structure
    """
    [dim,num_of_dat] = np.shape(dats)
    num_of_state = np.size(states)
    initmodel = hmm.GaussianHMM()






def trainHMMs(dat,M):
    """
    :param dat: data sample structure
    :param M: number of pdfs for each state
    :return: hmm : hmm structure of pdf for each state
    """

    [dim,num_of_dat] = np.shape(dat)



if __name__ == "__main__":
    #testing forward/Backward algorithm
    test_pi = np.mat('0.333  0.333  0.333')
    test_transition_matrix = np.mat('0.333 0.333 0.333 ; 0 0.5 0.5 ; 0 0 1.0')
    test_observation_prob = np.mat('1.0 0 ; 0.5 0.5 ; 0.333 0.666')
    dat = np.mat('1 0 1')

    forward_prob = foroward_alg(test_pi,test_transition_matrix,test_observation_prob,dat)
    backward_prob  = backward_alg(test_pi,test_transition_matrix,test_observation_prob,dat)

    print "forward algorithm probabilty : ",forward_prob
    print "backward algorithm probabilty  : ",backward_prob


    #forward algorithm
