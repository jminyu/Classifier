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

from HMMs import GeneralHMMs

def foroward_alg(initial,transition, observation,dat):
    """
    :param initial: start probabiltuy
    :param transition: transition probabilty matrix
    :param observation: observation probabilty matrix
    :return observ_prob :observation sequence probabilty
    """
    #print "Forward algorithm processing"
    temp_prob = 0.0;
    observation_prob = 0.0;

    #STEP1 initialization
    _sequence_length = np.size(dat)
    _state_scale = np.size(initial)
    temp_initial = np.mat(initial)
    forward_variable = np.zeros((_state_scale,_sequence_length))
    for i in range(0,_state_scale):
        forward_variable[i,0] = temp_initial[0,i] * observation[i,dat[0,0]]

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

    return forward_variable, observation_prob


def backward_alg(initial,transition, observation,dat):
    """
    :param initial: initial probabilty matrix
    :param transition: transition probabilty matrix
    :param observation: observation probabilty matrix
    :return observ_prob :  observation sequence probabilty
    """
    #print 'Backward algorithm processing'
    temp_prob = 0.0;
    observation_prob = 0.0;

    #STEP1 initialization
    _sequence_length = np.size(dat)
    _state_scale = np.size(initial)
    initial = np.mat(initial)
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


    return backward_variable, observation_prob


#hard development
def Baum_welch_test(initial, transition,observation,dat):
    [dim, num_of_state] = np.shape(transition)
    [num_of_observation,dim_ob] = np.shape(observation)
    temp_ = np.zeros((num_of_observation,1))

    renew_transi = np.zeros((dim,num_of_state))
    renew_observ = np.zeros((num_of_observation,dim_ob))


    gamma = np.zeros((num_of_observation,num_of_state))
    xi = np.zeros((num_of_observation,num_of_state,num_of_state))
    temp_xi = np.zeros((dim_ob,1))

    #computing forward variable using forward algorithm
    forward_variable,forward_prob = foroward_alg(initial, transition,observation,dat)

    #computing backward variable using backward algorithm
    backward_variable, backward_prob = backward_alg(initial, transition,observation,dat)

   #update - Maximization
    for t in range(0,num_of_observation):
        for j in range(0,num_of_state):
                temp_[t,0] = temp_[t,0] + forward_variable[j,t]*backward_variable[j,t]



    for t in range(0,num_of_observation):
        for i in range(0,num_of_state):
            gamma[t,i] = (forward_variable[i,t]*backward_variable[i,t])/temp_[t,0]

    #computing xi value
    for t in range(0,num_of_observation-1):
        for i in range(0,num_of_state):
            for j in range(0,num_of_state):
                xi[t,i,j] = (forward_variable[i,t]*transition[i,j]*observation[j,dat[0,t+1]]*backward_variable[j,t+1])/temp_[t,0]
                #if dat[0,t+1]==0: #temporary value for computing observation probabilty
                #    temp_xi[dat[0,t+1],0] = temp_xi[dat[0,t+1],0] + xi[t,i,j]
                #if dat[0,t+1]==1:
                #    temp_xi[dat[0,t+1],0] = temp_xi[dat[0,t+1],0] + xi[t,i,j]


    for i in range(0,num_of_state):
        for j in range(0,num_of_state):
            xi[num_of_state-1,i,j] = (forward_variable[i,num_of_state-2]*transition[i,j]*observation[j,dat[0,num_of_state-1]]*backward_variable[j,num_of_state-1])/temp_[num_of_state-2,0]



    #update - Estmiation
    new_initial = gamma[0,:]
    for i in range(0,num_of_state):
        for j in range(0,num_of_state):
            if sum(xi[0:num_of_observation-1,i,j])==0:
                renew_transi[i,j] = 0.0
            elif sum(gamma[0:num_of_observation-1,i])==0:
                 renew_transi[i,j] = sum(xi[0:num_of_observation-1,i,j])/0.00001
            else:
                renew_transi[i,j] = sum(xi[0:num_of_observation-1,i,j])/sum(gamma[0:num_of_observation-1,i])



    #observation probabilty updateing
    for j in range(0,num_of_state):
        for k in range(0,dim_ob):
            if k==1 or k==3:
                if (gamma[0,j]+gamma[2,j])==0:
                    renew_observ[j,k]=0;
                else:
                    renew_observ[j,k] = (gamma[0,j]+gamma[2,j])/sum(gamma[:,j])
            if k==2:
                if gamma[1,j]==0:
                    renew_observ[j,k]=0;
                else:
                    renew_observ[j,k] = gamma[1,j]/sum(gamma[:,j])


    return new_initial, renew_transi, renew_observ


#HMMs class
def _Baum_welch(hmm_model,train_dat):
    num_of_state = hmm_model.num_of_state


def initHMMs(dats,states):
    """
    initialization hmm paramters
    :param dats: data sample structure(matrix)
    :param M: number of pdfs for each state
    :return:hmm :initialized hmm structure
    """
    [dim,num_of_dat] = np.shape(dats)
    num_of_state = np.size(states)
    initmodel = GeneralHMMs()

    #initial probabilty
    initmodel.start_prob = zeros((num_of_state,1))
    initmodel.start_prob[0,0] = 1;


    #transition probabilty
    initmodel.transision_mtx = np.zeros((num_of_state,num_of_state))
    for i in range(0,num_of_state-1):
        initmodel.transision_mtx[i,i] = 0.5
        initmodel.transision_mtx[i,i+1] = 0.5

    initmodel.transision_mtx[num_of_state-1,num_of_state-1] = 1.0

    #initial cluster of pdf
    #equality seqmentation





def trainHMMs(dat,M):
    """
    :param dat: data sample structure
    :param M: number of pdfs for each state
    :return: hmm : hmm structure of pdf for each state
    """

    [dim,num_of_dat] = np.shape(dat)
    model_hmm = GeneralHMMs()




if __name__ == "__main__":
    #testing forward/Backward algorithm
    test_pi = np.mat('0.333  0.333  0.333')
    test_transition_matrix = np.mat('0.333 0.333 0.333 ; 0 0.5 0.5 ; 0 0 1.0')
    test_observation_prob = np.mat('1.0 0 ; 0.5 0.5 ; 0.333 0.666')
    dat = np.mat('1 0 1')

    renewal_pi = np.zeros((3,1))

    forward_variable, forward_prob = foroward_alg(test_pi,test_transition_matrix,test_observation_prob,dat)
    backward_variable, backward_prob  = backward_alg(test_pi,test_transition_matrix,test_observation_prob,dat)

    print "forward algorithm probabilty : ",forward_prob
    print "backward algorithm probabilty  : ",backward_prob

    renewal_pi, renewal_transition_matrix, renewal_observ_prob = Baum_welch_test(test_pi,test_transition_matrix,test_observation_prob,dat)
    print renewal_pi
    renewal_pi, renewal_transition_matrix, renewal_observ_prob = Baum_welch_test(renewal_pi, renewal_transition_matrix, renewal_observ_prob,dat)
    print renewal_pi

    #forward algorithm
