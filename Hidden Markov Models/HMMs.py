"""
hmm_model = dict(state,num_of_state,transition_matrix,observation_probabilty_matrx)
"""
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



__author__ = 'Schmidtz'



def trainHMMs(hmm_model, dat):
    """
    :param hmm_model: A recognizer object
    :param dat: A structure created by HOG-HOF
    :return: model - the trained model
    """
    [num_of_dat,dim] = np.shape(dat);


def initial_HMMs(dat):
    """
    build HMMs structure model
    :param dat: input data
    :return: initilized HMMs models
    """
    [num_of_dat,dim] = np.shape(dat)
    num_of_state = num_of_dat
    state = np.zeros((1,num_of_state))
    transition_mtx = np.zeros((num_of_state,num_of_state))
    observation_mtx = np.zeros(())

    hum_model =dict(transition_mtx, observation_mtx,state)

    return hum_model


class GeneralHMMs():
    def __init__(self,start_prob=None, transition_mtx=None, obsevation_prob=None,algorithm="viterbi"):
        self.start_prob = start_prob
        self.transision_mtx = transition_mtx
        self.observation_prob = obsevation_prob
        self.algorithm = algorithm
