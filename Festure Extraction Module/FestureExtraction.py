"""
Feature Extraction module
:Dev data: 2014.09.02
:J.m.yu @ GIST - Ph.D Candidate

Feature list
1. HOG(Histogram of oriented gradients
2.

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

