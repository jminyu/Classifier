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




def K_NearestNeighbor(N,k):
    print 'nothing'

if __name__ == "__main__":
    mean1 = 0.0
    variance1 = 1.0
    mean2 = 10.0
    variance2 = 4.0
    sigma1 = np.sqrt(variance1)
    sigma2 = np.sqrt(variance2)
    x = np.linspace(-5,20,1000)

    plt.figure(1)
    plt.plot(x,1.0/2.0*(mlab.normpdf(x,mean1,sigma1)+mlab.normpdf(x,mean2,sigma2)))
    plt.show()
