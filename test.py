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
    [Dim, num_of_data] = np.shape(dat)
    noc = 6 #number of cluster
    init_prob_noc = ones((1,k))/k
    mu = np.randn((Dim,noc))
    cov = np.zeros((Dim,Dim,noc))
    gaus_val = np.zeros((Dim,noc))
    iter = 10  #iteration account

    #initialization of covariance matrix value
    for i in range(0,noc):
        cov[:,:] = -100*diag(log10(random(Dim,1)))

    #EM algorithm
    for i in range(1,iter):
        print i,'th iteration steps'
        #Estimation Stpe(E Step)
        for j in range(i,noc):
            #2-d gasussian distribution function
            z[:,i] = init_prob_noc[i]*np.linalg.det(cov[:,:,j])

        #Maximization Step(M Step)
        for k in range(0,noc):
            mu[:,k] =






    '''
    red = ["r.","b.","g.","c."]
    plt.figure(1)
    for i in range(0,100):
        plt.plot(dat[0,i],dat[1,i],red[0])
    for i in range(100,300):
        plt.plot(dat[0,i],dat[1,i],red[1])
    for i in range(300,500):
        plt.plot(dat[0,i],dat[1,i],red[3])
    plt.title('Simplest default with labels')
    plt.show()
    '''