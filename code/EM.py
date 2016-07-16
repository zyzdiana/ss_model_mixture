import os
import glob
import csv
import time
import pandas as pd
from   pylab import *
from   urllib import urlopen
from   datetime import datetime
import numpy as np
import sys
import math
from   operator import truediv
from   pandas.io.json import json_normalize
import cPickle as pickle
from   random import *
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")
sys.path.insert(1,'/Users/zyzdiana/Github/AC299r/code')

def TransformToProb(meanv, sigma2, mu):
    '''
    Transform the computed mean, sigma2 into probability
    '''
    #compute upper and lower bounds of conf intervals by simulation
    NUM_SAMPS = 10000

    np.random.seed(0) # start at same random number each time
    T       = len(meanv)
    p       = np.zeros(T)
    pll     = np.zeros(T)
    pul     = np.zeros(T)
    pmode   = np.zeros(T)
    sigma   = np.sqrt(sigma2)

    for t in range(T):
        s         = np.random.normal(meanv[t], sigma[t], NUM_SAMPS)
        ps        = map(truediv, np.exp(s + mu), (1.0+np.exp(s + mu)))
        pmode[t]  = np.exp(meanv[t] + mu)/(1.0+np.exp(meanv[t] + mu))
        p[t]      = np.mean(ps)
        pll[t],pul[t] = np.percentile(ps,[5,95])
    return (pmode, p, pll, pul)

def NewtonSolve(x_prior, sigma_prior, N, Nmax, mu):
    '''
    Solve for posterior mode using Newton's method
    '''
    xp = x_prior
    sp = sigma_prior

    it = xp + sp*(N - Nmax*np.exp(mu+xp)/(1.0 + np.exp(mu+xp)))     #starting iteration  

    for i in range(30): 
        g     = xp + sp*(N - Nmax*np.exp(mu+it)/(1.0+np.exp(mu+it))) - it;
        gprime = -Nmax*sp*np.exp(mu+it)/(1.0+np.exp(mu+it))**2.0 - 1.0   
        x = it  - g/gprime 

        if np.abs(x-it)<1e-10:
            return x
        it = x

    #if no value found try different ICs: needed if there are consec same values
    it = -1
    for i in range(30): 
        g     = xp + sp*(N - Nmax*np.exp(mu+it)/(1.0+np.exp(mu+it))) - it
        gprime = -Nmax*sp*np.exp(mu+it)/(1.0+np.exp(mu+it))**2.0 - 1.0
        x = it  - g/gprime 

        if np.abs(x-it)<1e-10:
            return x
        it = x

    #if no value found try different ICs
    it = 1
    for i in range(30): 
        g     = xp + sp*(N - Nmax*np.exp(mu+it)/(1+np.exp(mu+it))) - it
        gprime = -Nmax*sp*np.exp(mu+it)/(1+np.exp(mu+it))**2 - 1.0
        x = it  - g/gprime 

        if np.abs(x-it)<1e-10:
            return x
        it = x

def FwdFilterEM(y,delta,x_init,sigma2_init,sigma2e, mu):
    '''
    EM step 1: The forward nonlinear recursive filter
    '''
    T = y.shape[1]
    # Data structures
    x_prior = zeros(T+1) # xk|k-1
    x_post  = zeros(T+1) # xk-1|k-1
    sd1     = zeros(T+1) 

    next_pred_error = zeros(T+1)

    sigma2_prior = zeros(T+1) # sigma2k|k-1
    sigma2_post  = zeros(T+1) # sigma2k-1|k-1

    # FORWARD FILTER
    x_post[0]      = x_init
    sigma2_post[0] = sigma2_init 

    for t in range(1,T+1):

        x_prior[t]      = x_post[t-1]
        sigma2_prior[t] = sigma2_post[t-1] + sigma2e
        
        N = np.sum(y[:,t-1])
        x_post[t]  = NewtonSolve(x_prior[t],sigma2_prior[t],N,len(y),mu)

        pt = exp(mu+x_post[t])/(1.0+exp(mu+x_post[t]))

        sigma2_post[t] = 1.0 / ( 1.0/sigma2_prior[t] + len(y)*pt*(1-pt))

        sd1[t] = np.sqrt(sigma2_post[t])
    ape = 0#next_pred_error.mean()

    return x_prior,x_post,sigma2_prior,sigma2_post, ape

def BackwardFilter(x_post,x_prior,sigma2_post, sigma2_prior):
    '''
    EM Step 1: Fixed Interval Smoothing Algorithm
    '''
    T = len(x_post)
    # Initial conditions
    x_T               = zeros(T)  
    x_T[T-1]          = x_post[T-1]
    sigma2_T          = zeros(T)
    sigma2_T[T-1]     = sigma2_post[T-1]
    A = np.zeros(T)

    for t in range(T-2,0,-1):
        A[t]            = sigma2_post[t]/sigma2_prior[t+1]
        x_T[t]          = x_post[t] + np.dot(A[t],x_T[t+1] - x_prior[t+1])
        Asq             = np.dot(A[t],A[t])
        diff_v          = sigma2_T[t+1] - sigma2_prior[t+1]
        sigma2_T[t]     = sigma2_post[t] + np.dot(Asq, diff_v)

    return x_T,sigma2_T,A

def MSTEP(xnew, signewsq, A):
    '''
    M step of EM
    '''
   
    T          = len(xnew)
    xnewt      = xnew[2:T]
    xnewtm1    = xnew[1:T-1]
    signewsqt  = signewsq[2:T]
    A          = A[1:T-1]
    covcalc    = np.multiply(signewsqt,A)

    term1      = np.dot(xnewt,xnewt) + np.sum(signewsqt)
    term2      = np.sum(covcalc) + np.dot(xnewt,xnewtm1)

    term3      = 2*xnew[1]*xnew[1] + 2*signewsq[1]
    term4      = xnew[T-1]**2 + signewsq[T-1]

    newsigsq   = (2*(term1-term2)+term3-term4)/T

    return newsigsq

def EM(xx, mu, sigma2e, x_init, sigma_init):
    '''
    xx : Neuron Spike Data
    '''
    num_its         = range(0,3000)   
    savesigma2_e    = np.zeros(len(num_its)+1)
    savesigma2_e[0] = sigma2e

    #run though until convergence
    its      = 0
    diff_its = 1
    max_its  = 3000

    while diff_its>0.00001 and its <= max_its:
        its +=  1

        x_prior,x_post,sigma2_prior,sigma2_post, ape = FwdFilterEM(xx, 1, x_init, sigma_init, sigma2e, mu) 


        x_T,sigma2_T,A   = BackwardFilter(x_post,x_prior,sigma2_post, sigma2_prior)

        x_T[0]     = 0               
        sigma2_T[0] = sigma2e

        sigma2e   = MSTEP(x_T, sigma2_T, A)  

        savesigma2_e[its+1]  = sigma2e       
        diff_its             = abs(savesigma2_e[its+1]-savesigma2_e[its])

        x_init     = 0               
        sigma_init = sigma2_T[0]


    if its == max_its:
        converge_flag = 1
        print 'Did not converge in 3000 iterations'
    else:
        converge_flag = 0
        print
        print 'Converged after ' + str(its) + ' iterations'
        print 'sigma2e is ', sigma2e

    x_post      = x_post[1:]
    sigma2_post = sigma2_post[1:]
    return x_T[1:],sigma2_T,sigma2e,sigma_init,converge_flag

def RunEM(values):
    '''
    Run the EM algorithm and return probability with confidence bands
    '''
    t0 = time.time()
    startflag  = 0
    sigma2e    = 0.5**2 #start guess
    sigma_init = sigma2e
    x_init     = 0.0
    mu = 0
    
    print 'initial sigma2e is', sigma2e
    x_post,sigma2_post,sigma2e,sigma_init,converge_flag =  EM(values, mu, sigma2e, x_init, sigma_init)
    pmode, p, pll, pul = TransformToProb(x_post, sigma2_post, mu)
    print 'runtime: %s seconds' % (time.time()-t0)
    return pmode, p, pll, pul,sigma2e
    
import matplotlib.patches as mpatches
def plot_results(values, pmode, p, pll, pul, plot_raster = False, ylim = [0.0,1.0]):
    '''
    Visualize the results EM
    '''
    fig = plt.figure(figsize = [15,5])
    ccc = 'b'
    line, = plt.plot(pmode,  linestyle = '-', color= 'b', alpha=0.9,lw=2,label='model probability')
    plt.fill_between(range(0,len(p)),pll,pul,color='blue',alpha=0.3)
    blue_patch = mpatches.Patch(color='blue', alpha = 0.3,label='Uncertainty')
    plt.legend(handles=[blue_patch,line])
    plt.xlim([0,len(p)+1])
    plt.ylim(ylim)
    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.grid('off')
    plt.show()
    
    if(plot_raster):
        I, J = np.where(values != 0)
        plt.figure(figsize = [15,4])
        plt.scatter(J,I,marker='x',color='green',alpha = 0.6)
        plt.title('Raster Plot from Data')
        plt.xlabel('Time')
        plt.ylabel('Trials')
        plt.ylabel
        plt.show()