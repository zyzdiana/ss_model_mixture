import os
import time
import numpy as np
import sys
import math
import cPickle as pickle
import multiprocessing as mp
import hips.distributions.polya_gamma as pg

def FwdFilterEM1_xk(y, w, x_r, x_init, sigma2_init, sigma2e_K):
    '''
    EM step 1: The forward nonlinear recursive filter
    Inputs:
    y - Recorded Neuron Responses
    w - Samples from Polya-gamma Distribution
    x_init, w_init: initial values for x and w
    R - Total number of trials
    '''
    K = y.shape[1]
    R = y.shape[0]

    x_prior = np.zeros(K+1) # xk|k-1
    x_post  = np.zeros(K+1) # xk-1|k-1

    sigma2_prior = np.zeros(K+1) # sigma2k|k-1
    sigma2_post  = np.zeros(K+1) # sigma2k-1|k-1

    # FORWARD FILTER
    x_post[0]      = x_init
    sigma2_post[0] = sigma2_init 

    for t in range(1,K+1):
        x_prior[t]      = x_post[t-1]
        sigma2_prior[t] = sigma2_post[t-1] + sigma2e_K
        
        N = np.sum(y[:,t-1])
        w_k = np.sum(w[:,t])
        x_post[t]  = (x_prior[t] + sigma2_prior[t]*(N-R/2.-np.sum(w[:,t]*x_r)))/ (1. + w_k*sigma2_prior[t])
        sigma2_post[t] = sigma2_prior[t] / ( 1.0 + w_k*sigma2_prior[t])

    return x_prior[1:],x_post[1:],sigma2_prior[1:],sigma2_post[1:]

def FwdFilterEM1_xr(y, w, x_k, x_init, sigma2_init, sigma2e_R):
    '''
    EM step 1: The forward nonlinear recursive filter
    Inputs:
    y - Recorded Neuron Responses
    w - Samples from Polya-gamma Distribution
    x_init, w_init: initial values for x and w
    R - Total number of trials
    '''
    K = y.shape[1]
    R = y.shape[0]

    x_prior = np.zeros(R+1) # xr|r-1
    x_post  = np.zeros(R+1) # xr-1|r-1

    sigma2_prior = np.zeros(R+1) # sigma2r|r-1
    sigma2_post  = np.zeros(R+1) # sigma2r-1|r-1

    # FORWARD FILTER
    x_post[0]      = x_init
    sigma2_post[0] = sigma2_init 

    for t in range(1,R+1):
        x_prior[t]      = x_post[t-1]
        sigma2_prior[t] = sigma2_post[t-1] + sigma2e_R
        
        N = np.sum(y[t-1,:])
        w_k = np.sum(w[t,:])
        x_post[t]  = (x_prior[t] + sigma2_prior[t]*(N-K/2.-np.sum(w[t,:]*x_k)))/ (1. + w_k*sigma2_prior[t])
        sigma2_post[t] = sigma2_prior[t] / ( 1.0 + w_k*sigma2_prior[t])

    return x_prior[1:],x_post[1:],sigma2_prior[1:],sigma2_post[1:]

# Function for Gibbs Sampling using filter backwards
def Gibbs_Sampler2(N, burnin, sigma2e_K, sigma2e_R, y, thin = 0, printTime = False):
    '''
    N: Number of Samples
    thin: thinning parameter
    burnin: Number of samples to burnin
    x_init, w_init: initial values for x and w
    R: Total number of trials
    '''
    K = y.shape[1]
    R = y.shape[0]
    
    # actual number of samples needed with thining and burin-in
    t0 = time.time()
    if(thin != 0):
        N_s = N * thin + burnin
    else:
        N_s = N + burnin
        
    samples_w = np.empty((N_s,R,K))
    samples_xk = np.empty((N_s,K))
    samples_xr = np.empty((N_s,R))
    
    w = np.zeros([R+1,K+1])
    x_k = np.zeros(K+1)
    x_r = np.zeros(R+1)
    A = np.ones(K+1)
    for i in xrange(N_s):
        #sample the conditional distributions x, w
        xkk, xrr = np.meshgrid(x_k,x_r)
        x = abs(xkk + xrr)
        for r in xrange(x.shape[0]):
            w[r,:] = pg.polya_gamma(a=A, c=abs(x[r,:]))
        #x_k = np.zeros(K+1)
        xk_prior,xk_post,sigma2k_prior,sigma2k_post = FwdFilterEM1_xk(y, w, x_r, x_init=0,sigma2_init=0,sigma2e_K=sigma2e_K)
        mean_k = xk_post[-1]
        var_k = sigma2k_post[-1]
        x_k[K] = np.random.normal(loc=mean_k, scale = np.sqrt(var_k))
        for k in xrange(K-1):
            # update equations
            xk_star_post = xk_post[K-k-2] + (sigma2k_post[K-k-2]/(sigma2e_K+sigma2k_post[K-k-2]))*(x_k[K-k] - xk_post[K-k-2])
            sigma2k_star_post = 1./(1./sigma2e_K+1./sigma2k_post[K-k-2])
            
            # Draw sample for x
            x_k[K-k-1] = np.random.normal(loc=xk_star_post, scale = np.sqrt(sigma2k_star_post))
        #x_r = np.zeros(R+1)
        xr_prior,xr_post,sigma2r_prior,sigma2r_post = FwdFilterEM1_xr(y, w, x_k, x_init=0,sigma2_init=0,sigma2e_R=sigma2e_R)
        mean_r = xr_post[-1]
        var_r = sigma2r_post[-1]
        x_r[R] = np.random.normal(loc=mean_r, scale = np.sqrt(var_r))
        for r in xrange(R-1):
            # update equations
            xr_star_post = xr_post[R-r-2] + (sigma2r_post[R-r-2]/(sigma2e_R+sigma2r_post[R-r-2]))*(x_r[R-r] - xr_post[R-r-2])
            sigma2r_star_post = 1./(1./sigma2e_R+1./sigma2r_post[R-r-2])
            
            # Draw sample for x
            x_r[R-r-1] = np.random.normal(loc=xr_star_post, scale = np.sqrt(sigma2r_star_post))
        samples_w[i,:,:] = w[1:,1:]
        samples_xk[i,:] = x_k[1:]
        samples_xr[i,:] = x_r[1:]
    if(printTime):
        print 'Time for Gibbs Sampling is %s seconds' % (time.time()-t0)
        
    if(thin == 0):
        return samples_w[burnin:,:,:],samples_xk[burnin:,:],samples_xr[burnin:,:]
    else:
        return samples_w[burnin:N_s:thin,:,:],samples_xk[burnin:N_s:thin,:],samples_xr[burnin:N_s:thin,:]

def BayesianEM(N_samples, burnin, sigma2eK,sigma2eR, y, thin=0, max_iter = 300, sampler = Gibbs_Sampler2):
    sigma_Ks = []
    sigma_Ks.append(sigma2eK)
    sigma_Rs = []
    sigma_Rs.append(sigma2eR)
    
    it = 0
    diff = 1
    while diff > 1e-5 and it <= max_iter:
        if it % 10 == 0:
            print it,
        w,x_k,x_r = sampler(N_samples,burnin, sigma2eK,sigma2eR, y, thin)
        x_k1 = np.roll(x_k,1)
        x_k1[:,0] = 0
        sigma2eK = np.mean((x_k-x_k1)**2)
        sigma_Ks.append(sigma2eK)
        sigmaK_post = np.var(x_k,axis=0)
        xk_post = np.mean(x_k,axis=0)
        
        x_r1 = np.roll(x_r,1)
        x_r1[:,0] = 0
        sigma2eR = np.mean((x_r-x_r1)**2)
        sigma_Rs.append(sigma2eR)
        sigmaR_post = np.var(x_r,axis=0)
        xr_post = np.mean(x_r,axis=0)
    
        diff = np.max(abs(sigma_Ks[it+1]-sigma_Ks[it]),abs(sigma_Rs[it+1]-sigma_Rs[it]))
        it +=  1

    if it >= max_iter:
        converge_flag = 1
        print 'Did not converge in %s iterations' % max_iter
        print 'sigma2e_K is ', sigma2eK
        print 'sigma2e_R is ', sigma2eR
    else:
        converge_flag = 0
        print
        print 'Converged after %d iterations' % (it)
        print 'sigma2e_K is ', sigma2eK
        print 'sigma2e_R is ', sigma2eR
    return x_k,x_r,sigma_Ks, sigmaK_post, xk_post, sigma_Rs, sigmaR_post, xr_post, converge_flag