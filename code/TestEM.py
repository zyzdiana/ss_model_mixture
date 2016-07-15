

#State-space EM code in python
#Anne Smith 2015
import os
import glob
import csv
import pandas as pd
from   pylab import *
from   urllib import urlopen
from   datetime import datetime	
import matplotlib.pyplot as plt
import numpy as np
import sys
import math
from   operator import truediv
from   pandas.io.json import json_normalize
import pickle
import FiltersEM as ff
from   random import *

reload(ff)


################################
def RunEM(df):

	startflag  = 0
	sigma2e    = 0.5**2 #start guess
	sigma_init = sigma2e

	p_init     = 0.5 #xx[0:2].mean()
	x_init     = 0.0

	if(p_init > 0.01 and p_init < 0.99):
		mu = np.log(p_init/(1-p_init))
	elif(p_init<=0.01):
		mu = -3.0
	else:
		mu = 3.0

	print sigma2e
	x_post,sigma2_post,sigma2e,sigma_init,converge_flag =  ff.EM(df.y, mu, sigma2e, x_init, sigma_init)
	

	pmode, p, pll, pul       = ff.TransformToProb(x_post, sigma2_post, mu)
	

	fig         = plt.figure(0)
	ccc = 'b'
	plt.plot(pmode,  linestyle = '-', color= 'r', alpha=0.9,lw=2)
	plt.plot(pll, linestyle = '-', color= ccc, alpha=0.9,lw=3)
	plt.plot(pul, linestyle = '-', color= ccc, alpha=0.9,lw=3)
	plt.plot([0, len(p)], [0.5, 0.5], 'k-', lw=1, alpha=0.9)
	plt.plot(range(1,len(df)+1), df['y'],'o')
	plt.locator_params(axis = 'y', nbins = 3)
	plt.ylim([0,1])
	#plt.xlim([1,len(p)+1])
	plt.show()
	return

###############################
def main():

	df = pd.DataFrame()

	resp_values = [ 0. ,1. ,0. , 1. ,1. ,1., 0. ,0. , 1. ,1. , 1. ,1., 0.,
	 1. ,1., 1., 0., 1., 1., 1., 0., 1., 0., 1., 1., 1.,
	 0., 0., 1., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1., 1.0 ,0.0]

	df['y'] = resp_values

	RunEM(df)


if __name__ == '__main__':
  main()



