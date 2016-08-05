#cython: boundscheck=False, wraparound=False, nonecheck=False
from libc.math cimport exp 
from libc.math cimport sqrt
from libc.math cimport M_PI
import numpy as np
from cython_gsl cimport *

cdef extern from "gsl/gsl_rng.h":
	ctypedef struct gsl_rng_type
	ctypedef struct gsl_rng
    
	gsl_rng_type *gsl_rng_mt19937
	gsl_rng *gsl_rng_alloc(gsl_rng_type * T) nogil
    
cdef extern from "gsl/gsl_randist.h":
	double gamma "gsl_ran_multinomial"(gsl_rng * r, int, int, double, int)

cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)

def smoothing(int T, int N, double Q, int M, double[:,:] X, double[:,:] W):
    
	cdef double[:] w_back = np.empty(N)
	cdef double[:] W_BACK = np.empty(N)
	cdef double[:,:] smoother = np.empty([T,M])
	cdef double w_back_sum
	cdef double const
	cdef int l
	cdef int i, j, k, t, tt, x, h, index
	cdef double[:] mutlinomialsamples
	
	const = 1/sqrt(2 * M_PI * Q)
	
	l = 0
	j = 0
	multinomialsamples = np.random.multinomial(M, W[T-1, :])
	while j< M:
		while multinomialsamples[l] == 0:
			l +=1
		for h in range(multinomialsamples[l]):
			smoother[T-1,j] = X[T,l]
			j +=1
		l +=1
	
	# for j in range(M):
# 		multinomialsamples = np.random.multinomial(1, W[T-1, :])
# 		l = 0
# 		while multinomialsamples[l] == 0:
# 			l += 1
# 		smoother[T-1, j] = X[T, l]

	for tt in range(T-1):
		t = T-2 - tt
		for j in range(M):
			w_back_sum = 0
			for k in range(N):
				w_back[k] = W[t,k] * const * exp( -(smoother[t+1,j]-X[t+1,k]) ** 2 / (2 * Q))
				w_back_sum += w_back[k]
			for k in range(N):
				W_BACK[k] = w_back[k]/w_back_sum
			multinomialsamples = np.random.multinomial(1, W_BACK)
			l = 0
			while multinomialsamples[l] == 0:
				l += 1
			smoother[t,j] = X[t+1, l]
	
	return smoother