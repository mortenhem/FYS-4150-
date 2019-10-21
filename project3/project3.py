#!/usr/bin/env python

import numpy as np
import scipy.special
import time



def integrand_cartesian(x1, y1, z1, x2, y2, z2):
	"""Integrand in cartesian coordinates."""
	
	alpha = 2.0
	
	r1 = np.sqrt(x1**2 + y1**2 + z1**2)
	r2 = np.sqrt(x2**2 + y2**2 + z2**2)

	r1_sub_r2 = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
	
		
	# Account for zero in denominator	
	if r1_sub_r2 < 1E-10:
		return 0.0

	else:
		return np.exp(-2*alpha*(r1 + r2))/r1_sub_r2



def integrand_spherical(r1, r2, theta1, theta2, phi1, phi2):
	"""
	Integrand in spherical coordinates.
	"""	

	alpha = 2.0
	beta = np.cos(theta1)*np.cos(theta2) + \
		   np.sin(theta1)*np.sin(theta2)*np.cos(phi1-phi2)

	f1 = r1*r2*np.sin(theta1)*np.sin(theta2)
	r12 = r1**2 + r2**2 - 2*r1*r2*np.cos(beta)
	
	#numerator = np.exp(-2*alpha*(r1+r2))*np.exp(r1+r2)*f1

	
	return f1/r12



def Gauss_Legendre(N):
	"""
	Computes mesh points and weights
	for given N.
	
	The mesh points are the roots of 
	a Legendre polynomial of degree N.
	"""

	
	L = np.zeros((N, N))
	weights = np.zeros(N)
	mesh_points = np.zeros(N)

	# Roots of Legendre polynomial of degree N
	poly_roots = scipy.special.roots_legendre(N)
		

	for i in range(N):
		mesh_points[i] = poly_roots[0][i]

		
	# Compute matrix L
	for k in range(N):
		L_poly = scipy.special.legendre(k)
	
		for j in range(N):		
			x = mesh_points[j]
	
			L[j, k] = L_poly(x)

	# Inverse of L
	L_inv = np.linalg.inv(L)

	# Compute weights
	for k in range(N):
		weights[k] = 2*L_inv[0,k]


	return weights, mesh_points


def Gauss_Laguerre(N):
	"""Computes weights and mesh points."""

	mesh, weights = np.polynomial.laguerre.laggauss(N)
	
	return mesh, weights



def Gauss_Legendre_integral(a, b, integrand, N):
	"""Computes integral by the Gauss-Legendre method."""
		

	# Compute weights and mesh points
	w, x = Gauss_Legendre(N=N)
	
	I = 0		

	t0 = time.clock()

	for i in range(N):
		for j in range(N):
			for k in range(N):
				for l in range(N):
					for m in range(N):
						for n in range(N):
							

							# Change of variables to account for all intervals									
							x1 = 0.5*(b-a)*x[i] + 0.5*(b+a)
							y1 = 0.5*(b-a)*x[j] + 0.5*(b+a)
							z1 = 0.5*(b-a)*x[k] + 0.5*(b+a)
							x2 = 0.5*(b-a)*x[l] + 0.5*(b+a)
							y2 = 0.5*(b-a)*x[m] + 0.5*(b+a)
							z2 = 0.5*(b-a)*x[n] + 0.5*(b+a)
					
							f = integrand(x1, y1, z1, x2, y2, z2)					
							weigths = w[i]*w[j]*w[k]*w[l]*w[m]*w[n]
							I += weigths*f

	t1 = time.clock()

	I *= (0.5*(b-a))**6

	cpu_time = t1 - t0

	return I, cpu_time


def Gauss_Laguerre_integral(N):
	"""
	Computes integral by using Laguerre polynomials
	for radial parts and Legendre polynomial for
	angular parts.
	"""
	
	x_lag, w_lag = Gauss_Laguerre(N)

	x_leg, w_leg = Gauss_Legendre(N)


	theta = 0.5*np.pi*x_leg + 0.5*np.pi	
	phi = 0.5*2.0*np.pi*x_leg + 0.5*2.0*np.pi
	

	# integrand_spherical(r1, r2, theta1, theta2, phi1, phi2)

	I = 0

	t0 = time.clock()

	for i in range(N):
		for j in range(N):
			for k in range(N):
					for l in range(N):
						for m in range(N):
							for n in range(N):
		

								I += w_leg[i]*w_leg[j]*w_lag[k]*w_lag[l]*integrand_spherical(x_lag[i], x_lag[j], theta[k], theta[l], phi[m], phi[n])
	t1 = time.clock()


		
	cpu_time = t1 - t0

	return I, cpu_time




def Monte_Carlo_integration(N):

	"""Using uniform distribution."""

	a = -4; b = 4


	I = 0
	
	t0 = time.clock()

	for i in range(N):
		for j in range(N):
			for k in range(N):
				for l in range(N):
					for m in range(N):
						for n in range(N):
							

								
							x1 = a - (b-a)*random.random()
							y1 = a - (b-a)*random.random()
							z1 = a - (b-a)*random.random()
							x2 = a - (b-a)*random.random()
							y2 = a - (b-a)*random.random()
							z2 = a - (b-a)*random.random()
	
							I += integrand_cartesian(x1, y1, z1, x2, y2, z2)
					
	t1 = time.clock()		

	I = (b-a)**6*I/float(N**6)

	return I, t1-t0	

	

	
						














def main(part):
	
	I_exact = 5.*(np.pi)**2/16**2
	header = "N       I        I_exact       error        cpu_time"
	N_vals = [2, 5, 10, 15, 20, 25, 30]	
	


	if part == 'a':
		print 'integration by Gauss-Legendre quadrature'
		
		# Integration limits. Function is very close to zero outside (-4, 4)
		a = -4; b = 4

		filename = 'Test_results_gauss_legendre.txt'
		outfile = open(filename, 'w+')

		for N in N_vals:

			I, cpu_time = Gauss_Legendre_integral(a, b, integrand_cartesian, N)
	
			outfile.write("%g %f %f %f, %f \n" % (N, I, I_exact, np.abs(I-I_exact), cpu_time))


		outfile.close()

		

	elif part == 'b':
		print 'integration by Gauss-Laguerre quadrature'

	
		filename = 'Test_results_gauss_laguerre.txt'
		outfile = open(filename, 'w+')
		
		for N in N_vals:
			I, cpu_time = Gauss_Laguerre_integral(N)
			 
			outfile.write("%g %f %f %f, %f \n" % (N, I, I_exact, np.abs(I-I_exact), cpu_time))


		outfile.close()
	
	else:
		import sys
		print 'please choose part a or b'
		sys.exit()	
		




if __name__=='__main__':
	
	#main(part='a')
	#main(part='b')




		

