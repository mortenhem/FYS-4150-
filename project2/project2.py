#!/usr/bin/env python


import numpy as np
import os
import time


def potential(rho):
	return rho*rho


def create_matrix(N, mesh_min, mesh_max, part):

	A = np.zeros((N, N))

	h = (mesh_max - mesh_min)/float(N)

	h2 = h**2

	rho = np.linspace(mesh_min, mesh_max, N)
	
	V = np.array([potential(rho[i]) for i in range(N)])


	if part == 'b':
		a = -1.0/h2
		d = 2.0/h2	

		for i in range(N-1):
			A[i,i] = d
			A[i, i+1] = a
			A[i+1, i] = a
		A[N-1, N-1] = d


	elif part == 'd':
		# Add potential to diagonal

		a = -1.0/h2
		d = np.array([2.0/h2 + V[i] for i in range(N)]) 
	

		for i in range(N-1):

			A[i,i] = d[i]
			A[i, i+1] = a
			A[i+1, i] = a
		A[N-1, N-1] = d[N-1]


	
	R = np.eye(A.shape[0])

	return A, R











def jacobi_method(A, R):
	
	B = A.copy()

	
	# Find largest off-diagonal value
	# Temporary variable

	B_ = B.copy(); np.fill_diagonal(B_, 0)

	B_ = np.abs(B_)

	result = np.where(B_ == np.max(B_))

	# Indicies of max value
	k = result[0][0]
	l = result[1][0]


	# t=tan(theta), s=sin(theta), c=cos(theta)

	tau = (A[l,l] - A[k,k])/(2.0*A[k,l])

	t_list = []
	
	t_list.append(-tau + np.sqrt(1 + tau**2))
	t_list.append(-tau - np.sqrt(1 + tau**2))

	# Choose smallest value
	t = min(t_list)
	c = 1./np.sqrt(1+t**2)
	s = t*c
	
	# Perform jacobi rotation

	B[k,l] = 0
	B[l,k] = 0
	
	for i in range(A.shape[1]):
		
		if i != k and i != l:
			
			B[i,i] = A[i,i]
			B[i,k] = A[i,k]*c - A[i,l]*s
			B[i,l] = A[i,l]*c + A[i,k]*s


		
	B[k,k] = A[k,k]*c**2 - 2*A[k,l]*c*s + A[l,l]*s**2
	B[l,l] = A[l,l]*c**2 + 2*A[k,l]*c*s + A[k,k]*s**2
	#B[k,l] = (A[k,k] -A[l,l])*s*c + A[k,l]*(c**2 - s**2)

	

	r_ik = R[i,k]
	r_il = R[i,l]

	R[i,k] = c*r_ik - s*r_il
	R[i,l] = c*r_il + s*r_ik


	return B, R


def Rotate(A, R, N, epsilon, mesh_min, mesh_max, part):

	# Start values
	A, R = create_matrix(N, mesh_min, mesh_max, part)
	A_temp = A.copy(); np.fill_diagonal(A_temp, 0)
	norm = np.linalg.norm(A_temp)


	# Perform rotation until A is diagonal	

	counter = 0
	while norm > epsilon:
					
		B, R_new = jacobi_method(A, R)

		# Find norm of off-diagonal elements
		B_temp = B.copy(); np.fill_diagonal(B_temp, 0)
		norm = np.linalg.norm(B_temp)	
	
		A = B

		counter += 1
	
	A_final = A
	R_final = R_new


	return A_final, R_final



	


def test_eigenvalues():

	N = 5
	mesh_min = 0.0
	mesh_max = 2.0
	h = (mesh_max - mesh_min)/float(N)
	h2 = h*h

	d = 2./h2
	a = -1./h2

	A = create_matrix(N, mesh_min, mesh_max, part='b')	
	
	
	Eigvals_analytical = np.array([d + 2*a*np.cos((j*np.pi)/(N+1.0)) for j in range(N)])

	tol = 1e-10
	diff = np.abs(Eigvals_analytical - np.linalg.eigvals(A)).max()

	assert diff < tol





