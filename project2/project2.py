#!/usr/bin/env python


import numpy as np
import os
import matplotlib.pyplot as plt
import time




def potential_term(rho, part):
	
	if part == 'd':
		return rho*rho
		
	elif part == 'e':
		return wr*rho**2 + 1.0/rho

	

def create_matrix(N, mesh_min, mesh_max, part):
		
	
	A = np.zeros((N, N))

	h = (mesh_max - mesh_min)/float(N)

	h2 = h**2

	rho = np.linspace(mesh_min, mesh_max, N)
	
	
	V = np.array([potential_term(rho[i], part) for i in range(N)])

	if part == 'b':
		a = -1.0/h2
		d = 2.0/h2	

		for i in range(N-1):
			A[i,i] = d
			A[i, i+1] = a
			A[i+1, i] = a
		A[N-1, N-1] = d


	elif part == 'd':
		# Add potential V(p)=p*p to diagonal

		a = -1.0/h2
		d = np.array([2.0/h2 + V[i] for i in range(N)]) 
	

		for i in range(N-1):

			A[i,i] = d[i]
			A[i, i+1] = a
			A[i+1, i] = a
		A[N-1, N-1] = d[N-1]

	elif part == 'e':
	
		# Add potential V(p)= wr*rho**2 + 1.rho to diagonal

		a = -1.0/h2
		d = np.array([2.0/h2 + wr*V[i]**2 + 1.0/V[i] for i in range(N)]) 
	

		for i in range(N-1):

			A[i,i] = d[i]
			A[i, i+1] = a
			A[i+1, i] = a
		A[N-1, N-1] = d[N-1]



	else:
		import sys
		print "part should be either 'b' or 'd'."
		sys.exit()

	

	# Eigenvectors
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
	#tau = (B[l,l] - B[k,k])/(2.0*B[k,l])

	t_list = []
	
	t_list.append(-tau + np.sqrt(1 + tau**2))
	t_list.append(-tau - np.sqrt(1 + tau**2))

	# Choose smallest value
	t = min(t_list)
	c = 1./np.sqrt(1+t**2)
	s = t*c
	
	# Perform jacobi rotation

	B[k,l] = 0.0
	B[l,k] = 0.0
	
	for i in range(A.shape[0]):
		
		if i != k and i != l:
			
			B[i,i] = A[i,i]
			B[i,k] = A[i,k]*c - A[i,l]*s
			B[i,l] = A[i,l]*c + A[i,k]*s


		
	B[k,k] = A[k,k]*c**2 - 2*A[k,l]*c*s + A[l,l]*s**2
	B[l,l] = A[l,l]*c**2 + 2*A[k,l]*c*s + A[k,k]*s**2
	
	#B[k,l] = (A[k,k] -A[l,l])*s*c + A[k,l]*(c**2 - s**2)

	
	# Update eigenvectors

	R_new = R.copy()

	r_ik = R_new[i,k]
	r_il = R_new[i,l]

	R_new[i,k] = c*r_ik - s*r_il
	R_new[i,l] = c*r_il + s*r_ik

	return B, R_new




def Rotate(N, epsilon, mesh_min, mesh_max, part):

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
		R = R_new		
		
		counter += 1
	
	A_rotated = A
	R_rotated = R		

	#print "A is diagonal after %g rotations" % counter

	return A_rotated, R_rotated, counter




def test_eigenvalues_schroedinger():
	os.system('clear')
	N = 3
	mesh_min = 0.0
	mesh_max = 3.3
	epsilon = 1e-8
	h = (mesh_max - mesh_min)/float(N)
	h2 = h*h

	eigvals_analytical = np.array([3, 7, 11])

	# Perform rotation
	A_rotated, R_rotated, counter = Rotate(N, epsilon, mesh_min, mesh_max, part='d')


	eigvals = np.sort(np.diag(A_rotated))

	print eigvals
	print eigvals_analytical
	tol = 1e-10

	diff = np.abs(eigvals - eigvals_analytical).max()

	print diff

	assert diff < tol


def test_eigenvalues():
	os.system('clear')
	#print "Testing eigenvalues of rotated matrix"

	N = 3
	mesh_min = 0.0
	mesh_max = 100.0
	epsilon = 1e-8
	h = (mesh_max - mesh_min)/float(N)
	h2 = h*h

	d = 2./h2
	a = -1./h2

	# Perform rotation
	A_rotated, R_rotated, counter = Rotate(N, epsilon, mesh_min, mesh_max, part='b')


	# Eigenvalues of original matrix A
	Eigvals = -np.sort(-(np.array([d + 2*a*np.cos((j*np.pi)/(N+1.0)) for j in range(1, N+1)])))


	# Eigenvalues of rotated matrix
	eigvals = -np.sort(-np.diag(A_rotated))

	print Eigvals
	print eigvals
	
	tol = 1e-6		# Tolerance

		
	diff = np.abs(eigvals - Eigvals).max()
	
	print "Difference:", diff
	assert diff < tol
	






if __name__=='__main__':

	#test_eigenvalues()
	test_eigenvalues_schroedinger()

	
	wr = 0.5

	#Ar1, Rr1,counter1 = Rotate(N=90, epsilon=1e-10, mesh_min=0.0, mesh_max=10.0, part='d')
	#Ar2, Rr2, counter2 = Rotate(N=500, epsilon=1e-10, mesh_min=0.0, mesh_max=10.0, part='e')



	



