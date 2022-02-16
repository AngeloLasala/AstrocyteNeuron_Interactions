"""
Mean field description of simple and tripartite synapses.

- "Gliotrasmitter Exocitosis and Its Consequences on Synaptic Transmission, De Pittà"
- "Activity-dependent transmission in neocortical synapses", Tsodyks
"""
import numpy as np 
import matplotlib.pyplot as plt 
from brian2 import *

def STP_mean_field(u_0, nu_S_start=-1, nu_S_stop=2, nu_S_number=200):
	"""
	Mean field solution of simple synapses (no gliotramission modulation)
	described by short-term plasticity.
	Return steady state of synaptic variable, u_S and x_S, for constant 
	synaptic input rate, nu_S (Hz)

	Parameters
	----------
	nu_S_start : integer 
				Order of magnitude of first nu_S value
		
	nu_S_stop : integer 
				Order of magnitude of last nu_S value

	nu_S_number : interger (optionl)
				Total sample's number of nu_S. Default=200
	Returns
	-------
	nu_S : 1D-array
			Sample of synaptic rates (Hz)
	u_S : 1D-array
		Steady states of u_S

	x_S : 1D-array
		Steady state of x_S

	"""
	nu_S = np.logspace(nu_S_start, nu_S_stop, nu_S_number)*Hz
	u_S =  (u_0*(Omega_f+nu_S))/(Omega_f+nu_S*u_0)
	x_S = Omega_d / (Omega_d + u_S*nu_S)

	return nu_S, u_S, x_S

def GRE_mean_field(nu_A_array=[], nu_A_start=-5, nu_A_stop=1, nu_A_number=200, select=True):
	"""
	Mean field solution of gliotrasmission modulation of synaptic release.
	Return the mean value of basal synaptic release probability u_0 from 
	mean field approximation values of astrocyte variable x_A and presynaptic 
	activated receptors Gamma_S for different values of gliorelease rate nu_A.

	Parameters
	----------
	nu_A_array: float or array
				sample of nu_A array

	nu_S_start : integer(optional)
				Order of magnitude of first nu_S value. Defaul=-5
		
	nu_S_stop : integer(optional)
				Order of magnitude of last nu_S value. Default=1

	nu_S_number : interger (optionl)
				Total sample's number of nu_S. Default=200
	Returns
	-------
	nu_A : 1D-array
			Sample of synaptic rates (Hz)
	
	u_0 : 1D-array
		steady state of u_0
	"""
	if select:
		nu_A = nu_A_array*Hz
	else:
		nu_A = np.logspace(nu_A_start, nu_A_stop, nu_A_number)*Hz

	x_A = Omega_A / (Omega_d + U_A*nu_A)
	Gamma_S = J_S*U_A*Omega_A*nu_A / (Omega_A*Omega_G + U_A*nu_A*(Omega_G+J_S*Omega_A))
	u_0 = (1-Gamma_S)*U_0
	

	return nu_A, u_0

if __name__ == "__main__":

	## Parameters #####################################################
	U_0 = 0.6
	U_A = 0.6
	Omega_f = 3.33/second
	Omega_d = 2/second
	Omega_G = 0.5/(60*second)
	Omega_e = 60/second
	Omega_A = 0.6/second
	G_T = 200*mmolar
	rho_e = 6.5e-4
	O_G = 1.5/umolar/second
	J_S = rho_e * O_G * G_T /Omega_e
	print(J_S*Omega_A/(J_S*Omega_A+Omega_G))
	###################################################################

	## STP steady states
	nu_S, u_S, x_S = STP_mean_field(u_0=U_0)

	## Gliotrasmission modulation - release-decresing alpha=0 
	nu_A, u_0 = GRE_mean_field(select=False)

	# CLose-loop filtering behaviorn - mean field solution
	# nu_A is a function of nu_S, I suppose an exponential one.
	# For each values of nu_A we have a single solutions for u_0 that 
	# I put in STD_mean_field solution
	nu_A_sim = (0.1 * np.exp(-nu_S/Hz))
	nu_A_close, u_0_close = GRE_mean_field(nu_A_array=nu_A_sim)
	nu_S_close, u_S_close, x_S_close = STP_mean_field(u_0_close)


	## Plots ##################################################################

	fig1, ax1 = plt.subplots(nrows=1, ncols=1, num='MF solution of filtering STP')
	ax1.plot(nu_S/Hz, u_S*x_S, color='k')
	ax1.set_xlabel(r'$\nu_S$ (Hz)')
	ax1.set_ylabel(r'$\langle \bar{r_S} \rangle$')
	ax1.set_xscale('log')
	ax1.grid(linestyle='dotted')
	ax1.legend()

	fig2, ax2 = plt.subplots(nrows=1, ncols=1, num='MF solution of relese-decreasing GRE')
	ax2.plot(nu_A, u_0, color='C2')
	ax2.set_xlabel(r'$\nu_A$ (Hz)')
	ax2.set_ylabel(r'$\langle \bar{u_0} \rangle$')
	ax2.grid(linestyle='dotted')
	ax2.set_xscale('log')

	fig3, ax3 = plt.subplots(nrows=1, ncols=2, figsize=(12,5),
							 num='MF solution of relese-decreasing: CLOSED')
	ax3[0].plot(nu_S_close, nu_A_close)
	ax3[0].set_ylabel(r'$\nu_A$ (Hz)')
	ax3[0].set_xlabel(r'$\nu_S$ (Hz)')
	ax3[0].grid(linestyle='dotted')
	ax3[0].set_xscale('log')

	ax3[1].plot(nu_S_close, u_S_close*x_S_close, color='C6')
	ax3[1].set_xlabel(r'$\nu_S$ (Hz)')
	ax3[1].set_ylabel(r'$\langle \bar{r_S} \rangle$')
	ax3[1].grid(linestyle='dotted')
	ax3[1].set_xscale('log')

	plt.show()