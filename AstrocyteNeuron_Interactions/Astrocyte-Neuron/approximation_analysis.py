import argparse
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
import makedir

set_device('cpp_standalone', directory=None)  # Use fast "C++ standalone mode"

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

def mean_error(values):
	r_mean, r_error = [], []
	for i in values:
		rrr = np.unique(i)
		r_mean.append(rrr.mean())
		if len(rrr)<30:
			error = (np.max(rrr)-np.min(rrr))/2
		else:
			error = rrr.std()/np.sqrt(len(rrr-1))
		r_error.append(error)
	return r_mean, r_error

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
	u_0 = (1-Gamma_S)*U_0__star
	

	return nu_A, u_0

def guess_fuction_bif(nu_S, nu_A0=0.16, nu_S_bif=1.0, tau_A=1):
	"""
	"""
	pos = np.where(nu_S<nu_S_bif)[0][-1]
	nu_A_1 = [nu_A0 for i in range(pos)]
	nu_A_2 = [nu_A0*np.exp(-(i-nu_S_bif)*tau_A) for i in nu_S[pos:]]
	nu_A = nu_A_1 + nu_A_2
	return np.asanyarray(nu_A)
 
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Tripartite synapses')
	parser.add_argument('O_beta', type=float, help='O_beta values')
	parser.add_argument('-p', action='store_true', help="show paramount plots, default=False")
	args = parser.parse_args()

	mod = {'F':[args.O_beta, 0.6]}
	## PARAMETERS ###################################################################
	# -- Synapse --
	rho_c = 0.005                # Synaptic vesicle-to-extracellular space volume ratio
	Y_T = 500.*mmolar            # Total vesicular neurotransmitter concentration
	Omega_c = 40/second          # Neurotransmitter clearance rate
	U_0__star = 0.6              # Resting synaptic release probability
	Omega_f = 3.33/second        # Synaptic facilitation rate
	Omega_d = 2.0/second         # Synaptic depression rate
	w_e = 0.05*nS                # Excitatory synaptic conductance
	w_i = 1.0*nS                 # Inhibitory synaptic conductance
	# - Presynaptic receptors
	O_G = 1.5/umolar/second      # Agonist binding (activating) rate
	Omega_G = 0.5/(60*second)    # Agonist release (deactivating) rate

	# -- Astrocyte --
	# CICR
	O_P = 0.9*umolar/second      # Maximal Ca^2+ uptake rate by SERCAs
	K_P = 0.05*umolar            # Ca2+ affinity of SERCAs
	C_T = 2*umolar               # Total cell free Ca^2+ content
	rho_A = 0.18                 # ER-to-cytoplasm volume ratio
	Omega_C = 6/second           # Maximal rate of Ca^2+ release by IP_3Rs
	Omega_L = 0.1/second         # Maximal rate of Ca^2+ leak from the ER
	d_1 = 0.13*umolar            # IP_3 binding affinity
	d_2 = 1.05*umolar            # Ca^2+ inactivation dissociation constant
	O_2 = 0.2/umolar/second      # IP_3R binding rate for Ca^2+ inhibition
	d_3 = 0.9434*umolar          # IP_3 dissociation constant
	d_5 = 0.08*umolar            # Ca^2+ activation dissociation constant
	#  IP_3 production
	# Agonist-dependent IP_3 production
	O_beta = mod['F'][0]*umolar/second   # Maximal rate of IP_3 production by PLCbeta
	O_N = 0.3/umolar/second      # Agonist binding rate
	Omega_N = 0.5/second         # Maximal inactivation rate
	K_KC = 0.5*umolar            # Ca^2+ affinity of PKC
	zeta = 10                    # Maximal reduction of receptor affinity by PKC
	# Endogenous IP3 production
	O_delta = mod['F'][1]*umolar/second  # Maximal rate of IP_3 production by PLCdelta
	kappa_delta = 1.5*umolar     # Inhibition constant of PLC_delta by IP_3
	K_delta = 0.1*umolar         # Ca^2+ affinity of PLCdelta
	# IP_3 degradation
	Omega_5P = 0.05/second       # Maximal rate of IP_3 degradation by IP-5P
	K_D = 0.7*umolar             # Ca^2+ affinity of IP3-3K
	K_3K = 1.0*umolar            # IP_3 affinity of IP_3-3K
	O_3K = 4.5*umolar/second     # Maximal rate of IP_3 degradation by IP_3-3K

	# Gliotransmitter release and time course
	C_Theta = 0.5*umolar         # Ca^2+ threshold for exocytosis
	Omega_A = 0.6/second         # Gliotransmitter recycling rate
	U_A = 0.6                    # Gliotransmitter release probability
	G_T = 200*mmolar             # Total vesicular gliotransmitter concentration
	rho_e = 6.5e-4               # Astrocytic vesicle-to-extracellular volume ratio
	Omega_e = 60/second          # Gliotransmitter clearance rate
	alpha = 0.0                  # Gliotransmission nature

	## Approximation 
	J_S = rho_e * O_G * G_T /Omega_e
	#################################################################################


	r_S_mean, r_S_error, rate_in = [], [], []
	for i in [0.2, 0.5, 1.0, 1.5, 2.3, 3.2, 4.6, 10.0, 21.0, 46.0, 100.0]:
		name = f'Data/{i}'
		r_S = np.load(f'{name}/r_S_mean.npy')
		r_S_err = np.load(f'{name}/r_S_error.npy')
		rate = np.load(f'{name}/rate_in.npy')

		r_S_mean.append(r_S)
		r_S_error.append(r_S_err)
		rate_in.append(rate[0])



	## Gliotrasmission modulation - release-decresing alpha=0 
	nu_A, u_0 = GRE_mean_field(select=False)

	if mod['F'][0] == 3.2:
		guess_par = {'nu_A0':0.15, 'nu_S_bif':1.0}

	if mod['F'][0] == 2.0:
		guess_par = {'nu_A0':0.15, 'nu_S_bif':1.5}

	# CLose-loop filtering behaviorn - mean field solution
	# nu_A is a function of nu_S, I suppose an exponential one.
	# For each values of nu_A we have a single solutions for u_0 that 
	# I put in STD_mean_field solution
	nu_A_sim = (0.18 * np.exp(-nu_S/Hz))
	nu_A_sim = guess_fuction_bif(nu_S/Hz, nu_A0=guess_par['nu_A0'], 
								nu_S_bif=guess_par['nu_S_bif'], tau_A=mod['F'][0])

	nu_A_close, u_0_close = GRE_mean_field(nu_A_array=nu_A_sim)
	nu_S_close, u_S_close, x_S_close = STP_mean_field(u_0_close)


	plt.figure(num='TM model solutions')
	plt.errorbar(rate_in, r_S_mean, r_S_error, 
					fmt='o', markersize=5, lw=1.5, capsize=2.0,color='C6', label='simulation')
	nu_S_app, u_S_app, x_S_app = STP_mean_field(u_0=U_0__star)
	plt.plot(nu_S_app/Hz, u_S_app*x_S_app, color='C6', label='mean field approximation')
	plt.xscale('log')
	plt.xlabel(r'$\nu_S$'+' (Hz)')
	plt.ylabel(r'$\langle r_S \rangle$')
	plt.grid(linestyle='dotted')
	plt.legend()
	plt.show()