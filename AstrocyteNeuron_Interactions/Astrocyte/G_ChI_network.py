"""
From analysis discused in Report about Tripartite synapses, it is clear the paramount role
of astrocyte dynamics. The stady state of I and C is one of main protagonist of facilitation effect 
and the bell-shape curve of filtering characteristi. 
Here is present an initial dynamical behavior analysis of G-ChI model with the same parameters present in
neuro-glia nework to undestang the type of modulation (Amplitude of Frequecy) and how the stady 
stady changes w.r.t. neurotrasmitter concentration
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from brian2 import *


def Biforcation_brian(variable, trans):
	"""
	Biforcation analysis of given dinamical variable computed
	throught Brian2 simulator.
    
	From continous variable is extrapolate the discrete map
	of maximum and minimum. 

	Parameters
	----------
	variable: 2d array 
			

	Returns
	-------
	"""
	I_list = list()
	Bif_list = list()
    
	for X in variable:
		X = X[t_relax:]

		max_loc = argrelextrema(X, np.greater)
		min_loc = argrelextrema(X, np.less)

		X_max = X[max_loc].tolist()
		X_min = X[min_loc].tolist()
		Bif_val = X_max + X_min
		I_x = [i for item in range(len(Bif_val))]

		I_list.append(I_x)
		Bif_list.append(Bif_val)  
        
	return I_list, Bif_list

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

if __name__ == "__main__":
	## PARAMETERS ###################################################################
	# -- Neuron --
	E_l = -60*mV                 # Leak reversal potential
	g_l = 9.99*nS                # Leak conductance
	E_e = 0*mV                   # Excitatory synaptic reversal potential
	E_i = -80*mV                 # Inhibitory synaptic reversal potential
	C_m = 198*pF                 # Membrane capacitance
	tau_e = 5*ms                 # Excitatory synaptic time constant
	tau_i = 10*ms                # Inhibitory synaptic time constant
	tau_r = 5*ms                 # Refractory period
	I_ex = 100*pA                # External current
	V_th = -50*mV                # Firing threshold
	V_r = E_l                    # Reset potential

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
	O_beta = 0.5*umolar/second   # Maximal rate of IP_3 production by PLCbeta
	O_N = 0.3/umolar/second      # Agonist binding rate
	Omega_N = 0.5/second         # Maximal inactivation rate
	K_KC = 0.5*umolar            # Ca^2+ affinity of PKC
	zeta = 10                    # Maximal reduction of receptor affinity by PKC
	# Endogenous IP3 production
	O_delta = 1.2*umolar/second  # Maximal rate of IP_3 production by PLCdelta
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
	#################################################################################
	U_0 = 0.6
	J_S = rho_e * O_G * G_T /Omega_e

	## TIME PARAMETERS ##############################################################
	defaultclock.dt = 0.01*second
	duration = 250*second
	#################################################################################

	## Astrocyte
	astro_eqs = """
	# Fraction of activated astrocyte receptors (1):
	dGamma_A/dt = O_N * Y_S * (1 - clip(Gamma_A,0,1)) -
				Omega_N*(1 + zeta * C/(C + K_KC)) * clip(Gamma_A,0,1) : 1

	# IP_3 dynamics (1)
	dI/dt = J_beta + J_delta - J_3K - J_5P                           : mmolar

	J_beta = O_beta * Gamma_A                                        : mmolar/second
	J_delta = O_delta/(1 + I/kappa_delta) * C**2/(C**2 + K_delta**2) : mmolar/second
	J_3K = O_3K * C**4/(C**4 + K_D**4) * I/(I + K_3K)                : mmolar/second
	J_5P = Omega_5P*I                                                : mmolar/second

	# Calcium dynamics (2):
	dC/dt = J_r + J_l - J_p                                   : mmolar
	dh/dt = (h_inf - h)/tau_h                                 : 1

	J_r = (Omega_C * m_inf**3 * h**3) * (C_T - (1 + rho_A)*C) : mmolar/second
	J_l = Omega_L * (C_T - (1 + rho_A)*C)                     : mmolar/second
	J_p = O_P * C**2/(C**2 + K_P**2)                          : mmolar/second
	m_inf = I/(I + d_1) * C/(C + d_5)                         : 1
	h_inf = Q_2/(Q_2 + C)                                     : 1
	tau_h = 1/(O_2 * (Q_2 + C))                               : second
	Q_2 = d_2 * (I + d_1)/(I + d_3)                           : mmolar

	# Fraction of gliotransmitter resources available for release (1):
	dx_A/dt = Omega_A * (1 - x_A) : 1
	# gliotransmitter concentration in the extracellular space (1):
	dG_A/dt = -Omega_e*G_A        : mmolar

	# Neurotransmitter concentration in the extracellular space:
	Y_S                         : mmolar
	"""
	nu_S, u_S, x_S = STP_mean_field(u_0=U_0)
	Y_S_daniele = (rho_c*Y_T*u_S*x_S*nu_S)/(Omega_c)
	for i,hh in zip (nu_S, Y_S_daniele):
		print(i,hh)

	plt.figure()
	plt.plot(nu_S/Hz, Y_S_daniele/umolar, color='k')

	# y_s_values = [0]*uM
	# Y_S_function = TimedArray(y_s_values, dt=duration)
	N_a = 500
	astrocyte = NeuronGroup(N_a, model=astro_eqs, method='rk4')
	astrocyte.Y_S = np.linspace(8,20,N_a)*umolar
	astrocyte.h = 0.9
	astrocyte.I = 0.01*umolar
	astrocyte.C = 0.01*umolar

	# Monitor and run
	astro_mon = StateMonitor(astrocyte, ['Y_S','C'], record=True)

	run(duration, report='text')
	
	par_list = list()
	Bif_list = list()

	for X, par in zip(astro_mon.C[:], astrocyte.Y_S[:]):
		X = X[5000:]

		max_loc = argrelextrema(X, np.greater)
		min_loc = argrelextrema(X, np.less)

		X_max = X[max_loc].tolist()
		X_min = X[min_loc].tolist()
		Bif_val = X_max + X_min
		par_x = [par for item in range(len(Bif_val))]

		par_list.append(par_x)
		Bif_list.append(Bif_val)  


	## Plots ##########################################################################

	fig1, ax1 = plt.subplots(nrows=1, ncols=1, num='Calcium stady state w.r.t. Y_S, network parameters')

	for i in range(5):
		ax1.plot(astro_mon.t[:]/second, astro_mon.C[i]/umolar, label = r'$Y_S$ ='+f' {astrocyte.Y_S[i]/umolar:.2f} uM')
	ax1.axhline(C_Theta/umolar,0,duration/second, ls='dashed', color='black')
	ax1.set_ylabel(r'C ($\mu M$)')
	ax1.set_xlabel('time (s)')
	ax1.legend()
	ax1.grid(linestyle='dotted')

	plt.figure()
	for y, bif in zip(par_list, Bif_list):
		plt.plot(y, bif, 'go', markersize=2)

	plt.show()


