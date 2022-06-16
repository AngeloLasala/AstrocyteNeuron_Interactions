"""
Validation of MF approximation of TM model
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib import cm as Cm
import makedir
from brian2 import *

set_device('cpp_standalone', directory=None) 

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
	u_0 = (1-Gamma_S)*U_0__star

	return nu_A, u_0

def CVsquare_u(nu_S, u_0=0.6):
	"""
	Coefficient of variation of us in TM model with respect to nu_S
	"""
	num = Omega_f*(1-u_0)**2*nu_S
	den = (Omega_f+nu_S)*(2*Omega_f+u_0*(2-u_0)*nu_S)
	
	return num/den


def CVsquare_x(nu_S,u_0=0.6):
	"""
	Coefficient of variation of us in TM model with respect to nu_S
	"""
	u_S =  (u_0*(Omega_f+nu_S))/(Omega_f+nu_S*u_0)
	u_S2 = u_S**2
	num = u_S2*nu_S
	den = 2*Omega_d+(2-u_S)*u_S*nu_S
	return num/den

def CVsquare_u1(nu_S, u_0=0.6):
	num = (1-u_0)*nu_S
	den = (Omega_f+nu_S)**2*(2*Omega_f+u_0*(2-u_0)*nu_S)
	return num/den

def CVsquare_xA(nu_A):
	num = U_A**2*nu_A
	den = 2*Omega_A+(2-U_A)*U_A*nu_A
	return num/den


def CVsquare_Gamma_S(nu_A):
	num = Omega_G**2
	den = (Omega_G + (1-beta)*nu_A)*(Omega_G+(1+beta)*nu_A)
	return num/den


def guess_fuction_bif(nu_S, nu_A0=0.16, nu_S_bif=1.0, tau_A=3.2):
	"""
	"""
	if nu_S_bif == 0 : pos = 0
	else: pos = np.where(nu_S<nu_S_bif)[0][-1]
	
	nu_A_1 = [nu_A0 for i in range(pos)]
	nu_A_2 = [nu_A0*np.exp(-(i-nu_S_bif)*tau_A) for i in nu_S[pos:]]
	nu_A = nu_A_1 + nu_A_2
	return np.asanyarray(nu_A)

def validity_TS(nu_S, O_G, Omega_G, nu_A0=0.16, nu_S_bif=1.0, tau_A=3.2):
	"""
	"""
	# nu_A from gues function
	nu_S_guess = nu_S/Hz
	nu_A = guess_fuction_bif(nu_S_guess, nu_A0=nu_A0, nu_S_bif=nu_S_bif, tau_A=tau_A)*Hz

	# parameters
	J_S = rho_e * O_G * G_T /Omega_e
	beta = np.exp(-J_S*U_A)
	Gamma_S_val = (1-beta)*nu_A / (Omega_G + (1-beta)*nu_A)
	u_0_din = (1-Gamma_S_val)*U_0__star	
	
	#C.V.
	cv_Gamma_s = CVsquare_Gamma_S(nu_A)
	cv_u_s = CVsquare_u(nu_S, u_0=0.6)
	
	K = Gamma_S_val / (1-Gamma_S_val)
	# print(X)
	return np.sqrt(cv_u_s), K*np.sqrt(cv_Gamma_s)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='TM model, validation of MF')
	parser.add_argument('-p', action='store_true', help="show paramount plots, default=False")
	args = parser.parse_args()

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
	O_G = 0.5/umolar/second      # Agonist binding (activating) rate
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
	O_beta = 3.2*umolar/second   # Maximal rate of IP_3 production by PLCbeta
	O_N = 0.3/umolar/second      # Agonist binding rate
	Omega_N = 0.5/second         # Maximal inactivation rate
	K_KC = 0.5*umolar            # Ca^2+ affinity of PKC
	zeta = 10                    # Maximal reduction of receptor affinity by PKC
	# Endogenous IP3 production
	O_delta = 0.6*umolar/second  # Maximal rate of IP_3 production by PLCdelta
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
	beta = np.exp(-J_S*U_A)
	#################################################################################

	# approximation
	plt.rc('font', size=13)   
	plt.rc('legend', fontsize=10) 

	nu_S_app, u_S_app, x_S_app = STP_mean_field(u_0=U_0__star)
	nu_A_app, u_0_app = GRE_mean_field(select=False)

	cv_us = CVsquare_u(nu_S_app, u_0=0.7)
	cv_xs = CVsquare_x(nu_S_app, u_0=0.7)
	cv_Gammas = CVsquare_Gamma_S(nu_A_app)
	cv_xa = CVsquare_xA(nu_A_app)

	plt.figure(num='STP')
	plt.plot(nu_S_app/Hz, np.sqrt(cv_us))
	plt.plot(nu_S_app/Hz, np.sqrt(cv_xs)*np.sqrt(cv_us))

	plt.figure(num='GRE')
	plt.plot(nu_A_app, np.sqrt(cv_Gammas))
	# plt.plot(nu_A_app, np.sqrt(cv_xa))
	plt.plot(nu_A_app, np.sqrt(cv_Gammas)*np.sqrt(cv_xa))
	


	# VALIDITY CLOSED-LOOP
	# 3.2 : nu_A0=0.16, nu_S_bif=1.0, tau_A=3.2
	# 2.0 : nu_A0=0.15, nu_S_bif=1.5, tau_A=2.0
	# 0.5 : nu_A0=0.1, nu_S_bif=0.0, tau_A=0.5
	cv_u0, cv_us = validity_TS(nu_S_app, O_G, Omega_G, nu_A0=0.1, nu_S_bif=0.0, tau_A=0.5)

	# validity with respect to O_G and Omega_G
	

	O_G_list = np.linspace(0.5,1.5,10)/umolar/second
	Omega_G_list = np.linspace(0.008,0.10,10)/second
	print()
	plt.figure(num='validity w.r.t. O_G')
	for o_g  in O_G_list:
		cv_u0, cv_us = validity_TS(nu_S_app, o_g, Omega_G=0.01/second, nu_A0=0.1, nu_S_bif=0.0, tau_A=0.5)
		plt.plot(nu_S_app, cv_us*cv_u0, label=f'{o_g*umolar*second}')
		plt.xscale('log')
		plt.legend()

	plt.figure(num='validity w.r.t. Omega_G')
	for omega_g  in Omega_G_list:
		cv_u0, cv_us = validity_TS(nu_S_app, 1.0/umolar/second, omega_g, nu_A0=0.1, nu_S_bif=0.0, tau_A=0.5)
		plt.plot(nu_S_app, cv_us*cv_u0, label=f'{omega_g/Hz}')
		plt.xscale('log')
		plt.legend()



	# plt.figure()
	# plt.plot(nu_S_app, u_S_app*x_S_app, label='STP')
	# plt.plot(nu_A_app, u_0_app, label='GRE')
	# plt.xscale('log')
	# plt.legend()

	
	

	fig1, ax1 = plt.subplots(num='Validity tripartite synapses')
	ax1.plot(nu_S_app, cv_us*cv_u0)
	ax1.set_xscale('log')
	ax1.grid(linestyle='dotted')
	ax1.set_xlabel(r'$\nu_S$'+r' ($\rm{spk/s}$)')
	ax1.set_ylabel(r'$\rm{CV}_{u_S}$ $\rm{CV}_{u_0}$')




	
	
	
	
	
	
	
	# nu_A_val = np.logspace(-3,1,200)*Hz
	# nu_S_val = np.logspace(-1,0,200)*Hz
	# X, Y, Z = validity_TS(nu_A_val, nu_S_val)

	# # Plot the surface.
	# surf = ax.plot_surface(X, Y, Z, cmap=Cm.coolwarm,linewidth=0, antialiased=False)

	# # Customize the z axis.
	# ax.set_zlim(-0.01, 0.15)
	# ax.set_xlabel(r'$\nu_A$ ($\rm{spk/s}$)')
	# ax.set_ylabel(r'$\nu_S$ ($\rm{spk/s}$)')
	# ax.zaxis.set_major_locator(LinearLocator(10))
	# # A StrMethodFormatter is used automatically
	# ax.zaxis.set_major_formatter('{x:.02f}')

	# # Add a color bar which maps values to colors.
	# fig.colorbar(surf, shrink=0.5, aspect=5)

	plt.show()
