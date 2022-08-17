"""
Simulated data and mean field description for heterosynapic connection
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from TM_analysis import chi_square_test, errore_in_quadrature
from scipy.optimize import bisect
import makedir
from brian2 import *
set_device('cpp_standalone', directory=None)

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
	# print(O_G)
	x_A = Omega_A / (Omega_d + U_A*nu_A)
	Gamma_S = J_S*U_A*Omega_A*nu_A / (Omega_A*Omega_G + U_A*nu_A*(Omega_G+J_S*Omega_A))
	# Gamma_S = J_S*U_A*x_A / (Omega_G + (J_S*U_A*x_A*nu_A))
	u_0 = U_0__star + (alpha - U_0__star)*Gamma_S

	# print(J_S*x_A*nu_A)

	return nu_A, u_0

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Bièartite sinapses filter characteristic')
	parser.add_argument('file', type=str, help="file's name of network in 'Synapses' folder")
	args = parser.parse_args()
	name = args.file

	## Parameters #####################################################
	U_0__star = 0.6
	alpha = 0
	U_0 = 0.6
	U_A = 0.6
	Omega_f = 3.33/second
	Omega_d = 2/second
	Omega_G = 0.008/second
	Omega_e = 60/second
	Omega_A = 0.6/second
	G_T = 200*mmolar
	rho_e = 6.5e-4
	O_G = 0.5/umolar/second
	J_S = rho_e * O_G * G_T /Omega_e
	# print(J_S*Omega_A/(J_S*Omega_A+Omega_G))
	u_theta = Omega_d / (Omega_f + Omega_d)

	defaultclock.dt = 1*ms
	###################################################################
	
	u_0_mean_trl = list()
	u_0_std_trl = list()
	for trl in range(1,27):
		rate_in = np.load(f'{name}'+f'/trial-{trl}/rate_in.npy')
		duration = np.load(f'{name}'+f'/trial-{trl}/duration.npy')
		N = np.load(f'{name}'+f'/trial-{trl}/N.npy')
		
		# time arrayd
		dt_syn = 1*ms
		t_relax = 20*second

		#variable array
		u_0 = np.load(f'{name}'+f'/trial-{trl}/u_0.npy')
		
		u_0_mean = u_0.mean(axis=-1)
		u_0_std = u_0.std(axis=-1)

		u_0_mean_trl.append(u_0_mean)
		u_0_std_trl.append(u_0_std)

	u_0_mean_trl = np.asarray(u_0_mean_trl)
	u_0_std_trl = np.asarray(u_0_std_trl)
	u_0_error = errore_in_quadrature(u_0_std_trl)
	
	## Gliotrasmission modulation - release-decresing alpha=0 
	nu_A, u_0 = GRE_mean_field( nu_A_start=-3, nu_A_stop=0, select=False)

	# chi square
	nu_a_chi, u_0_chi = GRE_mean_field(nu_A_array=rate_in, select=True)
	chi_tot = chi_square_test(u_0_chi,  u_0_mean_trl.mean(axis=0).flatten(), u_0_error.flatten())

	print('CHI SQUARE TEST')
	print(f'chi square : {chi_tot/N}')
	print(N)

	u_theta_plt = np.asarray([u_theta for i in range(len(u_0))])
	

	# Intersection of gurve: calcolo le differenze(f-g), 
	# prendo il segno (np.sing), 
	# faccio le differenze tra elementi congiunti (np.diff)
	# prendo di questi solo quello con il valore non nullo (np.argwhere) 
	nu_theta_pos = np.argwhere(np.diff(np.sign(u_0 - u_theta_plt)))[0,0]

	## Plots 
	plt.rc('font', size=13)
	plt.rc('legend', fontsize=10)
	fig2, ax2 = plt.subplots(nrows=1, ncols=1, num='MF solution of relese-decreasing GRE', tight_layout=True)
	ax2.errorbar(rate_in/Hz, u_0_mean_trl.mean(axis=0), u_0_error,
             				fmt='o', markersize=4, lw=0.6, capsize=2.0, color='C2', label='gliotransmission')
	ax2.plot(nu_A, u_0, color='C2', label='mean field approximation')
	ax2.plot(nu_A/Hz, u_theta_plt, color='C0', lw=1, alpha=0.6)
	# ax2.scatter(nu_A[nu_theta_pos], u_theta)
	ax2.axvline(x=nu_A[nu_theta_pos]/Hz, ymin=0, ymax=1, color='C1', lw=1, alpha=0.6)
	ax2.set_xlabel(r'$\nu_A$ ($\rm{gre/s})$')
	ax2.set_ylabel(r'$\langle \bar{u_0} \rangle$')
	ax2.fill_between(nu_A[nu_theta_pos:], u_theta_plt[nu_theta_pos:], color='C0', alpha=0.2)
	ax2.fill_between(nu_A[:nu_theta_pos]/Hz, u_theta_plt[:nu_theta_pos], u_theta_plt[:nu_theta_pos]+0.3,
					color='C1', alpha=0.2)
	ax2.legend()
	ax2.grid(linestyle='dotted')
	ax2.set_xscale('log')
	xticks = [0.001, 0.01, 0.1, 1.0]
	# ax4[2].set_ylim(yrange)
	ax2.set_xticks(np.logspace(-3, 0, 4))
	ax2.set_xticklabels(xticks)
	# ax2.xaxis.set_major_formatter(ScalarFormatter())
	device.delete()
	plt.show()

