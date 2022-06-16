"""
Simulated data and mean field description for heterosynapic connection
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from TM_model import chi_square_test
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
	print(O_G)
	x_A = Omega_A / (Omega_d + U_A*nu_A)
	Gamma_S = J_S*U_A*Omega_A*nu_A / (Omega_A*Omega_G + U_A*nu_A*(Omega_G+J_S*Omega_A))
	u_0 = U_0__star + (alpha - U_0__star)*Gamma_S

	# print(J_S*x_A*nu_A)


	return nu_A, u_0

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='TM model, approximation and simulation')
	parser.add_argument('-p', action='store_true', help="show paramount plots, default=False")
	args = parser.parse_args()

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
	U_thets = Omega_d / (Omega_f + Omega_d)

	defaultclock.dt = 1*ms
	duration = 1500*second
	trans = 20000*2   #trans*dt=20000*1*ms=20s
	# seed(28371)  # to get identical figures for repeated runs
	###################################################################

	## Gliorelese
	gliorelease_model = """
    dx_A/dt = Omega_A * (1 - x_A) : 1 (clock-driven)
    dG_A/dt = -Omega_e*G_A        : mmolar (clock-driven)
	dGamma_S/dt = O_G*G_A*(1-Gamma_S)-Omega_G*Gamma_S  : 1 (clock-driven)
	u_0 = U_0__star + (alpha - U_0__star)*Gamma_S : 1
	"""

	astro_release = """
    G_A += rho_e*G_T*U_A*x_A
    x_A -= U_A * x_A
    """

	N_syn = 25
	# rate_in = [args.r for i in range(N_syn)]*Hz
	rate_in = np.logspace(-3,0,N_syn)*Hz
	pre_neurons = PoissonGroup(N_syn, rates=rate_in)
	post_neurons = NeuronGroup(N_syn, model="")

	synapses = Synapses(pre_neurons, post_neurons, model=gliorelease_model, on_pre=astro_release, method='rk2')
	synapses.connect(j='i')
	synapses.x_A = 1.0

	synapse_mon = StateMonitor(synapses, ['u_0'], record=np.arange(0,N_syn), when='after_synapses')
	pre_mon = SpikeMonitor(pre_neurons)
	run(duration, report='text')

	## SAVE VARIABLE
	name = f"Data_hetero"
	makedir.smart_makedir(name, trial=True)
	trial_index = [int(trl.split('-')[-1]) for trl in os.listdir(name)]
	trial_free = max(trial_index)

	np.save(f'{name}'+f'/trial-{trial_free}/duration', duration)
	np.save(f'{name}'+f'/trial-{trial_free}/rate_in', rate_in)
	np.save(f'{name}'+f'/trial-{trial_free}/N', N_syn)
	np.save(f'{name}'+f'/trial-{trial_free}/u_0', synapse_mon.u_0[:,-int(duration/defaultclock.dt)//3:])

	## Gliotrasmission modulation - release-decresing alpha=0 
	nu_A, u_0 = GRE_mean_field( nu_A_start=-3, nu_A_stop=0, select=False)

	# chi square
	nu_a_chi, u_0_chi = GRE_mean_field(nu_A_array=rate_in/Hz, select=True)
	chi_tot = chi_square_test(u_0_chi, np.mean(synapse_mon.u_0[:,trans:], axis=1), 
							np.std(synapse_mon.u_0[:,trans:],ddof=1, axis=1))

	print('CHI SQUARE TEST')
	print(f'chi square : {chi_tot/N_syn}')

	# plt.figure()
	# plt.plot(synapse_mon.t[:]/second, synapse_mon.u_0[0,:], label=rate_in[0])
	# plt.plot(synapse_mon.t[:]/second, synapse_mon.u_0[25,:], label=rate_in[25])
	# plt.plot(synapse_mon.t[:]/second, synapse_mon.u_0[49,:], label=rate_in[49])
	# plt.legend()

	## Plots 
	if args.p:
		plt.rc('font', size=13)
		plt.rc('legend', fontsize=10)
		fig2, ax2 = plt.subplots(nrows=1, ncols=1, num='MF solution of relese-decreasing GRE', tight_layout=True)
		ax2.errorbar(rate_in/Hz, np.mean(synapse_mon.u_0[:,trans:], axis=1), 
								np.std(synapse_mon.u_0[:,trans:],ddof=1, axis=1),
								fmt='o', markersize=4, lw=0.6, color='C2', label='gliotransmission')
		ax2.plot(nu_A, u_0, color='C2', label='mean field approximation')
		ax2.set_xlabel(r'$\nu_A$ ($\rm{gre/s})$')
		ax2.set_ylabel(r'$\langle u_0 \rangle$')
		ax2.legend()
		ax2.grid(linestyle='dotted')
		ax2.set_xscale('log')
		xticks = [0.001, 0.010, 0.1, 1.0]
		# ax4[2].set_ylim(yrange)
		ax2.set_xticks(np.logspace(-3, 0, 4))
		ax2.set_xticklabels(xticks)
		# ax2.xaxis.set_major_formatter(ScalarFormatter())
	device.delete()
	plt.show()

