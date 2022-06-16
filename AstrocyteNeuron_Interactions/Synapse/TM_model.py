"""
TM model of synapse
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
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

def variance(x):
	"""
	Variance of finite size values. The correction is maded by
	the factor 1/(N-1)

	Parameters
	---------
	x : array
		array of data

	Returns
	-------
	var : float
		variance
	"""
	x = np.asarray(x)
	N = len(x)

	mean = np.mean(x)

	var = x-mean
	var = var**2
	var = var.sum()
	var = var / (N*(N-1))
	return var

def blocking(x ,k=10):
	"""
	Data blocking techniques to estimate the variance of
	correlated variable

	Parameters
	----------
	x : array
		data

	k : integer
		number of block

	Returns
	-------
	variances_list : list
		list of variances for each block
	"""
	x = np.array(x)

	variances_list = []
	for time in range(k):
		N = int(len(x))
		if N%2 != 0:
			N = N-1

		## index odd and even
		index = np.arange(N)
		odd = index[index%2==0]
		even = index[index%2!=0]
		# print(len(odd))
		# print(len(even))

		# variance
		x1 = x[odd]
		x2 = x[even]
		x_block = (x1+x2)/2.0
		var_block = variance(x_block)
		variances_list.append(var_block)
		x = x_block

	return variances_list

def standard_error_I(I, N_mean=10):
	"""
	Compute mean and standard error of recurrent current
	"""
	I = np.asarray(I)
	N = len(I)
	N_window = int(N/N_mean)

	I_list = []
	for i in range(N_mean):
		start = i*N_window
		stop = (i+1)*N_window
		I_mean = I[start:stop]
		I_list.append(I_mean.mean())

	I_list = np.asanyarray(I_list)

	if N_mean < 20 :
		error = (I_list.max()-I_list.min())/2
	else:
		error = I_list.std(ddof=1)/np.sqrt(N_mean)

	return I_list.mean(), error

def chi_square_test(fun, val, std):
	fun = np.asarray(fun)
	val = np.asarray(val)
	std = np.asarray(std)

	chi_e = []
	for f,v,s in zip(fun,val,std):
		# print((f-v)**2 / s**2)
		# print((f-v)**2,s**2)
		chi_e.append((f-v)**2/s**2)
	# print(chi_e)
	chi_e = np.asarray(chi_e)
	chi_square = chi_e.sum()
	return chi_square


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='TM model, approximation and simulation')
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


	## TIME PARAMETERS ##############################################################
	defaultclock.dt = 1*ms
	duration = 520*second
	# seed(28371)  # to get identical figures for repeated runs
	#################################################################################

	## SYNAPSES
	syn_model = """
	# Synaptic variable
	du_S/dt = -Omega_f * u_S : 1 (clock-driven)
	dx_S/dt = Omega_d * (1-x_S) : 1 (clock-driven)
	r_S : 1
	U_0 : 1
	"""

	action="""
	U_0 = U_0__star
	u_S += U_0*(1-u_S)
	r_S = u_S*x_S
	x_S -= r_S

	"""

	N_syn = 25
	# rate_in = [args.r for i in range(N_syn)]*Hz
	rate_in = np.logspace(-1,2,N_syn)*Hz
	pre_neurons = PoissonGroup(N_syn, rates=rate_in)
	post_neurons = NeuronGroup(N_syn, model="")

	synapses = Synapses(pre_neurons, post_neurons, model=syn_model, on_pre=action, method='linear')
	synapses.connect(j='i')
	synapses.x_S = 1.0

	synapse_mon = StateMonitor(synapses, ['u_S','x_S','r_S'], record=np.arange(0,N_syn), when='after_synapses')
	pre_mon = SpikeMonitor(pre_neurons)
	run(duration, report='text')

	trans = 20000   #trans*dt=300000*1*ms=300s

	## SAVE VARIABLE ###################################################################################
	name = f"Data"
	makedir.smart_makedir(name, trial=True)
	trial_index = [int(trl.split('-')[-1]) for trl in os.listdir(name)]
	trial_free = max(trial_index)

	np.save(f'{name}'+f'/trial-{trial_free}/duration', duration)
	np.save(f'{name}'+f'/trial-{trial_free}/rate_in', rate_in)
	np.save(f'{name}'+f'/trial-{trial_free}/N', N_syn)
	np.save(f'{name}'+f'/trial-{trial_free}/r_S', synapse_mon.r_S[:,-int(duration/defaultclock.dt)//3:])
	####################################################################################################

	## ERROR BLOCKING
	# print(len(synapse_mon.r_S[0,:]))
	# k=20
	# variances = blocking(synapse_mon.r_S[0,:], k=k)
	# print(synapse_mon.r_S[0,trans:].mean())
	# print(np.sqrt(variances[-1]))
	# print(np.std(synapse_mon.r_S[0,trans:]))

	# plt.figure()
	# plt.scatter([i+1 for i in range(k)], np.sqrt(variances)/np.sqrt(len(synapse_mon.r_S[0,50000:])))
	# plt.yscale('log')
	# plt.show()

	# r_S_mean, r_S_error = [], []
	# for rate, rs in zip(rate_in[:], synapse_mon.r_S[:,trans:]):
	# 	# print(rate, rs)
	# 	if rate/Hz<=1.0 : NN = int(duration*(rate/10))
	# 	else: NN = 60
	# 	mm, ss = standard_error_I(rs,N_mean=NN)
	# 	# print((ss/mm)*100)
	# 	r_S_mean.append(mm)
	# 	r_S_error.append(ss)

	# plt.figure()
	# plt.errorbar(rate_in/Hz, r_S_mean, r_S_error)
	# plt.plot(r_S_mean)



	# #DATA for ERROR OVER T
	# xxx=[]
	# r_S_mean, r_S_std = [],[]
	# for ind in range(len(rate_in)):
	# 	x = synapse_mon.r_S[ind,trans:]
	# 	sam = int((1/rate_in[ind])/defaultclock.dt)
	# 	indeces = np.arange(len(x))
	# 	r_S_sample = x[indeces%sam==0]
	# 	if ind == 30 : xxx.append(r_S_sample)
	# 	r_S_sample = np.array(r_S_sample)
	# 	r_S_mean.append(np.mean(r_S_sample))
	# 	r_S_std.append(np.std(r_S_sample,ddof=1)/np.sqrt(len(r_S_sample)))

	# plt.figure()
	# pois=xxx[0]
	# # print(len(pois))
	# plt.hist(pois, density=True)

	# nu_S_app, u_S_app, x_S_app = STP_mean_field(u_0=U_0__star,nu_S_start=-1,nu_S_stop=2,nu_S_number=N_syn)

	# chi_1 = chi_square_test(u_S_app[:N_syn//2]*x_S_app[:N_syn//2],
	# 						np.mean(synapse_mon.r_S[:N_syn//2,trans:], axis=1),
	# 						np.std(synapse_mon.r_S[:N_syn//2,trans:],ddof=1, axis=1))

	# chi_2 = chi_square_test(u_S_app[N_syn//2:]*x_S_app[N_syn//2:],
	# 						np.mean(synapse_mon.r_S[N_syn//2:,trans:], axis=1),
	# 						np.std(synapse_mon.r_S[N_syn//2:,trans:],ddof=1, axis=1))

	# # chi_tot = chi_square_test(u_S_app*x_S_app,
	# # 						np.mean(synapse_mon.r_S[:,trans:], axis=1),
	# # 						np.std(synapse_mon.r_S[:,trans:],ddof=1, axis=1))

	# chi_tot = chi_square_test(u_S_app*x_S_app, r_S_mean, r_S_error)


	# print("CHI TEST")
	# print(f'chi all : {chi_tot/(N_syn)}')
	# print(f'chi 1 : {chi_1/(N_syn//2)}')
	# print(f'chi 2 : {chi_2/(N_syn//2)}')

	## Plots #########################################################################################
	if args.p:
		plt.rc('font', size=13)
		plt.rc('legend', fontsize=10)
		# fig1, ax1 = plt.subplots(nrows=2, ncols=1, figsize=(10,6.5), sharex=True, num='TM model, synaptic varibles')
		# ni=0
		# spk_index = np.in1d(synapse_mon.t, pre_mon.t[pre_mon.i == ni])

		# # Super-impose reconstructed solutions
		# time = synapse_mon.t  # time vector
		# tspk = Quantity(synapse_mon.t, copy=True)  # Spike times
		# for ts in pre_mon.t[pre_mon.i == ni]:
		# 	tspk[time >= ts] = ts
		# ax1[0].plot(synapse_mon.t/second, 1 + (synapse_mon.x_S[0]-1)*exp(-(time-tspk)*Omega_d),
		# 		'-', color='C4', label=r'$x_S$')
		# ax1[0].plot(synapse_mon.t/second, synapse_mon.u_S[0]*exp(-(time-tspk)*Omega_f),
		# 		'-', color='C1', label=r'$u_S$')
		# ax1[0].set_ylabel(r'$u_S$, $x_S$')
		# ax1[0].grid(linestyle='dotted')
		# ax1[0].legend(loc='upper right')

		# nspikes = np.sum(spk_index)

		# x_S_spike = synapse_mon.x_S[0][spk_index]
		# u_S_spike = synapse_mon.u_S[0][spk_index]
		# ax1[1].vlines(synapse_mon.t[spk_index]/second, np.zeros(nspikes),
        # 				x_S_spike*u_S_spike/(1-u_S_spike), color='C8')
		# ax1[1].set_ylabel(r'$r_S$')
		# ax1[1].set_xlabel('time '+r'($\rm{s}$)')
		# ax1[1].grid(linestyle='dotted')

		fig2, ax2 = plt.subplots(num='STP approx vs data', tight_layout=True)
		ax2.errorbar(rate_in/Hz, np.mean(synapse_mon.r_S[:,trans:], axis=1), np.std(synapse_mon.r_S[:,trans:],ddof=1, axis=1),
                fmt='o', markersize=4, lw=0.6, color='black', label='STP')
		# ax2.errorbar(rate_in/Hz, r_S_mean, r_S_error,
                # fmt='o', markersize=4, lw=0.6, color='C0', label='STP')
		nu_S_app, u_S_app, x_S_app = STP_mean_field(u_0=U_0__star)
		ax2.plot(nu_S_app/Hz, u_S_app*x_S_app, color='k', label='mean field approximation')
		ax2.set_xscale('log')
		ax2.set_xlabel(r'$\nu_S$'+r' ($\rm{spk/s}$)')
		ax2.set_ylabel(r'$\langle {r_S} \rangle$')
		ax2.grid(linestyle='dotted')
		yticks = [ 0.1, 1.0, 10.0, 100.0]
		# ax4[2].set_ylim(yrange)
		ax2.set_xticks(np.logspace(-1, 2, 4))
		ax2.set_xticklabels(yticks)
		ax2.legend()
		print(rate_in.shape)
		print(synapse_mon.r_S[:, trans:].shape)
	device.delete()
	plt.show()




