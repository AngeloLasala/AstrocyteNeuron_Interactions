"""
TM model of synapse
"""
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

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='TM model, approximation and simulation')
	parser.add_argument('r', type=float, help="presynaptic firing rate")
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
	defaultclock.dt = 0.05*ms
	duration = 200*second
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

	N_syn = 20
	rate_in = [args.r for i in range(N_syn)]*Hz
	pre_neurons = PoissonGroup(N_syn, rates=rate_in)
	post_neurons = NeuronGroup(N_syn, model="")

	synapses = Synapses(pre_neurons, post_neurons, model=syn_model, on_pre=action, method='linear')
	synapses.connect(j='i')   
	synapses.x_S = 1.0

	synapse_mon = StateMonitor(synapses, ['u_S','x_S','r_S'], record=np.arange(0,N_syn), when='after_synapses')
	pre_mon = SpikeMonitor(pre_neurons)
	run(duration, report='text')


	name = f"Data/{rate_in[0]}/"
	makedir.smart_makedir(name)
	np.mean(synapse_mon.r_S[:]), np.std(synapse_mon.r_S[:])/np.sqrt(len(rate_in))
	np.save(f'{name}/rate_in',rate_in)
	np.save(f'{name}/r_S_mean',np.mean(synapse_mon.r_S[:]))
	np.save(f'{name}/r_S_error',np.std(synapse_mon.r_S[:])/np.sqrt(len(rate_in)))

	## Plots #########################################################################################
	if args.p:
		fig1, ax1 = plt.subplots(nrows=2, ncols=1, figsize=(10,6.5), sharex=True, num='TM model, synaptic varibles')
		ni=0
		spk_index = np.in1d(synapse_mon.t, pre_mon.t[pre_mon.i == ni])

		# Super-impose reconstructed solutions
		time = synapse_mon.t  # time vector
		tspk = Quantity(synapse_mon.t, copy=True)  # Spike times
		for ts in pre_mon.t[pre_mon.i == ni]:
			tspk[time >= ts] = ts
		ax1[0].plot(synapse_mon.t/second, 1 + (synapse_mon.x_S[0]-1)*exp(-(time-tspk)*Omega_d),
				'-', color='C1', label=r'$u_S$')
		ax1[0].plot(synapse_mon.t/second, synapse_mon.u_S[0]*exp(-(time-tspk)*Omega_f),
				'-', color='C4', label=r'$x_S$')
		ax1[0].set_ylabel(r'$u_S$, $x_S$')
		ax1[0].grid(linestyle='dotted')
		ax1[0].legend(loc='upper right')
		
		nspikes = np.sum(spk_index)

		x_S_spike = synapse_mon.x_S[0][spk_index]
		u_S_spike = synapse_mon.u_S[0][spk_index]
		ax1[1].vlines(synapse_mon.t[spk_index]/second, np.zeros(nspikes),
        				x_S_spike*u_S_spike/(1-u_S_spike), color='C8')
		ax1[1].set_ylabel(r'$r_S$')
		ax1[1].set_xlabel('time (s)')
		ax1[1].grid(linestyle='dotted')

		plt.figure()
		plt.errorbar(rate_in[0]/Hz, np.mean(synapse_mon.r_S[:]), np.std(synapse_mon.r_S[:])/np.sqrt(len(rate_in)), 
                fmt='o', markersize=4, lw=0.4, color='black', label='no gliotrasmission')
		nu_S_app, u_S_app, x_S_app = STP_mean_field(u_0=U_0__star)
		plt.plot(nu_S_app/Hz, u_S_app*x_S_app, color='k', label='mean field approximation')
		plt.xscale('log')
		print(rate_in.shape)
		print(synapse_mon.r_S[:].shape)
	device.delete()
	plt.show()
	
	


