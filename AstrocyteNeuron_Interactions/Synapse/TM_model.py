"""
TM model of synapse
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *

set_device('cpp_standalone', directory=None) 
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Tripartite synapses')
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
	duration = 3*second
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

	N_syn = 1
	rate_in = 4*Hz
	pre_neurons = PoissonGroup(N_syn, rates=rate_in)
	post_neurons = NeuronGroup(N_syn, model="")

	synapses = Synapses(pre_neurons, post_neurons, model=syn_model, on_pre=action, method='linear')
	synapses.connect(j='i')   
	synapses.x_S = 1.0

	synapse_mon = StateMonitor(synapses, ['u_S','x_S'], record=0, when='after_synapses')
	pre_mon = SpikeMonitor(pre_neurons)
	run(duration, report='text')

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
		

	device.delete()
	plt.show()
	
	


