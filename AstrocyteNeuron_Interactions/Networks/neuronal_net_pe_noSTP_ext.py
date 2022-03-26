"""
Neuronal network simulation using Brian 2.
Randomly connected networks with conductance-based synapses without any kind of plasticity.
Poisson heterogeneity input with different synaptic strenght for E and I

Plot and Save information: Computational time for each simulation is about 10 second, 
for this reason I chose to save only firing rate and LFP in external folder to future analysis.
If you want dynamical beahaviour of variable run this code, not 'plot_network.py'.

- "Modeling euron-glia interaction with Brian 2 simulator", Stimberg et al (2017)
"""
import argparse
import matplotlib.pyplot as plt
from scipy import signal
from random import randrange
from brian2 import *
from Module_network import neurons_firing, transient
import constant_EI as k_EI
from AstrocyteNeuron_Interactions import makedir

set_device('cpp_standalone', directory=None) 
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='EI network with costantexternal input (Poisson)')
	parser.add_argument('r', type=float, help="rate input of external poisson proces")
	parser.add_argument('-p', action='store_true', help="show paramount plots, default=False")
	parser.add_argument('-no_connection', action='store_true', 
						help="there is NO recurrent connection (no synaptic connection), default=False")
	args = parser.parse_args()
	## Parameters ########################################################################

	# Network size
	N_e = 3200               #Total number of excitatory neurons
	N_i = 800                #Total number of inhibitory neurons

	#Neurons parameters
	E_l = -60*mV           # Leak reversal potential
	g_l = 9.99*nS          # Leak conductance
	E_e = 0*mV             # Excitatory synaptic reversal potential
	E_i = -80*mV           # Inhibitory synaptic reversal potential
	C_m = 198*pF           # Membrane capacitance
	tau_e = 5*ms           # Excitatory synaptic time constant
	tau_i = 10*ms          # Inhibitory synaptic time constant
	tau_r = 5*ms           # Refractory period
	I_ex = 110*pA          # External current
	V_th = -50*mV          # Firing threshold
	V_r = E_l              # Reset potential 

	#Synapse parameters
	w_e = 0.05*nS          # Excitatory synaptic conductance
	w_i = 1.0*nS           # Inhibitory synaptic conductance
	U_0 = 0.6              # Synaptic release probability at rest
	Omega_d = 2.0/second   # Synaptic depression rate
	Omega_f = 3.33/second  # Synaptic facilitation rate
	#############################################################################################

	## MODEL   ##################################################################################
	defaultclock.dt = k_EI.dt*ms
	duration = k_EI.duration*second 
	seed(19958)

	#Neurons
	neuron_eqs = """
	# External input from external synapses
	I_syn_ext = w_ext * (E_e-v) * X_ext : ampere
	dv/dt = (g_l*(E_l-v) + g_e*(E_e-v) + g_i*(E_i-v) + I_syn_ext)/C_m : volt (unless refractory)
	dg_e/dt = -g_e/tau_e : siemens   # post-synaptic excitatory conductance
	dg_i/dt = -g_i/tau_i : siemens   # post-synaptic inhibitory conductance
	dX_ext/dt = -X_ext/tau_e :  1    # post-synaptic external input

	LFP = (abs(g_e*(E_e-v)) + abs(g_i*(E_i-v)) + abs(I_syn_ext))/g_l : volt
	w_ext : siemens                  # external conductance
	"""
	neurons = NeuronGroup(N_e+N_i, model=neuron_eqs, method='rk4',
						threshold='v>V_th', reset='v=V_r', refractory='tau_r')

	#External input
	rate_in = args.r*Hz
	poisson = PoissonInput(neurons, 'X_ext', 160 , rate=rate_in, weight='1')

	neurons.v = 'E_l + rand()*(V_th-E_l)'
	neurons.g_e = 'rand()*w_e'
	neurons.g_i = 'rand()*w_i'

	exc_neurons = neurons[:N_e]
	inh_neurons = neurons[N_e:]

	# Synapses connection
	s = k_EI.s
	exc_neurons.w_ext = w_e
	inh_neurons.w_ext = s*w_e 
	
	g = k_EI.g
	p_e = k_EI.p_e
	p_i = p_e/g

	if not(args.no_connection):
		## actually synaptic connection
		exc="g_e_post+=w_e"
		inh="g_i_post+=w_i"

		exc_syn = Synapses(exc_neurons, neurons, model="", on_pre=exc)
		inh_syn = Synapses(inh_neurons, neurons, model="", on_pre=inh)

		exc_syn.connect(p=p_e)
		inh_syn.connect(p=p_i)
	else:
		#no recurrent connection
		pass   

	#############################################################################################

	## RUN and MONITOR  ######################################################################### 
	spikes_exc_mon = SpikeMonitor(exc_neurons)
	spikes_inh_mon = SpikeMonitor(inh_neurons)
	spikes_mon = SpikeMonitor(neurons)
	firing_rate_exc = PopulationRateMonitor(exc_neurons)
	firing_rate_inh = PopulationRateMonitor(inh_neurons)
	firing_rate = PopulationRateMonitor(neurons)

	# select random excitatory neurons
	index = 488
	state_exc_mon = StateMonitor(exc_neurons, ['v', 'g_e', 'g_i', 'LFP', 'I_syn_ext'], record=np.arange(N_e))
	state_inh_mon = StateMonitor(inh_neurons, ['v', 'g_e', 'g_i', 'LFP', 'I_syn_ext'], record=index)

	run(duration, report='text')
	print('NETWORK')
	print(f'g: {g}')
	print(f's: {s}')
	if not(args.no_connection):
		print(f'exc syn: {exc_syn.N[:]}')
		print(f'inh syn: {inh_syn.N[:]}')
	print('\n')

	#Transient time
	trans_time = k_EI.trans_time
	trans = transient(state_exc_mon.t[:]/second*second, trans_time)

	## Network variable
	LFP = state_exc_mon.LFP[:].sum(axis=0)
	fr_exc = firing_rate_exc.smooth_rate(window='gaussian', width=1*ms)
	fr_inh = firing_rate_inh.smooth_rate(window='gaussian', width=1*ms)
	fr = firing_rate.smooth_rate(window="gaussian", width=1*ms)

	print('Average population firing rate')
	print(f'exc: {fr_exc[trans:].mean()}')
	print(f'inh: {fr_inh[trans:].mean()}')
	print(f'population: {fr[trans:].mean()}')
	print(f'I/E: {fr_inh[trans:].mean()/fr_exc[trans:].mean()}')

	## Spectra Analysis and Neuron's firing rates distribuction
	freq_fr, spectrum_fr = signal.welch(firing_rate.rate[trans:], fs=1/defaultclock.dt/Hz, nperseg=len(firing_rate.rate[:])//4)
	freq_LFP, spectrum_LFP = signal.welch(LFP[trans:], fs=1/defaultclock.dt/Hz, nperseg=len(LFP)//4)

	neurons_fr = neurons_firing(spikes_mon.t[:]/second, spikes_mon.i[:], time_start=0.5, time_stop=2.3)
	exc_neurons_fr = neurons_firing(spikes_exc_mon.t[:]/second, spikes_exc_mon.i[:], time_start=0.5, time_stop=2.3)
	inh_neurons_fr = neurons_firing(spikes_inh_mon.t[:]/second, spikes_inh_mon.i[:], time_start=0.5, time_stop=2.3)
	#########################################################################################################
	## SAVE VARIABLES #######################################################################################
	if not(args.no_connection): name = f"Neural_network/EI_net_noSTP/Network_g{g}_s{s}_v_in{rate_in/Hz}"
	else: name = f"Neural_network/EI_net_noSTP/Network_g{g}_s{s}_v_in{rate_in/Hz}_no_connection"
	
	makedir.smart_makedir(name)

	# Time evolytion variable
	np.save(f'{name}/trans',trans)

	# Excitatory neurons variable
	np.save(f'{name}/state_exc_mon.t',state_exc_mon.t)
	np.save(f'{name}/state_exc_mon.I_syn_ext',state_exc_mon.I_syn_ext[index])
	np.save(f'{name}/state_inh_mon.I_syn_ext',state_inh_mon.I_syn_ext)

	# Population istantaneus firing rate
	np.save(f'{name}/firing_rate_exc.t',firing_rate_exc.t)
	np.save(f'{name}/firing_rate_inh.t',firing_rate_inh.t)
	np.save(f'{name}/fr_exc',fr_exc)
	np.save(f'{name}/fr_inh',fr_inh)
	#########################################################################################################

	# Plots  ################################################################################################
	if args.p:
		fig1, ax1 = plt.subplots(nrows=3, ncols=1, sharex=True, 
								num=f'exc variable dynamic, v_in={rate_in/Hz} (no STP)',figsize=(9,10))

		ax1[0].plot(state_exc_mon.t[trans:]/ms, state_exc_mon.I_syn_ext[index,trans:]/pA,
					color='C3', label=f'{index}'+r' $I_{syn ext}$')
		ax1[0].set_ylabel(r'$I_{syn ext} (pA)$')
		ax1[0].grid(linestyle='dotted')
		ax1[0].legend(loc = 'upper right')

		ax1[1].plot(state_exc_mon.t[trans:]/ms, state_exc_mon.v[index,trans:]/mV, label=f' exc neuron {index}')
		ax1[1].axhline(V_th/mV, color='C2', linestyle=':')
		for spk in spikes_exc_mon.t[spikes_exc_mon.i == index]:
			if (spk>trans_time*ms): ax1[1].axvline(x=spk/ms, ymin=0.15, ymax=0.95 )
		ax1[1].set_ylim(bottom=-60.8, top=0.1)
		ax1[1].set_ylabel('Membran potential (V)')
		ax1[1].grid(linestyle='dotted')
		ax1[1].legend(loc = 'upper right')

		ax1[2].plot(state_exc_mon.t[trans:]/ms, state_exc_mon.g_i[index,trans:]/nS, color='C4', label=f'{index}'+r' $g_i$')
		ax1[2].plot(state_exc_mon.t[trans:]/ms, state_exc_mon.g_e[index,trans:]/nS, color='C5', label=f'{index}'+r' $g_e$')
		ax1[2].set_ylabel('Conductance (nS)')
		ax1[2].grid(linestyle='dotted')
		ax1[2].legend(loc = 'upper right')

		plt.savefig(name+f'/exc variable dynamic, v_in={rate_in/Hz} (no STP).png')

		fig3, ax3 = plt.subplots(nrows=3, ncols=1, sharex=True, 
								num=f'inh variable dynamic, v_in={rate_in/Hz} (no STP)',figsize=(9,10))

		ax3[0].plot(state_inh_mon.t[trans:]/ms, state_inh_mon.I_syn_ext[0,trans:]/pA,
					color='C3', label=f'{index}'+r' $I_{syn ext}$')
		ax3[0].set_ylabel(r'$I_{syn ext} (pA)$')
		ax3[0].grid(linestyle='dotted')
		ax3[0].legend(loc = 'upper right')

		ax3[1].plot(state_inh_mon.t[trans:]/ms, state_inh_mon.v[0,trans:]/mV, label=f'inh neuron {index}')
		ax3[1].axhline(V_th/mV, color='C2', linestyle=':')
		for spk in spikes_inh_mon.t[spikes_inh_mon.i == index]:
			if (spk>trans_time*ms): ax3[1].axvline(x=spk/ms, ymin=0.15, ymax=0.95 )
		ax3[1].set_ylim(bottom=-60.8, top=0.1)
		ax3[1].set_ylabel('Membran potential (V)')
		ax3[1].grid(linestyle='dotted')
		ax3[1].legend(loc = 'upper right')

		ax3[2].plot(state_inh_mon.t[trans:]/ms, state_inh_mon.g_i[0,trans:]/nS, color='C4', label=f'{index}'+r' $g_i$')
		ax3[2].plot(state_inh_mon.t[trans:]/ms, state_inh_mon.g_e[0,trans:]/nS, color='C5', label=f'{index}'+r' $g_e$')
		ax3[2].set_ylabel('Conductance (nS)')
		ax3[2].grid(linestyle='dotted')
		ax3[2].legend(loc = 'upper right')

		plt.savefig(name+f'/inh variable dynamic, v_in={rate_in/Hz} (no STP).png')

		fig2, ax2 = plt.subplots(nrows=4, ncols=1, sharex=True, gridspec_kw={'height_ratios': [2,0.6,0.6,1]},
								num=f'Raster plot, v_in={rate_in/Hz} (no STP)', figsize=(8,10))

		ax2[0].scatter(spikes_exc_mon.t[:]/second, spikes_exc_mon.i[:], color='C3', marker='|')
		ax2[0].scatter(spikes_inh_mon.t[:]/second, spikes_inh_mon.i[:]+N_e, color='C0', marker='|')
		ax2[0].set_ylabel('neuron index')
		ax2[0].set_xlim([0.3,2.3])
		ax2[0].set_title('Raster plot')

		ax2[1].plot(firing_rate_exc.t[trans:]/second, fr_exc[trans:], color='C3')
		ax2[2].plot(firing_rate_inh.t[trans:]/second, fr_inh[trans:], color='C0')
		ax2[1].set_ylabel('rate (Hz)')
		ax2[2].set_ylabel('rate (Hz)')
		ax2[1].set_xlim([0.25,2.4])
		ax2[2].set_xlim([0.25,2.4])
		ax2[1].grid(linestyle='dotted')
		ax2[2].grid(linestyle='dotted')

		ax2[3].plot(state_exc_mon.t[trans:]/second, LFP[trans:], color='C5')
		ax2[3].set_ylabel('LFP (mV)')
		ax2[3].set_xlabel('time (s)')
		ax2[3].set_xlim([0.25,2.4])
		ax2[3].grid(linestyle='dotted')

		plt.savefig(name+f'/Raster plot, v_in={rate_in/Hz} (no STP).png')

		fig4, ax4 = plt.subplots(nrows=1, ncols=1, figsize=(7,6),
								num="Neuron's firing rate distribuction")

		ax4.hist(neurons_fr/Hz, bins=13, color='k', alpha=0.5, density=True, label='Total population')
		ax4.hist(exc_neurons_fr/Hz, bins=13, color='C3', density=True, label='Exc', alpha=0.65)
		ax4.hist(inh_neurons_fr/Hz, bins=13, color='C0', density=True, label='Inh', alpha=0.65)
		ax4.set_xlabel('frequency (Hz)')
		ax4.set_ylabel('Numeber of neurons')
		ax4.legend()

		plt.savefig(name+f"/Neuron's firing rate distribuction - {rate_in/Hz}.png")

		fig5, ax5 = plt.subplots(nrows=1, ncols=2, figsize=(12,6),
								num=f"Power Spectrum - {rate_in/Hz}")

		ax5[0].set_title('Population Firing Rate')
		ax5[0].plot(freq_fr, spectrum_fr, color='k')
		ax5[0].set_xlabel('frequency (Hz)')
		ax5[0].grid(linestyle='dotted')
		
		ax5[1].set_title('LFP')
		ax5[1].plot(freq_LFP, spectrum_LFP, color='C5')
		ax5[1].plot(freq_LFP, 1/(freq_LFP**2), lw=0.7, alpha=0.9)
		ax5[1].set_xscale('log')
		ax5[1].set_yscale('log')
		ax5[1].set_xlabel('frequency (Hz)')
		ax5[1].grid(linestyle='dotted')
		plt.savefig(name+f"/Power Spectrum - {rate_in/Hz}.png")

		device.delete()
		plt.show()

	device.delete()
