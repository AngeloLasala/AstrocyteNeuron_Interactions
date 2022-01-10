"""
Neuronal network simulation using Brian 2.
Randomly connected networks with conductance-based synapses.

- "Modeling euron-glia interaction with Brian 2 simulator", Stimberg et al (2017)
"""
import argparse
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from random import randrange
from brian2 import *
from Neuro_Astro_network.network_analysis import transient
from AstrocyteNeuron_Interactions import makedir

# set_device('cpp_standalone', directory=None)  #1% gain 

parser = argparse.ArgumentParser(description='EI network with costantexternal input (Poisson)')
parser.add_argument('r', type=float, help="rate input of external poisson proces")
parser.add_argument('-p', action='store_true', help="show paramount plots, default=False")
parser.add_argument('-b', action='store_true', help="EI balance, default=False")
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
defaultclock.dt = 0.05*ms
duration = 2.3*second  # Total simulation time
seed(19958)

#Neurons
neuron_eqs = """
# External input from external synapses
I_syn_ext = w_e * (E_e-v) * X_ext : ampere
dv/dt = (g_l*(E_l-v) + g_e*(E_e-v) + g_i*(E_i-v) + I_syn_ext)/C_m : volt (unless refractory)
dg_e/dt = -g_e/tau_e : siemens   # post-synaptic excitatory conductance
dg_i/dt = -g_i/tau_i : siemens   # post-synaptic inhibitory conductance
dX_ext/dt = -X_ext/tau_e :  1    # post-synaptic external input

LFP = (abs(g_e*(E_e-v)) + abs(g_i*(E_i-v)) + abs(I_syn_ext))/g_l : volt
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

syn_model = """
du_S/dt = -Omega_f * u_S : 1 (clock-driven)
dx_S/dt = Omega_d * (1-x_S) : 1 (clock-driven)
"""

action="""
u_S += U_0*(1-u_S)
r_S = u_S*x_S
x_S -= r_S
"""
exc="g_e_post+=w_e*r_S"
inh="g_i_post+=w_i*r_S"

exc_syn = Synapses(exc_neurons, neurons, model= syn_model, on_pre=action+exc)
inh_syn = Synapses(inh_neurons, neurons, model= syn_model, on_pre=action+inh)

if args.b:
	exc_syn.connect(p=0.05)
	inh_syn.connect(p=0.01)
else:
	exc_syn.connect(p=0.05)
	inh_syn.connect(p=0.2)


exc_syn.x_S = 1
inh_syn.x_S = 1
#############################################################################################

## RUN and MONITOR  ######################################################################### 
spikes_exc_mon = SpikeMonitor(exc_neurons)
spikes_inh_mon = SpikeMonitor(inh_neurons)
firing_rate_exc = PopulationRateMonitor(exc_neurons)
firing_rate_inh = PopulationRateMonitor(inh_neurons)
firing_rate = PopulationRateMonitor(neurons)

# select random excitatory neurons
index = 488
state_exc_mon = StateMonitor(exc_neurons, ['v', 'g_e', 'g_i', 'LFP', 'I_syn_ext'], record=True)
syn_exc_mon = StateMonitor(exc_syn, ['u_S','x_S'], record=exc_syn[index, :]) 
syn_inh_mon = StateMonitor(inh_syn, ['u_S','x_S'], record=inh_syn[index, :])
#record=exc_syn[index, :], outgoing synapses from neurons labeled by index


run(duration, report='text')
print('NETWORK')
print(f'exc syn: {exc_syn.N[:]}')
print(f'inh syn: {inh_syn.N[:]}')
print()
print(f'exc neuron number: {index}')
print(f'exc syn: {syn_exc_mon.u_S[:].shape}')
print('\n')

#Transient time
trans_time = 300
trans = transient(state_exc_mon.t[:]/second*second, trans_time)

## Network variable

LFP = state_exc_mon.LFP[:].sum(axis=0)
fr_exc = firing_rate_exc.smooth_rate(window='gaussian', width=1*ms)
fr_inh = firing_rate_inh.smooth_rate(window='gaussian', width=1*ms)
fr = firing_rate.smooth_rate(window="gaussian", width=1*ms)

fr_trans = fr[trans:]
fr_fft = fft(fr_trans)
N = len(fr_trans)
fr_freq = fftfreq(N,defaultclock.dt)

plt.figure()
plt.plot(firing_rate.t[:], fr[:])

plt.figure()
plt.plot(fr_freq[:N//2], np.abs(fr_fft[:N//2]))

#########################################################################################################
## SAVE VARIABLES #######################################################################################

name = f"Neural_network/EI_net_STP/Network_pe_v_in{rate_in}"
if args.b: name = f"Neural_network/EI_net_STP_balanceSTP/Network_pe_v_in{rate_in}"
makedir.smart_makedir(name)

# Time evolytion variable
np.save(f'{name}/duration',duration)
np.save(f'{name}/trans_time',trans_time)
np.save(f'{name}/trans',trans)
np.save(f'{name}/rate_in',rate_in)

# Excitatory neurons variable
np.save(f'{name}/state_exc_mon.t',state_exc_mon.t)
np.save(f'{name}/state_exc_mon.LFP',state_exc_mon.LFP)

# Population istantaneus firing rate
np.save(f'{name}/firing_rate_exc.t',firing_rate_exc.t)
np.save(f'{name}/firing_rate_inh.t',firing_rate_inh.t)
np.save(f'{name}/fr_exc',fr_exc)
np.save(f'{name}/fr_inh',fr_inh)
np.save(f'{name}/firing_rate',fr)

#########################################################################################################

# Plots  ################################################################################################
if args.p:
	fig1, ax1 = plt.subplots(nrows=4, ncols=1, sharex=True, 
							num=f'exc variable dynamic, v_in={rate_in/Hz}',figsize=(9,10))

	ax1[0].plot(state_exc_mon.t[trans:]/ms, state_exc_mon.I_syn_ext[index,trans:]/pA,
				color='C3', label=f'{index}'+r' $I_{syn ext}$')
	ax1[0].set_ylabel(r'$I_{syn ext} (pA)$')
	ax1[0].grid(linestyle='dotted')
	ax1[0].legend(loc = 'upper right')

	ax1[1].plot(state_exc_mon.t[trans:]/ms, state_exc_mon.v[index,trans:]/mV, label=f'neuron {index}')
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

	ax1[3].plot(syn_exc_mon.t[trans:]/ms, syn_exc_mon.u_S[0,trans:], label=f'{index}'+r' $u_S$', color='C1')
	ax1[3].plot(syn_exc_mon.t[trans:]/ms, syn_exc_mon.x_S[0,trans:], label=f'{index}'+r' $x_S$', color='C4')
	ax1[3].set_xlabel('time (ms)')
	ax1[3].set_ylabel(r'$u_S$, $x_S$')
	ax1[3].grid(linestyle='dotted')
	ax1[3].legend(loc = 'upper right')

	plt.savefig(name+f'/exc variable dynamic, v_in={rate_in/Hz} (STP).png')

	fig2, ax2 = plt.subplots(nrows=4, ncols=1, sharex=True, gridspec_kw={'height_ratios': [2,0.6,0.6,1]},
							num=f'Raster plot, v_in={rate_in/Hz}', figsize=(8,10))

	ax2[0].scatter(spikes_exc_mon.t[:]/second, spikes_exc_mon.i[:], color='C3', marker='|')
	ax2[0].scatter(spikes_inh_mon.t[:]/second, spikes_inh_mon.i[:]+N_e, color='C0', marker='|')
	ax2[0].set_ylabel('neuron index')
	ax2[0].set_title('Raster plot')

	ax2[1].plot(firing_rate_exc.t[trans:]/second, fr_exc[trans:], color='C3')
	ax2[2].plot(firing_rate_inh.t[trans:]/second, fr_inh[trans:], color='C0')
	ax2[1].set_ylabel('rate (Hz)')
	ax2[2].set_ylabel('rate (Hz)')
	ax2[1].grid(linestyle='dotted')
	ax2[2].grid(linestyle='dotted')

	ax2[3].plot(state_exc_mon.t[trans:]/second, LFP[trans:], color='C5')
	ax2[3].set_ylabel('LFP (mV)')
	ax2[3].set_xlabel('time (s)')
	ax2[3].grid(linestyle='dotted')

	plt.show()
