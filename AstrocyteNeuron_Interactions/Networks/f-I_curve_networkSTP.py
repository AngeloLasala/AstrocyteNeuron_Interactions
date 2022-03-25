"""
Recurrent Network of E/I neurons with short-term plasticity (STP), f-I curve of 
single excitatory and inhibitory neurons. Basically it is the v_in vs nu_S curve.
"""
import argparse
import matplotlib.pyplot as plt
from scipy import signal
from brian2 import *
from Module_network import neurons_firing, transient
from AstrocyteNeuron_Interactions import makedir
import constant_EI as k_EI

set_device('cpp_standalone', directory=None)  #1% gain 

parser = argparse.ArgumentParser(description="""EI network with costant external input (Poisson)
                                                you find the constants in 'constant_EI.py' module""")
parser.add_argument('r', type=float, help="rate input of external poisson proces")
parser.add_argument('-p', action='store_true', help="show paramount plots, default=False")
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
duration = k_EI.duration*second  # Total simulation time
seed(19958)

#Neurons
neuron_eqs = """
# External input from external synapses
I_syn_ext = w_ext * (E_e-v) * X_ext : ampere
w_ext : siemens                  # external conductance

dv/dt = (g_l*(E_l-v) + g_e*(E_e-v) + g_i*(E_i-v) + I_syn_ext)/C_m : volt (unless refractory)
dg_e/dt = -g_e/tau_e : siemens   # post-synaptic excitatory conductance
dg_i/dt = -g_i/tau_i : siemens   # post-synaptic inhibitory conductance
dX_ext/dt = -X_ext/tau_e :  1    # post-synaptic external input

LFP = (abs(g_e*(E_e-v)) + abs(g_i*(E_i-v)) + abs(I_syn_ext))/g_l : volt
"""
neurons = NeuronGroup(N_e+N_i, model=neuron_eqs, method='rk4',
                     threshold='v>V_th', reset='v=V_r', refractory='tau_r')

neurons.v = 'E_l + rand()*(V_th-E_l)'
neurons.g_e = 'rand()*w_e'
neurons.g_i = 'rand()*w_i'

exc_neurons = neurons[:N_e]
inh_neurons = neurons[N_e:]

#External input
rate_in = args.r*Hz
poisson = PoissonInput(neurons, 'X_ext', 160 , rate=rate_in, weight='1')
s = k_EI.s
exc_neurons.w_ext = w_e
inh_neurons.w_ext = s*w_e

syn_model = """
du_S/dt = -Omega_f * u_S : 1 (event-driven)
dx_S/dt = Omega_d * (1-x_S) : 1 (event-driven)
r_S : 1
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

# Balance degree 
g = k_EI.g 
p_e = k_EI.p_e
p_i = p_e/g
exc_syn.connect(p=p_e)
inh_syn.connect(p=p_i)

exc_syn.x_S = 1
inh_syn.x_S = 1
#############################################################################################

## RUN and MONITOR  ######################################################################### 
spikes_exc_mon = SpikeMonitor(exc_neurons)
spikes_inh_mon = SpikeMonitor(inh_neurons)
spikes_mon = SpikeMonitor(neurons)

population_fr_exc = PopulationRateMonitor(exc_neurons)
population_fr_inh = PopulationRateMonitor(inh_neurons)
population_fr = PopulationRateMonitor(neurons)

state_exc_mon = StateMonitor(exc_neurons, 'LFP', record=True)
run(duration, report='text')
#################################################################################################
## SAVE VARIABLE ################################################################################
name = f"Neural_network/EI_net_STP/f-I_curve/Network_pe_v_in{rate_in}_g{g}_s{s}_we{w_e/nS:.2f}_fIcurve"
makedir.smart_makedir(name)

np.save(f'{name}/duration',duration)
np.save(f'{name}/g',g)
np.save(f'{name}/s',s)
np.save(f'{name}/rate_in',rate_in)

np.save(f'{name}/fr_exc', population_fr_exc.rate[:])
np.save(f'{name}/fr_inh', population_fr_exc.rate[:])
np.save(f'{name}/fr', population_fr.rate[:])
#################################################################################################

## ANALYSIS #####################################################################################
#Transient time
trans_time = k_EI.trans_time
trans = transient(population_fr.t[:]/second*second, trans_time)

## Spectral analysis and Neuron's firing rates distribuction
LFP = state_exc_mon.LFP[:].sum(axis=0)
firing_rate_exc = population_fr_exc.smooth_rate(window="gaussian", width=0.05*ms)
firing_rate_inh = population_fr_inh.smooth_rate(window="gaussian", width=0.05*ms)

freq_fr_exc, spect_fr_exc = signal.welch(firing_rate_exc[trans:], fs=1/defaultclock.dt, nperseg=len(firing_rate_exc[trans:])//4)
freq_fr_inh, spect_fr_inh = signal.welch(firing_rate_inh[trans:], fs=1/defaultclock.dt, nperseg=len(firing_rate_inh[trans:])//4)
freq_fr, spectrum_fr = signal.welch(population_fr.rate[trans:], fs=1/defaultclock.dt, nperseg=len(population_fr.rate[trans:])//4)
freq_LFP, spectrum_LFP = signal.welch(LFP[trans:], fs=1/defaultclock.dt/Hz, nperseg=len(LFP[trans:])//4)

neurons_fr = neurons_firing(spikes_mon.t[:]/second, spikes_mon.i[:], time_start=0.5, time_stop=2.3)
exc_neurons_fr = neurons_firing(spikes_exc_mon.t[:]/second, spikes_exc_mon.i[:], time_start=0.5, time_stop=2.3)
inh_neurons_fr = neurons_firing(spikes_inh_mon.t[:]/second, spikes_inh_mon.i[:], time_start=0.5, time_stop=2.3)

## some information
print('NETWORK')
print(f'transient time = {trans_time} ms')
print(f'g = {g}')
print(f's = {s}')
print(f'pop-exc: {firing_rate_exc[trans:].mean()}')
print(f'pop-inh: {firing_rate_inh[trans:].mean()}')

plt.figure(num='Firing rate distribuction')
plt.hist(neurons_fr/Hz, bins=13)
plt.hist(exc_neurons_fr/Hz, bins=13, label='Exc')
plt.hist(inh_neurons_fr/Hz, bins=13, label='Inh')
plt.legend()

#################################################################################################
if args.p:
    fig1, ax1 = plt.subplots(nrows=4, ncols=1, sharex=True, gridspec_kw={'height_ratios': [2,0.6,0.6,1]},
								num=f'Raster plot, v_in={rate_in/Hz} (no STP)', figsize=(8,10))

    ax1[0].scatter(spikes_exc_mon.t[:]/second, spikes_exc_mon.i[:], color='C3', marker='|')
    ax1[0].scatter(spikes_inh_mon.t[:]/second, spikes_inh_mon.i[:]+N_e, color='C0', marker='|')
    ax1[0].set_ylabel('neuron index')
    ax1[0].set_xlim([0.3,2.3])
    ax1[0].set_title('Raster plot')

    ax1[1].plot(population_fr_exc.t[trans:]/second, firing_rate_exc[trans:], color='C3')
    ax1[2].plot(population_fr_inh.t[trans:]/second, firing_rate_inh[trans:], color='C0')
    ax1[1].set_ylabel('rate (Hz)')
    ax1[2].set_ylabel('rate (Hz)')
    ax1[1].set_xlim([0.25,2.4])
    ax1[2].set_xlim([0.25,2.4])
    ax1[1].grid(linestyle='dotted')
    ax1[2].grid(linestyle='dotted')

    ax1[3].plot(state_exc_mon.t[trans:]/second, LFP[trans:], color='C5')
    ax1[3].set_ylabel('LFP (mV)')
    ax1[3].set_xlabel('time (s)')
    ax1[3].set_xlim([0.25,2.4])
    ax1[3].grid(linestyle='dotted')

    plt.savefig(name+f'/Raster plot, v_in={rate_in/Hz} (no STP).png')

    fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(7,6),
								num=f"Neuron's firing rate distribuction - {rate_in/Hz}")

    ax2.hist(neurons_fr/Hz, bins=12, color='k', alpha=0.5, label='Total population' )
    ax2.hist(exc_neurons_fr/Hz, bins=6, color='C3', label='Exc', alpha=0.65 )
    ax2.hist(inh_neurons_fr/Hz, bins=6, color='C0', label='Inh', alpha=0.65)
    ax2.set_xlabel('frequency (Hz)')
    ax2.set_ylabel("Number of neurons")
    ax2.legend()

    plt.savefig(name+f"/Neuron's firing rate distribuction - {rate_in/Hz}.png")


    fig3, ax3 = plt.subplots(nrows=1, ncols=2, figsize=(12,6),
								num=f"Power Spectrum - {rate_in/Hz}")

    ax3[0].set_title('Population Firing Rate')
    ax3[0].plot(freq_fr, spectrum_fr, color='k')
    ax3[0].set_xlabel('frequency (Hz)')
    ax3[0].grid(linestyle='dotted')
    
    ax3[1].set_title('LFP')
    ax3[1].plot(freq_LFP, spectrum_LFP, color='C5')
    ax3[1].plot(freq_LFP, 1/(freq_LFP**2), lw=0.7, alpha=0.9)
    ax3[1].set_xscale('log')
    ax3[1].set_yscale('log')
    ax3[1].set_xlabel('frequency (Hz)')
    ax3[1].grid(linestyle='dotted')
    plt.savefig(name+f"/Power Spectrum - {rate_in/Hz}.png")


    device.delete()
    plt.show()

device.delete()
