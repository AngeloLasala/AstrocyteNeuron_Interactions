"""
Recurrent Network of E/I neurons with short-term plasticity (STP), f-I curve of 
single excitatory and inhibitory neurons. Basically it is the v_in vs nu_S curve.
"""
import argparse
import matplotlib.pyplot as plt
from random import randrange
from brian2 import *
from Neuro_Astro_network.network_analysis import transient
import constant_EI as k_EI
from AstrocyteNeuron_Interactions import makedir

set_device('cpp_standalone', directory=None)  #1% gain 

parser = argparse.ArgumentParser(description='EI network with costantexternal input (Poisson)')
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
s = k_EI.alpha
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
population_fr_exc = PopulationRateMonitor(exc_neurons)
population_fr_inh = PopulationRateMonitor(inh_neurons)

run(duration, report='text')

#Transient time
trans_time = k_EI.trans_time
print(f'g = {g}')
print(f's = {s}')
pop_exc = population_fr_exc.smooth_rate(window="gaussian", width=0.05*ms)
pop_inh = population_fr_inh.smooth_rate(window="gaussian", width=0.05*ms)
print(f'pop-exc: {pop_exc.mean()}')
print(f'pop-inh: {pop_inh.mean()}')


name = f"Neural_network/EI_net_STP/f-I_curve/Network_pe_v_in{rate_in}_g{g}_s{s}_we{w_e/nS:.2f}_fIcurve"
makedir.smart_makedir(name)

## SAVE VARIABLE ################################################################################
np.save(f'{name}/duration',duration)
np.save(f'{name}/g',g)
np.save(f'{name}/s',s)
np.save(f'{name}/rate_in',rate_in)

np.save(f'{name}/pop_exc',pop_exc)
np.save(f'{name}/pop_inh',pop_inh)
#################################################################################################
#################################################################################################
plt.figure()
plt.scatter(spikes_exc_mon.t[:], spikes_exc_mon.i[:], marker='|', color='C3')
plt.scatter(spikes_inh_mon.t[:], spikes_inh_mon.i[:]+N_e, marker='|', color='C0')

device.delete()
# plt.show()