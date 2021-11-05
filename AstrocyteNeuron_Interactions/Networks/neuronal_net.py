"""
Neuronal network simulation using Brian 2.
Randomly connected networks with conductance-based synapses.

- "Modeling euron-glia interaction with Brian 2 simulator", Stimberg et al (2017)
"""
import matplotlib.pyplot as plt
from random import randrange
from brian2 import *
from AstrocyteNeuron_Interactions.Brian2_utils.connectivity import connectivity_EIring

## Parameters ########################################################################

# Network size
N_e = 3200                #Total number of excitatory neurons
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
I_ex = 150*pA          # External current
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

duration = 1.0*second  # Total simulation time
seed(19958)

#Neurons
neuron_eqs = """
dv/dt = (g_l*(E_l-v) + g_e*(E_e-v) + g_i*(E_i-v) + I_ex)/C_m : volt (unless refractory)
dg_e/dt = -g_e/tau_e : siemens
dg_i/dt = -g_i/tau_i : siemens
"""
Np = 0
poisson = PoissonGroup(Np, rates=500.0*Hz)
neurons = NeuronGroup(N_e+N_i, model=neuron_eqs, method='rk4',
                     threshold='v>V_th', reset='v=V_r', refractory='tau_r')

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

exc_syn.connect(p=0.05)
inh_syn.connect(p=0.2)

exc_syn.x_S = 1
inh_syn.x_S = 1
#############################################################################################

## RUN and MONITOR  ######################################################################### 
spikes_exc_mon = SpikeMonitor(exc_neurons)
spikes_inh_mon = SpikeMonitor(inh_neurons)

# select random excitatory neurons
index = randrange(N_e) 

state_exc_mon = StateMonitor(exc_neurons, ['v', 'g_e', 'g_i'], record=index)
syn_exc_mon = StateMonitor(exc_syn, ['u_S','x_S'], record=exc_syn[index, :]) 
syn_inh_mon = StateMonitor(inh_syn, ['u_S','x_S'], record=inh_syn[index, :])
#record=exc_syn[index, :], outgoing synapses from neurons labeled by index
spikes_mon = SpikeMonitor(neurons)

run(duration, report='text')
print(f'exc neuron number: {index}')
print(f'exc syn: {len(syn_exc_mon.u_S[:])}')
print(f'inh syn: {len(syn_inh_mon.u_S[:])}')
print()
print(exc_syn[index, :])
print(inh_syn[index, :])
print()
print(syn_exc_mon.u_S)
print('\n\n')
#########################################################################################################

# Plots  ################################################################################################
fig1, ax1 = plt.subplots(nrows=3, ncols=1, sharex=True, 
                         num=f'exc variable dynamics, Ne:{N_e} Ni:{N_i}, Iex={I_ex/pA}',figsize=(9,10))


ax1[0].plot(state_exc_mon.t/ms, state_exc_mon.v[0]/mV, label=f'neuron {index}')
ax1[0].axhline(V_th/mV, color='C2', linestyle=':')
for spk in spikes_exc_mon.t[spikes_exc_mon.i == index]:
    ax1[0].axvline(x=spk/ms, ymin=0.15, ymax=0.95 )
ax1[0].set_ylim(bottom=-60.8, top=0.1)
ax1[0].set_ylabel('Membran potential (V)')
ax1[0].set_title('Neurons dynamics')
ax1[0].grid(linestyle='dotted')
ax1[0].legend(loc = 'upper right')

ax1[1].plot(state_exc_mon.t/ms, state_exc_mon.g_i[0]/nS, color='C4', label=f'{index}'+r' $g_i$')
ax1[1].plot(state_exc_mon.t/ms, state_exc_mon.g_e[0]/nS, color='C5', label=f'{index}'+r' $g_e$')
ax1[1].set_ylabel('Conductance (nS)')
ax1[1].grid(linestyle='dotted')
ax1[1].legend(loc = 'upper right')

ax1[2].plot(syn_exc_mon.t/ms, syn_exc_mon.u_S[0], label=f'{index}'+r' $u_S$', color='C1')
ax1[2].plot(syn_exc_mon.t/ms, syn_exc_mon.x_S[0], label=f'{index}'+r' $x_S$', color='C4')
ax1[2].set_xlabel('time (ms)')
ax1[2].set_ylabel(r'$u_S$, $x_S$')
ax1[2].grid(linestyle='dotted')
ax1[2].legend(loc = 'upper right')

fig2, ax2 = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios': [3, 1]},
                         num=f'Raster plot, Ne:{N_e} Ni:{N_i}, Iex={I_ex/pA}', figsize=(8,10))

ax2[0].scatter(spikes_exc_mon.t[:]/ms, spikes_exc_mon.i[:], color='C3', marker='|')
ax2[0].scatter(spikes_inh_mon.t[:]/ms, spikes_inh_mon.i[:]+N_e, color='C0', marker='|')
ax2[0].set_ylabel('neuron index')
ax2[0].set_title('Raster plot')

hist_step = 1
bin_size = (duration/ms)/((duration/ms)//hist_step)*ms
spk_count, bin_edges = np.histogram(np.r_[spikes_exc_mon.t/ms,spikes_inh_mon.t/ms], 
                                    int(duration/ms)//hist_step)
# POPULATION ACTIVITY, ISTANTANEUS FIRING RATE
# numero di spikes emesso in un breve istante di tempo 
# meiato su tutta la popolazione
rate = double(spk_count)/(N_e+N_i)/bin_size
ax2[1].plot(bin_edges[:-1], rate, '-', color='k')
ax2[1].set_ylabel('rate (Hz)')
ax2[1].set_xlabel('time (ms)')
ax2[1].grid(linestyle='dotted')

connectivity_EIring(exc_syn, inh_syn)
# connectivity_plot(exc_syn, source='Exc', target='Exc+Inh', color_s='red', color_t='indigo', size=10)
# connectivity_plot(inh_syn, source='Inh', target='Exc+Inh', color_s='C0', color_t='indigo', size=10)

plt.show()
