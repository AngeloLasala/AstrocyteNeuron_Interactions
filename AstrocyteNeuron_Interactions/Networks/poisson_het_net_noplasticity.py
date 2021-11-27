"""
Poisson heterogeneity for input stimulus.
Each neurons in the network is connected with an external neuron with poisson spiking train
instead of a costant input. The goal is to find the poisson rate such thet the firing rate 
is equals to the firing rate with costant external stimulus.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *


## PARAMETERS ##########################################################################
# -- Network size --
N_e = 3200                #Total number of excitatory neurons
N_i = 800                #Total number of inhibitory neurons

# -- Neuron --
E_l = -60*mV                 # Leak reversal potential
g_l = 9.99*nS                # Leak conductance
E_e = 0*mV                   # Excitatory synaptic reversal potential
E_i = -80*mV                 # Inhibitory synaptic reversal potential
C_m = 198*pF                 # Membrane capacitance
tau_e = 5*ms                 # Excitatory synaptic time constant
tau_i = 10*ms                # Inhibitory synaptic time constant
tau_r = 5*ms                 # Refractory period
V_th = -50*mV                # Firing threshold
V_r = E_l                    # Reset potential

## --Synapse parameters--
w_e = 0.05*nS          # Excitatory synaptic conductance
w_i = 1.0*nS           # Inhibitory synaptic conductance
U_0 = 0.6              # Synaptic release probability at rest
Omega_d = 2.0/second   # Synaptic depression rate
Omega_f = 3.33/second  # Synaptic facilitation rate

## Time evolution and Stimulus
duration = 1*second
I_ex = 120*pA   #default 100*pA

seed(19958)
#############################################################################################

## NETWORK I_ext=cost ######################################################################
neuron_eqs = """
dv/dt = (g_l*(E_l-v) + g_e*(E_e-v) + g_i*(E_i-v) + I_ex)/C_m : volt (unless refractory)
dg_e/dt = -g_e/tau_e : siemens
dg_i/dt = -g_i/tau_i : siemens
"""
neurons = NeuronGroup(N_e+N_i, model=neuron_eqs, method='euler',
                    threshold='v>V_th', reset='v=V_r', refractory='tau_r')
neurons.v = 'E_l + rand()*(V_th-E_l)'
neurons.g_e = 'rand()*w_e'
neurons.g_i = 'rand()*w_i'

exc_neurons = neurons[:N_e]
inh_neurons = neurons[N_e:]

Synapses
syn_model = """
du_S/dt = -Omega_f * u_S : 1 (event-driven)
dx_S/dt = Omega_d * (1-x_S) : 1 (event-driven)
"""
action="""
u_S += U_0*(1-u_S)
r_S = u_S*x_S
x_S -= r_S
"""
exc="g_e_post+=w_e*r_S"
inh="g_i_post+=w_i*r_S"

exc_syn = Synapses(exc_neurons, neurons, model= syn_model, on_pre=action+exc, method='exact')
inh_syn = Synapses(inh_neurons, neurons, model= syn_model, on_pre=action+inh, method='exact')

exc_syn.connect(p=0.05)
inh_syn.connect(p=0.2)

exc_syn.x_S = 1
inh_syn.x_S = 1

# Monitor
# monitor = StateMonitor(neurons, ['v'], record=True)
monitor_spk = SpikeMonitor(neurons)

# Network and Run
net_cost = Network(neurons, exc_syn, inh_syn, monitor_spk)
net_cost.run(duration, report='text')

firing_rate_costant = monitor_spk.count[:].sum()/duration
print(f'firing rate I_ex=cost: {firing_rate_costant}')

## PLOTS
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios': [3, 1]},
                         num=f'Raster plot, Iex={I_ex/pA} costant', figsize=(8,10))

ax[0].scatter(monitor_spk.t[:]/ms, monitor_spk.i[:], marker='|')
ax[0].set_ylabel('neuron index')

hist_step = 10
bin_size = (duration/ms)/((duration/ms)//hist_step)*ms
spk_count, bin_edges = np.histogram(monitor_spk.t[:]/ms, int(duration/ms)//hist_step)
# POPULATION ACTIVITY, ISTANTANEUS FIRING RATE
# numero di spikes emesso in un breve istante di tempo 
# mediato su tutta la popolazione
rate = double(spk_count)/(N_e+N_i)/bin_size
ax[1].plot(bin_edges[:-1], rate, '-', color='k')
ax[1].set_ylabel('rate (Hz)')
ax[1].set_xlabel('time (ms)')
ax[1].grid(linestyle='dotted')

# ## NETWORK I_ext=Poisson ##########################################################################
# each neurons in the network is connected with an external neuron with poisson firing rate
# the gaol is to find the f_poisson such that network's firing rate is equals to the previous case with I_ext
# costant. 
# We assume the synapses of this external input is uquals to the internal ones (same parameters)
# abut without any kind of plasticity dynamics.

# Poisson input rates
N_poisson = 160     
rate_num = 30    # number of v_poisson
stats_num = 10   # how many time each v_poisson is shown to the network for statistic  
# 100pA -> 7826Hz
# 120pA -> 9304Hz           
rate_in_list = np.linspace(37.5,70,rate_num)*Hz            # list of v_poisson
rate_range= np.tile(rate_in_list,(stats_num,1)).T.flatten()   # repeated list for mean and std            
      
neuron_eqs_p = """
# Neurons dynamics
I_syn_ext = w_e * (E_e-v) * X_ext : ampere
dv/dt = (g_l*(E_l-v) + g_e*(E_e-v) + g_i*(E_i-v) + I_syn_ext)/C_m : volt (unless refractory)
dg_e/dt = -g_e/tau_e :      siemens  # post-synaptic excitatory conductance
dg_i/dt = -g_i/tau_i :      siemens  # post-synaptic inhibitory conductance
dX_ext/dt = -X_ext/tau_e :  1        # post-synaptic external input
"""
firing_rates = []
for rate_in in rate_range:
	# Neurons
	neurons = NeuronGroup(N_e+N_i, model=neuron_eqs_p, method='euler',
						threshold='v>V_th', reset='v=V_r', refractory='tau_r')

	neurons.v = 'E_l + rand()*(V_th-E_l)'
	neurons.g_e = 'rand()*w_e'
	neurons.g_i = 'rand()*w_i'

	exc_neurons = neurons[:N_e]
	inh_neurons = neurons[N_e:]

	# Synapses
	exc_syn = Synapses(exc_neurons, neurons, model= syn_model, on_pre=action+exc, method='exact')
	inh_syn = Synapses(inh_neurons, neurons, model= syn_model, on_pre=action+inh, method='exact')
	exc_syn.connect(p=0.05)
	inh_syn.connect(p=0.2)

	exc_syn.x_S = 1
	inh_syn.x_S = 1

	# Poisson input
	poisson = PoissonInput(neurons, 'X_ext', N_poisson , rate=rate_in, weight='1')

	# Monitor
	monitor_spk = SpikeMonitor(neurons)
	monitor = StateMonitor(neurons, ['v','g_e','g_i','I_syn_ext'], record=range(20))

	net_poisson = Network(neurons, poisson, exc_syn, inh_syn, monitor, monitor_spk)
	net_poisson.run(duration, report='text')

	firing_rate_poisson = monitor_spk.count[:].sum()/duration
	print(f'v_poisson : {poisson.rate}')
	print(f'firing rate I_ex=poisson {firing_rate_poisson}')
	print('___________________________________')
	firing_rates.append(firing_rate_poisson)

# Mean and std for each v_poisson
fr_p = np.array(firing_rates)
fr_poisson_plot = []
for i in range(rate_num):
    print(f'poisson rate_in: {rate_in_list[i]}')
    mean_f = fr_p[i*stats_num:(i+1)*stats_num].mean()
    std_f = fr_p[i*stats_num:(i+1)*stats_num].std()
    print(f'fr_poisson(out) mean: {mean_f}')
    print(f'fr_poisson(out) std: {std_f}')
    fr_poisson_plot.append([mean_f,std_f])

fr_poisson_plot = np.array(fr_poisson_plot)

## Plots ##################################################################################
fig1, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True,
                         num=f'Caratteristic curve v_out-v_poiss, N_poisson={N_poisson}')
ax1.axhline(1500, ls='dashed', color='black', label=r'$I_{ex}$'+f'=[100,120] pA')
ax1.axhline(6680, ls='dashed', color='black')
ax1.errorbar(rate_in_list, fr_poisson_plot[:,0], fr_poisson_plot[:,1]/(stats_num-1**0.5),
            fmt='o', markersize=3, lw=1, color='C9')
ax1.set_xlabel(r'$\nu_{Poisson}$ $(Hz)$ ')
ax1.set_ylabel(r'$\nu_{out}$ $(Hz)$ ')
ax1.grid(linestyle='dotted')
ax1.legend()

fig2, ax2 = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios': [3, 1]},
                         num=f'Raster plot Poisson input, N_poisson={N_poisson} rate:{rate_in_list[-1]}Hz', 
                         figsize=(8,10))

ax2[0].scatter(monitor_spk.t[:]/ms, monitor_spk.i[:], marker='|')
ax2[0].set_ylabel('neuron index')

hist_step = 10
bin_size = (duration/ms)/((duration/ms)//hist_step)*ms
spk_count, bin_edges = np.histogram(monitor_spk.t[:]/ms, int(duration/ms)//hist_step)
# POPULATION ACTIVITY, ISTANTANEUS FIRING RATE
# numero di spikes emesso in un breve istante di tempo 
# mediato su tutta la popolazione
rate = double(spk_count)/(N_e+N_i)/bin_size
ax2[1].plot(bin_edges[:-1], rate, '-', color='k')
ax2[1].set_ylabel('rate (Hz)')
ax2[1].set_xlabel('time (s)')
ax2[1].grid(linestyle='dotted')

fig3, ax3 = plt.subplots(nrows=4, ncols=1, sharex=True,
                         num=f'variables dynamics Poisson input, N_poisson={N_poisson} rate:{rate_in_list[-1]}Hz')
ax3[0].plot(monitor.t[:]/second, monitor.I_syn_ext[5]/pA)
ax3[0].set_ylabel(r'$I_{ext}$ (pA)')
ax3[0].grid(linestyle='dotted')
ax3[1].plot(monitor.t[:]/second, monitor.g_e[5]/nS)
ax3[1].set_ylabel(r'$g_e$ (nS)')
ax3[1].grid(linestyle='dotted')
ax3[2].plot(monitor.t[:]/second, monitor.g_i[5]/nS)
ax3[2].set_ylabel(r'$g_i$ (nS)')
ax3[2].grid(linestyle='dotted')
ax3[3].plot(monitor.t[:]/second, monitor.v[5]/mV)
ax3[3].set_ylabel(r'$v$ (mV)')
ax3[3].grid(linestyle='dotted')
ax3[3].set_xlabel('time (s)')

plt.show()