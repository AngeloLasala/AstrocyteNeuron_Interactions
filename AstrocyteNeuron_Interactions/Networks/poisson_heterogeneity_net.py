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
I_ex = 100*pA   #default 100*pA

# seed(19958)
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

# Synapses
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

hist_step = 1
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
# costant. We assume the synapses of this external input is uquals to the internal ones but, because of 
# the size network, a scaling on synaptic streght w_e is needed
scaling = 8000
w_e_stm = scaling * w_e

# Poisson input rates
rate_num = 15          # number of rates
stats_num = 1          # how manu time compute network values for each rate 
rate_in_list = np.linspace(0.1, 30, rate_num)*Hz     # list of rates
rate_range= np.tile(rate_in_list,(stats_num,1)).T.flatten()   # repeated list for mean and std

neuron_eqs_stm = """
dv/dt = (g_l*(E_l-v) + g_e*(E_e-v) + g_i*(E_i-v))/C_m : volt (unless refractory)
dg_e/dt = -g_e/tau_e : siemens
dg_i/dt = -g_i/tau_i : siemens
"""
neurons = NeuronGroup(N_e+N_i, model=neuron_eqs_stm, method='euler',
                    threshold='v>V_th', reset='v=V_r', refractory='tau_r')

neurons.v = 'E_l + rand()*(V_th-E_l)'
neurons.g_e = 'rand()*w_e'
neurons.g_i = 'rand()*w_i'

exc_neurons = neurons[:N_e]
inh_neurons = neurons[N_e:]

poisson = PoissonGroup(N_e+N_i, rates='rate_in')

# Synapses
stm_action = "g_e_post+=w_e_stm*r_S"
stm_syn = Synapses(poisson, neurons, model=syn_model, on_pre=action+stm_action, method='exact')
stm_syn.connect(j='i')

exc_syn = Synapses(exc_neurons, neurons, model= syn_model, on_pre=action+exc, method='exact')
inh_syn = Synapses(inh_neurons, neurons, model= syn_model, on_pre=action+inh, method='exact')
exc_syn.connect(p=0.05)
inh_syn.connect(p=0.2)

stm_syn.x_S = 1
exc_syn.x_S = 1
inh_syn.x_S = 1

monitor = StateMonitor(neurons, ['v','g_e'], record=range(5))
monitor_spk = SpikeMonitor(neurons)

store()
firing_rates_poisson = []
for rate_in in rate_range:
    restore()
    run(duration, report='text')
    print(f'rate_in = {rate_in}')
    print(f'fr_poisson = {monitor_spk.count[:].sum()/duration}')
    firing_rates_poisson.append(monitor_spk.count[:].sum()/duration)
print('______________________')

fr_p = np.array(firing_rates_poisson)
fr_poisson_plot = []
for i in range(rate_num):
    print(f'poisson rate_in: {rate_in_list[i]}')
    mean_f = fr_p[i*stats_num:(i+1)*stats_num].mean()
    std_f = fr_p[i*stats_num:(i+1)*stats_num].std()
    print(f'fr_poisson(out) mean: {mean_f}')
    print(f'fr_poisson(out) std: {std_f}')
    fr_poisson_plot.append([mean_f,std_f])

fr_poisson_plot = np.array(fr_poisson_plot)


## PLOTS  #########################################################################################
fig1, ax1 = plt.subplots(nrows=2, ncols=1, sharex=True,
                         num=f'variables dynamics Poisson input, rate:{rate_in_list[-1]}Hz, scaling:{scaling}')
ax1[0].plot(monitor.t[:], monitor.g_e[0]/nS)
ax1[0].set_ylabel(r'$g_e$ (nS)')
ax1[1].plot(monitor.t[:], monitor.v[0]/mV)
ax1[1].set_ylabel(r'$v$ (mV)')

fig2, ax2 = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios': [3, 1]},
                         num=f'Raster plot Poisson input, rate:{rate_in_list[-1]}Hz, scaling:{scaling}', 
                         figsize=(8,10))

ax2[0].scatter(monitor_spk.t[:]/ms, monitor_spk.i[:], marker='|')
ax2[0].set_ylabel('neuron index')

hist_step = 1
bin_size = (duration/ms)/((duration/ms)//hist_step)*ms
spk_count, bin_edges = np.histogram(monitor_spk.t[:]/ms, int(duration/ms)//hist_step)
# POPULATION ACTIVITY, ISTANTANEUS FIRING RATE
# numero di spikes emesso in un breve istante di tempo 
# mediato su tutta la popolazione
rate = double(spk_count)/(N_e+N_i)/bin_size
ax2[1].plot(bin_edges[:-1], rate, '-', color='k')
ax2[1].set_ylabel('rate (Hz)')
ax2[1].set_xlabel('time (ms)')
ax2[1].grid(linestyle='dotted')

fig3, ax3 = plt.subplots(nrows=1, ncols=1, sharex=True,
                         num=f'Caratteristic curve v_out-v_poiss, scaling:{scaling}')
ax3.axhline(firing_rate_costant/Hz, ls='dashed', color='black', label=r'$I_{ex}$'+f' {I_ex/pA}')
ax3.errorbar(rate_in_list, fr_poisson_plot[:,0], fr_poisson_plot[:,1],
            fmt='o', markersize=3, lw=1, color='C9')
ax3.set_xlabel(r'$\nu_{Poisson}$ $(Hz)$ ')
ax3.set_ylabel(r'$\nu_{out}$ $(Hz)$ ')
ax3.grid(linestyle='dotted')
ax3.legend()

plt.show()
####################################################################################################

                