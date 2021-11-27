"""
Poisson heterogeneity of external stimulus throught extenal synapses
without plasticity dynamics (x_S and u_S)
"""
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *

## PARAMETERS ##########################################################################
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
dtt = 0.1*ms
I_ex = [100,120]*pA
#############################################################################################

## NETWORK I_ext=cost ######################################################################
N_e = 2
neuron_eqs = """
# Neurons dynamics
I_ext : ampere
dv/dt = (g_l*(E_l-v)+I_ext)/C_m : volt (unless refractory)
dg_e/dt = -g_e/tau_e : siemens  # post-synaptic excitatory conductance
dg_i/dt = -g_i/tau_i : siemens  # post-synaptic inhibitory conductance
"""
neurons = NeuronGroup(N_e, model=neuron_eqs, method='euler',
                    threshold='v>V_th', reset='v=V_r', refractory='tau_r')
neurons.v = -55*mV
neurons.I_ext = I_ex

monitor_spk = SpikeMonitor(neurons)

net_cost = Network(neurons, monitor_spk)
net_cost.run(duration, report='text')

fr_costant_0 = monitor_spk.count[0]/duration
fr_costant_1 = monitor_spk.count[1]/duration


## NETWORK I_ext=Poisson ##########################################################################
N_e = 300  # numer over wich compute mean adn std for each rate input

# Poisson input rates
N_poisson = 160     #how many inputs receive each neuron (rate_in = N_poisson*rate_poisson)
rate_num = 50       
# 100pA -> 7826Hz
# 120pA -> 9304Hz                       
rate_in_list = np.linspace(37.5,200,rate_num)*Hz       

neuron_eqs = """
# Neurons dynamics
I_syn_ext = w_e * (E_e-v) * X_ext : ampere
dv/dt = (g_l*(E_l-v) + g_e*(E_e-v) + I_syn_ext)/C_m : volt (unless refractory)
dg_e/dt = -g_e/tau_e :      siemens  # post-synaptic excitatory conductance
dg_i/dt = -g_i/tau_i :      siemens  # post-synaptic inhibitory conductance
dX_ext/dt = -X_ext/tau_e :  1        # post-synaptic external input
"""
firing_rates = []
for rate_in in rate_in_list:
	neurons = NeuronGroup(N_e, model=neuron_eqs, method='euler',
						threshold='v>V_th', reset='v=V_r', refractory='tau_r')
	neurons.v = 'E_l + rand()*(V_th-E_l)'

	poisson = PoissonInput(neurons, 'X_ext', N_poisson , rate=rate_in, weight='1')

	monitor_spk = SpikeMonitor(neurons)
	monitor = StateMonitor(neurons, ['v','g_e','I_syn_ext'], record=True)

	net_stm = Network(neurons, poisson, monitor_spk, monitor)
	net_stm.run(duration, report='text')


	print(f'poisson rate: {poisson.rate}')
	firing_rate = monitor_spk.count/duration
	mean_f = np.array(firing_rate).mean()
	std_f = np.array(firing_rate).std()
	print(f'mean: {mean_f}')
	print(f'std: {std_f}')
	print('_______________')
	firing_rates.append([mean_f,std_f])

firing_rates = np.array(firing_rates)


## PLOTS ###########################################################################################
# plt.figure()
# plt.scatter(monitor_spk.t[:], monitor_spk.i[:], marker='|')
fig1,ax1 = plt.subplots(nrows=1, ncols=1, sharex=True,
                         num=f'Characteristic curve rate_out vs rate_in, N_poisson={N_poisson}')
ax1.axhline(fr_costant_0/Hz, ls='dashed', color='black', label=r'$I_{ex}$'+f' {I_ex[0]/pA}')
ax1.axhline(fr_costant_1/Hz, ls='dotted', color='black', label=r'$I_{ex}$'+f' {I_ex[1]/pA}')
ax1.errorbar(rate_in_list, firing_rates[:,0], firing_rates[:,1]/(stats_num-1**0.5),
             fmt='o', markersize=2, lw=0.4, color='C9')
ax1.set_xlabel(r'$\nu_{Poisson}$ $(Hz)$ ')
ax1.set_ylabel(r'$\nu_{out}$ $(Hz)$ ')
ax1.grid(linestyle='dotted')
ax1.legend()

fig2, ax2 = plt.subplots(nrows=1, ncols=1, sharex=True,
                         num=f'Raster plot Poisson input, N_poisson={N_poisson} rate:{rate_in_list[-1]}Hz', figsize=(8,8))

ax2.scatter(monitor_spk.t[:]/second, monitor_spk.i[:], marker='|')
ax2.set_ylabel('neuron index')
ax2.set_xlabel('time (s)')

fig3, ax3 = plt.subplots(nrows=3, ncols=1, sharex=True,
                         num=f'variables dynamics Poisson input, N_poisson={N_poisson} rate:{rate_in_list[-1]}Hz')
ax3[0].plot(monitor.t[:]/second, monitor.I_syn_ext[0]/pA)
ax3[0].set_ylabel(r'$I_{ext}$ (pA)')
ax3[0].grid(linestyle='dotted')
ax3[1].plot(monitor.t[:]/second, monitor.g_e[0]/nS)
ax3[1].set_ylabel(r'$g_e$ (nS)')
ax3[1].grid(linestyle='dotted')
ax3[2].plot(monitor.t[:]/second, monitor.v[0]/mV)
ax3[2].set_ylabel(r'$v$ (mV)')
ax3[2].grid(linestyle='dotted')
ax3[2].set_xlabel('time (s)')

plt.show()

####################################################################################################
