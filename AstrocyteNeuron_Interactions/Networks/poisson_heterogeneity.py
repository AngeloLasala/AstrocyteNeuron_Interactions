"""
Poisson heterogeneity of external stimulus
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

scaling = 8000
w_e_stm = scaling * w_e
# Poisson input rates
rate_num = 70                              # total numer of rate
rate = np.linspace(0,400,rate_num)*Hz     # range
rate_in = np.tile(rate, (N_e,1)).T.flatten()   # reshaping: [0:N_e]=150, [N_e:2*N_e]=150*(300-150)/rate_in ...

neuron_eqs = """
# Neurons dynamics
dv/dt = (g_l*(E_l-v) + g_e*(E_e-v))/C_m : volt (unless refractory)
dg_e/dt = -g_e/tau_e : siemens  # post-synaptic excitatory conductance
dg_i/dt = -g_i/tau_i : siemens  # post-synaptic inhibitory conductance
"""
neurons = NeuronGroup((N_e)*rate_num, model=neuron_eqs, method='euler',
                    threshold='v>V_th', reset='v=V_r', refractory='tau_r')
neurons.v = -55*mV

poisson = PoissonGroup((N_e)*rate_num, rates=rate_in)

syn_model = """
du_S/dt = -Omega_f * u_S : 1 (event-driven)
dx_S/dt = Omega_d * (1-x_S) : 1 (event-driven)
"""
action="""
u_S += U_0*(1-u_S)
r_S = u_S*x_S
x_S -= r_S
"""
stimulus_action="g_e_post+=w_e_stm*r_S"

synapses = Synapses(poisson, neurons, model=syn_model, on_pre=action+stimulus_action,
					method='exact')
synapses.connect(j='i')

monitor_spk = SpikeMonitor(neurons)

firing_rates = []
net_stm = Network(neurons, poisson, synapses, monitor_spk)
net_stm.run(duration, report='text')

for i in range(rate_num):
    print(f'poisson rate: {rate_in[i*N_e]}')
    firing_rate = monitor_spk.count[i*N_e:(i+1)*N_e]/duration
    # print(f'firing rate: {firing_rate}')
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
                         num=f'Characteristic curve rate_out vs rate_in, scaling:{scaling}')
ax1.axhline(fr_costant_0/Hz, ls='dashed', color='black', label=r'$I_{ex}$'+f' {I_ex[0]/pA}')
ax1.axhline(fr_costant_1/Hz, ls='dotted', color='black', label=r'$I_{ex}$'+f' {I_ex[1]/pA}')
ax1.errorbar(rate, firing_rates[:,0], firing_rates[:,1],
             fmt='o', markersize=2, lw=0.4, color='C9')
ax1.set_xlabel(r'$\nu_{Poisson}$ $(Hz)$ ')
ax1.set_ylabel(r'$\nu_{out}$ $(Hz)$ ')
ax1.grid(linestyle='dotted')
ax1.legend()
plt.show()

####################################################################################################
