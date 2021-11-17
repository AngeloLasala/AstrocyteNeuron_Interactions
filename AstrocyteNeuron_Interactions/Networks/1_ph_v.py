"""
"""

import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
# -- Neuron --
E_l = -60*mV                 # Leak reversal potential
g_l = 9.99*nS                # Leak conductance
E_e = 0*mV                   # Excitatory synaptic reversal potential
E_i = -80*mV                 # Inhibitory synaptic reversal potential
C_m = 198*pF                 # Membrane capacitance
tau_e = 5*ms                 # Excitatory synaptic time constant
tau_i = 10*ms                # Inhibitory synaptic time constant
tau_r = 5*ms                 # Refractory period
# I_ex = 100*pA                # External current
V_th = -50*mV                # Firing threshold
V_r = E_l                    # Reset potential


N_poisson=1
N_e=40
N_i=10

duration = 1*second
dt = 0.1*ms
I_ext = 120*pA

stimulus = TimedArray(np.random.poisson(I_ext/pA, (int(duration/defaultclock.dt),N_e+N_i)),
                      dt=defaultclock.dt) 

neuron_eqs = """
# Neurons dynamics
dv/dt = (g_l*(E_l-v))/C_m : volt (unless refractory)
dg_e/dt = -g_e/tau_e : siemens  # post-synaptic excitatory conductance
dg_i/dt = -g_i/tau_i : siemens  # post-synaptic inhibitory conductance
"""
neurons = NeuronGroup(N_e+N_i, model=neuron_eqs, method='euler',
                    threshold='v>V_th', reset='v=V_r', refractory='tau_r')
neurons.v = -55*mV
poisson = PoissonGroup(N_e+N_i, (1/dt)/1.2)
synapses = Synapses(poisson, neurons, on_pre='v += I_ext/C_m*dt')
synapses.connect(j='i')

monitor = StateMonitor(neurons, ['v'], record=True)
monitor_spk = SpikeMonitor(neurons)

run(duration, report='text')
print(f'firing rate: {monitor_spk.count[0]/duration}')

plt.figure()
plt.plot(monitor.t[:], monitor.v[0])

plt.figure()
plt.plot(monitor_spk.t[:], monitor_spk.i[:], '|')
plt.show()