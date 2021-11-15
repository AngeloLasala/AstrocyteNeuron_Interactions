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
N_e=20
N_i=10

duration = 2*second
dt = 0.1*ms
I_ext = 100*pA

stimulus = TimedArray(np.random.poisson(I_ext/pA, (int(duration/defaultclock.dt),N_e+N_i)),
                      dt=defaultclock.dt) 

neuron_eqs = """
# Neurons dynamics
I_test = stimulus(t,i%(N_e+N_i))*pA : ampere
dv/dt = I_test/C_m : volt (unless refractory)
dg_e/dt = -g_e/tau_e : siemens  # post-synaptic excitatory conductance
dg_i/dt = -g_i/tau_i : siemens  # post-synaptic inhibitory conductance

"""

neurons = NeuronGroup(N_e+N_i, model=neuron_eqs, method='euler',
                    threshold='v>V_th', reset='v=V_r', refractory='tau_r')

monitor = StateMonitor(neurons, ['v','I_test'], record=True)

run(duration, report='text')

plt.figure()
for i in range(N_e):
    plt.plot(monitor.t[:], monitor.I_test[i])

plt.figure()
plt.plot(monitor.t[:], monitor.v[0])
plt.plot(monitor.t[:], monitor.v[4])
plt.show()

                