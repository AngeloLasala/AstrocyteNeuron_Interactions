"""
 firing rate of a leaky integrate-and-fire neuron 
 driven by Poisson spiking neurons change depending on 
its membrane time constant
"""
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *

duration = 1000*ms
rate = 10.0*Hz    #
weight = 0.5*mV  #synaptic weight
tau = 10*ms      # 1 to 10


#input spikes train - poisson neuron
Np=10
P = PoissonGroup(Np, rates=rate)

#neuron
neuron_eqs = """
dv/dt = (v0-v)/tau : volt (unless refractory)
v0 : volt
"""
G = NeuronGroup(1, model=neuron_eqs, threshold='v>1.0*mV', reset='v=0.0*mV', refractory=2*ms, method='rk4')
G.v0 = 0*mV

#Synapses connects input and neuron
S = Synapses(P, G, on_pre='v += weight')
S.connect(i=np.arange(Np), j=0)

#Monitor
M = StateMonitor(G,'v', record=True)
Spikes = SpikeMonitor(G)

run(duration)
print(f'spike timing: {Spikes.t[:]}')
print(f'spikes count: {Spikes.count[:]}')
print(f'firing rate: {Spikes.count[:]/duration}')

plt.figure()
plt.plot(M.t/ms, M.v[0]*1000)
plt.xlabel('t (ms)')
plt.ylabel('v (mv)')
plt.show()


