"""
Multi neurons network (not interactiong) with Brian2
"""
import matplotlib.pyplot as plt
from brian2 import *

N = 100 #numer of neurons
tau = 10*ms
eqs = """
dv/dt = (2*mV-v)/tau : volt
"""

G = NeuronGroup(N, model=eqs, threshold='v>1.0*mV', reset='v=0*mV', method='rk4')
M = StateMonitor(G, 'v', record=True)
Spikes = SpikeMonitor(G)
G.v = 'rand()*mV' #set random initial conditions for each neuron - uniform dist

run(1000*ms)
print(Spikes.i)


fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ax1.plot(M.t/ms, M.v[0], label='1 neuron')
ax1.plot(M.t/ms, M.v[1], label='2 neuron')
ax1.set_xlabel('time (ms)')
ax1.set_ylabel('v (Volt)')
ax1.set_title('Neurons dynamics')
ax1.legend()

ax2.plot(Spikes.t[:], Spikes.i[:], '.k')
ax2.set_xlabel('time (ms)')
ax2.set_ylabel('v (Volt)')
ax2.set_title('Raster plot')


plt.show()