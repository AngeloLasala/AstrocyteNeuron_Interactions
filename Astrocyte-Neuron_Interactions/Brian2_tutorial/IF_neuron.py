"""
Firing rate of a leaky integrate-and-fire neuron 
driven by Poisson spiking neurons change depending on 
its membrane time constant
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from brian2 import *


#Parameters
duration = 1000*ms
rate = 10.0*Hz    #
weight = 0.1*mV  #synaptic weight
tau = 10*ms      # 1 to 10


#input spikes train - poisson neuron
Np = 100
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

# We store a copy of the state of the network before the loop,
#  and restore it at the beginning of each iteration.
store()
tau_range = np.linspace(1,10,30)*ms
rate_output = list()
start_time1 = time.time()
for tau in tau_range:
    restore()
    run(duration)
    print(f'tau: {tau}, firing rate: {Spikes.count[:]/duration}')
    rate_output.append(Spikes.count/duration)
end_time1 = time.time()
loop_time = end_time1-start_time1
print(f'Loop time: {"%.2f" % loop_time}')
#Look Brian 2 documantation for more efficient way to do this


fig = plt.figure(num='LIF neuron - dumb synapses', figsize=(10,5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ax1.plot(M.t/ms, M.v[0]*1000)
ax1.set_xlabel('t (ms)')
ax1.set_ylabel('v (mv)')
ax1.set_title('Membrane Potential evolution')

ax2.plot(tau_range/ms, rate_output/Hz, color='orange')
ax2.set_xlabel('tau (ms)')
ax2.set_ylabel('firing rate (spikes/s)')
ax2.set_title('Firing rate vs tau')


plt.show()


