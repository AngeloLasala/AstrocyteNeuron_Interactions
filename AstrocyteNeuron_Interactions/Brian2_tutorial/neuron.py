"""
Simple tutorial to simulate a single spiking neuron with Brian2
"""
import matplotlib.pyplot as plt
from brian2 import *

tau = 10*ms

#Models are multi-line sting object
#for (unless refractory) see brain2 documantation
eqs = """
dv/dt = (1-v)/tau : 1 (unless refractory)
"""

#Create a neuron (without spikes)
G = NeuronGroup(1, eqs, method='exact', dt=0.01*ms) #create a class of nueron
M = StateMonitor(G, 'v', record=True)  #how variable evolves over time
G.v = 0.2  #before run() set initial condition


#Spiking neuron 
#treshold: The condition which produces spikes. Should be a single line boolean expressio
#reset: The (possibly multi-line) string with the code to execute on reset
G2 = NeuronGroup(1, eqs, threshold='v>0.8', reset='v=0.1',refractory=2.0*ms, method='exact')
M2 = StateMonitor(G2, 'v', record=True)

#how monitor the spikes
spikes = SpikeMonitor(G2)

run(50*ms)
print(spikes.t[:])

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)


ax.plot(M.t/ms, M.v[0], label='Brian2')
ax.plot(M.t/ms, 1-exp(-M.t/tau), 'C1--',label='Analytic')
ax.set_xlabel('time (ms)')
ax.set_ylabel('v')
ax.set_title('ODE')
ax.legend()

ax2.plot(M2.t/ms, M2.v[0], label='v dinymic')
for i, spk_t  in enumerate(spikes.t[:]):
    if i == 0:
        ax2.axvline(spk_t/ms, ls='--', c='C1', lw=2, label='Spike events')
    else: ax2.axvline(spk_t/ms, ls='--', c='C1', lw=2)
ax2.set_xlabel('time (ms)')
ax2.set_ylabel('v')
ax2.set_title('Spiking neuron')
ax2.legend(loc='upper right')


plt.show()