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


#More intersting - parameters
N2 = 100
tau2 = 10*ms
refr2 = 5*ms
v0_max = 3.0*mV
duration = 1000*ms

eqs2 = """
dv/dt = (v0-v)/tau2 : volt (unless refractory)
v0 : volt
"""
G2 = NeuronGroup(N2, eqs2, threshold='v>1.5*mV', reset='v=0*mV', refractory=refr2, method='rk4')
M2 = StateMonitor(G2, 'v', record=True)
Spikes2 = SpikeMonitor(G2)

G2.v0 = 'i*v0_max/(N-1)' #i in this case rappresente the neuron index

run(duration)

#Firing rate
count = Spikes2.count[:] #total spikes for each neurons
firing_rate = count/duration


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
ax2.set_ylabel('neuron index')
ax2.set_title('Raster plot')

fig2 = plt.figure(figsize=(10,5))
ax3 = fig2.add_subplot(1,2,1)
ax4 = fig2.add_subplot(1,2,2)

ax3.plot(Spikes2.t[:], Spikes2.i[:], '.k')
ax3.set_xlabel('time (ms)')
ax3.set_ylabel('neuron index')
ax3.set_title('Neurons dynamics')
ax3.legend()

ax4.plot(G2.v0[:]*1000, firing_rate)
ax4.set_xlabel('v0 (mV)')
ax4.set_ylabel('firing rate (spikes/s)')
ax4.set_title('Firing rate vs v0')

plt.show()