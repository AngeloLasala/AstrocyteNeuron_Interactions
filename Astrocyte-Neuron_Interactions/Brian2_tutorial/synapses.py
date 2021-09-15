"""
Synapsis with Brian2, interactiong neurons
"""
import matplotlib.pyplot as plt
from brian2 import *

duration = 100*ms
eqs = """
dv/dt = (I-v)/tau : volt
I : volt
tau : second
"""

G = NeuronGroup(2, eqs, threshold='v>1.0*mV', reset='v=0.0*mV', method='rk4')
#Difference I and tau values for neurons
G.I = [2, 0]*mV
G.tau = [10, 100]*ms
M = StateMonitor(G,'v',record=True)

S = Synapses(G, G, model='w : volt', on_pre='v_post+=w')
# The syntax on_pre='v_post += 0.2' means that when a spike occurs in the presynaptic neuron 
# (hence on_pre) it causes an instantaneous change to happen v_post += 0.2
# Note: += is excitatory, -= inhibitory
S.connect(i=0, j=1) #i:pre_syn - j:post_syn
S.w = 'j*0.2*mV'    #usefull for multiple post syn neurons

run(duration)

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

ax1.plot(M.t/ms, M.v[0]*1000, label='1 neuron')
ax1.plot(M.t/ms, M.v[1]*1000, label='2 neuron')
ax1.set_xlabel('time (ms)')
ax1.set_ylabel('v (mV)')
ax1.set_title('Neurons dynamics')
ax1.legend()

plt.show()



