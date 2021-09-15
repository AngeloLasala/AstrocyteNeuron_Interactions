"""
Network connectivity using Brian2
"""
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *

def Connectivity(Syn):
    """
    Connectiovity of neuronal network

    Parameters
    ----------
    Syn: 'brian2.synapses.synapses.Synapses'
        Synapses object when the connectivity is defined

    Returns
    -------

    """
    Ns = len(Syn.source) #number of neuron in Source
    Nt = len(Syn.target) #number of neuron in Target

    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    ax1.plot(np.zeros(Ns), np.arange(Ns), 'ok', ms=10)
    ax1.plot(np.ones(Nt), np.arange(Nt), 'ok', ms=10)
    for i,j in zip(S.i,S.j):
        ax1.plot([0,1], [i,j], '-k')
    ax1.set_xticks([0,1])
    ax1.set_xticklabels(['Source', 'Target'])
    ax1.set_ylabel('Neuron index')
    ax1.set_title('Connectivity')

if __name__ == "__main__":

    N = 10
    G = NeuronGroup(N, 'v:1')  #dumb neuron, costant 
    S = Synapses(source=G, target=G)
    
    S.connect(condition='i!=j', p=0.2)
    # S.connect(condition='i!=j', p=0.2) will connect all pairs of neurons i and j 
    # with probability 0.2 as long as the condition i!=j holds

    M = StateMonitor(G, 'v', record=True)
    G.v = 'rand()'

    run(100*ms)

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)

    ax1.plot(M.t/ms, M.v[0], label='1 neuron')
    ax1.plot(M.t/ms, M.v[1], label='2 neuron')
    ax1.set_xlabel('time (ms)')
    ax1.set_ylabel('v')
    ax1.set_title('Neurons dynamics')
    ax1.legend()

    Connectivity(S)

    plt.show()
