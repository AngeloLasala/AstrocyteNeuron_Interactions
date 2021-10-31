"""
Collection of usefull pyhton fuction to plot network connectivity 
defined by Brian2 simulator
"""
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *

def connectivity_plot(Syn, name='Source_to_Target', source='Source', target='Target', 
                    color_s='k', color_t='k', size=35):
    """
    Easy rapresentation of network connectivity

    Parameters
    ----------
    Syn : 'brian2.synapses.synapses.Synapses'
        Synapses object when the connectivity is defined

    name : string (optional)
        main name of figure

    source : string (optional)
            name of Sourge network. Default='Source'

    target : string (optional)
            name of Target network. Default='Target'

    color_s : array-like or list of colors or color (optional)
        color of Source network. See matplotlib.color for more information about color.

    color_s : array-like or list of colors or color (optional)
        color of Target network. See matplotlib.color for more information about color.
    
    size : integer or float (optional)
        marker size. Default=35
    Returns
    -------
    """

    Ns = len(Syn.source) #number of neuron in Source
    Nt = len(Syn.target) #number of neuron in Target

    fig = plt.figure(num='Connectivity'+' '+name, figsize=(10,5))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    ax1.scatter(np.zeros(Ns), np.arange(Ns), marker='o',s=size, color=color_s) 
    ax1.scatter(np.ones(Nt), np.arange(Nt), marker='o', s=size, color=color_t)
    for i,j in zip(Syn.i,Syn.j):
        ax1.plot([0,1], [i,j], lw=0.4, color=color_s)
    ax1.set_xticks([0,1])
    ax1.set_xticklabels([source, target])
    ax1.set_ylabel('Neuron index')
    ax1.set_title('Connectivity')

    ax2.scatter(Syn.i, Syn.j, marker='o', s=size, color='k')
    ax2.set_xlabel('Source neuron index')
    ax2.set_ylabel('Target neuron index')
    ax2.set_title('Source vs Target connectivity')

def connectivity_ring(Syn, r=10):
    """
    Network connectivity in a ring fashion
    version0.1: works only if Syn.source==Syn.target

    Parameters
    ----------
    Syn : 'brian2.synapses.synapses.Synapses'
        Synapses object when the connectivity is defined 

    r : integer (optional)
        ring's radius
    """
    N = len(Syn.source)
    theta = np.linspace(2*np.pi/N, 2*np.pi, N)
    xx = r*np.cos(theta)
    yy = r*np.sin(theta)
    
    plt.figure(figsize=(10,10))
    for i,j in zip(Syn.i,Syn.j):
        plt.plot([xx[i],xx[j]], [yy[i],yy[j]], lw=0.4, color='C0') 
    plt.scatter(xx,yy)


if __name__ == "__main__":

    N = 20
    G = NeuronGroup(N, 'v:1')  #dumb neuron, costant
    G1= NeuronGroup(N+10, 'v:1') 

    S = Synapses(source=G, target=G)
    S.connect(condition='i!=j', p=0.2)
    # S.connect(condition='i!=j', p=0.2) will connect all pairs of neurons i and j 
    # with probability 0.2 as long as the condition i!=j holds

    S1 = Synapses(source=G, target=G1)
    S1.connect(p=0.2)
    

    #only connect neighbouring neurons.
    S2 = Synapses(G,G)
    S2.connect(condition='abs(i-j)<4 and i!=j')
    #S2.connect(j='k for k in range(i-3, i+4) if i!=k', skip_if_invalid=True)
    #more appropriate with larger network
    run(100*ms)

    connectivity_plot(S)
    connectivity_plot(S2)

    connectivity_ring(S)

    plt.show()
