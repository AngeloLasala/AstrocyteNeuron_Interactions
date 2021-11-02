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

def connectivity_ring(Syn, r=10, color='C0', size=20):
    """
    Network connectivity in a ring fashion
    version0.1: works only if Syn.source==Syn.target

    Parameters
    ----------
    Syn : 'brian2.synapses.synapses.Synapses'
        Synapses object when the connectivity is defined 

    r : integer (optional)
        ring's radius

    color : array-like or list of colors or color (optional)
        color of Source network. See matplotlib.color for more information about color.
    
    size : integer or float (optional)
        marker size. Default=20
    """
    N = len(Syn.source)
    theta = np.linspace(2*np.pi/N, 2*np.pi, N)
    xx = r*np.cos(theta)
    yy = r*np.sin(theta)
    
    plt.figure(figsize=(10,10))
    for i,j in zip(Syn.i,Syn.j):
        plt.plot([xx[i],xx[j]], [yy[i],yy[j]], lw=0.4, color=color) 
    plt.scatter(xx,yy, color=color, s=size)

def connectivity_EIring(Syn_exc, Syn_inh, r=1, step=1, color='C0', size=10):
    """
    Network connectivity of a Exitatory-Inhibitory Neural Network in a ring fashion.
    Neurons are defined in a unique NeuronGroup objet with N_e+N_I elements
    when first N_e neurons are excitatory while the others are inhibitory.
    The connectivity is built up by two Synapses class, one 'exc' and the other 'inh':
    - exc_syn = Synapses(exc_neurons, neurons,...)
    - inh_syn = Synapses(inh_neurons, neurons, ...)

    Parameters
    ----------
    Syn_exc : 'brian2.synapses.synapses.Synapses'
        Synapses object when the excitatory connectivity is defined 

    Syn_inh : 'brian2.synapses.synapses.Synapses'
        Synapses object when the inhibitory connectivity is defined

    r : integer (optional)
        ring's radius

    step : integer (oprional)
        step therby the neurons are effectivly plot on th ring, usefull for a large network
        example: with Ne=400 Ni=100 step=10 the neurons ploted are [0,10,20,..]
        so only the 10% of the network is plotted. Default=1

    color : array-like or list of colors or color (optional)
        color of Source network. See matplotlib.color for more information about color.
    
    size : integer or float (optional)
        marker size. Default=20
    """
    N_e = len(Syn_exc.source) #exc neurons
    N_i = len(Syn_inh.source) #inh neurons
    N = N_e + N_i

    theta = np.linspace(2*np.pi/N, 2*np.pi, N)
    xx = r*np.cos(theta)
    yy = r*np.sin(theta)
   
    plt.figure(figsize=(10,10))
    # for i,j in zip(Syn_exc.i,Syn_exc.j):
    #     plt.plot([xx[:N_e][i],xx[j]], [yy[:N_e][i],yy[j]], lw=0.2, color='C3') 
    for i,j in zip(Syn_inh.i,Syn_inh.j):
        plt.plot([xx[N_e:][i],xx[j]], [yy[N_e:][i],yy[j]], lw=0.2, color='C0')
    plt.scatter(xx[:N_e],yy[:N_e], color='C3',  marker='o', s=size)
    plt.scatter(xx[N_e:],yy[N_e:], color='C0',  marker='o', s=size)



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
    
    N_e = 40
    N_i = 10
    neurons = NeuronGroup(N_e+N_i, model='')
    exc_neurons = neurons[:N_e]
    inh_neurons = neurons[N_e:]

    exc_syn = Synapses(exc_neurons, neurons)
    inh_syn = Synapses(inh_neurons, neurons)
    exc_syn.connect(True, p=0.05)
    inh_syn.connect(True, p=0.2)


    run(100*ms)

    connectivity_plot(S)
    connectivity_plot(S2)
    connectivity_ring(S)

    connectivity_EIring(exc_syn, inh_syn, step=1)

    plt.show()
