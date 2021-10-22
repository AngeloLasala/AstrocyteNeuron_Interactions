"""
'G-ChI' model for astrocyte dynamics tacking into account both 
endogenous (J_delta) and external (J_beta) mechanism of IP3 production
and degradetion without quasi-static approximation. 


- "G Protein-Coupled Receptor-Mediated Calcium Signaling in Astrocytes" De Pittà et al (2015)

- "Modelling neuro-glia interactions with the Brian2 simulator" Stimberg et al (2017)
"""

import argparse
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from random import randrange
from brian2 import *

def Biforcation_brian2(N_a, eqs):
    """
    Biforcation analysis of dynamical system used brian 2.

    Note: variable name and genaral notation refers to astrocyte dynamics

    Parameters
    ----------
    N_a : integer
        Total number of astrocyte defined by the same equation. Each astrocytes share 
        the same parameters except that over wich compute the biforcation

    eqs : string
        Astrocyte model


    Returns
    -------
    par_list : list
        List of lists contained parameter values

    Bif_list : list
        List of lists, each one contains max-min discrete maps for biforcation analysis 
    """
    astrocyte = NeuronGroup(N_a, astro_eqs)
    astrocyte.Y_S = np.linspace(0.04,1,N_a)*umolar

    astro_mon = StateMonitor(astrocyte, 'C', record=True)

    net = Network(astrocyte)  # automatically include G and S
    net.add(astro_mon)  # manually add the monitors

    net.run(300*second, report='text')
   

    #Biforcation - later define a function for it
    C = np.array(astro_mon.C/umolar)[:,1500000:]

    par_list = []
    Bif_list = []
    for i in range(N_a):
        max_loc = argrelextrema(C[i], np.greater)
        min_loc = argrelextrema(C[i], np.less)
        
        C_max = C[i][max_loc].tolist()
        C_min = C[i][min_loc].tolist()

        Bif_val = C_max + C_min
        par_x = [astrocyte.Y_S[i]/umolar for item in range(len(Bif_val))]
        
        par_list.append(par_x)
        Bif_list.append(Bif_val)
    
    return par_list, Bif_list



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='G-ChI model')
    parser.add_argument('-b', action='store_true', help="biforcation plots, default=False")
    args = parser.parse_args()

    #####  PARAMETERS  #######################################################
    ## General Parameters ##
    N_a = 2                    # Total number of astrocyte
    duration = 50*second       # Integration time

    ## Synapses parameters ##
    f_0 = 0.5*Hz                  # Spike rate of the "source" neurons
    rho_c = 0.001                 # Synaptic vesicle-to-extracellular space volume ratio
    Y_T = 500*mmolar              # Total vesicular neurotransmitter concentration
    Omega_c = 40/second           # Neurotransmitter clearance rate

    ## Astrocye parameters ##
    # -- Calcium - CICR --
    Omega_C = 6.0/second       # Maximal rate of Ca^2+ release by IP_3Rs
    Omega_L = 0.1/second       # Maximal rate of calcium leak from the ER
    O_P = 0.9*umolar/second    # Maxium rate of SERCA uptake
    d_1 = 0.13*umolar       # IP_3 binding affinity
    d_2 = 1.049*umolar      # Ca inactivation dissociation costant
    d_3 = 0.9434*umolar     # IP3 dissociation constant
    d_5 = 0.08234*umolar    # Ca activation dissociation costant
    C_T = 2.0*umolar        # Total cell free Calcium concentration
    rho_A = 0.185           # Ratio between ER volume and cytosol
    O_2 = 0.2/umolar/second    # IP3R binding rate for Ca inhibition
    K_P = 0.1*umolar        # SERCA calcium affinity

    # -- IP3 methabolism --
    #degradation
    Omega_5P = 0.1/second        # Maximal rate of degradation by IP3-5P
    O_3K = 4.5*umolar/second     # Maximal rate of degradation by IP3-3K
    K_3K = 1.0*umolar         # IP3 affinity of IP3-3K, muM
    K_D = 0.5*umolar          # Ca affinity of IP3-3K, muM

    #PLC_delta production 
    O_delta = 0.2*umolar/second  # Maximal rate of IP3 production by PLC_delta
    kappa_delta = 1.5*umolar  # Inhibition constant of PLC_delta activity
    K_delta = 0.3*umolar      # Ca affinity of PLC_delta

    #PLC_beta production, agonist(glutammate) dependent
    O_beta = 5*umolar/second   # Maximal rate of IP3 production by PLC_beta
    K_KC = 0.5*umolar       # Ca affinity of PKC

    # -- Gamma_A - fraction of activated astrocyte (Gprotein receptors) --
    O_N = 0.3/umolar/second    # Agonist binding rate
    Omega_N =0.5/second        # Maximal inactivation rate
    zeta = 10                  # Maximal reduction of receptor affinity by PKC
    ##############################################################################

    seed(1224689)

    astro_eqs = """
    # Fraction of activated astrocyte receptors (1):
    dGamma_A/dt = O_N * Y_S * (1 - Gamma_A) -
                Omega_N*(1 + zeta * C/(C + K_KC)) * Gamma_A : 1

    # IP_3 dynamics (1):
    dI/dt = J_beta + J_delta - J_3K - J_5P       : mmolar

    J_beta = O_beta * Gamma_A                         : mmolar/second
    J_delta = O_delta/(1 + I/kappa_delta) *
                            C**2/(C**2 + K_delta**2) : mmolar/second
    J_3K = O_3K * C**4/(C**4 + K_D**4) * I/(I + K_3K) : mmolar/second
    J_5P = Omega_5P*I                                 : mmolar/second

    # Calcium dynamics (2):
    dC/dt = J_r + J_l - J_p                          : mmolar
    dh/dt = (h_inf - h_clipped) / tau_h  * (1+noise*xi*tau_h**0.5): 1
    # dh/dt = alpha *(1-h_clipped) - beta * h_clipped 
    #         + ((alpha *(1-h_clipped) + beta * h_clipped) / N_ch )*xi*second**0.5 : 1
    
    h_clipped = clip(h,0,1)                            : 1

    J_r = Omega_C*(m_inf**3)*(h_clipped**3)*(C_T-(1+rho_A)*C)  : mmolar/second
    J_l = Omega_L*(C_T-(1+rho_A)*C)                    : mmolar/second
    J_p = (O_P*C**2)/(K_P**2+C**2)                     : mmolar/second

    Q_2 = d_2*((I+d_1)/(I+d_3))                  : mmolar
    m_inf = (I/(I+d_1))*(C/(C+d_5))              : 1
    tau_h = 1 / (O_2*(Q_2+C))                    : second
    h_inf = Q_2/(Q_2+C)                          : 1
    alpha =  O_2 * Q_2                           : 1/second
    beta =  O_2 * C                              : 1/second  

    # Neurotransmitter concentration in the extracellular space
    Y_S     : mmolar

    #Stochastic parameters
    N_ch                                         : 1 (constant) 
    noise                                        : 1 (constant) 
    """

    # Neuron 
    pre_synaptic_eqs = """
    dv/dt = f_0 : 1
    """
    pre_synaptic = NeuronGroup(1, model=pre_synaptic_eqs, threshold='v>1', reset='v=0', method='rk4')
    
    post_synaptic = NeuronGroup(1, model='')

    #Synapses
    syn_eqs = "dY_S/dt = -Omega_c*Y_S  : mmolar (clock-driven)"
    syn_action = "Y_S += rho_c*Y_T"
    synapses = Synapses(pre_synaptic, post_synaptic, model=syn_eqs, on_pre=syn_action, method='linear')
    synapses.connect()

    # Astrocyte
    astrocyte = NeuronGroup(N_a, astro_eqs, method='milstein')
    astrocyte.noise = [0,1]
    astrocyte.N_ch = [1000,3]
    astrocyte.h = 0.9
    
    # Connection between synapses and astrocytes 
    ecs_syn_to_astro = Synapses(synapses, astrocyte, 
                                model = "Y_S_post = Y_S_pre : mmolar (summed)")
    ecs_syn_to_astro.connect()    

    # Monitor
    spike_mon = SpikeMonitor(pre_synaptic)
    syn_mon = StateMonitor(synapses, 'Y_S', record=True)
    astro_mon = StateMonitor(astrocyte, ['Gamma_A','I','C','h'], record=True)

    run(duration, report='text')

    # Free astrocyte Biforcation 
    if args.b:
        Y_s_val, Bif_val = Biforcation_brian2(50, astro_eqs)
    


    #Plots
    for astro_i in range(N_a):
        stoc=''
        if astro_i>0 : stoc = ' stocasticity'
        fig1 = plt.figure(num=f'Synaptically activated astrocyte {astro_i}'+ stoc,figsize=(10,10))
        ax11 = fig1.add_subplot(5,1,1)
        ax12 = fig1.add_subplot(5,1,2)
        ax13 = fig1.add_subplot(5,1,3)
        ax14 = fig1.add_subplot(5,1,4)
        ax15 = fig1.add_subplot(5,1,5)

        ax11.plot(syn_mon.t, syn_mon.Y_S[0]/umolar, color='C6')
        ax11.set_ylabel(r'$Y_S$ ($\mu$M)')
        ax11.grid(linestyle='dotted')

        ax12.plot(astro_mon.t, astro_mon.Gamma_A[astro_i], color='C4')
        if astro_i>0: ax12.plot(astro_mon.t, astro_mon.Gamma_A[0], color='k', alpha=0.4)
        ax12.set_ylabel(r'$\Gamma_A$')
        ax12.grid(linestyle='dotted')

        ax13.plot(astro_mon.t, astro_mon.I[astro_i]/umolar, color='blue')
        if astro_i>0: ax13.plot(astro_mon.t, astro_mon.I[0]/umolar, color='k', alpha=0.4)
        ax13.set_ylabel(r'$I$ ($\mu$M)') 
        ax13.grid(linestyle='dotted')

        ax14.plot(astro_mon.t, astro_mon.C[astro_i]/umolar, color='red')
        if astro_i>0: ax14.plot(astro_mon.t, astro_mon.C[0]/umolar, color='k', alpha=0.4)
        ax14.set_ylabel(r'$Ca^{2\plus}$ ($\mu$M)') 
        ax14.grid(linestyle='dotted')

        ax15.plot(astro_mon.t, astro_mon.h[astro_i], color='C5')
        if astro_i>0: ax15.plot(astro_mon.t, astro_mon.h[0], color='k', alpha=0.4)
        ax15.set_ylabel(r'h')
        ax15.set_xlabel('time (s)')
        ax15.grid(linestyle='dotted')


    if args.b:
        fig2 = plt.figure(num='Biforcation')
        ax21 = fig2.add_subplot(1,1,1)

        for par, bif in zip(Y_s_val, Bif_val):
                ax21.plot(par, bif, 'go', markersize=2)
        ax21.set_xlabel(r'$Y_S$ ($\mu$M)')
        ax21.set_ylabel(r'$Ca^{2\plus}$ ($\mu$M)')  
        ax21.set_title('Biforcation')
        ax21.grid(linestyle='dotted')


    plt.show()