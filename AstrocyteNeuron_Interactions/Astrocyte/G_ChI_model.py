"""
'G-ChI' model, extention of Li Rinzel that takes into account
the IP3 metabolism and the glutamate-dependent production.
The later mechanism is treated in quasi-static approximation

- "Glutamate regulation of calcium and IP3 oscillating 
   and pulsating dynamics in astrocytes" De Pittà et al (2009a)

- "Modelling neuro-glia interactions with the Brian2 simulator" Stimberg et al (2017)
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate
from scipy.signal import argrelextrema
from sympy.solvers import solve
from sympy import re, im, symbols

def Hill(L, K, exp):
    """
    Hill equation reflect the binding of ligands to macromolecules, 
    as C or IP3 with related-receptors in astrocyte membran

    Paramters
    ---------
    L: float
        Ligands concentration
    
    K:  float
        ligand concentration producing half occupation

    exp: integer or float
        exponent of Hill fuction

    Returns
    -------
    theta: float
        fraction of the receptor protein concentration that is bound by the ligand
        (fraction of receptor binding sites)
    """
    theta = L**exp/(L**exp+K**exp)
    return theta

def G_ChI(X, t, gamma):
    """
    Model of calcium dynamics with endogenous IP3 metabolism based on
    Li Rinzel (C and h variable)model and IP3 (I) concentration
    provided by a third couple model and the glutamate-deoendent IP3 production.

    Parameters
    ----------
    X: list, numpy array
        array of dynamical variables

    t: float
        time variable

    gamma: float
        Extracellular concentrations of glutamate, control parameters

    Returns
    -------
    dvdt: numpy.array
        numpy array of vector field
    """
    C,h,I = X

    Q2 = d2 * ((I+d1)/(I+d3))
    m_inf = Hill(I,d1,1) * Hill(C,d5,1)
    h_inf = Q2 / (Q2+C)
    tau_h = 1 / (a2*(Q2+C))

    J_leak = v2 * (C0-(1+c1)*C)
    J_pump = v3 * Hill(C,K3,2)
    J_chan = v1 * (m_inf**3) * (h**3) * (C0-(1+c1)*C)

    J_beta = v_beta * Hill(gamma, K_R*(1+(K_P/K_R)*Hill(C,K_phi,1)), 0.7) 
    J_delta = v_delta * (k_delta/(k_delta+I)) * Hill(C,K_PLCdelta,2)
    J_3K = v_3k * Hill(C,K_D,4) * Hill(I,K_3k,1)
    J_5P = r_5p *I

    dvdt = [J_chan + J_leak - J_pump,
            (h_inf - h) / tau_h,
            J_beta + J_delta - J_3K - J_5P]

    return np.array(dvdt)

def Biforcation3D(model, par_start, par_stop, par_tot=300, X0_ic=0.0,X1_ic=0.0,X2_ic=0.0,t0=0., t_stop=500., dt=2E-2, t_relax=-5000):
    """
    Biforcation analysis of continous 3D dynamical system
    throught maximum and minimum discete mapping

    To taking account relaxation time avoiding transient regime, 
    local extremes is found only at the end of variable evolution, 
    the extation of this time regione is set by t_relax.

    Note: In this version the bifurcation is computed only respect to 
    the only control parameters

    Parameters
    ----------
    model: callable(y, t, ...) or callable(t, y, ...) 
        Computes the derivative of y at t. If the signature is callable(t, y, ...), then the argument tfirst must be set True.
        Model codimension must be 1 thereby bifurcation analysis concerns only the parameters.
        from scipy.integrate.odeint

    par_stat: integer or float
        initial value of parameter

    par_stop: integer or float
        final value of parameter

    par_tot: integer(optional)
        total number of parameter value. Default par_tot=300

    X0_ic: float (optional)
        first variable initial condiction. Defaul=0

    X1_ic: float (optional)
        second variable initial condiction. Defaul=0
    
    X2_ic: float (optional)
        third variable initial condiction. Defaul=0
    
    t0: integer or float(optional)
        initial time. Default t0=0

    t_stop: integer or float(optional)
        final time. Default t_stop=200

    dt: integer or float(optional)
        integration step. Default dt=2E-2

    t_relax: negative integer(optional)
        time window to taking account relaxation time. Default t_relax=-5000
    """
    t0 = t0      #sec
    t_stop = t_stop
    dt = dt
      
    t = np.arange(t0, t_stop, dt)
    X0 = np.array([X0_ic,X1_ic,X2_ic])

    I_list = list()
    Bif_list = list()
    
    for i in np.linspace(par_start, par_stop, par_tot):
        sol = integrate.odeint(model, X0, t, args=(i,))
        X = sol[:,0]
        Y = sol[:,1]
        X = X[t_relax:]
        
        max_loc = argrelextrema(X, np.greater)
        min_loc = argrelextrema(X, np.less)

        X_max = X[max_loc].tolist()
        X_min = X[min_loc].tolist()
        Bif_val = X_max + X_min
        I_x = [i for item in range(len(Bif_val))]

        I_list.append(I_x)
        Bif_list.append(Bif_val)
        
        
    return I_list, Bif_list

def Period3D(model, par_start, par_stop, par_tot=300, t0=0., t_stop=500., dt=2E-2):
    """
    Oscillation periods of 2D dynamical system
    concern different values of the parameter.


    Parameters
    ----------
    model: callable(y, t, ...) or callable(t, y, ...) 
        Computes the derivative of y at t. If the signature is callable(t, y, ...), then the argument tfirst must be set True.
        Model codimension must be 1 thereby bifurcation analysis concerns only the parameters.
        from scipy.integrate.odeint

    par_stat: integer or float
        initial value of parameter

    par_stop: integer or float
        final value of parameter

    par_tot: integer(optional)
        total number of parameter value. Default par_tot=300

    t0: integer or float(optional)
        initial time. Default t0=0

    t_stop: integer or float(optional)
        final time. Default t_stop=200

    dt: integer or float(optional)
        integration step. Default dt=2E-2

    Returns
    -------
    par_list: list
        paremeters list over compute the oscillation periods

    period_list: list
        list of oscillation periods
    """
    t0 = t0      #sec
    t_stop = t_stop
    dt = dt
    t = np.arange(t0, t_stop, dt)

    X0 = np.array([0.0,0.0,0.0])

    par_list = np.linspace(par_start, par_stop, par_tot)
    period_list = list()
    for i in par_list:
        sol  = integrate.odeint(model, X0, t, args=(i,))
        X = sol[:,0]
        Y = sol[:,1]

        max_loc = argrelextrema(X, np.greater)

        X_max = X[max_loc]
        t_max0 = t[max_loc]
        t_max1 = t_max0[1:]
        
        per = t_max1 - t_max0[:-1]
        period = np.mean(per)

        period_list.append(period)

    return par_list, period_list

def Encoding(model, *G_values, X0_ic=0, X1_ic=0, X2_ic=0, t_wind=100):
    """
    Different type of encoding modes of a dynamical 3D model, as G-ChI.
    The external stimulus is steps fuction defined by G_values parameters

    Parameters
    ----------
    model: callable(y, t, ...) or callable(t, y, ...) 
        Computes the derivative of y at t. If the signature is callable(t, y, ...), then the argument tfirst must be set True.
        from scipy.integrate.odeint

    G_values: integer or float
        External stumuls concentration 

    X0_ic: float (optional)
        first variable initial condiction. Defaul=0

    X1_ic: float (optional)
        second variable initial condiction. Defaul=0
    
    X2_ic: float (optional)
        third variable initial condiction. Defaul=0

    t_wind: float (optional)
        time window over which the external signal concentration is costant. Defaul=100

    Returns
    -------
    t_tot: list
        each item of the list is the time widow where the IP3
        concentration is costan

    C_tot: list
        Each item of the list is the dynamical behaviour 
        of Calcium concentration inside time window with constant 
        external signal concentration
    """

    t0 = 0.      
    t_wind = t_wind
    dt = 2E-2
    t_fin = t_wind
    X0 = np.array([X0_ic,X1_ic,X2_ic])

    C_tot = list()
    I_tot = list()
    t_tot = list()
    G_tot = list()
    
    for G_item in G_values: 
        t = np.arange(t0, t_fin, dt)
        
        sol  = integrate.odeint(model, X0, t, args=(G_item,))
        C = sol[:,0]
        h = sol[:,1]
        I = sol[:,2]

        C_tot.append(C)
        I_tot.append(I)
        t_tot.append(t)

        t0 = t_fin
        t_fin = t0 +t_wind
        X0 = np.array([C[-1],h[-1], I[-1]])

        G_graph = np.repeat(G_item, t.shape[0])
        G_tot.append(G_graph)

    return t_tot, C_tot , I_tot, G_tot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Dynamic analysis of ChI model')
    parser.add_argument("-K3", type=float,
                        help="""K3 parameter descriminates Amplitude Modulation (AM) to Frequency Modelation (FM):
                                 K3=0.1 AM; K3=0.051 FM""")
    parser.add_argument("-gamma", type=float, help="""gamma parameter: extracellular Glutammate concentration:""")
    parser.add_argument('-biforcation', action='store_true', help="""Different type of excitability, default=False""")

    args = parser.parse_args()

    # Parameter CICR
    v1 = 6.0       # Maximal CICR rate, sec-1
    v2 = 0.11      # Maximal rate of calcium leak from the ER, sec-1
    v3 = 0.9       # Maxium rate of SERCA uptake, muM*sec-1
    d1 = 0.13      # IP3 dissociation constant, muM
    d2 = 1.049     # Ca inactivation dissociation costant, muM
    d3 = 0.9434    # IP3 dissociation constant, muM 
    d5 = 0.08234   # Ca activation dissociation costant, muM
    C0 = 2.0       # Total cell free Calcium concentration, muM
    c1 = 0.185     # Ratio between cytosol and ER volume, adimensional
    a2 = 0.2       # IP3R binding rate for Ca inhibition, muM-1*sec-1
    K3 = args.K3   # SERCA calcium affinity, muM

    # Parameters IP3 metabolism
    #PLC delta production - glutamate 
    if K3 == 0.1: v_beta = 0.2    # Maximal rate of IP3 production by PLC_beta, muM*sec-1
    if K3 == 0.051: v_beta = 0.5 
    K_R = 1.3     # Glutamate affinity withreceptor, muM
    K_P = 10      # Ca-PKC-dependet inhibition factor
    K_phi = 0.6   # Ca affinity of PKC

    #PLC delta production 
    if K3 == 0.1:  v_delta = 0.02  # Maximal rate of IP3 production by PLC_delta, muM*sec-1
    if K3 ==0.051: v_delta = 0.05
    k_delta = 1.5      # Inhibition constant of PLC_delta activity, muM
    K_PLCdelta = 0.1   # Ca affinity of PLC_delta, muM
    #degradation
    v_3k = 2.0         # Maximal rate of degradation by IP3-3K, muM*sec-1
    K_3k = 1.0         # IP3 affinity of IP3-3K, muM
    K_D = 0.7          # Ca affinity of IP3-3K, muM
    if K3 == 0.1:   r_5p = 0.04    # Maximal rate of degradation by IP3-5P, muM*sec-1
    if K3 == 0.051: r_5p = 0.05
      
    gamma = args.gamma #muM

    # Parameters - time
    t0 = 0.      #sec
    t_fin = 700.
    dt = 2E-2

    t = np.arange(t0, t_fin, dt)
    X0 = np.array([0.0,0.0,0.0])

    sol  = integrate.odeint(G_ChI, X0, t, args=(gamma,))
    C = sol[:,0]
    h = sol[:,1]
    I = sol[:,2]

    #Encoding modes
    if K3 == 0.1:
        t_AFM, C_AFM , I_AFM, gamma_AFM = Encoding(G_ChI, 0.002,3,0.002,3,0.002)

    if K3 == 0.051:
        t_AFM, C_AFM , I_AFM, gamma_AFM = Encoding(G_ChI, 0.001,3,0.001,3,0.001)


    
    #Biforcations and Periods
    if args.biforcation:
        if args.K3 == 0.1:
            gamma_l1, bif_l1 = Biforcation3D(G_ChI, par_start=0.006, par_stop=0.014, par_tot=20,t0=0.,t_stop=400.,dt=2E-2,t_relax=-20000)
            gamma_l2, bif_l2 = Biforcation3D(G_ChI, par_start=0.013, par_stop=3.5, par_tot=250,t0=0.,t_stop=500.,dt=2E-2,t_relax=-15000)
            gamma_l3, bif_l3 = Biforcation3D(G_ChI, par_start=3.5, par_stop=6, par_tot=20,t0=0.,t_stop=500.,dt=2E-2,t_relax=-7000)

            gamma_list, Per_list = Period3D(G_ChI, 0.002, 3.8, par_tot=75)
            

        if args.K3 == 0.051:
            gamma_l1, bif_l1 = Biforcation3D(G_ChI, par_start=0.0001, par_stop=0.01, par_tot=10,t0=0.,t_stop=500.,dt=2E-2,t_relax=-20000)
            gamma_l2, bif_l2 = Biforcation3D(G_ChI, par_start=0.01, par_stop=15, par_tot=100,t0=0.,t_stop=500.,dt=2E-2,t_relax=-15000)
            gamma_l3, bif_l3 = Biforcation3D(G_ChI, par_start=12, par_stop=25, X0_ic=0.4, X2_ic=1.0,par_tot=50,t0=0.,t_stop=300.,dt=2E-2,t_relax=-7000)

            gamma_list, Per_list = Period3D(G_ChI, 0.01, 0.6, par_tot=75)

    

    #Plots
    if K3==0.1: title='G-ChI Amplitude Modulation'
    if K3==0.051: title='G-ChI Frequency Modulation'

    #Dynamics behaviour
    fig1 = plt.figure(num=title+' - Time evolution',figsize=(10,5))
    ax1 = fig1.add_subplot(1,2,1)
    ax2 = fig1.add_subplot(1,2,2)

    ax1.plot(t[-10000:], C[-10000:], 'r-', label=r'$Ca^{2\plus}$')  
    ax1.plot(t[-10000:], I[-10000:], 'b-', label=r'$IP_3$')  
    ax1.set_title(fr"ChI dynamics - $\gamma$ = {gamma}")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel(r'C,I ($\mu$M)')
    ax1.legend(loc='lower right')
    ax1.grid(linestyle='dotted')

    ax2.plot(C[-10000:], I[-10000:], 'm-')
    ax2.set_xlabel(r'C ($\mu$M)')
    ax2.set_ylabel(r'I ($\mu$M)')
    ax2.set_title(f"C-I Phase space")
    ax2.grid(linestyle='dotted')

    #Encoding mode
    if K3==0.1: mod='AM'
    if K3==0.051: mod='FM'
    fig2 = plt.figure(num=title+' - Encoding modes',figsize=(10,5))
    ax21 = fig2.add_subplot(3,1,1)
    ax22 = fig2.add_subplot(3,1,2)
    ax23 = fig2.add_subplot(3,1,3)

    for t_i,C_i, in zip(t_AFM,C_AFM):
        ax21.plot(t_i, C_i, linestyle='-', color='red', label=r'$Ca^{2\plus}$ - AM')
    ax21.set_title(f"{mod} - Encoding Mode")
    ax21.set_ylabel(r'C ($\mu$M)')
    ax21.grid(linestyle='dotted')

    for t_i,I_i, in zip(t_AFM,I_AFM):
        ax22.plot(t_i, I_i, linestyle='-', color='blue', label=r'$Ca^{2\plus}$ - AM')
    ax22.set_ylabel(r'I ($\mu$M)')
    ax22.grid(linestyle='dotted')

    for t_i,gamma_i, in zip(t_AFM,gamma_AFM):
        ax23.plot(t_i, gamma_i, linestyle='-', color='black', label=r'$Ca^{2\plus}$ - AM')
    ax23.set_ylabel(r'$\gamma$ ($\mu$M)')
    ax23.set_xlabel('time (s)')
    ax23.grid(linestyle='dotted')

    #Biforcation
    if args.biforcation:
        fig3 = plt.figure(num=title+' - Biforcation', figsize=(12,8))
        ax31 = fig3.add_subplot(1,2,1)
        ax32 = fig3.add_subplot(1,2,2)

        for gam, bif in zip(gamma_l1, bif_l1):
            ax31.plot(gam, bif, 'go', markersize=2)
        for gam, bif in zip(gamma_l2, bif_l2):
            ax31.plot(gam, bif, 'go', markersize=2)
        for gam, bif in zip(gamma_l3, bif_l3):
            ax31.plot(gam, bif, 'go', markersize=2)
        
        ax31.set_xlabel(r'$\gamma$ ($\mu$M)')
        ax31.set_ylabel(r'$Ca^{2\plus} ($\mu$M)$')  
        ax31.set_title(r'Biforcation respect to $\gamma$')
        ax31.grid(linestyle='dotted')

        ax32.scatter(gamma_list, Per_list, marker="^")
        ax32.set_xlabel(r'$\gamma$ ($\mu$M)')
        ax32.set_ylabel('Period [s]')  
        ax32.set_title('Periods')
        ax32.grid(linestyle='dotted')


    plt.show()