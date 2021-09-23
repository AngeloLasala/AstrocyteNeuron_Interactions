"""
'ChI' model, extention of Li Rinzel that takes into account
the IP3 production and degeneration.

- "Glutamate regulation of calcium and IP3 oscillating 
   and pulsating dynamics in astrocytes" De Pittà et al (2009a)

- "Modelling neuro-glia interactions with the Brian2 simulator" Stimberg et al (2017)
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy.signal import argrelextrema
from sympy import Eq, symbols, solve


def ChI(X, t, v_delta):
    """
    Model of calcium dynamics with endogenous IP3 metabolism based on
    Li Rinzel (C and h variable)model and IP3 (I) concentration
    provided by a third couple model

    Parameters
    ----------
    X: list, numpy array
        array of dynamical variables

    t: float
        time variable

    Returns
    -------
    dvdt: numpy.array
        numpy array of vector field

    """
    C,h,I = X

    Q2 = d2 * ((I+d1)/(I+d3))
    m_inf = (I/(I+d1)) * (C/(C+d5))
    h_inf = Q2 / (Q2+C)
    tau_h = 1 / (a2*(Q2+C))

    J_leak = v2 * (C0-(1+c1)*C)
    J_pump = (v3*C**2) / (K3**2+C**2)
    J_chan = v1 * (m_inf**3) * (h**3) * (C0-(1+c1)*C)

    J_delta = v_delta * (k_delta/(k_delta+I)) * (C**2/(C**2+K_PLCdelta**2))
    J_3K = v_3k * (C**4/(C**4+K_D**4)) * (I/(I+K_3k))
    J_5P = r_5p *I

    dvdt = [J_chan + J_leak - J_pump,
            (h_inf - h) / tau_h,
            J_delta - J_3K - J_5P]

    return np.array(dvdt)

def Biforcation3D(model, par_start, par_stop, par_tot=300, t0=0., t_stop=500., dt=2E-2, t_relax=-5000):
    """
    Biforcation analysis of continous 3D dynamical system
    throught maximum and minimum discete mapping

    To taking account relaxation time avoiding transient regime, 
    local extremes is found only at the end of variable evolution, 
    the extation of this time regione is set by t_relax.

    Note: In this version the bifurcation is cpmputed only respect to first variable in the model

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

    t_relax: negative integer(optional)
        time window to taking account relaxation time. Default t_relax=-5000
    """
    t0 = t0      #sec
    t_stop = t_stop
    dt = dt
      
    t = np.arange(t0, t_stop, dt)
    X0 = np.array([0.0,0.0,0.0])

    I_list = list()
    Bif_list = list()
    
    for i in np.linspace(par_start, par_stop, par_tot):
        sol  = integrate.odeint(model, X0, t, args=(i,))
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

# Nunclines 3D
def h_nunc(h, C=0.25):
    nunc1 = (d3*h*C-d1*d2*(1-h))/(d2*(1-h)-h*C)
    return nunc1

def C_nunc(C, h=0.71):
    nunc2 = d1*((C+d5)/(C*h))*((v3/v1)*(C**2/(C**2+K3**2))/(C0-(1+c1)*C)-(v2/v1))**1/3 / (1-((C+d5)/(C*h))*((v3/v1)*(C**2/(C**2+K3**2))*(C0-(1+c1)*C)-(v2/v1))**1/3)
    return nunc2
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Dynamic analysis of ChI model')
    parser.add_argument("-K3", type=float,
                        help="""K3 parameter descriminates Amplitude Modulation (AM) to Frequency Modelation (FM):
                                 K3=0.1 AM; K3=0.051 FM""")
    parser.add_argument("-r_5p", type=float,
                        help="""r_5p parameter descriminates Amplitude Modulation (AM) to Frequency Modelation (FM):
                                 r_5p=0.04 AM; r_5p=0.05 FM""")
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
    #PLC delta production 
    v_delta = 0.5    # Maximal rate of IP3 production by PLC_delta, muM*sec-1
    k_delta = 1.5      # Inhibition constant of PLC_delta activity, muM
    K_PLCdelta = 0.1   # Ca affinity of PLC_delta, muM
    #degradation
    v_3k = 2.0         # Maximal rate of degradation by IP3-3K, muM*sec-1
    K_3k = 1.0         # IP3 affinity of IP3-3K, muM
    K_D = 0.7          # Ca affinity of IP3-3K, muM
    r_5p = args.r_5p   # Maximal rate of degradation by IP3-5P, muM*sec-1

    # Parameters - time
    t0 = 0.      #sec
    t_fin = 500.
    dt = 2E-2

    t = np.arange(t0, t_fin, dt)
    X0 = np.array([0.4,0.4,0.4])

    sol  = integrate.odeint(ChI, X0, t, args=(v_delta,))
    C = sol[:,0]
    h = sol[:,1]
    I = sol[:,2]


    #Biforcations
    if args.K3 == 0.1:
        v_delta_l1, bif_l1 = Biforcation3D(ChI, par_start=0.005, par_stop=0.025, par_tot=70,t0=0.,t_stop=700.,dt=2E-2,t_relax=-20000)
        v_delta_l2, bif_l2 = Biforcation3D(ChI, par_start=0.020, par_stop=0.025, par_tot=70,t0=0.,t_stop=1000.,dt=2E-2,t_relax=-15000)
        v_delta_l3, bif_l3 = Biforcation3D(ChI, par_start=0.025, par_stop=0.14, par_tot=60,t0=0.,t_stop=400.,dt=2E-2,t_relax=-7000)
        v_delta_l4, bif_l4 = Biforcation3D(ChI, par_start=0.14, par_stop=0.16, par_tot=20,t0=0.,t_stop=800.,dt=2E-2,t_relax=-5000)

    if args.K3 == 0.051:
        v_delta_l1, bif_l1 = Biforcation3D(ChI, par_start=0.01, par_stop=1.0, par_tot=100,t0=0.,t_stop=700.,dt=2E-2,t_relax=-20000)

    #Periods
    if args.K3 == 0.1:
        v_delta_list, Per_list = Period3D(ChI, 0.027, 0.147, par_tot=30)
    
    if args.K3 == 0.051:
        v_delta_list, Per_list = Period3D(ChI, 0.18, 0.53, par_tot=30)
        

    # I_s = symbols('I_s')
    # C_s = 0.2
    # A = v_delta*k_delta*(C_s**2/(C_s**2-K_PLCdelta**2))
    # B = v_3k*(C_s**4/(C_s**4-K_D**2))
    # J_5P = r_5p *I_s
    # eqs = Eq(A*(I_s+K_3k)-B*(I_s**2+I_s*k_delta)-r_5p*I_s*(I_s+K_3k)*(I_s+k_delta),0)
    # print(solve(eqs,I_s))
         

    # cc = np.linspace(0, 0.8, 100)
    # II = h_nunc(cc)
    # print(II)

    if K3==0.1: title='ChI Amplitude Modulation'
    if K3==0.051: title='ChI Frequency Modulation'

    fig = plt.figure(num=title+' - Time evolution',figsize=(10,5))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    ax1.plot(t[-10000:], C[-10000:], 'r-', label=r'$Ca^{2\plus}$')  
    ax1.plot(t[-10000:], I[-10000:], 'b-', label=r'$IP_3$')  
    ax1.set_title(f"ChI dynamics")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel(r'C,I ($\mu$M)')
    ax1.legend(loc='lower right')
    ax1.grid(linestyle='dotted')

    ax2.plot(C[-10000:], I[-10000:], 'm-')
    ax2.set_xlabel(r'C ($\mu$M)')
    ax2.set_ylabel(r'I ($\mu$M)')
    ax2.set_title(f"C-I Phase space")
    ax2.grid(linestyle='dotted')

    

    fig2 = plt.figure(num=title+' - Biforcation', figsize=(10,5))
    ax21 = fig2.add_subplot(1,2,1)
    ax22 = fig2.add_subplot(1,2,2)

    if args.K3 == 0.1:
        for I, bif in zip(v_delta_l1, bif_l1):
            ax21.plot(I, bif, 'go', markersize=2)
        for I, bif in zip(v_delta_l2, bif_l2):
            ax21.plot(I, bif, 'go', markersize=2)
        for I, bif in zip(v_delta_l3, bif_l3):
            ax21.plot(I, bif, 'go', markersize=2)
        for I, bif in zip(v_delta_l4, bif_l4):
            ax21.plot(I, bif, 'go', markersize=2)
        ax21.set_xlabel(r'$v_{\delta}$')
        ax21.set_ylabel(r'$Ca^{2\plus}$')  
        ax21.set_title(r'Biforcation with regard $v_{\delta}$')
        ax21.grid(linestyle='dotted')

        ax22.scatter(v_delta_list, Per_list, marker="^")
        ax22.set_xlabel(r'$v_{\delta}$')
        ax22.set_ylabel('Period [s]')  
        ax22.set_title('Periods')
        ax22.grid(linestyle='dotted')

    if args.K3 == 0.051:
        for v, bif in zip(v_delta_l1, bif_l1):
            ax21.plot(v, bif, 'go', markersize=2)
        ax21.set_xlabel(r'$v_{\delta}$')
        ax21.set_ylabel(r'$Ca^{2\plus}$')  
        ax21.set_title(r'Biforcation with regard $v_{\delta}$')
        ax21.grid(linestyle='dotted')

        ax22.scatter(v_delta_list, Per_list, marker="^")
        ax22.set_xlabel(r'$v_{\delta}$')
        ax22.set_ylabel('Period [s]')  
        ax22.set_title('Periods')
        ax22.grid(linestyle='dotted')
    plt.show()