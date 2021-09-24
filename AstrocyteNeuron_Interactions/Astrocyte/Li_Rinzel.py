"""
Calcium oscillations in single astrocyte.
Dynamic analysis of Li-Rinzel model, for details see
- De Pittà et al, 'Coexistence of amplitude and frequency 
  modulations in intracellular calcium dynamics' (2008)
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from scipy import integrate
from scipy.optimize import fsolve
from scipy.signal import argrelextrema

def LiRinzel(X, t, I):
    """
    Li-Rinzel model, dynamical behavior of calcium oscillation
    in a single astrocyte.
    Set of two time-indipendent nonlinear ODEs where the main 
    variable is calcium concentration into the cytosol.

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
    C,h = X

    Q2 = d2 * ((I+d1)/(I+d3))
    m_inf = (I/(I+d1)) * (C/(C+d5))
    h_inf = Q2 / (Q2+C)
    tau_h = 1 / (a2*(Q2+C))

    J_leak = v2 * (C0-(1+c1)*C)
    J_pump = (v3*C**2) / (K3**2+C**2)
    J_chan = v1 * (m_inf**3) * (h**3) * (C0-(1+c1)*C)

    dvdt = [J_chan + J_leak - J_pump,
            (h_inf - h) / tau_h]

    return np.array(dvdt)

def LiRinzel_nunc(C_start=0, C_stop=0.8, steps=1000):
    """
    Nunclines of Li-Rinzel model, analytic expression and 
    just one stable state.

    Parameters
    ----------
    C_start: integer,float(optional)
        initial value of C 

    C_stop: integer or float(optional)
        final value of C

    steps: integer(optional)
        total number of C value

    Returns
    -------
    C_nunc: numpy array
        numpy array of C value

    h_nunc1: numpy array
        numpy array of h values of first nuncline

    h_nunc2: numpy array 
        numpy array of h values of second nuncline
    """
    C_nunc = np.linspace(C_start,C_stop,steps)

    Q2 = d2 * ((I+d1)/(I+d3))
    
    h_nunc1 = Q2/(Q2+C_nunc)
    h_nunc2 = ((((v3*C_nunc**2)/(K3**2+C_nunc**2))-v2*(C0-(1+c1)*C_nunc))/
               (v1*(((I/(I+d1))*(C_nunc/(C_nunc+d5)))**3)*(C0-(1+c1)*C_nunc)))**(1/3)
    
    return np.array(C_nunc), np.array(h_nunc1), np.array(h_nunc2)

def LiRinzel_stable(X):
    """
    Stable state of Li Rinzel model by implicit equations
    of nunclines
    
    Parameters
    ----------
    X: array
        state variables

    Returns
    -------
    implicit_form: tuple
        Implicit form of nunclines
    """
    C,h = X

    Q2 = d2 * ((I+d1)/(I+d3))
    eqs1 = h - Q2/(Q2+C)
    eqs2 =  h - ((((v3*C**2)/(K3**2+C**2))-v2*(C0-(1+c1)*C))/
               (v1*(((I/(I+d1))*(C/(C+d5)))**3)*(C0-(1+c1)*C)))**(1/3)
        
    implicit_form = (eqs1, eqs2)

    return implicit_form

def LiRinzel_Jacobian(C,h):
    """
    Jacobian matrix of Li Rinzel model

    Parameters
    ----------

    Returns
    -------
    """
    Q2 = d2 * ((I+d1)/(I+d3))

    Jac_21 = -a2*h
    Jac_22 = -a2*Q2
    pass


def Biforcation(model, par_start, par_stop, par_tot=300, t0=0., t_stop=500., dt=2E-2, t_relax=-5000):
    """
    Biforcation analysis of continous 2D dynamical system
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
    X0 = np.array([0.2,0.2])

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


def Period(model, par_start, par_stop, par_tot=300, t0=0., t_stop=500., dt=2E-2):
    """
    Oscillation periods of 2D dynamical system
    concern different values of the parameter

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

    X0 = np.array([0.0,0.0])

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
    

def AF_Modulation(model, *I_values):
    """
    Different type of excitability of Li-Rinzel. Amplitude and Frequency
    modulation concern different values of IP3 concentration.
    IP3 signals is a steps function with increasing values of IP3 concentration

    Parameters
    ----------
    model: callable(y, t, ...) or callable(t, y, ...) 
        Computes the derivative of y at t. If the signature is callable(t, y, ...), then the argument tfirst must be set True.
        from scipy.integrate.odeint

    I_values: integer or float
        IP3 values concentration of signal

    Returns
    -------
    t_tot: list
        each item of the list is the time widow where the IP3
        concentration is costan

    C_tot: list
        Each item of the list is the dynamical behaviour 
        of Calcium concentration inside time window with constant 
        IP3 concentration
    """

    t0 = 0.      
    t_wind = 100.
    dt = 2E-2
    t_fin = t_wind
    X0 = np.array([0.0,0.0])

    C_tot = list()
    t_tot = list()
    I_tot = list()
    
    for I_item in I_values: 
        t = np.arange(t0, t_fin, dt)
        
        sol  = integrate.odeint(model, X0, t, args=(I_item,))
        C = sol[:,0]
        h = sol[:,1]

        C_tot.append(C)
        t_tot.append(t)

        t0 = t_fin
        t_fin = t0 +t_wind
        X0 = np.array([C[-1],h[-1]])

        I_graph = np.repeat(I_item, t.shape[0])
        I_tot.append(I_graph)

    return t_tot, C_tot , I_tot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Dynamic analysis of Li Rinzel model')
    parser.add_argument("-K3", type=float,
                        help="""K3 parameter descriminates Amplitude Modulation (AM) to Frequency Modelation (FM):
                                 K3=0.1 AM; K3=0.051 FM""")
    parser.add_argument("-I", type=float,
                        help="""I parameter determines single dynamic behaviour: suggest range [0.1-1.5]""")
    parser.add_argument('-excitability', action='store_false', help="""Different type of excitability, default=True""")
    args = parser.parse_args()
    
    #Parameters
    v1 = 6.0      #sec-1
    v2 = 0.11     #sec-1
    v3 = 0.9      #muM*sec-1
    d1 = 0.13     #muM
    d2 = 1.049
    d3 = 0.9434
    d5 = 0.08234
    C0 = 2.0      #muM
    c1 = 0.185    #adimensional
    a2 = 0.2      #muM-1*sec-1

    I = args.I  #muM
    K3 = args.K3  #muM

    #Stable state
    C_stable, h_stable =  fsolve(LiRinzel_stable, (0.2,0.7))
    print (f'Stable state K3 = {K3}, I = {I}:  C={C_stable:.4f} h={h_stable:.4f}')

    #Nunclines 
    C_nunc, h_nunc1, h_nunc2 = LiRinzel_nunc()
    
    #Dynamical behavior - solution
    t0 = 0.      #sec
    t_fin = 600.
    dt = 2E-2

    t = np.arange(t0, t_fin, dt)
    X0 = np.array([0.2,0.2])

    sol  = integrate.odeint(LiRinzel, X0, t, args=(I,))
    C = sol[:,0]
    h = sol[:,1]

    #Qualitative analysis - Arrow field rapr
    xx = np.linspace(0.0, 0.8, 20)
    yy = np.linspace(0.0, 1.0, 20)

    XX, YY = np.meshgrid(xx, yy)    #create grid
    DX1, DY1 = LiRinzel([XX,YY],t,I)  #arrows' lenghts in cartesian cordinate
    
    M = np.hypot(DX1,DY1)  #normalization with square root
    M[M==0] = 1

    DX1 = DX1/M
    DY1 = DY1/M

    #Biforcation
    if args.K3 == 0.1:
        I_l1, Bif_l1 = Biforcation(LiRinzel,0.1,0.4,par_tot=50,t0=0.,t_stop=400.,dt=2E-2,t_relax=-17000)
        I_l2, Bif_l2 = Biforcation(LiRinzel,0.4,0.7,par_tot=70,t0=0.,t_stop=700.,dt=2E-2,t_relax=-10000)

    if args.K3 == 0.051:
        I_l1, Bif_l1 = Biforcation(LiRinzel,0.1,0.5,par_tot=50,t0=0.,t_stop=400.,dt=2E-2,t_relax=-15000)
        I_l2, Bif_l2 = Biforcation(LiRinzel,0.5,1.1,par_tot=60,t0=0.,t_stop=700.,dt=2E-2,t_relax=-10000)
        I_l3, Bif_l3 = Biforcation(LiRinzel,1.1,1.5,par_tot=50,t0=0.,t_stop=400.,dt=2E-2,t_relax=-15000)


    #Periods
    if args.K3 == 0.1:
        I_list, Per_list = Period(LiRinzel, 0.35, 0.7, par_tot=30)
    if args.K3 == 0.051:
        I_list, Per_list = Period(LiRinzel, 0.5, 1.1, par_tot=50)


    #Differnt type of exatability
    if args.excitability:
        K3 = 0.1
        t_AM, C_AM ,I_AM = AF_Modulation(LiRinzel, 0.2,0.4,0.5,0.6,0.8,1.5)

        K3= 0.051
        t_FM, C_FM, I_FM = AF_Modulation(LiRinzel, 0.2,0.4,0.5,0.6,0.8,1.5)

        
    #Plots
    if args.K3 == 0.1:
        title=f'Li-Rinzel Amplitude Modulation - I:{I}'
    if args.K3 == 0.051:
        title=f'Li-Rinzel Frequency Modulation - I:{I}'

    fig = plt.figure(num=title, figsize=(12,9))
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    ax31 = fig.add_subplot(2,2,4)

    ax1.plot(t[-10000:], C[-10000:], 'r-', label=r'$Ca^{2\plus}$')  
    ax1.set_title(f"Calcium dynamic - I = {I}")
    ax1.set_xlabel("time")
    ax1.set_ylabel(r'$Ca^{2\plus}$')
    ax1.grid(linestyle='dotted')
    
    ax2.plot(C, h, 'm-', label='dynamic')
    ax2.quiver(XX, YY, DX1, DY1, color='orange', pivot='mid', alpha=0.5)
    ax2.plot(C_nunc,h_nunc1, color="blue", linewidth=0.7, alpha=0.5, label="nunclines")
    ax2.plot(C_nunc,h_nunc2, color="blue", linewidth=0.7, alpha=0.5)
    ax2.set_xlabel(r'$Ca^{2\plus}$')
    ax2.set_ylabel("h")  
    ax2.set_title("Phase space")
    ax2.grid(linestyle='dotted')
    ax2.legend(loc='upper right')

    if args.K3 == 0.1:
        for I, bif in zip(I_l1,Bif_l1):
            ax3.plot(I, bif, 'go', markersize=2)
        for I, bif in zip(I_l2,Bif_l2):
            ax3.plot(I, bif, 'go', markersize=2)
        ax3.set_xlabel('I')
        ax3.set_ylabel(r'$Ca^{2\plus}$')  
        ax3.set_title('Biforcation')
        ax3.grid(linestyle='dotted')

    if args.K3 == 0.051:
        for I, bif in zip(I_l1,Bif_l1):
            ax3.plot(I, bif, 'go', markersize=2)
        for I, bif in zip(I_l2,Bif_l2):
            ax3.plot(I, bif, 'go', markersize=2)
        for I, bif in zip(I_l3,Bif_l3):
            ax3.plot(I, bif, 'go', markersize=2)
        ax3.set_xlabel('I')
        ax3.set_ylabel(r'$Ca^{2\plus}$')  
        ax3.set_title('Biforcation')
        ax3.grid(linestyle='dotted')
    
    ax31.scatter(I_list, Per_list, marker="^")
    ax31.set_xlabel('I')
    ax31.set_ylabel('Period [s]')  
    ax31.set_title('Periods')
    ax31.grid(linestyle='dotted')

    fig1 = plt.figure(f'Different type of excitability', figsize=(15,8))
    ax4 = fig1.add_subplot(3,1,1)
    ax5 = fig1.add_subplot(3,1,2)
    ax6 = fig1.add_subplot(3,1,3)

    for t_i,C_i, in zip(t_AM,C_AM):
        ax4.plot(t_i, C_i, linestyle='-', color='indigo', label=r'$Ca^{2\plus}$ - AM')
    ax4.set_title(f"Different type of excitability")
    ax4.set_ylabel(r'$Ca^{2\plus}$')
    ax4.grid(linestyle='dotted')

    for t_i,C_i, in zip(t_FM,C_FM):
        ax5.plot(t_i, C_i, linestyle='-', color='indigo', label=r'$Ca^{2\plus} - FM$')
    ax5.set_ylabel(r'$Ca^{2\plus}$')
    ax5.grid(linestyle='dotted')
    
    for t_i,I_i, in zip(t_FM,I_FM):
        ax6.plot(t_i, I_i, linestyle='-', color='indigo', label='I signal')
    ax6.set_xlabel("time (s)")
    ax6.set_ylabel(r'I')
    ax6.grid(linestyle='dotted')

    plt.show()


