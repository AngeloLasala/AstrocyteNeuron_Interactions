"""
Calcium oscillations in single astrocyte.
Dynamical analysis of Li-Rinzel model, for details see
- De Pittà et al, 'Coexistence of amplitude and frequency 
  modulations in intracellular calcium dynamics' (2008)
"""
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from scipy import integrate

def LiRinzel(X, t):
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
    

    """
    C,h = X

    Q2 = d2 * ((I+d1)/(I+d3))
    m_inf = (I/(I+d1)) * (C/(C+d5))
    h_inf = Q2/(Q2+C)
    tau_h = 1/(a2*(Q2+C))

    J_leak = v2 * (C0-(1+c1)*C)
    J_pump = (v3*C**2) / (K3**2+C**2)
    J_chan = v1 * (m_inf**3) * (h**3) * (C0-(1+c1)*C)

    dvdt = [J_chan + J_leak - J_pump,
            (h_inf - h)/tau_h]

    return np.array(dvdt)
     

if __name__ == "__main__":
    
    #Parameters
    v1 = 6.0      #sec-1
    v2 = 0.11     #sec-1
    v3 = 0.9      #muM*sec-1
    d1 = 0.13     #muM
    d2 = 1.049
    d3 = 0.9434
    d5 = 0.08234
    C0 = 2.0      #muM
    K3 = 0.1
    c1 = 0.185    #adimensional
    a2 = 0.2      #muM-1*sec-1

    I = 0.2    #muM

    #Dynamical behavior - solution
    t0 = 0.      #sec
    t_fin = 60.
    dt = 2E-2

    t = np.arange(t0, t_fin, dt)
    X0 = np.array([0.2,0.2])

    sol  = integrate.odeint(LiRinzel, X0, t)
    C = sol[:,0]
    h = sol[:,1]

    #Plots
    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    ax1.plot(t, C, 'r-', label=r'$Ca^{2\plus}$')
    ax1.set_title("Dynamics in time")
    ax1.set_xlabel("time")
    ax1.set_ylabel(r'$Ca^{2\plus}$')
    ax1.grid(linestyle='dotted')
    ax1.legend(loc='best')


    ax2.plot(C, h, color="orange", label='dynamic')
    ax2.set_xlabel(r'$Ca^{2\plus}$')
    ax2.set_ylabel("h")  
    ax2.set_title("Phase space")
    ax2.grid(linestyle='dotted')
    ax2.legend(loc='upper right')

    plt.show()


