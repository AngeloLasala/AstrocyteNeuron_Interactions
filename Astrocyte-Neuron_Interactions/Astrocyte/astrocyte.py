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
    dvdt: numpy.array
        numpy array of vector field

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

def LiRinzel_nunc(C_start=0, C_stop=0.8, steps=1000):
    """
    Nunclines of Li-Rinzel model

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
    C_nunc = np.linspace(0,0.8,100)

    Q2 = d2 * ((I+d1)/(I+d3))
    
    h_nunc1 = Q2/(Q2+C_nunc)
    h_nunc2 = ((((v3*C_nunc**2)/(K3**2+C_nunc**2))-v2*(C0-(1+c1)*C_nunc))/
               (v1*(((I/(I+d1))*(C_nunc/(C_nunc+d5)))**3)*(C0-(1+c1)*C_nunc)))**(1/3)
    
    return np.array(C_nunc), np.array(h_nunc1), np.array(h_nunc2)


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

    I = 0.9    #muM


    #Nunclines - just 1 (NOT WORK!!)
    C_nunc, h_nunc1, h_nunc2 = LiRinzel_nunc()
    print(h_nunc1[h_nunc1==h_nunc2])

    #Dynamical behavior - solution
    t0 = 0.      #sec
    t_fin = 100.
    dt = 2E-2

    t = np.arange(t0, t_fin, dt)
    X0 = np.array([0.2,0.2])

    sol  = integrate.odeint(LiRinzel, X0, t)
    C = sol[:,0]
    h = sol[:,1]

    #Qualitative analysis - Arrow field rapr
    xx = np.linspace(0.0, 0.8, 20)
    yy = np.linspace(0.0, 1.0, 20)

    XX, YY = np.meshgrid(xx, yy)    #create grid
    DX1, DY1 = LiRinzel([XX,YY],t)  #arrows' lenghts in cartesian cordinate
    
    M = np.hypot(DX1,DY1)  #normalization with square root
    M[M==0] = 1

    DX1 = DX1/M
    DY1 = DY1/M

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


    ax2.plot(C, h, color="red", label='dynamic')
    ax2.quiver(XX, YY, DX1, DY1, color='orange', pivot='mid', alpha=0.5)
    ax2.plot(C_nunc,h_nunc1, color="blue", linewidth=0.7, alpha=0.5, label="nunclines")
    ax2.plot(C_nunc,h_nunc2, color="blue", linewidth=0.7, alpha=0.5)
    ax2.set_xlabel(r'$Ca^{2\plus}$')
    ax2.set_ylabel("h")  
    ax2.set_title("Phase space")
    ax2.grid(linestyle='dotted')
    ax2.legend(loc='upper right')

    plt.show()


