"""
'ChI' model, extention of Li Rinzel that takes into account
the IP3 production and degeneration.

- "Glutamate regulation of calcium and IP3 oscillating 
   and pulsating dynamics in astrocytes" De Pittà et al (2009a)

- "Modelling neuro-glia interactions with the Brian2 simulator" Stimberg et al (2017)
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy.signal import argrelextrema
from astrocyte import LiRinzel


def ChI(X, t):
    """
    

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

if __name__ == "__main__":

    # Parameter CICR
    v1 = 6.0      # Maximal CICR rate, sec-1
    v2 = 0.11     # Maximal rate of calcium leak from the ER, sec-1
    v3 = 0.9      # Maxium rate of SERCA uptake, muM*sec-1
    d1 = 0.13     # IP3 dissociation constant, muM
    d2 = 1.049    # Ca inactivation dissociation costant, muM
    d3 = 0.9434   # IP3 dissociation constant, muM 
    d5 = 0.08234  # Ca activation dissociation costant, muM
    C0 = 2.0      # Total cell free Calcium concentration, muM
    c1 = 0.185    # Ratio between cytosol and ER volume, adimensional
    a2 = 0.2      # IP3R binding rate for Ca inhibition, muM-1*sec-1
    K3 = 0.1     # SERCA calcium affinity, muM

    # Parameters IP3 metabolism
    #PLC delta production 
    v_delta = 0.02    # Maximal rate of IP3 production by PLC_delta, muM*sec-1
    k_delta = 1.5    # Inhibition constant of PLC_delta activity, muM
    K_PLCdelta = 0.1 # Ca affinity of PLC_delta, muM
    #degradation
    v_3k = 2.0       # Maximal rate of degradation by IP3-3K, muM*sec-1
    K_3k = 1.0       # IP3 affinity of IP3-3K, muM
    K_D = 0.7        # Ca affinity of IP3-3K, muM
    r_5p = 0.04     # Maximal rate of degradation by IP3-5P, muM*sec-1

    # Parameters - time
    t0 = 0.      #sec
    t_fin = 600.
    dt = 2E-2


    t = np.arange(t0, t_fin, dt)
    X0 = np.array([0.4,0.4,0.4])

    sol  = integrate.odeint(ChI, X0, t)
    C = sol[:,0]
    h = sol[:,1]
    I = sol[:,2]


    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    ax1.plot(t, C, 'r-', label=r'$Ca^{2\plus}$')  
    ax1.plot(t, I, 'b-', label=r'$IP_3$')  
    ax1.set_title(f"ChI dynamics")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel(r'C,I ($\mu$M)')
    ax1.legend(loc='lower right')
    ax1.grid(linestyle='dotted')

    ax2.plot(C, I)
    ax2.set_xlabel(r'C ($\mu$M)')
    ax2.set_ylabel(r'I ($\mu$M)')
    ax2.set_title(f"C-I Phase space")
    ax2.grid(linestyle='dotted')
    plt.show()