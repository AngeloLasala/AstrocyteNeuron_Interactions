"""
Deterministic LR model deal with Calcium oscillations with
large number of chanels. When the number of channels is finite
the stochastic nature of opening-closing emerge.

Numerical solutions of stochastic LR model with differents number 
of channel

- "Modeling of Stochastic Ca 2+ Signals" Rudiger, Shuai (2017)

- "Modeling euron-glia interaction with Brian 2 simulator", Stimberg et al (2017)
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import integrate

def LiRinzel(X, t, I):
    """
    Li-Rinzel model, dynamical behavior of calcium oscillation
    in a single astrocyte.
    Set of two time-indipendent nonlinear ODEs where the main 
    variable is calcium concentration into the cytosol.

    Parameters
    ----------
    X : list, numpy array
        array of dynamical variables

    t : float
        time variable

    I : float
        IP3 concentration

    Returns
    -------
    dvdt: numpy.array
        numpy array of vector field

    """
    C,h = X

    Q_2 = d_2*((I+d_1)/(I+d_3))                 
    m_inf = (I/(I+d_1))*(C/(C+d_5))              
    alpha =  O_2 * Q_2
    beta =  O_2 * C

    J_r = Omega_C*(m_inf**3)*(h**3)*(C_T-(1+rho_A)*C)  
    J_l = Omega_L*(C_T-(1+rho_A)*C)                    
    J_p = (O_P*C**2)/(K_P**2+C**2)                    

    dvdt = [J_r + J_l - J_p,
            alpha * (1-h) - beta * h]

    return np.array(dvdt)

def stochastic_LiRinzel(X, t, I, N):
    """
    Stochastic Li-Rinzel model, dynamical behavior of calcium oscillation
    in a single astrocyte. The stochasticity is added to inactive varible h
    simulating the randomicity due to finite number of chanels.

    Set of two time-indipendent nonlinear ODEs where the main 
    variable is calcium concentration into the cytosol.

    Parameters
    ----------
    X : list, numpy array
        array of dynamical variables

    t : float
        time variable
    
    I : float
        IP3 concentration

    N : integer
        Number of channels


    Returns
    -------
    dvdt: numpy.array
        numpy array of vector field

    """

    C,h = X

    Q_2 = d_2*((I+d_1)/(I+d_3))                 
    m_inf = (I/(I+d_1))*(C/(C+d_5))
    alpha =  O_2 * Q_2
    beta =  O_2 * C

    J_r = Omega_C*(m_inf**3)*(h**3)*(C_T-(1+rho_A)*C)  
    J_l = Omega_L*(C_T-(1+rho_A)*C)                    
    J_p = (O_P*C**2)/(K_P**2+C**2)                    

    dvdt = [0,
            (alpha * (1-h) + beta * h) / N]

    return np.array(dvdt)

def der_stochastic_LiRinzel(X, t, I, N):
    """
    derivate of Stochastic Li-Rinzel model.

    Parameters
    ----------
    X : list, numpy array
        array of dynamical variables

    t : float
        time variable
    
    I : float
        IP3 concentration

    N : integer
        Number of channels


    Returns
    -------
    dvdt: numpy.array
        numpy array of vector field

    """

    C,h = X

    Q_2 = d_2*((I+d_1)/(I+d_3))                 
    m_inf = (I/(I+d_1))*(C/(C+d_5))              
    tau_h = 1 / (O_2*(Q_2+C))                    
    h_inf = Q_2/(Q_2+C)

    alpha =  O_2 * Q_2
    beta =  O_2 * C

    J_r = Omega_C*(m_inf**3)*(h**3)*(C_T-(1+rho_A)*C)  
    J_l = Omega_L*(C_T-(1+rho_A)*C)                    
    J_p = (O_P*C**2)/(K_P**2+C**2)                    

    dvdt = [0,
            (-alpha + beta) / N]

    return np.array(dvdt)

def LiRinzel_multiplicative(X, t, I):
    """
    Li-Rinzel model with multiplicative noise, dynamical behavior of calcium oscillation
    in a single astrocyte.
.
    Parameters
    ----------
    X : list, numpy array
        array of dynamical variables

    t : float
        time variable

    I : float
        IP3 concentration

    Returns
    -------
    dvdt: numpy.array
        numpy array of vector field

    """
    C,h = X

    Q_2 = d_2*((I+d_1)/(I+d_3))                 
    m_inf = (I/(I+d_1))*(C/(C+d_5))              
    tau_h = 1 / (O_2*(Q_2+C))                    
    h_inf = Q_2/(Q_2+C)

    J_r = Omega_C*(m_inf**3)*(h**3)*(C_T-(1+rho_A)*C)  
    J_l = Omega_L*(C_T-(1+rho_A)*C)                    
    J_p = (O_P*C**2)/(K_P**2+C**2)

    dvdt = [J_r + J_l - J_p,
            (h_inf - h) / tau_h]

    return np.array(dvdt)

def stochastic_LiRinzel_multiplicative(X, t, I):
    """
    Li-Rinzel model with multiplicative noise, dynamical behavior of calcium oscillation
    in a single astrocyte.
.
    Parameters
    ----------
    X : list, numpy array
        array of dynamical variables

    t : float
        time variable

    I : float
        IP3 concentration

    Returns
    -------
    dvdt: numpy.array
        numpy array of vector field

    """
    C,h = X

    Q_2 = d_2*((I+d_1)/(I+d_3))                 
    m_inf = (I/(I+d_1))*(C/(C+d_5))              
    tau_h = 1 / (O_2*(Q_2+C))                    
    h_inf = Q_2/(Q_2+C)
                    
    dvdt = [0,
            (h_inf - h) / (math.sqrt(tau_h))]

    return np.array(dvdt)

def der_stochastic_LiRinzel_multiplicative(X, t, I):
    """
    Li-Rinzel model with multiplicative noise, dynamical behavior of calcium oscillation
    in a single astrocyte.
.
    Parameters
    ----------
    X : list, numpy array
        array of dynamical variables

    t : float
        time variable

    I : float
        IP3 concentration

    Returns
    -------
    dvdt: numpy.array
        numpy array of vector field

    """
    C,h = X

    Q_2 = d_2*((I+d_1)/(I+d_3))                 
    m_inf = (I/(I+d_1))*(C/(C+d_5))              
    tau_h = 1 / (O_2*(Q_2+C))                    
    h_inf = Q_2/(Q_2+C)
                    
    dvdt = [0,
            -1/(math.sqrt(tau_h))]

    return np.array(dvdt)

if __name__ == "__main__":

    #Parameters
    # -- Calcium - CICR --
    Omega_C = 6.0       # Maximal rate of Ca^2+ release by IP_3Rs
    Omega_L = 0.1       # Maximal rate of calcium leak from the ER
    O_P = 0.9           # Maxium rate of SERCA uptake
    d_1 = 0.13          # IP_3 binding affinity
    d_2 = 1.049         # Ca inactivation dissociation costant
    d_3 = 0.9434        # IP3 dissociation constant
    d_5 = 0.08234       # Ca activation dissociation costant
    C_T = 2.0           # Total cell free Calcium concentration
    rho_A = 0.185       # Ratio between ER volume and cytosol
    O_2 = 0.2           # IP3R binding rate for Ca inhibition
    K_P = 0.1           # SERCA calcium affinity

    I = 0.4

    #fixed ramdom seed for reproducibility
    # np.random.seed(199586) 
    
    #Deterministic solutions
    t0 = 0.      #sec
    t_fin = 100.
    dt = 2E-2

    t = np.arange(t0, t_fin, dt)
    X0 = np.array([0.2,0.2])

    sol  = integrate.odeint(LiRinzel, X0, t, args=(I,))
    C_exc = sol[:,0]
    h_exc = sol[:,1]

    #Euler-Milstein stochastic LR
    N_values = [1,10,100,1000]
    sol = []
    Y1_list = []
    for N in N_values:
        X0 =  np.array([0.2,0.2])
        sol_N = []
        for i in t:
            Y1 = np.random.normal(0,1)
            Y1_list.append(Y1)
            X_new = X0 + dt * LiRinzel(X0, i, I) + stochastic_LiRinzel(X0, i, I, N)*math.sqrt(dt)*Y1 
            + 0.5*der_stochastic_LiRinzel(X0, i, I, N)*stochastic_LiRinzel(X0, i, I, N)*dt*Y1**2
            sol_N.append(X_new)
            X0 = X_new
        sol.append(sol_N)

    sol = np.array(sol)
    
    
    #Euler-Milstein stochastic LR_multiplicative
    X0 =  np.array([0.2,0.2])
    sol_m = []
    for en,i in enumerate(t):
        Y1 = np.random.normal(0,1)
        X_new = X0 + dt * LiRinzel_multiplicative(X0, i, I)+ stochastic_LiRinzel_multiplicative(X0, i, I)*math.sqrt(dt)*Y1 
        + 0.5*der_stochastic_LiRinzel_multiplicative(X0, i, I)*stochastic_LiRinzel_multiplicative(X0, i, I)*dt*Y1**2
        sol_m.append(X_new)
        X0 = X_new
    sol_m = np.array(sol_m)

    print(sol_m.shape)

    

    # Plots
    fig1 = plt.figure(num='Stochastic Li Rinzel model', figsize=(12,10))
    ax11 = fig1.add_subplot(2,1,1)
    ax12 = fig1.add_subplot(2,1,2)

    ax11.plot(t, C_exc, c='black', lw=2.5, label='C deterministic')
    for n in range(len(N_values)):
        ax11.plot(t,sol[n,:,0], label=f'C stochastic, N={N_values[n]}')
    ax11.set_ylabel(r'$Ca^{2\plus}$ ($\mu$M)')
    ax11.grid(linestyle='dotted')
    ax11.legend()

    ax12.plot(t, h_exc, c='black', lw=2.5, label='h deterministic')
    for n in range(len(N_values)):
        ax12.plot(t,sol[n,:,1], label=f'h stochastic, N={N_values[n]}')
    ax12.set_ylabel('h')
    ax12.set_xlabel('time (s)')
    ax12.grid(linestyle='dotted')
    ax12.legend()

    fig2 = plt.figure(num='Stochastic Li Rinzel model - multiplicative noise', figsize=(12,10))
    ax21 = fig2.add_subplot(2,1,1)
    ax22 = fig2.add_subplot(2,1,2)

    ax21.plot(t, sol_m[:,0], label='C mult_noise')
    ax21.set_ylabel(r'$Ca^{2\plus}$ ($\mu$M)')
    ax21.grid(linestyle='dotted')
    ax21.legend()

    ax22.plot(t, sol_m[:,1], label='h mult_noise') 
    ax22.set_ylabel('h')
    ax22.set_xlabel('time (s)')
    ax22.grid(linestyle='dotted')
    ax22.legend()

    plt.show()
