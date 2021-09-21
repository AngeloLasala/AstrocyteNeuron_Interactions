"""
Noninteractive astrocytes behavior with IP3 dynamics
"""
import matplotlib.pyplot as plt
import time
import numpy as np
from brian2 import *

if __name__ == "__main__":

    #Parameters
    d1 = 0.13*umolar
    d2 = 1.049*umolar
    d3 = 0.9434*umolar
    d5 = 0.08234*umolar
    v1 = 6/second
    v2 = 0.11/second
    v3 = 0.9*umolar/second
    c1 = 0.185
    C0 = 2*umolar
    a2 = 0.2/umolar/second

    astro_eqs = """
    dC/dt = J_chan + J_leak - J_pump: mmolar
    dh/dt = (h_inf - h) / tau_h: 1
    I : mmolar
    K3 : mmolar

    J_chan = v1*(m_inf**3)*(h**3)*(C0-(1+c1)*C): mmolar/second
    J_leak = v2*(C0-(1+c1)*C)                  : mmolar/second
    J_pump = (v3*C**2)/(K3**2+C**2)            : mmolar/second

    Q2 = d2*((I+d1)/(I+d3))                    : mmolar
    m_inf = (I/(I+d1))*(C/(C+d5))              : 1
    tau_h = 1 / (a2*(Q2+C))                    : second
    h_inf = Q2/(Q2+C)                          : 1
    """

    start = time.time()

    Astrocyte = NeuronGroup(10, astro_eqs, method='rk4')
    Astrocyte.K3 = np.linspace(0.051,0.1,10)*umolar

    print(f'K3: {Astrocyte.K3[:]}')
    
    # Astrocyte.I = 0.7*umolar

    M_Astro = StateMonitor(Astrocyte, ['C','h'], record=True)

    # This makes the specified block of code run every dt=100*ms. 
    # The run_regularly lets you run code specific 
    # to a single NeuronGroup
    dt_window = 100*second
    Astrocyte.run_regularly('I += 0.2*umolar', dt=dt_window)
    run(5*dt_window)

    stop = time.time()
    print(stop-start)

    fig = plt.figure(figsize=(10,10))

    ax1 = fig.add_subplot(2,1,1)
    for l in range(5):
        ax1.axvline(l*dt_window/second, ls='--', color='lightgrey')
    ax1.plot(M_Astro.t/second, M_Astro.C[0]*1000, label=f'K3={Astrocyte.K3[0]}')
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel(r'C ($\mu$M)')
    ax1.set_title('Calcium dynamics - CICR')
    ax1.legend()

    ax2 = fig.add_subplot(2,1,2)
    for l in range(5):
        ax2.axvline(l*dt_window/second, ls='--', color='lightgrey')
    ax2.plot(M_Astro.t/second, M_Astro.C[-1]*1000, label=f'K3={Astrocyte.K3[-1]}')
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel(r'C ($\mu$M)')
    ax2.set_title('Calcium dynamics - CICR')
    ax2.legend()

    plt.show()