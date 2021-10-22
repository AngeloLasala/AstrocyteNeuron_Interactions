"""
Gliotrasmission release from an astrocyte and how could change 
synaptic neurotrasmitter release.

- "Modelling neuro-glia interactions with the Brian2 simulator" Stimberg et al (2017)
"""
import matplotlib.pyplot as plt
from brian2 import *

#Parameters
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

I = 0.4*umolar

astro_eqs = """
# Calcium dynamics (2):
dC/dt = J_r + J_l - J_p: mmolar
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
N_ch                                         : 1 (constant) 
noise                                        : 1 (constant) 
"""
N_astro = 4
astrocyte = NeuronGroup(N_astro, model=astro_eqs, method='milstein')
# astrocyte.N_ch = [1,5,10,100]
astrocyte.noise = [0,0.25,0.75,1]


astro_mon = StateMonitor(astrocyte, ['C', 'h'], record=True)

run(80*second, report='text')

# Plots
fig1 = plt.figure(num='Stochastic Li Rinzel model', figsize=(12,10))
ax11 = fig1.add_subplot(2,1,1)
ax12 = fig1.add_subplot(2,1,2)

for i in range(N_astro):
    ax11.plot(astro_mon.t[:], astro_mon.C[i]/umolar, label=f'C stochastic, N={astrocyte.noise[i]}')
ax11.set_ylabel(r'$Ca^{2\plus}$ ($\mu$M)')
ax11.grid(linestyle='dotted')
ax11.legend()

for i in range(N_astro):
    ax12.plot(astro_mon.t[:], astro_mon.h[i], label=f'h stochastic, N={astrocyte.noise[i]}')
ax12.set_ylabel('h')
ax12.set_xlabel('time (s)')
ax12.grid(linestyle='dotted')
ax12.legend()

plt.show()