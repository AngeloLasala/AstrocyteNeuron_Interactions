"""
Gliotrasmission release from an astrocyte and how could change 
synaptic neurotrasmitter release.

- "Modelling neuro-glia interactions with the Brian2 simulator" Stimberg et al (2017)
"""
import matplotlib.pyplot as plt
from brian2 import *

#####  PARAMETERS  #######################################################
## General Parameters ##
N_a = 1                    # Total number of astrocyte
transient = 16.5*second
duration = transient + 3000*ms   # Total simulation time

## Synapses parameters ##
rho_c = 0.005                 # Synaptic vesicle-to-extracellular space volume ratio (bigger then G_ChI_astrpcyte.py)
Y_T = 500*mmolar              # Total vesicular neurotransmitter concentration
Omega_c = 40/second            # Neurotransmitter clearance rate
U_0__star = 0.6               # Resting synaptic release probabilityOmega_d = 2.0/second          # Synaptic depression rate
Omega_f = 3.33/second         # Synaptic facilitation rate
Omega_d = 2.0/second          # Synaptic depression rate
O_G = 1.5/umolar/second       # Agonist binding (activating) rate
Omega_G = 0.5/(60*second)     # Agonist release (deactivating) rate


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

# -- Gliotrasmitter --
C_Theta = 0.5*umolar        # Ca^2+ threshold for exocytosis
Omega_A = 0.6/second        # Gliotransmitter recycling rate
U_A = 0.6                   # Gliotransmitter release probability
G_T = 200*mmolar            # Total vesicular gliotransmitter concentration
rho_e = 6.5e-4              # Astrocytic vesicle-to-extracellular volume ratio
Omega_e = 60/second         # Gliotransmitter clearance rate
alpha = 0.0                   # Gliotransmission nature
##############################################################################

# Neurons
spikes = [0, 50, 100, 150, 200,
          300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400]*ms
spikes += transient
print(spikes)
pre_synaptic = SpikeGeneratorGroup(1, np.zeros(len(spikes)), spikes)
post_synaptic = NeuronGroup(1, "")

# Synapses
syn_eqs = """
du_S/dt = -Omega_f * u_S                           : 1 (clock-driven)
dx_S/dt = Omega_d * (1-x_S)                        : 1 (clock-driven)
dY_S/dt = -Omega_c*Y_S                             : mmolar (clock-driven)
dGamma_S/dt = O_G*G_A*(1-Gamma_S)-Omega_G*Gamma_S  : 1 (clock-driven)
G_A                                                : mmolar
r_S                                                : 1
"""
syn_action = """
U_0 = (1 - Gamma_S) * U_0__star + alpha * Gamma_S
u_S += U_0 * (1 - u_S)
r_S = u_S * x_S
x_S -= r_S
Y_S += rho_c * Y_T * r_S
"""

synapses = Synapses(pre_synaptic, post_synaptic, model=syn_eqs, on_pre=syn_action, method='linear')
synapses.connect(True, n=2) # connect pre and post with 2 identical synapses
synapses.x_S = 1.0


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
dC/dt = J_r + J_l - J_p: mmolar
dh/dt = (h_inf - h) / tau_h: 1

J_r = Omega_C*(m_inf**3)*(h**3)*(C_T-(1+rho_A)*C)  : mmolar/second
J_l = Omega_L*(C_T-(1+rho_A)*C)                    : mmolar/second
J_p = (O_P*C**2)/(K_P**2+C**2)                     : mmolar/second

Q_2 = d_2*((I+d_1)/(I+d_3))                  : mmolar
m_inf = (I/(I+d_1))*(C/(C+d_5))              : 1
tau_h = 1 / (O_2*(Q_2+C))                    : second
h_inf = Q_2/(Q_2+C)                          : 1


# Fraction of gliotransmitter resources available for release
dx_A/dt = Omega_A * (1 - x_A) : 1

# gliotransmitter concentration in the extracellular space
dG_A/dt = -Omega_e*G_A        : mmolar

# Neurotransmitter concentration in the extracellular space
Y_S     : mmolar
"""

astro_release = """
G_A += rho_e*G_T*U_A*x_A
"""

astrocyte = NeuronGroup(1, model=astro_eqs, method='rk4',
                        threshold='C>C_Theta', refractory='C>C_Theta', reset='G_A += rho_e*G_T*U_A*x_A')
astrocyte.x_A = 1.0
astrocyte.h = 0.9
astrocyte.I = 0.4*umolar

# Synaptic to Astrocyte connection
ecs_syn_to_astro = Synapses(synapses, astrocyte,
                            'Y_S_post = Y_S_pre : mmolar (summed)')
ecs_syn_to_astro.connect(i=1, j=0)

# Astrocyte to Synapses connection
ecs_astro_to_syn = Synapses(astrocyte, synapses, model= "G_A_post = G_A_pre : mmolar (summed)")
ecs_astro_to_syn.connect(i=0, j=1)

#Monitor
spike_mon = SpikeMonitor(pre_synaptic)
syn_mon = StateMonitor(synapses, ['u_S','x_S','Y_S','Gamma_S'], record=True)
astro_mon = StateMonitor(astrocyte, ['C', 'G_A', 'Y_S'], record=True)

run(duration, report='text')
print(spike_mon.t)
print(syn_mon.u_S)


# Plots
fig1 = plt.figure(num='Extracellular gliotrasmitter and Calcium dynamics',
                  figsize=(10,10))

ax11 = fig1.add_subplot(3,1,1)
ax12 = fig1.add_subplot(3,1,2)
ax13 = fig1.add_subplot(3,1,3)

ax11.plot(astro_mon.t[:], astro_mon.Y_S[0]/umolar, color='C3')
ax11.set_ylabel(r'$Y_S$ ($\mu$M)')
ax11.grid(linestyle='dotted')

ax12.plot(astro_mon.t[:], astro_mon.C[0]/umolar, color='red')
ax12.set_ylabel(r'$Ca^{2\plus}$ ($\mu$M)')
ax12.axhline(C_Theta/umolar,0,duration/second, ls='dashed', color='black')
ax12.grid(linestyle='dotted')

ax13.plot(astro_mon.t[:], astro_mon.G_A[0]/umolar, color='C5')
ax13.set_ylabel(r'$G_A$ ($\mu$M)')
ax13.set_xlabel('time (s)')
ax13.grid(linestyle='dotted')

fig2 = plt.figure(num='Modulation of synapses by gliotrasmission',
                  figsize=(10,10))
ax21 = fig2.add_subplot(4,1,1)
ax22 = fig2.add_subplot(4,1,2)
ax23 = fig2.add_subplot(4,1,3)
ax24 = fig2.add_subplot(4,1,4)

ax21.plot(syn_mon.t[:], syn_mon.u_S[0], color='C2', label='no gliotrasmission')
ax21.plot(syn_mon.t[:], syn_mon.u_S[1], color='C3', label='gliotrasmission')
ax21.set_ylabel(r'$u_S$')
ax21.legend()
ax21.grid(linestyle='dotted')

ax22.plot(syn_mon.t[:], syn_mon.x_S[0], color='C2', label='no gliotrasmission')
ax22.plot(syn_mon.t[:], syn_mon.x_S[1], color='C3', label='gliotrasmission')
ax22.set_ylabel(r'$x_S$')
ax22.legend()
ax22.grid(linestyle='dotted')

ax23.plot(syn_mon.t[:], syn_mon.Y_S[0]/umolar, color='C2', label='no gliotrasmission')
ax23.plot(syn_mon.t[:], syn_mon.Y_S[1]/umolar, color='C3', label='gliotrasmission')
ax23.set_ylabel(r'$Y_S$ ($\mu$M)')
ax23.legend()
ax23.grid(linestyle='dotted')

ax24.plot(syn_mon.t[:], syn_mon.Gamma_S[0], color='C2', label='no gliotrasmission')
ax24.plot(syn_mon.t[:], syn_mon.Gamma_S[1], color='C3', label='gliotrasmission')
ax24.set_ylabel(r'$\Gamma_S$')
ax24.set_xlabel('time (s)')
ax24.legend()
ax24.grid(linestyle='dotted')

plt.show()
