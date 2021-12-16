"""
Astrocyte dynamics using brian2 simulator. the model is the same as the one used
in NG network.

The main goal is to understand the activation of gliorelease based on different values 
of some control parameters
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *

parser = argparse.ArgumentParser(description='dependence of GRE with rispect of time parameters')
parser.add_argument('-o_beta', action='store_false', help="""select parameter over compute the GRE dependece.
															default=True => omega_c,
															-o_beta (False) => o_beta""" )
args = parser.parse_args()

## Parameters ####################################################
# -- Astrocyte --
# CICR
O_P = 0.9*umolar/second      # Maximal Ca^2+ uptake rate by SERCAs
K_P = 0.05*umolar            # Ca2+ affinity of SERCAs
C_T = 2*umolar               # Total cell free Ca^2+ content
rho_A = 0.18                 # ER-to-cytoplasm volume ratio
Omega_L = 0.1/second         # Maximal rate of Ca^2+ leak from the ER
d_1 = 0.13*umolar            # IP_3 binding affinity
d_2 = 1.05*umolar            # Ca^2+ inactivation dissociation constant
O_2 = 0.2/umolar/second      # IP_3R binding rate for Ca^2+ inhibition
d_3 = 0.9434*umolar          # IP_3 dissociation constant
d_5 = 0.08*umolar            # Ca^2+ activation dissociation constant
#  IP_3 production
# Agonist-dependent IP_3 production
O_N = 0.3/umolar/second      # Agonist binding rate
Omega_N = 0.5/second         # Maximal inactivation rate
K_KC = 0.5*umolar            # Ca^2+ affinity of PKC
zeta = 10                    # Maximal reduction of receptor affinity by PKC
# Endogenous IP3 production
O_delta = 1.2*umolar/second  # Maximal rate of IP_3 production by PLCdelta
kappa_delta = 1.5*umolar     # Inhibition constant of PLC_delta by IP_3
K_delta = 0.1*umolar         # Ca^2+ affinity of PLCdelta
# IP_3 degradation
Omega_5P = 0.05/second       # Maximal rate of IP_3 degradation by IP-5P
K_D = 0.7*umolar             # Ca^2+ affinity of IP3-3K
K_3K = 1.0*umolar            # IP_3 affinity of IP_3-3K
O_3K = 4.5*umolar/second     # Maximal rate of IP_3 degradation by IP_3-3K
# IP_3 diffusion (astrocyte coupling)
F = 0.09*umolar/second       # GJC IP_3 permeability
I_Theta = 0.3*umolar         # Threshold gradient for IP_3 diffusion
omega_I = 0.05*umolar        # Scaling factor of diffusion
# Gliotransmitter release and time course
C_Theta = 0.5*umolar         # Ca^2+ threshold for exocytosis
Omega_A = 0.6/second         # Gliotransmitter recycling rate
U_A = 0.6                    # Gliotransmitter release probability
G_T = 200*mmolar             # Total vesicular gliotransmitter concentration
rho_e = 6.5e-4               # Astrocytic vesicle-to-extracellular volume ratio
Omega_e = 60/second          # Gliotransmitter clearance rate
alpha = 0.0                  # Gliotransmission nature
#####################################################################################

## Select
omega_c = args.o_beta
o_beta = not(omega_c)

## Astrocyte ######################################################################## 
astro_eqs = """
# Fraction of activated astrocyte receptors (1):
dGamma_A/dt = O_N * Y_S * (1 - clip(Gamma_A,0,1)) -
			Omega_N*(1 + zeta * C/(C + K_KC)) * clip(Gamma_A,0,1) : 1

# IP_3 dynamics (1)
dI/dt = J_beta + J_delta - J_3K - J_5P + J_coupling              : mmolar

J_beta = O_beta * Gamma_A                                        : mmolar/second
J_delta = O_delta/(1 + I/kappa_delta) * C**2/(C**2 + K_delta**2) : mmolar/second
J_3K = O_3K * C**4/(C**4 + K_D**4) * I/(I + K_3K)                : mmolar/second
J_5P = Omega_5P*I                                                : mmolar/second
# Diffusion between astrocytes (1):
J_coupling                                                       : mmolar/second

# Calcium dynamics (2):
dC/dt = J_r + J_l - J_p                                   : mmolar
dh/dt = (h_inf - h)/tau_h                                 : 1

J_r = (Omega_C * m_inf**3 * h**3) * (C_T - (1 + rho_A)*C) : mmolar/second
J_l = Omega_L * (C_T - (1 + rho_A)*C)                     : mmolar/second
J_p = O_P * C**2/(C**2 + K_P**2)                          : mmolar/second
m_inf = I/(I + d_1) * C/(C + d_5)                         : 1
h_inf = Q_2/(Q_2 + C)                                     : 1
tau_h = 1/(O_2 * (Q_2 + C))                               : second
Q_2 = d_2 * (I + d_1)/(I + d_3)                           : mmolar

# Fraction of gliotransmitter resources available for release (1):
dx_A/dt = Omega_A * (1 - x_A) : 1
# gliotransmitter concentration in the extracellular space (1):
dG_A/dt = -Omega_e*G_A        : mmolar

# Neurotransmitter concentration in the extracellular space:
Y_S                           : mmolar
O_beta                        : mmolar/second
Omega_C                       : 1/second
"""

astro_release = """
G_A += rho_e*G_T*U_A*x_A
x_A -= U_A * x_A
"""
N_a = 300
omega_c_range = np.linspace(4,20,100)/second
o_beta_range = np.linspace(0.1,8,100)*umolar/second
gre_timing = []

if omega_c : parameters = omega_c_range
if o_beta : parameters = o_beta_range

for par in parameters:

	astrocyte = NeuronGroup(N_a, astro_eqs, 
							threshold='C>C_Theta', refractory='C>C_Theta', reset=astro_release,
							method='rk4', dt=1e-2*second)

	if omega_c:
		astrocyte.O_beta = 0.5*umolar/second    # Maximal rate of IP_3 production by PLCbeta
		astrocyte.Omega_C = par                 # Maximal rate of Ca^2+ release by IP_3Rs

	if o_beta:
		astrocyte.O_beta = par                  # Maximal rate of IP_3 production by PLCbeta
		astrocyte.Omega_C = 6/second            # Maximal rate of Ca^2+ release by IP_3Rs

	astrocyte.Y_S = 50*uM
	astrocyte.C ="0.005*umolar + rand()*(0.015-0.005)*umolar"
	astrocyte.h = "0.85 + rand()*(0.95-0.85)"
	astrocyte.I = "0.005*umolar + rand()*(0.015-0.005)*umolar"
	astrocyte.x_A = 1.0

	## Monitor
	gliorelease_mon = SpikeMonitor(astrocyte)
	astro_mon = StateMonitor(astrocyte, ['C','I','Gamma_A','Y_S','G_A'], record=True)

	run(6*second, report='text')
	print(f'O_beta = {astrocyte.O_beta[0]/(umolar/second)} uM/s')
	print(f'Omega_C = {astrocyte.Omega_C[0]*second} Hz')
	mean_gre = gliorelease_mon.t[:].mean()
	std_gre = gliorelease_mon.t[:].std()
	print(f'{mean_gre:.3f} +/- {std_gre/((N_a-1)**0.5):.3f}')
	print('============================')

	gre_timing.append([mean_gre,std_gre/((N_a-1)**0.5)])

gre_timing = np.array(gre_timing)

## Single astrocite
astro_eqs_s = """
# Fraction of activated astrocyte receptors (1):
dGamma_A/dt = O_N * Y_S * (1 - clip(Gamma_A,0,1)) -
			Omega_N*(1 + zeta * C/(C + K_KC)) * clip(Gamma_A,0,1) : 1

# IP_3 dynamics (1)
dI/dt = J_beta + J_delta - J_3K - J_5P + J_coupling              : mmolar

J_beta = O_beta * Gamma_A                                        : mmolar/second
J_delta = O_delta/(1 + I/kappa_delta) * C**2/(C**2 + K_delta**2) : mmolar/second
J_3K = O_3K * C**4/(C**4 + K_D**4) * I/(I + K_3K)                : mmolar/second
J_5P = Omega_5P*I                                                : mmolar/second
# Diffusion between astrocytes (1):
J_coupling                                                       : mmolar/second

# Calcium dynamics (2):
dC/dt = J_r + J_l - J_p                                   : mmolar
dh/dt = (h_inf - h)/tau_h                                 : 1

J_r = (Omega_C * m_inf**3 * h**3) * (C_T - (1 + rho_A)*C) : mmolar/second
J_l = Omega_L * (C_T - (1 + rho_A)*C)                     : mmolar/second
J_p = O_P * C**2/(C**2 + K_P**2)                          : mmolar/second
m_inf = I/(I + d_1) * C/(C + d_5)                         : 1
h_inf = Q_2/(Q_2 + C)                                     : 1
tau_h = 1/(O_2 * (Q_2 + C))                               : second
Q_2 = d_2 * (I + d_1)/(I + d_3)                           : mmolar

# Fraction of gliotransmitter resources available for release (1):
dx_A/dt = Omega_A * (1 - x_A) : 1
# gliotransmitter concentration in the extracellular space (1):
dG_A/dt = -Omega_e*G_A        : mmolar

# Neurotransmitter concentration in the extracellular space:
Y_S = Y_S_function(t)  : mmolar
O_beta                 : mmolar/second
Omega_C                : 1/second
"""
y_s_values = [10]*uM
Y_S_function = TimedArray(y_s_values, dt=10*second)

astrocyte_s = NeuronGroup(1, astro_eqs_s, 
							threshold='C>C_Theta', refractory='C>C_Theta', reset=astro_release,
							method='rk4', dt=1e-2*second)
astrocyte_s.O_beta = 0.005*umolar/second
astrocyte_s.Omega_C = 6/second
astrocyte_s.C ="0.005*umolar + rand()*(0.015-0.005)*umolar"
astrocyte_s.h = "0.85 + rand()*(0.95-0.85)"
astrocyte_s.I = "0.005*umolar + rand()*(0.015-0.005)*umolar"
astrocyte_s.x_A = 1.0

astro_mon_s = StateMonitor(astrocyte_s, ['C','I','Gamma_A','Y_S','G_A'], record=True)

net_s = Network(astrocyte_s, astro_mon_s)
net_s.run(60*second, report='text')



## Plots #####################################################################################
if omega_c:
	fig1, ax1 = plt.subplots(nrows=1, ncols=1, 
							num=f'GRE_Omega_C, Y_S={y_s_values/umolar} um')
						
	ax1.errorbar(omega_c_range, gre_timing[:,0], gre_timing[:,1],
				fmt='o', markersize=2.5, lw=0.6, color='C2')
	ax1.set_xlabel(r'$\Omega_C$ (Hz)')
	ax1.set_ylabel(f'GRE (s)')
	ax1.grid(linestyle='dotted')

	plt.savefig('G_ChI_images'+'/GRE_Omega_C.png')


if o_beta:
	fig1, ax1 = plt.subplots(nrows=1, ncols=1, 
							num=f'GRE_O_beta')
						
	ax1.errorbar(o_beta_range/(umolar/second), gre_timing[:,0], gre_timing[:,1],
				fmt='o', markersize=2.5, lw=0.6, color='C2')
	ax1.set_xlabel(r'$O_\beta$ ($\mu$m/s)')
	ax1.set_ylabel(f'GRE (s)')
	ax1.grid(linestyle='dotted')

	plt.savefig('G_ChI_images'+'/GRE_O_beta.png')

fig2, ax2 = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(13, 9), 
                        num=f'astrocyte dynamics, Y_S={y_s_values/umolar} um')

ax2[0].plot(astro_mon_s.t[:], astro_mon_s.Y_S[0,:]/umolar, color='C3')
ax2[0].set_ylabel(r'$Y_S$ ($\mu$M)')
ax2[0].grid(linestyle='dotted')

ax2[1].plot(astro_mon_s.t[:], astro_mon_s.Gamma_A[0,:], color='C7')
ax2[1].set_ylabel(r'$\Gamma_A$ ')
ax2[1].grid(linestyle='dotted')

ax2[2].plot(astro_mon_s.t[:], astro_mon_s.I[0,:]/umolar, color='C0')
ax2[2].set_ylabel(r'$I$ ($\mu$M)')
ax2[2].grid(linestyle='dotted')

ax2[3].plot(astro_mon_s.t[:], astro_mon_s.C[0,:]/umolar, color='red')
ax2[3].set_ylabel(r'$Ca^{2\plus}$ ($\mu$M)')
ax2[3].set_xlabel('time (s)')
ax2[3].plot(astro_mon_s.t[:], np.full(astro_mon_s.t[:].shape[0], C_Theta/umolar), ls='dashed', color='black')
ax2[3].grid(linestyle='dotted')

plt.show()