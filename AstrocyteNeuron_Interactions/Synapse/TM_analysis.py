import numpy as np
import matplotlib.pyplot as plt
from brian2 import *

def STP_mean_field(u_0, nu_S_start=-1, nu_S_stop=2, nu_S_number=200):
	"""
	Mean field solution of simple synapses (no gliotramission modulation)
	described by short-term plasticity.
	Return steady state of synaptic variable, u_S and x_S, for constant 
	synaptic input rate, nu_S (Hz)

	Parameters
	----------
	nu_S_start : integer 
				Order of magnitude of first nu_S value
		
	nu_S_stop : integer 
				Order of magnitude of last nu_S value

	nu_S_number : interger (optionl)
				Total sample's number of nu_S. Default=200
	Returns
	-------
	nu_S : 1D-array
			Sample of synaptic rates (Hz)
	u_S : 1D-array
		Steady states of u_S

	x_S : 1D-array
		Steady state of x_S

	"""
	nu_S = np.logspace(nu_S_start, nu_S_stop, nu_S_number)*Hz
	u_S =  (u_0*(Omega_f+nu_S))/(Omega_f+nu_S*u_0)
	x_S = Omega_d / (Omega_d + u_S*nu_S)

	return nu_S, u_S, x_S

## PARAMETERS ###################################################################
# -- Synapse --
rho_c = 0.005                # Synaptic vesicle-to-extracellular space volume ratio
Y_T = 500.*mmolar            # Total vesicular neurotransmitter concentration
Omega_c = 40/second          # Neurotransmitter clearance rate
U_0__star = 0.6              # Resting synaptic release probability
Omega_f = 3.33/second        # Synaptic facilitation rate
Omega_d = 2.0/second         # Synaptic depression rate
w_e = 0.05*nS                # Excitatory synaptic conductance
w_i = 1.0*nS                 # Inhibitory synaptic conductance

r_S_mean, r_S_error, rate_in = [], [], []
for i in [0.2, 0.46, 1.0, 2.15, 4.6, 10.0, 21.0, 46.0, 100.0]:
	name = f'Data/{i}'
	r_S = np.load(f'{name}/r_S_mean.npy')
	r_S_err = np.load(f'{name}/r_S_error.npy')
	rate = np.load(f'{name}/rate_in.npy')

	r_S_mean.append(r_S)
	r_S_error.append(r_S_err)
	rate_in.append(rate[0])

plt.figure(num='TM model solutions')
plt.errorbar(rate_in, r_S_mean, r_S_error, 
                fmt='o', markersize=5, lw=1.5, capsize=2.0,color='black', label='simulation')
nu_S_app, u_S_app, x_S_app = STP_mean_field(u_0=U_0__star)
plt.plot(nu_S_app/Hz, u_S_app*x_S_app, color='k', label='mean field approximation')
plt.xscale('log')
plt.xlabel(r'$\nu_S$'+' (Hz)')
plt.ylabel(r'$\langle r_S \rangle$')
plt.grid(linestyle='dotted')
plt.legend()
plt.show()


