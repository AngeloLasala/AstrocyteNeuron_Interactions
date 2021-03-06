"""
Analysis about membrane conductance and spiking action
"""
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
from scipy.signal import argrelextrema

def jumps(conductance):
	"""
	Conductance jumps due to presynaptic action. The type of synaptic 
	(exc or inh) depends on input: g_i stands for presinaptic inhibitory 
	and g_e stands for excitatory.

	Parameters
	----------
	conductance: 1d array of conductance variable, for istance g_e_STP[index,trans:]

	Returns
	-------
	jump: array. Conductance jumps
	"""
	max_loc=argrelextrema(conductance, np.greater)
	min_loc=argrelextrema(conductance, np.less)

	jump = conductance[max_loc] - conductance[min_loc]

	return jump

## Load variable #######################################################
#Synapse parameters
w_e = 0.05*nS          # Excitatory synaptic conductance
w_i = 1.0*nS           # Inhibitory synaptic conductance
U_0 = 0.6              # Synaptic release probability at rest
Omega_d = 2.0/second   # Synaptic depression rate
Omega_f = 3.33/second  # Synaptic facilitation rate

index=0
# EI_net_STP/Network_pe_g0.25_s1.0_we0.05/v_in50.0/f-I_curve
name_STP = 'EI_net_STP/Network_pe_g0.25_s1.0_we0.05/v_in50.0/f-I_curve'
name_noSTP = 'EI_net_noSTP/Network_pe_g0.25_s1.0_we0.05/v_in50.0/f-I_curve_fixed'

# t = np.load(f'{name_STP}/state_exc_mon.t.npy')
trans = 5000

g_e_STP = np.load(f'{name_STP}/g_e_inh.npy')*siemens
g_i_STP = np.load(f'{name_STP}/g_i_inh.npy')*siemens
g_e_noSTP = np.load(f'{name_noSTP}/g_e_inh.npy')*siemens
g_i_noSTP = np.load(f'{name_noSTP}/g_i_inh.npy')*siemens
#######################################################################################################

# Mean values for index neurons (exc)
g_i_STP_mean = g_i_STP[index,trans:].mean()
g_i_STP_std = g_i_STP[index,trans:].std()
g_i_noSTP_mean = g_i_noSTP[index,trans:].mean()
g_i_noSTP_std = g_i_noSTP[index,trans:].std()
g_e_noSTP_mean = g_e_noSTP[index,trans:].mean()
g_e_STP_mean = g_e_STP[index,trans:].mean()


print(f'g_i_STP mean: {g_i_STP_mean/nS}')
print(f'g_i_STP std: {g_i_STP_std/nS}')
print(f'g_i_noSTP mean: {g_i_noSTP_mean/nS}')
print(f'g_i_noSTP std: {g_i_noSTP_std/nS}')

print(f'g_e_noSTP mean: {g_e_noSTP_mean/nS}')
print(f'g_e_STP mean: {g_e_STP_mean/nS}')

# Estimate mean of r_S
inh_noSTP = jumps(g_i_noSTP[index,trans:])
inh_STP = jumps(g_i_STP[index,trans:])
exc_noSTP = jumps(g_e_noSTP[index,trans:])
exc_STP = jumps(g_e_STP[index,trans:])

r_S_inhSTP = inh_STP.mean()/w_i
r_S_inhnoSTP = inh_noSTP.mean()/w_i	
r_S_excSTP = exc_STP.mean()/w_e
# r_S_excnoSTP = exc_noSTP.mean()/w_e
print(len(inh_STP))

print('r_S mean value')
print(f'STP : r_S_inh = {inh_STP.mean()/w_i:.4f} +- {inh_STP.std()/w_i/np.sqrt(len(inh_STP)):.4f}')
print(f'STP : r_S_exc = {exc_STP.mean()/w_e:.4f} +- {exc_STP.std()/w_e/np.sqrt(len(exc_STP)):.4f}')
print(f'noSTP : r_S_inh = {inh_noSTP.mean()/w_i:.4f} +- {inh_noSTP.std()/w_i/np.sqrt(len(inh_noSTP)):.4f}')
print(f'noSTP : r_S_exc = {exc_noSTP.mean()/w_e:.4f} +- {exc_noSTP.std()/w_e/np.sqrt(len(exc_noSTP)):.4f}')



print('Computation on exc neuron')
print(f'STP   -> r_s inh {r_S_inhSTP:.4f}, r_S exc {r_S_excSTP:.4f}')
print(f'noSTP -> r_s inh {r_S_inhnoSTP:.4f}')#, r_S exc {r_S_excnoSTP:.4f}')
	


########################################################################################################
# plt.figure()
# # plt.plot(t[trans:]/ms,g_e_STP[index,trans:]/nS)
# plt.plot(t[trans:]/ms,g_i_STP[index,trans:]/nS)
# # plt.plot(t[trans:]/ms,g_e_noSTP[index,trans:]/nS)
# plt.plot(t[trans:]/ms,g_i_noSTP[index,trans:]/nS)
# plt.show()