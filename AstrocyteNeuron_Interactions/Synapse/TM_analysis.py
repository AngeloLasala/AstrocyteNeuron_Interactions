import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
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

def errore_in_quadrature(std_array):
	"""
	fist index trial, second index number of indipendent measure
	"""
	std_square = std_array*std_array
	std_sum = std_square.sum(axis=0)
	return np.sqrt(std_sum)/std_array.shape[0]

def chi_square_test(fun, val, std):
	fun = np.asarray(fun)
	val = np.asarray(val)
	std = np.asarray(std)

	chi_e = []
	for f,v,s in zip(fun,val,std):
		# print((f-v)**2 / s**2)
		# print((f-v)**2/s**2)
		chi_e.append((f-v)**2/s**2)
	chi_e = np.asarray(chi_e)
	chi_square = chi_e.sum()
	return chi_square

def CV_test(fun, val, std):
	"""
	"""
	CV_data = std/val
	error_app = (val - fun)/fun
	return (error_app**2/ CV_data**2)

def normality_test(x):
	"""
	Condition for normality of data over trial
	"""
	shapiro_test = stats.shapiro(x)
	if shapiro_test.pvalue <= 0.05 : print(f'p_val={shapiro_test.pvalue:.5f}:posso rifiutare H_0 - NO NORMALI')
	else :  print(f'p_val={shapiro_test.pvalue:.5f}: non posso rifiutare H_0 - SI NORMALI')
	return shapiro_test.pvalue

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Bièartite sinapses filter characteristic')
	parser.add_argument('file', type=str, help="file's name of network in 'Synapses' folder")
	args = parser.parse_args()
	name = args.file

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

	r_S_mean_trl = list()
	r_S_std_trl = list()
	r_S_dist = ()
	for trl in range(1,60):
		# rate array
		rate_in = np.load(f'{name}'+f'/trial-{trl}/rate_in.npy')
		duration = np.load(f'{name}'+f'/trial-{trl}/duration.npy')
		N = np.load(f'{name}'+f'/trial-{trl}/N.npy')
		
		# time array
		duration = 520*second
		dt_syn = 1*ms
		t_relax = 20*second

		#variable array
		r_S = np.load(f'{name}'+f'/trial-{trl}/r_S.npy')
		r_S_mean = r_S.mean(axis=-1)
		r_S_std = r_S.std(axis=-1)
		
		r_S_mean_trl.append(r_S_mean)
		r_S_std_trl.append(r_S_std)
		

	r_S_mean_trl = np.asarray(r_S_mean_trl)
	r_S_std_trl = np.asarray(r_S_std_trl)
	r_S_error = errore_in_quadrature(r_S_std_trl)

	print(r_S_mean_trl.shape)
	print(r_S_std_trl.shape)

	x_mean_recurvively = list()
	x_mean_n = 0
	x_var_n = 0
	for ii in range(r_S_mean_trl.shape[0]):
		x_mean_n1 = x_mean_n + (r_S_mean_trl[ii,15]-x_mean_n)/ (ii+1)
		x_var_n1 = x_var_n + x_mean_n**2 - x_mean_n1**2 + (r_S_mean_trl[ii,15]**2 - x_var_n - x_mean_n**2)/(ii+1)
		x_mean_n = x_mean_n1
		x_var_n = x_var_n1
	# 	print(ii+1, x_mean_n1, np.sqrt(x_var_n1))
	# 	plt.scatter(ii+1, x_mean_n1)
	# print(r_S_mean_trl.shape[0], r_S_mean_trl.mean(axis=0)[15], r_S_mean_trl.std(axis=0)[15])
	# print(r_S_mean_trl.shape[0], r_S_mean_trl.mean(axis=0)[15], r_S_error[15])

	# print()

	# Normality condition
	# for i in range(len(rate_in)):
		# normality_test(r_S_mean_trl[:,i])

	## Chi square test
	nu_S_app, u_S_app, x_S_app = STP_mean_field(u_0=U_0__star, nu_S_start=-1,nu_S_stop=2,nu_S_number=N)
	chi_tot = chi_square_test(u_S_app*x_S_app, r_S_mean_trl.mean(axis=0), r_S_error.flatten())
	print(chi_tot/len(u_S_app))

	## C.V. test
	a = CV_test(u_S_app*x_S_app, r_S_mean_trl.mean(axis=0), r_S_error)
	print(a.mean())
	

	## Plots
	plt.rc('font', size=13)
	plt.rc('legend', fontsize=10)
	fig2, ax2 = plt.subplots(num='STP approx vs data', tight_layout=True)
	ax2.errorbar(rate_in/Hz, r_S_mean_trl.mean(axis=0), r_S_error,
			fmt='o', markersize=4, lw=0.6, capsize=2.0, color='k', label='STP')
	nu_S_app, u_S_app, x_S_app = STP_mean_field(u_0=U_0__star)
	ax2.plot(nu_S_app/Hz, u_S_app*x_S_app, color='k', label='mean field approximation')
	ax2.set_xscale('log')
	ax2.set_xlabel(r'$\nu_S$'+r' ($\rm{spk/s}$)')
	ax2.set_ylabel(r'$\langle \bar{r_S} \rangle$')
	ax2.grid(linestyle='dotted')
	yticks = [ 0.1, 1.0, 10.0, 100.0]
	# ax4[2].set_ylim(yrange)
	ax2.set_xticks(np.logspace(-1, 2, 4))
	ax2.set_xticklabels(yticks)
	ax2.legend()

	plt.show()



