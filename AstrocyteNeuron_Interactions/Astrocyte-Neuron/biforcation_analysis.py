"""
Biforcation analysis fro saved data
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, welch
from scipy.fft import fft, ifft
from brian2 import *

def Biforcation_brian(variable, control_par, t_relax):
	"""
	Biforcation analysis of given dinamical variable computed
	throught Brian2 simulator.

	From continous variable is extrapolate the discrete map
	of maximum and minimum. 

	Parameters
	----------
	variable: 2d array 
		

	Returns
	-------
	"""
	I_list = list()
	Bif_list = list()
	print(t_relax)
	for i,X in enumerate(variable):
		X = X[t_relax:]

		max_loc = np.argmax(X)
		min_loc = np.argmin(X)

		X_max = X[max_loc].tolist()
		X_min = X[min_loc].tolist()
		Bif_val = [X_max] + [X_min]
		I_x = [control_par[i] for item in range(len(Bif_val))]

		I_list.append(I_x)
		Bif_list.append(Bif_val)  
        
	return I_list, Bif_list

def Period_brian(variable, t, t_relax):
    """
    Oscillation periods of 2D dynamical system
    concern different values of the parameter

    Parameters
    ----------
    model: callable(y, t, ...) or callable(t, y, ...) 
        Computes the derivative of y at t. If the signature is callable(t, y, ...), then the argument tfirst must be set True.
        Model codimension must be 1 thereby bifurcation analysis concerns only the parameters.
        from scipy.integrate.odeint

    par_stat: integer or float
        initial value of parameter

    par_stop: integer or float
        final value of parameter

    par_tot: integer(optional)
        total number of parameter value. Default par_tot=300

    t0: integer or float(optional)
        initial time. Default t0=0

    t_stop: integer or float(optional)
        final time. Default t_stop=200

    dt: integer or float(optional)
        integration step. Default dt=2E-2

    Returns
    -------
    par_list: list
        paremeters list over compute the oscillation periods

    period_list: list
        list of oscillation periods
    """
    period_list = list()
    period_list_error = list()
    for X in variable:
        X = X[t_relax:]

        max_loc = argrelextrema(X, np.greater)

        X_max = X[max_loc]
        t_max0 = t[max_loc]
        t_max1 = t_max0[1:]
        
        per = t_max1 - t_max0[:-1]
        period = np.mean(per)
        period_error = np.std(per)/np.sqrt(len(per))

        period_list.append(period)
        period_list_error.append(period_error)

    return period_list, period_list_error

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Advanced connectivity connection')
	parser.add_argument('file', type=str, help="file's name of network in 'Neuro_Glia_network' folder")
	args = parser.parse_args()

	## load variable 
	name = args.file

	# rate array
	start = np.load(f'{name}/rate_start.npy')
	stop = np.load(f'{name}/rate_stop.npy')
	N = np.load(f'{name}/N.npy')
	rate_in = np.linspace(start,stop,N)*Hz

	# time array
	d = np.load(f'{name}/duration.npy')
	duration = d*second
	dt_astro = 10*ms
	dt_syn = 1*ms
	t_relax = 400*second

	t_astro = np.linspace(0, duration, int(duration/dt_astro))
	t_syn = np.linspace(0, duration, int(duration/dt_syn))

	#variable array
	C = np.load(f'{name}/C.npy')
	I = np.load(f'{name}/I.npy')
	Gamma_A = np.load(f'{name}/Gamma_A.npy')
	Y_S = np.load(f'{name}/Y_S.npy')
	r_S = np.load(f'{name}/r_S.npy')
	Gamma_S = np.load(f'{name}/Gamma_S.npy')
	#################################################################################################
	
	array_plot = np.arange(0,35,2)
	# print(C.shape)
	## Biforcation
	par_list_c, C_bif = Biforcation_brian(C, rate_in, t_relax=int(t_relax/dt_astro))

	## Period
	period_list, period_list_err = Period_brian(C, t_astro, t_relax=int(t_relax/dt_astro))
	
	## Fourier analysis -and periods estimation
	plt.rc('font', size=13)
	plt.rc('legend', fontsize=10)
	fig4, ax4 = plt.subplots(nrows=1, ncols=1, num='Spectral analysis', tight_layout=True )
	for r in array_plot:
		# yf = fft(C[r])
		# xf = fftfreq(NN, dt_astro)[:NN//2]
		# plt.plot(xf, 2.0/NN * np.abs(yf[0:NN//2]), label='fft')
		NN = len(C[r])
		f, Pxx_den = welch(C[r], 1/dt_astro, nperseg=NN//4)		
		ax4.plot(f, Pxx_den, label=r'$\nu_S$ = '+f'{rate_in[r]:.2f}'+r' ($\rm{spk/s}$)')
	ax4.legend()
	ax4.set_xlabel('frequency (Hz)')
	ax4.set_ylabel('PSD '+r'($\rm{C^2/Hz}$)')
	ax4.set_xlim([-0.2, 1.0])
	ax4.grid(linestyle='dotted')
	ax4.legend(loc='upper right')
	
	# plt.figure()
	# plt.errorbar(rate_in, period_list, period_list_err,fmt='o', markersize=4, lw=0.4)

	## Plots #######################################################################################
	
	print(f'rate_in = {rate_in[0]/Hz} - {rate_in[-1]/Hz}')

	C_theta = 0.0005
	fig1, ax1 = plt.subplots(nrows=1,ncols=1, num='bifurcation', tight_layout=True) 
	ax1.hlines(C_theta/umolar, rate_in[0], rate_in[-1], color='k', ls='dashed')
	for p, c in zip(par_list_c, C_bif):
		ax1.scatter(p, c/umolar, color='C3', s=8.0)
	ax1.set_xlabel(r'$\nu_S$'+r' ($\rm{spk/s}$)')
	ax1.set_ylabel(r'C ($\mu\rm{M}$)')
	ax1.grid(linestyle='dotted')

	


	# fig2, ax2 = plt.subplots(nrows=1,ncols=1, num='astro variable C') 
	# for r in array_plot:
	# 	ax2.plot(t_astro[int(t_relax/dt_astro):], C[r,int(t_relax/dt_astro):]/umolar, label=f'{rate_in[r]/Hz:.2f} (Hz)')
	# ax2.legend()
	# ax2.set_xlabel('time (s)')
	# ax2.set_ylabel(r'C ($\mu M$)')
	# ax2.grid(linestyle='dotted')
	

	# fig3, ax3 = plt.subplots(nrows=1,ncols=1, num='synaptic variable Y_S') 
	# for r in array_plot:
	# 	plt.plot(t_syn, Y_S[r], label=f'{rate_in[r]/Hz:.2f} (Hz)')
	# 	plt.plot()
	# plt.legend()

	# fig3, ax3 = plt.subplots(nrows=1,ncols=1, num='sybaptic variable r_S Gamma_S') 
	# for i,r in enumerate(array_plot):
	# 	plt.plot(t_syn, r_S[r], c=f'C{i}',label=f'{rate_in[r]/Hz:.2f} (Hz)')
	# 	plt.plot(t_syn, Gamma_S[r], c=f'C{i}')
	# 	plt.plot()
	# plt.legend()
	plt.show()
	

