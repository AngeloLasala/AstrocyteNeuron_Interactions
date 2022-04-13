"""
Analysis of gliotrasmission modulation of single synapses 
in astrocytic steady state
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from brian2 import*

def standard_error(X):
	"""
	Time series standard error for independent measurements

	Parameters
	----------
	X : numpy 2darray
		array of data, first index is the number of indipent trial 
		second one is the number of value fro each mesurements.
		Size = (N_syn, duration/dt)
	"""

	## sample average
	X_mean = np.mean(X, axis=0)
	X_std = np.std(X, axis=0)
	X_error = X_std/np.sqrt(X.shape[0])

	return X_mean, X_error

def standard_error_mean(X_mean, X_error):
	"""
	Estimation of absolute error of average value

	Parameters
	----------

	X_mean : array
			sample values

	X_error : array
			sample absolute error

	Returns
	------- 
	"""
	X_mean_time = np.mean(X_mean)
	X_mean_error = X_error.sum()/len(X_error)

	return X_mean_time, X_mean_error

def built_time(window, dt=0.1, duration=40):
	"""
	Return time array of proprer time window simulation

	Parameters
	----------

	dt : float optional
		integration step, in ms. Default = 0.1*ms

	duration : float optional
		total time simulation in selected window, in second. Default = 40*second


	Return
	------
	t : array
		time array, in second
	"""
	duration = duration*second
	dt = dt * ms
	N = int(duration/dt)
	t = np.linspace((window-1)*duration, window*duration, N)

	return t

def variance(x):
	"""
	Variance of finite size values. The correction is maded by 
	the factor 1/(N-1)

	Parameters
	---------
	x : array
		array of data

	Returns
	-------
	var : float
		variance
	"""
	x = np.array(x)
	N = len(x)

	mean = np.mean(x)
	var = ((x-mean)**2).sum()/(N*(N-1))
	return var

def blocking(x ,k=10):
	"""
	Data blocking techniques to estimate the variance of 
	correlated variable

	Parameters
	----------
	x : array
		data 

	k : integer
		number of block
	
	Returns
	-------
	variances_list : list
		list of variances for each block
	"""
	x = np.array(x)

	variances_list = []
	for time in range(k):
		N = int(len(x))
		if N%2 != 0:
			N = N-1

		## index odd and even
		index = np.arange(N)
		odd = index[index%2==0]
		even = index[index%2!=0]

		# variance 
		x1 = x[odd]
		x2 = x[even]
		x_block = (x1+x2)/2.0
		var_block = variance(x_block)
		variances_list.append(var_block)
		x = x_block

	return variances_list

def crosscorr(x, y, max_lag, bootstrap_test=False):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    cross_corr = []
    for d in range(max_lag):
        cc = 0
        for i in range(len(x)-d):
            cc += (x[i] - x_mean) * (y[i+d] - y_mean)
        cc = cc / np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
        cross_corr.append(cc)
    plt.plot(cross_corr,'k')
    plt.title('Cross-correlation function')
    plt.xlabel('Lags')


    if bootstrap_test:
        cross_corr_s = []
        for i in range(100):
            xs, ys = x, y
            np.random.shuffle(xs)
            np.random.shuffle(ys)
            xs_mean = np.mean(xs)
            ys_mean = np.mean(ys)
            cross_corr_s_i = []
            for d in range(max_lag):
                cc = 0
                for i in range(len(x)-d):
                    cc += (xs[i] - xs_mean) * (ys[i+d] - ys_mean)
                cc = cc / np.sqrt(np.sum((xs - xs_mean)**2) * np.sum((ys- ys_mean)**2))
                cross_corr_s_i.append(cc)
            cross_corr_s.append(cross_corr_s_i)
        meancc = np.mean(np.array(cross_corr_s), axis=0)
        stdcc = np.std(np.array(cross_corr_s), axis=0)
        plt.plot(meancc - 3*stdcc, 'crimson', lw=0.5)
        plt.plot(meancc, 'crimson', lw=0.5)
        plt.plot(meancc + 3*stdcc, 'crimson', lw=0.5)
        plt.fill_between(np.arange(0, max_lag), meancc + 3*stdcc, meancc - 3*stdcc, color='crimson', alpha=0.6)
    plt.grid()
    return cross_corr


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Tripartite synapses')
	parser.add_argument('v_in', type=float, help="input frequency value")
	parser.add_argument('w', type=int, help="index of time window")
	parser.add_argument('-p', action='store_true', help="show paramount plots, default=False")
	args = parser.parse_args()
		
	## load variable
	rate_in = args.v_in
	w = args.w

	saturation_time = []
	saturation_mean_stp, saturation_error_stp = [], []
	saturation_mean_glia, saturation_error_glia = [], []
	for w in [2,3,4,5,6]:
		name = f"Tripartite_synapses/Astrocycitc_steady_state/rate_in{rate_in}/time_windows{w}"
		Y_S_noglio = np.load(f'{name}/Y_S_noglio.npy')*mmolar
		Y_S_astro = np.load(f'{name}/Y_S_astro.npy')*mmolar
		t = built_time(w)
		
		## Analysis #######################################################################################
		Y_S_mean, Y_S_error = standard_error(Y_S_noglio)
		Y_S_astro_mean, Y_S_astro_err = standard_error(Y_S_astro)

		Y_S_mean_time = np.mean(Y_S_mean) 
		Y_S_mean_time_error = np.sqrt(blocking(Y_S_mean/mmolar, k=15)[-1])*mmolar

		Y_S_astro_mean_time = np.mean(Y_S_astro_mean)
		Y_S_astro_time_error = np.sqrt(blocking(Y_S_astro_mean/mmolar, k=15)[-1])*mmolar

		# plt.figure()
		# plt.scatter([i+1 for i in range(40)], Y_S_astro_time_error, label='glia')
		# plt.scatter([i+1 for i in range(40)], blocking(Y_S_mean, k=40), label='STP')
		# plt.yscale('log')
		# plt.legend()
		# plt.show()
		print(f'MEAN VALUES ACCROS ENTIRE WINDOW {w}')
		print('STP ')
		print(f'{Y_S_mean_time/umolar:.4f} +- {Y_S_mean_time_error/umolar:.4f} uM')
		print('GLIOMODULATION')
		print(f'{Y_S_astro_mean_time/umolar:.4f} +- {Y_S_astro_time_error/umolar:.4f} uM')
		print('____________________________________________________')

		## Saturation
		saturation_window = 10*second
		time_wind= int(saturation_window/(0.1*ms))
		time_start = 0
		time_stop = time_wind

		for tt in range(4):			
			t_sat = t[time_start:time_stop]
			Y_S_astro_sat = Y_S_astro_mean[time_start:time_stop]
			Y_S_stp_sat = Y_S_mean[time_start:time_stop]

			Y_S_astro_sat_err = np.sqrt(blocking(Y_S_astro_sat/mmolar, k=14)[-1])*mmolar
			Y_S_stp_sat_err = np.sqrt(blocking(Y_S_stp_sat/mmolar, k=14)[-1])*mmolar

			print()
			print(f'MEAN TIME IN {t_sat.mean()/second:.1f} +- 5 s')
			print(f'Y_S = {Y_S_astro_sat.mean()/umolar:.4f}+-{Y_S_astro_sat_err/umolar:.4f} uM')
			# plt.figure()
			# plt.scatter([i+1 for i in range(15)], blocking(Y_S_astro_sat, k=15), label='glia')
			# plt.scatter([i+1 for i in range(15)], blocking(Y_S_stp_sat, k=15), label='STP')
			# plt.yscale('log')
			# plt.legend()
			# plt.show()

			saturation_time.append(t_sat.mean())
			saturation_mean_glia.append(Y_S_astro_sat.mean())
			saturation_error_glia.append(Y_S_astro_sat_err)
			saturation_mean_stp.append(Y_S_stp_sat.mean())
			saturation_error_stp.append(Y_S_stp_sat_err)

			time_start = time_stop
			time_stop = time_start + time_wind
		####################################################################################################

		## Plots ###########################################################################################
		fig1, ax1 = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12,6), 
								num=f"Y_S noglio mean - window {w}")

		ax1[0].plot(t, Y_S_mean/umolar, lw=0.7, color='black', label='STP')
		ax1[0].fill_between(t, (Y_S_mean+Y_S_error)/umolar, (Y_S_mean-Y_S_error)/umolar, color='black', alpha=0.5)
		ax1[0].grid(linestyle='dotted')
		ax1[0].set_ylabel(r'$\langle Y_S \rangle$ ($\mu$M)')
		ax1[0].legend()

		Y_S_noglio_plot = np.full(len(t), Y_S_mean_time)*mmolar
		ax1[1].plot(t, Y_S_noglio_plot/umolar, ls='dashed', color='black', label='mean STP')
		ax1[1].fill_between(t, (Y_S_noglio_plot+Y_S_mean_time_error)/umolar, 
							(Y_S_noglio_plot-Y_S_mean_time_error)/umolar, color='black', alpha=0.3)
		ax1[1].plot(t, Y_S_astro_mean/umolar, color='C6', label='GLIO')
		ax1[1].fill_between(t, (Y_S_astro_mean+Y_S_astro_err)/umolar, (Y_S_astro_mean-Y_S_astro_err)/umolar, color='C6', alpha=0.5)
		ax1[1].grid(linestyle='dotted')
		ax1[1].legend()
		ax1[1].set_ylabel(r'$\langle Y_S \rangle$ ($\mu$M)')
		ax1[1].set_xlabel('time (s)')

	fig2, ax2 = plt.subplots(nrows=1, ncols=1, num=f'Saturation - rate {rate_in}')

	ax2.errorbar(saturation_time, saturation_mean_stp/umolar, saturation_error_stp/umolar, 
				fmt='o', color='k', markersize=3, label='GLIO')
	ax2.errorbar(saturation_time, saturation_mean_glia/umolar, saturation_error_glia/umolar, 
				fmt='o', color='C6', markersize=3, label='GLIO')
	ax2.grid(linestyle='dotted')
	ax2.set_ylabel(r'$\langle Y_S \rangle$ ($\mu$M)')
	ax2.set_xlabel('time (s)')

	plt.show()
