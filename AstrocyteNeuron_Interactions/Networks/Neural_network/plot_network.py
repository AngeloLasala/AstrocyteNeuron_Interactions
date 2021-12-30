import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from brian2 import *
from AstrocyteNeuron_Interactions import makedir


def smoothing_b(x, window='gaussian', width=None, ddt=defaultclock.dt):
	"""
	Return a smooth version of signal x(t).

	Reference: https://brian2.readthedocs.io/en/2.5.0.1/_modules/brian2/monitors/ratemonitor.html

	Parameters
	----------
	x : ndarray
		input signal 

	window : str
		The window to use for smoothing. It is a string to chose a
		predefined window(``'flat'`` for a rectangular, and ``'gaussian'``
		for a Gaussian-shaped window). In this case the width of the window
		is determined by the ``width`` argument. Note that for the Gaussian
		window, the ``width`` parameter specifies the standard deviation of
		the Gaussian, the width of the actual window is ``4*width + ddt``
		(rounded to the nearest dt). For the flat window, the width is
		rounded to the nearest odd multiple of dt to avoid shifting the rate
		in time.

	width : `Quantity`, optional
		The width of the ``window`` in seconds (for a predefined window).

	ddt : `Quantity`, optional
		sampling sepatation 

	Returns
	-------
	x(t) : `Quantity`
		x(t), smoothed with the given window. Note that
		the values are smoothed and not re-binned, i.e. the length of the
		returned array is the same as the length of input array
		and can be plotted against the same time 't'.
	"""
	if window == 'gaussian':
		width_dt = int(np.round(2*width / ddt))
		# Rounding only for the size of the window, not for the standard
		# deviation of the Gaussian
		window = np.exp(-np.arange(-width_dt,
								width_dt + 1)**2 *
						1. / (2 * (width/ddt) ** 2))
	elif window == 'flat':
		width_dt = int(width / 2 / ddt)*2 + 1
		used_width = width_dt * ddt
		if abs(used_width - width) > 1e-6*ddt:
			logger.info(f'width adjusted from {width} to {used_width}',
						'adjusted_width', once=True)
		window = np.ones(width_dt)
	else:
		raise NotImplementedError(f'Unknown pre-defined window "{window}"')

	return np.convolve(x, window * 1. / sum(window), mode='same')

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='EI Network - variable and spectral analysis plot')
	parser.add_argument('folder', type=str, help="""folder's name of network in 'Neural_network:
													- EI_net_noSTP
													- EI_net_STP""")
	parser.add_argument('-p', action='store_true', help="show plots, default=False")
	args = parser.parse_args()

	## Load variables ######################################################################################
	name_folder = args.folder+'Network_pe_v_in'

	N_e = 3200
	N_i = 800
	V_th = -50*mV          # Firing threshold
	defaultclock.dt = 0.05*ms

	v_in_range = [44.0, 47.7, 54.0, 58.8, 64.0, 74.0, 84.0]
	# v_in_range = [42.0, 43.5, 44.0, 45.0, 47.7, 58.8]
	for v_in in v_in_range:
		name = name_folder+str(v_in)
		print(name)

		duration = np.load(f'{name}/duration.npy')*second
		trans_time = np.load(f'{name}/trans_time.npy')
		trans = np.load(f'{name}/trans.npy')
		rate_in = np.load(f'{name}/rate_in.npy')

		# Excitatory neurons variable
		t_exc = np.load(f'{name}/state_exc_mon.t.npy')
		LFP = np.load(f'{name}/state_exc_mon.LFP.npy')

		# Population istantaneus firing rate
		firing_rate_exc_t= np.load(f'{name}/firing_rate_exc.t.npy')
		firing_rate_inh_t = np.load(f'{name}/firing_rate_inh.t.npy')
		fr_exc = np.load(f'{name}/fr_exc.npy')
		fr_inh = np.load(f'{name}/fr_inh.npy')
		fr = np.load(f'{name}/firing_rate.npy')

		########################################################################################################

		## LFP - spectral analysis
		LFP = LFP[:].sum(axis=0)

		LFP_trans = LFP[trans:]
		fr_exc_trans = fr_exc[trans:]
		fr_inh_trans = fr_inh[trans:]
		fr_trans = fr[trans:]

		LFP_fft = fft(LFP_trans)
		N = len(LFP_trans)
	
		print(f'sampling freq Whelch : {1/((N//4+1)*defaultclock.dt)} Hz')

		f_LFP, spect_den_LFP = signal.welch(LFP_trans,fs=1/defaultclock.dt, nperseg=N//4)
		f_fr_exc, spect_den_fr_exc = signal.welch(fr_exc_trans,fs=1/defaultclock.dt, nperseg=N//4)
		f_fr_inh, spect_den_fr_inh = signal.welch(fr_inh_trans,fs=1/defaultclock.dt, nperseg=N//4)
		f_fr, spect_den_fr = signal.welch(fr_trans,fs=1/defaultclock.dt, nperseg=N//4)

		## SAVE VARIABLE #######################################################
		plasticity = args.folder.split('_')[-1]
		name_save = 'Spectral_analysis/'+plasticity+'net_v_in'+str(v_in)
		makedir.smart_makedir(name_save)

		# LFP
		np.save(f'{name_save}/LFP',LFP)
		np.save(f'{name_save}/f_LFP',f_LFP)
		np.save(f'{name_save}/spect_den_LFP',spect_den_LFP)
		np.save(f'{name_save}/f_fr',f_fr)
		np.save(f'{name_save}/spect_den_fr',spect_den_fr)
		

		## Comparison between 47.7 and 58.8
		fig, ax = plt.subplots(nrows=1, ncols=2 , figsize=(10,5), num=f'Compare spectrum ({plasticity})')
		for v_in in [43.0, 43.5, 44.0]:
			name_1 = args.folder+'Network_pe_v_in'+str(v_in)

			LFP_1 = np.load(f'{name_1}/state_exc_mon.LFP.npy')
			LFP_1 = LFP_1.sum(axis=0)
			fr_1 = np.load(f'{name_1}/firing_rate.npy')

			f_LFP_1, spect_den_LFP_1 = signal.welch(LFP_1[trans:],fs=1/defaultclock.dt, nperseg=N//4)
			f_fr_1, spect_den_fr_1 = signal.welch(fr_1[trans:],fs=1/defaultclock.dt, nperseg=N//4)

			ax[0].plot(f_LFP_1[:150], spect_den_LFP_1[:150], label=r'$\nu_{in}$='+f'{v_in}')
			ax[0].set_title('LFP')
			ax[0].set_xlabel('frequancy (Hz)')
			ax[0].legend()
			ax[0].grid(linestyle='dotted')

			ax[1].plot(f_fr_1[:150], spect_den_fr_1[:150], label=r'v_{in}='+f'{v_in}')
			ax[1].set_title('Population Firing Rate')
			ax[1].set_xlabel('frequancy (Hz)')
			ax[1].grid(linestyle='dotted')
			

		## Average spiking activity
		activity_exc_noSTP = []
		activity_inh_noSTP = []
		activity_exc_STP = []
		activity_inh_STP = []
		activity_LFP_noSTP = []
		activity_LFP_STP = []
		for v_in in v_in_range:
			name_f_noSTP = 'EI_net_noSTP/'+'Network_pe_v_in'+str(v_in)
			name_f_STP = 'EI_net_STP/'+'Network_pe_v_in'+str(v_in)

			fr_exc_noSTP = np.load(f'{name_f_noSTP}/fr_exc.npy')
			fr_inh_noSTP = np.load(f'{name_f_noSTP}/fr_inh.npy')
			fr_exc_STP = np.load(f'{name_f_STP}/fr_exc.npy')
			fr_inh_STP = np.load(f'{name_f_STP}/fr_inh.npy')
			LFP_noSTP = np.load(f'{name_f_noSTP}/state_exc_mon.LFP.npy')
			LFP_STP = np.load(f'{name_f_STP}/state_exc_mon.LFP.npy')

			LFP_noSTP = LFP_noSTP.sum(axis=0)
			LFP_STP = LFP_STP.sum(axis=0)

			activity_exc_noSTP.append(fr_exc_noSTP.mean())
			activity_inh_noSTP.append(fr_inh_noSTP.mean())
			activity_exc_STP.append(fr_exc_STP.mean())
			activity_inh_STP.append(fr_inh_STP.mean())
			activity_LFP_noSTP.append(LFP_noSTP.mean())
			activity_LFP_STP.append(LFP_STP.mean())
		

		# Plots  ################################################################################################
		if args.p:
			fig1, ax1 = plt.subplots(nrows=2, ncols=1, sharex=True,
									num=f'Population activity, v_in={rate_in/Hz}', figsize=(8,7))

			ax1[0].plot(firing_rate_exc_t[trans:]/second, fr[trans:], color='indigo')
			ax1[0].set_ylabel('Population rate (Hz)')
			ax1[0].grid(linestyle='dotted')

			ax1[1].plot(t_exc[trans:]/second, LFP[trans:], color='C5')
			ax1[1].set_ylabel('LFP (mV)')
			ax1[1].set_xlabel('time (s)')
			ax1[1].grid(linestyle='dotted')

			fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=(12,6),
									num=f'Average Population Activity')

			ax2[0].plot(v_in_range, activity_exc_noSTP, ls='', marker="$\u25EF$", color='C3', label="exc no STP")
			ax2[0].plot(v_in_range, activity_inh_noSTP, ls='', marker="$\u25A1$", color='C0', label="inh no STP")
			ax2[0].plot(v_in_range, activity_exc_STP, ls='', marker="o", color='C3', label="exc STP")
			ax2[0].plot(v_in_range, activity_inh_STP, ls='', marker="s", color='C0', label="inh STP")
			ax2[0].legend()
			ax2[0].set_xlabel(r'$v_{input}$ (Hz)')
			ax2[0].set_ylabel(r'Mean firing rate (Hz)')
			ax2[0].grid(linestyle='dotted')

			ax2[1].plot(v_in_range, activity_LFP_noSTP, ls='', marker="$\u25C7$", color='C5', label="LFP no STP")
			ax2[1].plot(v_in_range, activity_LFP_STP, ls='', marker="$\u25C6$", color='C5', label="LFP STP")
			ax2[1].legend()
			ax2[1].set_xlabel(r'$v_{input}$ (Hz)')
			ax2[1].set_ylabel(r'Mean LFP (mV)')
			ax2[1].grid(linestyle='dotted')

			fig3, ax3 = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(10,7),
									num=f'Spectral_Density, v_in={rate_in/Hz}')

			ax3[0,0].semilogy(f_LFP, spect_den_LFP, color='C5')
			ax3[0,0].set_ylabel('LFP')
			ax3[0,0].set_xlim([-10, 200])
			ax3[0,0].grid(linestyle='dotted')

			ax3[0,1].plot(f_LFP, spect_den_LFP, color='C5')
			ax3[0,1].set_xlim([-10, 200])
			ax3[0,1].grid(linestyle='dotted')

			ax3[1,0].semilogy(f_fr, spect_den_fr, color='indigo')
			ax3[1,0].set_ylabel('Population firing rate')
			ax3[1,0].set_xlim([-10, 200])
			ax3[1,0].set_xlabel('frequency (Hz)')
			ax3[1,0].grid(linestyle='dotted')

			ax3[1,1].plot(f_fr, spect_den_fr, color='indigo')
			ax3[1,1].set_xlim([-10, 200])
			ax3[1,1].set_xlabel('frequency (Hz)')
			ax3[1,1].grid(linestyle='dotted')

			plt.show()
