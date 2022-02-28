"""
load network variables and makes some analysis and intersting plot:
- Raster plot
- Population firing rate
- LFP
- Variable dynamics

Note: To obtain usefull information about network dynamics it is very usefull
run 'connectivity_analysis.py' to know advanced information of connectivity
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from brian2 import *
from network_analysis import transient

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Advanced connectivity connection')
    parser.add_argument('file', type=str, help="file's name of network in 'Neuro_Glia_network' folder")
    args = parser.parse_args()

    ## Load variables ######################################################################################
    name=args.file

    duration = np.load(f'{name}/duration.npy')*second
    rate_in = np.load(f'{name}/rate_in.npy')

    t_exc = np.load(f'{name}/spikes_exc_mon.t.npy')
    exc_neurons_i = np.load(f'{name}/spikes_exc_mon.i.npy')
    t_inh = np.load(f'{name}/spikes_inh_mon.t.npy')
    inh_neurons_i = np.load(f'{name}/spikes_inh_mon.i.npy')
    t_astro = np.load(f'{name}/astro_mon.t.npy')
    astro_i = np.load(f'{name}/astro_mon.i.npy')

    t = np.load(f'{name}/var_astro_mon.t.npy')
    Y_S = np.load(f'{name}/var_astro_mon.Y_S.npy')
    Gamma_A =np.load(f'{name}/var_astro_mon.Gamma_A.npy')
    I = np.load(f'{name}/var_astro_mon.I.npy')
    C = np.load(f'{name}/var_astro_mon.C.npy')
    h = np.load(f'{name}/var_astro_mon.h.npy')
    x_A = np.load(f'{name}/var_astro_mon.x_A.npy')
    G_A = np.load(f'{name}/var_astro_mon.G_A.npy')

    #[:200] = excitatory , [200:] = inhibitory
    mon_v = np.load(f'{name}/neurons_mon.v.npy')
    mon_g_e = np.load(f'{name}/neurons_mon.g_e.npy')
    mon_g_i = np.load(f'{name}/neurons_mon.g_i.npy')
    mon_t = np.load(f'{name}/neurons_mon.t.npy')
    I_exc = np.load(f'{name}/neurons_mon.I_exc.npy')
    I_inh = np.load(f'{name}/neurons_mon.I_inh.npy')
    I_external = np.load(f'{name}/neurons_mon.I_syn_ext.npy')
    firing_rate_exc_t = np.load(f'{name}/firing_rate_exc.t.npy')
    firing_rate_exc = np.load(f'{name}/firing_rate_exc.rate.npy')
    firing_rate_inh_t = np.load(f'{name}/firing_rate_inh.t.npy')
    firing_rate_inh = np.load(f'{name}/firing_rate_inh.rate.npy')


    astro_connected = np.load(f'{name}/ecs_astro_to_syn.i.npy')
    syn_connected = np.load(f'{name}/ecs_astro_to_syn.j.npy')
    astro_to_syn_i = np.load(f'{name}/ecs_astro_to_syn.i.npy')
    astro_to_syn_j = np.load(f'{name}/ecs_astro_to_syn.j.npy')
    syn_to_astro_i = np.load(f'{name}/ecs_syn_to_astro.i.npy')
    syn_to_astro_j = np.load(f'{name}/ecs_syn_to_astro.j.npy')

    N_e = 3200
    N_i = 800
    N_a = 3200
    C_Theta = 0.5*umolar
    defaultclock.dt = 0.1*ms
    
    #######################################################################################################

    ## Analysis ##############################################################################
    # transient time
    trans = transient(t*second, 50000)
    firing_rate_exc = smoothing_b(firing_rate_exc, width=1*ms)
    firing_rate_inh = smoothing_b(firing_rate_inh, width=1*ms)
    
    # Mean firing rate and Recurrent current before and after GRE
    # before: trans- 2 second
    # after: 3 - 5 second
    fr_exc_before = firing_rate_exc[trans:20000]    
    fr_exc_after = firing_rate_exc[30000:50000]  
    fr_inh_before = firing_rate_inh[trans:20000]    
    fr_inh_after = firing_rate_inh[30000:50000]

    I_exc_before = I_exc[:200,trans:20000]
    I_exc_after = I_exc[:200,30000:50000]
    I_inh_before = I_inh[200:,trans:20000]
    I_inh_after = I_inh[200:,30000:50000]

    print(f'POPULATION FIRING RATES')
    print(f'before GRE')
    print(f'exc: mean={fr_exc_before.mean():.4f} std={fr_exc_before.std():.4f} Hz')
    print(f'inh: mean={fr_inh_before.mean():.4f} std={fr_inh_before.std():.4f} Hz')
    print(f'after GRE')
    print(f'exc: mean={fr_exc_after.mean():.4f} std={fr_exc_after.std():.4f} Hz')
    print(f'inh: mean={fr_inh_after.mean():.4f} std={fr_inh_after.std():.4f} Hz')
    print(f'total time simulation')
    print(f'exc: mean={firing_rate_exc[trans:].mean():.4f} std={firing_rate_exc[trans:].std():.4f} Hz')
    print(f'inh: mean={firing_rate_inh[trans:].mean():.4f} std={firing_rate_inh[trans:].std():.4f} Hz')

    N = len(firing_rate_exc)
    #  print(f'sampling freq Whelch : {1/((N//4+1)*defaultclock.dt)} Hz')
    
    f_fr_exc, spect_fr_exc = signal.welch(firing_rate_exc[trans:],fs=1/defaultclock.dt, nperseg=N//4)
    f_fr_inh, spect_fr_inh = signal.welch(firing_rate_inh[trans:],fs=1/defaultclock.dt, nperseg=N//4)
    ###########################################################################################

    ## Information #############################################################################
    print('')
    print('RECURRENT CURRENTS')
    print(f'I_external on exc: {I_external[:200].mean()/pA:.4f} +- {(I_external[:200].mean()/pA)/np.sqrt(200):.4f} pA')
    print(f'I_external on inh: {I_external[200:].mean()/pA:.4f} +- {(I_external[200:].mean()/pA)/np.sqrt(200):.4f} pA')
    print(f'before GRE')
    print(f'exc: mean={I_exc_before.mean()/pA:.4f} std={I_exc_before.std()/pA:.4f} pA')
    print(f'inh: mean={I_inh_before.mean()/pA:.4f} std={I_inh_after.std()/pA:.4f} pA')
    print(f'after GRE')
    print(f'exc: mean={I_exc_after.mean()/pA:.4f} std={I_exc_after.std()/pA:.4f} pA')
    print(f'inh: mean={I_inh_after.mean()/pA:.4f} std={I_inh_after.std()/pA:.4f} pA')
    print(f'OVERALL')
    print(f'I_exc : {I_exc[:200].mean()/pA:.4f}  +- {(I_exc[:200].std()/pA)/np.sqrt(200):.4f} pA')
    print(f'I_inh : {I_inh[:200].mean()/pA:.4f}  +- {(I_inh[:200].std()/pA)/np.sqrt(200):.4f} pA')

    ############################################################################################
    ## PLOTS ######################################################################################
    
    
    fig1, ax1 = plt.subplots(nrows=3, ncols=1, sharex=True, gridspec_kw={'height_ratios': [2.5,1,1]},
                            figsize=(12, 14), num=f'Raster plot')
    step = 4
    ax1[0].plot(t_exc[exc_neurons_i%step==0]/second, 
            exc_neurons_i[exc_neurons_i%step==0], '|', color='C3')
    ax1[0].plot(t_inh[inh_neurons_i%step==0]/second, 
            inh_neurons_i[inh_neurons_i%step==0]+N_e, '|', color='C0',)
    ax1[0].plot(t_astro[astro_i%step==0]/second, 
            astro_i[astro_i%step==0]+(N_e+N_i),'|' , color='green')
    ax1[0].set_ylabel('cell index')

#     firing_rate_exc = smoothing_b(firing_rate_exc, width=1*ms)
#     firing_rate_inh = smoothing_b(firing_rate_inh, width=1*ms)
   
    ax1[1].plot(firing_rate_exc_t[trans:]/second, firing_rate_exc[trans:]/Hz, color='C3')
    ax1[1].set_ylabel('FR_exc (Hz)')
    ax1[1].grid(linestyle='dotted')

    ax1[2].plot(firing_rate_inh_t[trans:]/second, firing_rate_inh[trans:]/Hz, color='C0')
    ax1[2].set_ylabel('FR_inh (Hz)')
    ax1[2].set_xlabel('time (s)')
    ax1[2].grid(linestyle='dotted')

    plt.savefig(name+f'Raster plot.png')

    fig2, ax2 = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(12, 7),
                            num='External and Recurrent current')

    ax2[0].plot(mon_t[trans:]/second, I_external[:200].mean(axis=0)[trans:]/pA, color='C1', label='on Exc')
    ax2[0].plot(mon_t[trans:]/second, I_external[200:].mean(axis=0)[trans:]/pA, color='C4', label='on Inh')
    ax2[0].legend()
    ax2[0].set_ylabel(r'$I_{external}$ (pA)')
    ax2[0].grid(linestyle='dotted')

    ax2[1].plot(mon_t[trans:]/second, I_exc[:200].mean(axis=0)[trans:]/pA, color='C3')
    ax2[1].set_ylabel(r'$I_{exc}^{rec}$ (pA)')
    ax2[1].grid(linestyle='dotted')

    ax2[2].plot(mon_t[trans:]/second, I_inh[:200].mean(axis=0)[trans:]/pA, color='C0')
    ax2[2].set_ylabel(r'$I_{inh}^{rec}$ (pA)')
    ax2[2].grid(linestyle='dotted')
    ax2[2].set_xlabel('time (s)')

    plt.savefig(name+f'External and Recurrent current.png')

    fig3, ax3 = plt.subplots(nrows=2, ncols=1, sharex=True,
                            num='Spectral analisis - firing rate')

    ax3[0].plot(f_fr_exc, spect_fr_exc)
    ax3[0].grid(linestyle='dotted')

    ax3[1].plot(f_fr_inh, spect_fr_inh)
    ax3[1].grid(linestyle='dotted')

    # fig2, ax2 = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(13, 9), 
    #                         num=f'astrocyte dynamics')
    # con_index = connected_astro[0:1] # synaptically connected astrocytes
    # free_index = free_astro[0:1]     # not synaptically connected astrocytes

    # ax2[0].plot(t[trans:], Y_S[con_index][0,trans:]/umolar, color='C3', label='synaptically connected')
    # ax2[0].plot(t[trans:], Y_S[free_index][0,trans:]/umolar, color='C3', ls='dashed', label='free')
    # ax2[0].set_ylabel(r'$Y_S$ ($\mu$M)')
    # ax2[0].grid(linestyle='dotted')
    # ax2[0].legend()

    # ax2[1].plot(t[trans:], Gamma_A[con_index][0,trans:], color='C7', label='synaptically connected')
    # ax2[1].plot(t[trans:], Gamma_A[free_index][0,trans:], color='C7', ls='dashed', label='free')
    # ax2[1].set_ylabel(r'$\Gamma_A$ ')
    # ax2[1].grid(linestyle='dotted')
    # ax2[1].legend()

    # ax2[2].plot(t[trans:], I[con_index][0,trans:]/umolar, color='C0', label='synaptically connected')
    # ax2[2].plot(t[trans:], I[free_index][0,trans:]/umolar, color='C0', ls='dashed', label='free')
    # ax2[2].set_ylabel(r'$I$ ($\mu$M)')
    # ax2[2].grid(linestyle='dotted')
    # ax2[2].legend()

    # ax2[3].plot(t[trans:], C[con_index][0,trans:]/umolar, color='red', label='synaptically connected')
    # ax2[3].plot(t[trans:], C[free_index][0,trans:]/umolar, color='red', ls='dashed', label='free')
    # ax2[3].set_ylabel(r'$Ca^{2\plus}$ ($\mu$M)')
    # ax2[3].set_xlabel('time (s)')
    # ax2[3].plot(t[trans:], np.full(t[trans:].shape[0], C_Theta/umolar), ls='dashed', color='black')
    # ax2[3].grid(linestyle='dotted')
    # ax2[3].legend()

    # plt.savefig(name+f'astrocyte dynamics.png')

    # fig3, ax3 = plt.subplots(nrows=1, ncols=1, 
    #                         num=f'gliorelease hist - connected astro')
    # ax3.hist(gliorelease_conn, bins=20)
    # ax3.set_xlabel('time (s)')

    # plt.savefig(name+f'gliorelease hist - connected astro.png')

   

    plt.show()

