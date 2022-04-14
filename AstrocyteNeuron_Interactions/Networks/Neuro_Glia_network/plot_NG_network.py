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
from network_analysis import transient, selected_window

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

def neurons_firing(t_spikes, neurons_i, time_start, time_stop):
    """
    Firing rate of single neurons
    Distribuction of neurons spikes activity

    Parameters
    ----------
    t_spikes : array
            spiking time of all neurons (comes from SpikesMonitor group of Brian2),
            there are expressed in second. For istance:
            monitor = SpikesMonitor(neurons)
            t_spikes = monitor.t

    neurons_i : array
            indexes array come from SpikesMonitor group of Brian2, for istance
            monitor = SpikesMonitor(neurons)
            neurons_i = monitor.i
    
    time_start : float
                start of time window, second
    
    stop_start : float
                stop of time window, second

    Returns
    -------
    neurons_fr : array
                list of neuron's firing rate in time_start-time_stop windows
    """
    
    # indeces: indeces of firing neurons
    indeces = np.unique(neurons_i)
    time = (time_stop-time_start)*second

    neurons_fr = []
    for ind in indeces:
        spikes = [spk for spk in t_spikes[neurons_i==ind] if (spk > time_start and spk < time_stop)]
        firing_rate = len(spikes)/time
        neurons_fr.append(firing_rate)

    return neurons_fr


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
    mon_LFP = np.load(f'{name}/mon_LFP.LFP.npy')

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
    firing_rate_t = np.load(f'{name}/firing_rate.t.npy')
    firing_rate = np.load(f'{name}/firing_rate.rate.npy')



    astro_connected = np.load(f'{name}/ecs_astro_to_syn.i.npy')
    syn_connected = np.load(f'{name}/ecs_astro_to_syn.j.npy')
    astro_to_syn_i = np.load(f'{name}/ecs_astro_to_syn.i.npy')
    astro_to_syn_j = np.load(f'{name}/ecs_astro_to_syn.j.npy')
    syn_to_astro_i = np.load(f'{name}/ecs_syn_to_astro.i.npy')
    syn_to_astro_j = np.load(f'{name}/ecs_syn_to_astro.j.npy')

    N_e = 3200
    N_i = 800
    N_a = 4000
    C_Theta = 0.5*umolar
    defaultclock.dt = 0.1*ms
    
    #######################################################################################################

    ## Analysis ##############################################################################
    # transient time
    trans = transient(t*second, 50000)
#     firing_rate_exc = smoothing_b(firing_rate_exc, width=1*ms)
#     firing_rate_inh = smoothing_b(firing_rate_inh, width=1*ms)
    LFP = mon_LFP.sum(axis=0)

    # Network analysis concern mean values and spectral analysis of
    # population firing rate and LFP
    analysis = {'BASE': [0.5*second, 1.6*second],'GRE1': [4*second, 6.5*second], 'GRE2': [9*second, 11.5*second]}

    fig4, ax4 = plt.subplots(nrows=2, ncols=1, sharex=True,
                            num='Firing rate and LFP ')
    fr_t = firing_rate_t[:]/second
    fr_smooth = smoothing_b(firing_rate, width=5*ms)
    ax4[0].plot(fr_t[5000:], fr_smooth[5000:], color='k', alpha=0.7)
    ax4[0].grid(linestyle='dotted')
    ax4[0].set_xlabel('time (s)')
    ax4[0].set_ylabel('firing rate (Hz)')
    

    ax4[1].plot(mon_t[5000:]/second, LFP[5000:], color='C5', alpha=0.7)
    ax4[1].grid(linestyle='dotted')
    ax4[1].set_xlabel('time (s)')
    ax4[1].set_ylabel('LFP (?)')
    
    for zone in analysis.keys():
        fig3, ax3 = plt.subplots(nrows=1, ncols=2, figsize=(12,6),
                            num='Spectral analysis '+zone)
        
        fr_exc = selected_window(firing_rate_exc, analysis[zone][0], analysis[zone][1])
        fr_inh = selected_window(firing_rate_inh, analysis[zone][0], analysis[zone][1])
        fr = selected_window(firing_rate, analysis[zone][0], analysis[zone][1])
        LFP_w = selected_window(LFP, analysis[zone][0], analysis[zone][1])

        N = len(fr_exc)
        NN = len(LFP_w)

        freq_fr_exc, spectrum_fr_exc = signal.welch(fr_exc, fs=1/defaultclock.dt/Hz, nperseg=N//3)
        freq_fr_inh, spectrum_fr_inh = signal.welch(fr_inh, fs=1/defaultclock.dt/Hz, nperseg=N//3)
        freq_fr, spectrum_fr = signal.welch(fr, fs=1/defaultclock.dt/Hz, nperseg=N//3)
        freq_LFP_w, spectrum_LFP_w = signal.welch(LFP_w, fs=1/defaultclock.dt/Hz, nperseg=NN//3)
        print(len(freq_LFP_w))

        print(f' {zone} - time window: {analysis[zone][0]/second:.2f} - {analysis[zone][1]/second:.2f} s')
        print(f'exc: mean={fr_exc.mean():.4f} std={fr_exc.std():.4f} Hz')
        print(f'inh: mean={fr_inh.mean():.4f} std={fr_inh.std():.4f} Hz')
        print(f'LFP: mean={LFP_w.mean():.4f} std={LFP_w.std():.4f} volt') 
        print(f'frequency resolution Whelch : {1/((N//3+1)*defaultclock.dt)} Hz')
        print(f'frequency resolution Whelch : {1/((NN//3+1)*defaultclock.dt)} Hz')

        ax3[0].set_title(zone)
        ax3[0].plot(freq_fr_exc, spectrum_fr_exc, color='k')
        ax3[0].set_xlim([-10,500])
        ax3[0].set_ylabel('spectrum fr')
        ax3[0].grid(linestyle='dotted')
        ax3[0].set_xlabel('frequecy (Hz)')
        # ax3[0].set_yscale('log')
        
        ax3[1].plot(freq_LFP_w, spectrum_LFP_w, color='C5')
        ax3[1].plot(freq_LFP_w[1:], 1/(freq_LFP_w[1:]**2), alpha=0.5)
        ax3[1].set_xscale('log')
        ax3[1].set_yscale('log')
        ax3[1].set_ylabel('spectrum LFP')
        ax3[1].set_xlabel('frequecy (Hz)')
        ax3[1].grid(linestyle='dotted')

        # Underline selected zone
        ax4[0].plot(selected_window(fr_t, analysis[zone][0], analysis[zone][1]),
                    selected_window(fr_smooth, analysis[zone][0], analysis[zone][1]), color='C1')
        
        ax4[1].plot(selected_window(mon_t, analysis[zone][0], analysis[zone][1]),
                    LFP_w, color='C1')
        
    # Neurons firnig rate distribuction
    neuron_fr_BASE = neurons_firing(t_exc, exc_neurons_i, 
                                time_start=analysis['BASE'][0]/second, time_stop=analysis['BASE'][1]/second)
    neuron_fr_GRE1 = neurons_firing(t_exc, exc_neurons_i, 
                                time_start=analysis['GRE1'][0]/second, time_stop=analysis['GRE1'][1]/second)
    neuron_fr_GRE2 = neurons_firing(t_exc, exc_neurons_i, 
                                time_start=analysis['GRE2'][0]/second, time_stop=analysis['GRE2'][1]/second)
    
    # Mean firing rate and Recurrent current before and after GRE
    # before: trans- 2 second
    # after: 4 - 6.5 second
    fr_exc_base = selected_window(firing_rate_exc, analysis['BASE'][0], analysis['BASE'][1])  
    fr_inh_base = selected_window(firing_rate_inh, analysis['BASE'][0], analysis['BASE'][1])
    fr_exc_gre1 = selected_window(firing_rate_exc, analysis['GRE1'][0], analysis['GRE1'][1])  
    fr_inh_gre1 = selected_window(firing_rate_inh, analysis['GRE1'][0], analysis['GRE1'][1])

    I_exc_base = selected_window(I_exc[0], 1.2*second, 5.5*second)
   
    ###########################################################################################
    ## Information #############################################################################
    print('')
    print('EXTERNAL AND RECURRENT CURRENTS')
    print(f'I_external on exc: {I_external[:200].mean()/pA:.4f} +- {(I_external[:200].mean()/pA)/np.sqrt(200):.4f} pA')
    print(f'I_external on inh: {I_external[200:].mean()/pA:.4f} +- {(I_external[200:].mean()/pA)/np.sqrt(200):.4f} pA')
    for zone in analysis.keys():
        print(zone)
        I_exc_zone = selected_window(I_exc[:200].mean(axis=0), analysis[zone][0], analysis[zone][1])
        I_inh_zone = selected_window(I_inh[:200].mean(axis=0), analysis[zone][0], analysis[zone][1])
        print(f'I_exc : {I_exc_zone.mean()/pA:.4f} +- {I_exc_zone.std()/pA/np.sqrt(len(I_exc_zone)):.4f} pA')
        print(f'I_inh : {I_inh_zone.mean()/pA:.4f} pA +- {I_inh_zone.std()/pA/np.sqrt(len(I_exc_zone)):.4f} pA')
    print('')
    print('FIRING RATE')
    for zone in analysis.keys():
        print(zone)
        fr_smooth_zone = selected_window(fr_smooth, analysis[zone][0], analysis[zone][1])  
        fr_exc_zone = selected_window(firing_rate_exc, analysis[zone][0], analysis[zone][1])  
        fr_inh_zone = selected_window(firing_rate_inh, analysis[zone][0], analysis[zone][1])
        print(f'fr : {fr_smooth_zone.mean()/Hz:.4f} +- {fr_smooth_zone.std()/Hz/np.sqrt(len(I_exc_zone)):.4f} Hz')
        print(f'fr_exc : {fr_exc_zone.mean()/Hz:.4f} +- {fr_exc_zone.std()/Hz/np.sqrt(len(I_exc_zone)):.4f} Hz')
        print(f'fr_inh : {fr_inh_zone.mean()/Hz:.4f} +- {fr_inh_zone.std()/Hz/np.sqrt(len(I_exc_zone)):.4f} Hz')
    print('')
    print('LFP')
    for zone in analysis.keys():
        print(zone)
        LFP_zone = selected_window(LFP, analysis[zone][0], analysis[zone][1])  
        print(f'LFP : {LFP_zone.mean()/volt:.4f} +- {LFP_zone.std()/volt:.4f} V')
        
    ############################################################################################
    ## PLOTS ######################################################################################
    fig1, ax1 = plt.subplots(nrows=3, ncols=1, sharex=True, gridspec_kw={'height_ratios': [2.5,1,1]},
                            figsize=(12, 14), num=f'Raster plot')
    step = 10
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

    fig6, ax6 = plt.subplots(nrows=1, ncols=1, num='Single firing rate distribuction - Exc population')
    
    ax6.hist(x=neuron_fr_BASE/Hz, bins=7, alpha=0.5, label='BASE')
    ax6.hist(x=neuron_fr_GRE1/Hz, bins=7, alpha=0.6, label='GRE1')
    ax6.hist(x=neuron_fr_GRE2/Hz, bins=7, alpha=0.75, label='GRE2')
    ax6.set_xlabel('frequency (Hz)')
    ax6.set_ylabel('Number of neurons')
    ax6.legend()

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

