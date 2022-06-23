"""
Statistical analysis of long-term simulation, i.e, t > 20 second.
This module include the dynamical ration of E and I recurrent currents,
the modulation of spectral analysis with respect to baseline condiction and
the study og syncronization regime.

Note: statistical analysis for simulation t < 12 s is implemanted in plot_NG_network.py
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy import signal
from scipy import stats
from brian2 import *
from plot_NG_network import smoothing_b
from network_analysis import transient, selected_window
from plot_NG_network import *

def synchrony_v(v):
    """
    Membran syncrony index of large neural network.

    Reference: http://www.scholarpedia.org/article/Neuronal_synchrony_measures

    Parameters
    ----------
    v : 2d array
        menbran potential of monitored neurons

    Returns
    -------
    msi : float
        Membran syncrony index
    """
    V_tot = v.mean(axis=0)
    V_var = stats.tvar(V_tot)

    V_var_single = np.asarray([stats.tvar(v_i) for v_i in v])
    V_var_single = V_var_single.mean()

    chi_square = V_var / V_var_single
    return np.sqrt(chi_square)

def correlation(x,y):
    """
    Correlation between signal x and y. the signals must have the same size

    Parameters
    ----------

    Return
    ------

    """
    x = np.asarray(x)
    y = np.asarray(y)
    x_mean = x.mean()
    y_mean = y.mean()

    cov_xy = (x-x_mean) * (y-y_mean) 
    cov_xy = cov_xy.sum()

    std_x = (x-x_mean)**2
    std_x = std_x.sum()

    std_y = (y-y_mean)**2
    std_y = std_y.sum()

    corr = cov_xy / (np.sqrt(std_x*std_y))
    return corr


def MPS_index(v):
    """
    Membran Potential Syncrony (MPS) index
    See: Brunel 2003

    Parameters
    ----------
    v : 2d array
        menbran potential of monitored neurons

    Returns
    -------
    """
    N = v.shape(0)

def Modulation(baseline, signal):
    """
    Modulation of density spectral analysis with respect to bese line condiction.
    Modulation is defined as the difference of 'signal' and 'baseline, normalized to the latter

    Note: baseline and sugnal must have save size
    """
    base = np.asarray(baseline)
    sig = np.asarray(signal)

    mod = (sig - base)/base

    return mod

def standard_error_I(I, N_mean=10):
	"""
	Compute mean and standard error of recurrent current
	"""
	I = np.asarray(I)
	N = len(I)
	N_window = int(N/N_mean)

	I_list = []
	for i in range(N_mean):
		start = i*N_window
		stop = (i+1)*N_window
		I_mean = I[start:stop]
		I_list.append(I_mean.mean())
	
	I_list = np.asanyarray(I_list)
	
	if N_mean < 30 : 
		error = (I_list.max()-I_list.min())/2
	else:
		error = I_list.std()/np.sqrt(N_mean)
	
	return I_list.mean(), error



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Advanced connectivity connection')
    parser.add_argument('file', type=str, help="file's name of network in 'Neuro_Glia_network' folder")
    args = parser.parse_args()

    ## hyperparameters
    total_window=8

    ## Temporal parameters
    duration = 10*second
    defaultclock.dt = 0.1*ms
    dt_astro = 1e-2*second
    dt = 0.1*ms
    N = duration/dt
    N_wind = int(N/4)
    t_window = np.arange(0,N)*dt

	## E/I recurrent currents ratio #################################################################
    #initialize fig setting
    fig1, ax1 = plt.subplots(nrows=3, ncols=1, num='EI ratio', figsize=(12,8))
    ax1[0].set_ylabel('E/I')
    ax1[0].grid(linestyle='dotted')

    ax1[1].grid(linestyle='dotted')
    ax1[1].set_ylabel(r'$I_{exc}$ (pA)' )
    for i_time,w in enumerate(range(1,total_window)):
        name = args.file+f'/time_windows_{w}'
        I_exc = np.load(f'{name}/neurons_mon.I_exc.npy')
        I_inh = np.load(f'{name}/neurons_mon.I_inh.npy')

        t_plot = t_window + duration*i_time
        
        ## select consecutivewindows of 2.5 seconds
        I_exc = I_exc.mean(axis=0)
        I_inh = I_inh.mean(axis=0)

        ax1[1].plot(t_plot, I_exc/pA, color='C3', label='I_exc')
        
        if w==1:
            print('baseline for current')
            E_base = I_exc[2000:9000]/pA
            I_base = I_inh[2000:9000]/pA
            print(E_base.shape)

            E_base_m , E_base_err = standard_error_I(E_base, N_mean=30)
            I_base_m , I_base_err = standard_error_I(I_base, N_mean=30)

            EI_ratio_base = E_base_m/I_base_m
            EI_ratio_base_err = ((E_base_err/E_base_m) + (I_base_err/I_base_m))*EI_ratio_base
            print(f'baseline: E = {E_base_m:.4f} +- {E_base_err:.4f} pA')
            print(f'baseline: I = {I_base_m:.4f} +- {I_base_err:.4f} pA')
            print(f'baseline: E/I = {EI_ratio_base:.4f} +- {EI_ratio_base_err:.4f}')

        N_wind_p = int(N/2)
        for ii in range(2):
            start = ii*N_wind_p
            stop = (ii+1)*N_wind_p
            t = (duration/4)+(duration/2*ii)+(w-1)*duration 

            I_exc_ii =  I_exc[start:stop]
            I_inh_ii = I_inh[start:stop]

            I_exc_ii_m, I_exc_ii_err = standard_error_I(I_exc_ii, N_mean=30)
            I_inh_ii_m, I_inh_ii_err = standard_error_I(I_inh_ii, N_mean=30)

            EI_ratio = I_exc_ii_m/I_inh_ii_m
            EI_ratio_err = ((I_exc_ii_err/I_exc_ii_m) + (I_inh_ii_err/I_inh_ii_m))*EI_ratio
        
            ax1[0].errorbar(t, EI_ratio, EI_ratio_err, fmt='o', color='C0', capsize=1.5)
            # if w == 5:
            #     plt.figure(num=f'currents blocking {w}')
            #     plt.scatter([i+1 for i in range(13)], np.sqrt(blocking(I_exc_ii/pA, k=13)), label=f'{ii+1} exc')
            #     plt.scatter([i+1 for i in range(13)], np.sqrt(blocking(I_inh_ii/pA, k=13)), label=f'{ii+1} inh')
            #     plt.scatter([i+1 for i in range(15)], np.sqrt(blocking(I_exc/pA, k=15)), label=f'{ii+1} exc')
            #     plt.yscale('log')
            #     plt.legend()
            print(f'{t} second: E/I = {EI_ratio:.5f} +- {EI_ratio_err:.5f}')
    print()
    #################################################################################################

    ## Astrocitic dynamics #################################################################################################
    fig3, ax3 = plt.subplots(nrows=2, ncols=1, sharex=True, num='Astro activity')
    fig5, ax5 = plt.subplots(nrows=1, ncols=1, sharex=True, num='Astro oscillation')
    for w in range(1,total_window):
        name = args.file+f'/time_windows_{w}'
        astro_i = np.load(f'{name}/astro_mon.i.npy')
        astro_t = np.load(f'{name}/astro_mon.t.npy')
        C = np.load(f'{name}/var_astro_mon.C.npy')
        Y_S = np.load(f'{name}/var_astro_mon.Y_S.npy')
        ttt = np.load(f'{name}/var_astro_mon.t.npy')

        # ax3.scatter(astro_t,astro_i, marker='|', color='green')
        for i in range(10):
            for axis in [ax1[2], ax3[1]]:	
                axis.plot(ttt, C[i]/umolar, c='C'+f'{i}')
                axis.axhline(0.5, color='k', ls='dashed')
                axis.set_xlabel('time (s)')
                axis.set_ylabel(r'C ($\mu$M)')
                axis.grid(linestyle='dotted')

        
            # astro_oscillation.append(np.where(np.max(Pxx_den))[0])    

            ax3[0].plot(ttt, Y_S[i]/umolar, c='C'+f'{i}')
            ax3[0].set_xlabel('time (s)')
            ax3[0].set_ylabel(r'$Y_S$ ($\mu$M)')
            ax3[0].grid(linestyle='dotted')
    ########################################################################################################

    ## Spectral analysis and Modulation ####################################################################
    # base_fr = np.load(f'Baseline_g0.25_s1.0_fixed/spectrum_fr.npy')
    # base_LFP = np.load(f'Baseline_g0.25_s1.0_fixed/spectrum_LFP.npy')

    # fig4, ax4 = plt.subplots(nrows=1, ncols=2, num='Spectral Analysis')
    # ax4[0].set_title('Population firing rate')
    # ax4[0].plot(base_fr, color='k')
    # ax4[0].set_xlabel('frequency (Hz)')
    # ax4[0].set_xlim([-10,200])
    # ax4[0].grid(linestyle='dotted')

    # ax4[1].set_title('LFP')
    # ax4[1].plot(base_LFP, color='C5') 
    # ax4[1].set_xlabel('frequency (Hz)')
    # ax4[1].set_xlim([-10,200])
    # ax4[1].grid(linestyle='dotted')

    # glio_LFP = []
    # for i_time,w in enumerate(range(9,11)):
    #     name = args.file+f'/time_windows_{w}'

    #     fr_t = np.load(f'{name}/firing_rate_exc.t.npy')
    #     fr_exc = np.load(f'{name}/firing_rate_exc.rate.npy')
    #     fr_inh = np.load(f'{name}/firing_rate_inh.rate.npy')
    #     fr = np.load(f'{name}/firing_rate.rate.npy')
    #     LFP = np.load(f'{name}/mon_LFP.LFP.npy')
    #     LFP = LFP.sum(axis=0)

    #     t_plot = t_window + duration*i_time
        
        
    #     for ii in range(4):
    #         start = ii*N_wind
    #         stop = (ii+1)*N_wind
    #         t = (duration/8)+(duration/4*ii)+(w-1)*duration

    #         fr_exc_ii =  fr_exc[start:stop] 
    #         fr_inh_ii =  fr_inh[start:stop]
    #         fr_ii =  fr[start:stop]
    #         LFP_ii = LFP[start:stop]
    #         N_ii = len(fr_ii)

    #         freq_fr_exc_ii, spectrum_fr_exc_ii = signal.welch(fr_exc_ii, fs=1/defaultclock.dt/Hz, nperseg=N_ii//2)
    #         freq_fr_inh_ii, spectrum_fr_inh_ii = signal.welch(fr_inh_ii, fs=1/defaultclock.dt/Hz, nperseg=N_ii//2)
    #         freq_fr_ii, spectrum_fr_ii = signal.welch(fr_ii, fs=1/defaultclock.dt/Hz, nperseg=N_ii//3)
    #         freq_LFP_ii, spectrum_LFP_ii = signal.welch(LFP_ii, fs=1/defaultclock.dt/Hz, nperseg=N_ii//3)
    #         glio_LFP.append(spectrum_LFP_ii)

    #         plt.figure(num=f'LFP spectrum: {(w-1)*10 + ii*2.5} - {((w-1)*10 + ii*2.5)+2.5} s')
    #         plt.plot(freq_LFP_ii, spectrum_LFP_ii, label='gliotransmission')
    #         plt.plot(freq_LFP_ii, base_LFP, label='baseline')
    #         plt.xlabel('frequency (Hz)')
    #         plt.ylabel('PSD ' + r'($\rm{V^2/Hz}$)')
    #         plt.xlim([-10,200])
    #         plt.grid(linestyle='dotted')
    #         plt.legend()

    # glio_LFP = np.asanyarray(glio_LFP)
    # glio_LFP = glio_LFP.mean(axis=0)

    # plt.figure(num='Baseline vs glio LFP spectrum')
    # plt.plot(freq_LFP_ii, base_LFP, label='baseline')
    # plt.plot(freq_LFP_ii, glio_LFP, label='gliotransmission')
    # plt.xlabel('frequency (Hz)')
    # plt.ylabel('PSD ' + r'($\rm{V^2/Hz}$)')
    # plt.xlim([-10,200])
    # plt.grid(linestyle='dotted')
    # plt.legend()
        
    #     # LFP_astro = np.asanyarray(LFP_astro)
    #     # LFP_astro = LFP_astro.mean(axis=0)
    #     # mod_LFP = Modulation(base_LFP,LFP_astro)
    #     # print(base_LFP[25], spectrum_LFP_ii[25], mod_LFP[25])
    #     # plt.figure()
    #     # plt.plot(freq_LFP_ii, LFP_astro)
    #     # plt.plot(freq_LFP_ii, base_LFP)
    #     # plt.xlim([-10,200])
               
        

     
    #################################################################################################
    
    # ## Syncrony index ###############################################################################
    fig2, ax2 = plt.subplots(nrows=1, ncols=1, num='Synchrony')
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel(r'$\chi$')
    ax2.grid(linestyle='dotted')

    for w in range(1,total_window):
        name = args.file+f'/time_windows_{w}'
        v = np.load(f'{name}/neurons_mon.v.npy')
        v_exc = v[:200]
        v_inh = v[200:]

        ## select consecutivewindows of 2.5 seconds
        for ii in range(4):
            start = ii*N_wind
            stop = (ii+1)*N_wind
            t = (duration/8)+(duration/4*ii)+(w-1)*duration 

            v_ii = v[:,start:stop]
            v_exc_ii = v_exc[:,start:stop]
            v_inh_ii = v_inh[:,start:stop]

            # print(f'{t} second : chi = {synchrony_v(v_inh_ii)}')
            ax2.scatter(t, synchrony_v(v_ii), color='k')
            ax2.scatter(t, synchrony_v(v_exc_ii), color='C3')
            ax2.scatter(t, synchrony_v(v_inh_ii), color='C0')
        
    ax2.scatter(t, synchrony_v(v_ii), color='k', label='network')
    ax2.scatter(t, synchrony_v(v_exc_ii), color='C3', label='E')
    ax2.scatter(t, synchrony_v(v_inh_ii), color='C0', label='I')
    ax2.legend()
           
    #################################################################################################
           
    plt.show()
