import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from brian2 import *

parser = argparse.ArgumentParser(description='Variable dynamics of Neuronal Network')
parser.add_argument('file', type=str, help="file's name of network in 'Neural_network' folder")
args = parser.parse_args()

## Load variables ######################################################################################
name=args.file

N_e = 3200
N_i = 800
V_th = -50*mV          # Firing threshold
defaultclock.dt = 0.1*ms

duration = np.load(f'{name}/duration.npy')*second
trans_time = np.load(f'{name}/trans_time.npy')
trans = np.load(f'{name}/trans.npy')
index = np.load(f'{name}/index.npy')
rate_in = np.load(f'{name}/rate_in.npy')

# Excitatory neurons variable
t_exc = np.load(f'{name}/state_exc_mon.t.npy')
I_syn_ext = np.load(f'{name}/state_exc_mon.I_syn_ext.npy')
v = np.load(f'{name}/state_exc_mon.v.npy')
g_i = np.load(f'{name}/state_exc_mon.g_i.npy')
g_e = np.load(f'{name}/state_exc_mon.g_e.npy')
LFP = np.load(f'{name}/state_exc_mon.LFP.npy')

# Population istantaneus firing rate
firing_rate_exc_t= np.load(f'{name}/firing_rate_exc.t.npy')
firing_rate_inh_t = np.load(f'{name}/firing_rate_inh.t.npy')
fr_exc = np.load(f'{name}/fr_exc.npy')
fr_inh = np.load(f'{name}/fr_inh.npy')

# Spike variable
spikes_exc_mon_t = np.load(f'{name}/spikes_exc_mon.t.npy')
spikes_inh_mon_t = np.load(f'{name}/spikes_inh_mon.t.npy')
spikes_exc_mon_i = np.load(f'{name}/spikes_exc_mon.i.npy')
spikes_inh_mon_i = np.load(f'{name}/spikes_inh_mon.i.npy')
########################################################################################################

## LFP analysis
LFP = LFP[:].sum(axis=0)

LFP_trans = LFP[trans:]
LFP_fft = fft(LFP_trans)
N = len(LFP_trans)
LFP_freq = fftfreq(N,defaultclock.dt)

f, Pxx_den = signal.welch(LFP_trans,fs=1/defaultclock.dt, nperseg=N//2)


# Plots  ################################################################################################
fig1, ax1 = plt.subplots(nrows=3, ncols=1, sharex=True, 
                         num=f'exc variable dynamic, v_in={rate_in/Hz}',figsize=(9,10))

ax1[0].plot(t_exc[trans:]/second, I_syn_ext[index,trans:]/pA,
			color='C3', label=f'{index}'+r' $I_{syn ext}$')
ax1[0].set_ylabel(r'$I_{syn ext} (pA)$')
ax1[0].grid(linestyle='dotted')
ax1[0].legend(loc = 'upper right')

ax1[1].plot(t_exc[trans:]/second, v[index,trans:]/mV, label=f'neuron {index}')
ax1[1].axhline(V_th/mV, color='C2', linestyle=':')
for spk in spikes_exc_mon_t[spikes_exc_mon_i == index]:
	if (spk>trans_time/1000): ax1[1].axvline(x=spk, ymin=0.15, ymax=0.95 )
ax1[1].set_ylim(bottom=-60.8, top=0.1)
ax1[1].set_ylabel('Membran potential (V)')
ax1[1].set_title('Neurons dynamics')
ax1[1].grid(linestyle='dotted')
ax1[1].legend(loc = 'upper right')

ax1[2].plot(t_exc[trans:]/second, g_i[index,trans:]/nS, color='C4', label=f'{index}'+r' $g_i$')
ax1[2].plot(t_exc[trans:]/second, g_e[index,trans:]/nS, color='C5', label=f'{index}'+r' $g_e$')
ax1[2].set_ylabel('Conductance (nS)')
ax1[2].grid(linestyle='dotted')
ax1[2].legend(loc = 'upper right')


fig2, ax2 = plt.subplots(nrows=4, ncols=1, sharex=True, gridspec_kw={'height_ratios': [2,0.6,0.6,1]},
                         num=f'Raster plot, v_in={rate_in/Hz}', figsize=(8,10))

ax2[0].scatter(spikes_exc_mon_t[:]/second, spikes_exc_mon_i[:], color='C3', marker='|')
ax2[0].scatter(spikes_inh_mon_t[:]/second, spikes_inh_mon_i[:]+N_e, color='C0', marker='|')
ax2[0].set_ylabel('neuron index')
ax2[0].set_title('Raster plot')

ax2[1].plot(firing_rate_exc_t[trans:]/second, fr_exc[trans:], color='C3')
ax2[2].plot(firing_rate_inh_t[trans:]/second, fr_inh[trans:], color='C0')
ax2[1].set_ylabel('rate (Hz)')
ax2[2].set_ylabel('rate (Hz)')
ax2[1].grid(linestyle='dotted')
ax2[2].grid(linestyle='dotted')

ax2[3].plot(t_exc[trans:]/second, LFP[trans:], color='C5')
ax2[3].set_ylabel('LFP (mV)')
ax2[3].set_xlabel('time (s)')
ax2[3].grid(linestyle='dotted')

fig3, ax3 = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(12,6),
						num=f'Spectral_Density, v_in={rate_in/Hz}')

ax3[0].semilogy(f, Pxx_den, color='C8')
ax3[0].set_ylim([1e-4,1e-0])
ax3[0].set_xlim([-10, 300])
ax3[0].grid(linestyle='dotted')

ax3[1].plot(f, Pxx_den, color='C8')
ax3[1].set_xlim([-10, 300])
ax3[1].grid(linestyle='dotted')

plt.show()
