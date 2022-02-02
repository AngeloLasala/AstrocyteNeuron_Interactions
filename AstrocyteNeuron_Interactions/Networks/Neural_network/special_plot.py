"""
Additional code to generate specific plots
"""
import numpy as np 
import matplotlib.pyplot as plt

rate = 44.0
trans = np.load(f'EI_net_noSTP/Network_pe_v_in{rate}/trans.npy')
t =np.load(f'EI_net_noSTP/Network_pe_v_in{rate}/state_exc_mon.t.npy')
LFP_nobalance = np.load(f'EI_net_noSTP/Network_pe_v_in{rate}/state_exc_mon.LFP.npy').sum(axis=0)
LFP_balance = np.load(f'EI_net_noSTP_balancenoSTP/Network_pe_v_in{rate}/state_exc_mon.LFP.npy').sum(axis=0)

f_LFP_nobalance = np.load(f'Spectral_analysis/noSTP/net_v_in{rate}/f_LFP.npy')
f_LFP_balance = np.load(f'Spectral_analysis/balancenoSTP/net_v_in{rate}/f_LFP.npy')
spect_den_LFP_nobalance = np.load(f'Spectral_analysis/noSTP/net_v_in{rate}/spect_den_LFP.npy')
spect_den_LFP_balance = np.load(f'Spectral_analysis/balancenoSTP/net_v_in{rate}/spect_den_LFP.npy')

## Plots ################################################################################################
fig1, ax1 = plt.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': [4.3,2.2]}, figsize=(15,6),
						num=f'Compare balanced and unbalanced networks, rate input {rate}')

ax1[0].plot(t[trans:], LFP_nobalance[trans:], lw=1, label='unbalanced')
ax1[0].plot(t[trans:], LFP_balance[trans:],lw=1, label='balanced')
ax1[0].set_ylabel('LFP (mV)')
ax1[0].set_xlabel('time (s)')
ax1[0].grid(linestyle='dotted')
ax1[0].legend()

ax1[1].plot(f_LFP_nobalance, spect_den_LFP_nobalance)
ax1[1].plot(f_LFP_nobalance, spect_den_LFP_balance)
ax1[1].set_xlabel('freq (Hz)')
ax1[1].set_xlim([-10,300])
ax1[1].grid(linestyle='dotted')

plt.show()