"""
Intersting plots concern population firing rate in the case og stroger external input
on the inhibitory population
"""
import numpy as np 
import matplotlib.pyplot as plt
from brian2 import *

## load variable ##########################################################################
g = 7.5
alpha = 1.0
name_folder = f'EI_net_noSTP_ext/EI_net_strong_I_input_g{g}_alpha{alpha}/Network_pe_v_in'

## Comparison of two signals
v_in_range = [47.7, 58.8]
fig1, ax1 = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12,5),
							num=f'I and E firing rate (no STP) g={g}')
for i, v_in in enumerate(v_in_range):
	name = name_folder+str(v_in)
	print(v_in,i)

	trans = np.load(f'{name}/trans.npy')

	# Population istantaneus firing rate
	firing_rate_exc_t= np.load(f'{name}/firing_rate_exc.t.npy')
	firing_rate_inh_t = np.load(f'{name}/firing_rate_inh.t.npy')
	fr_exc = np.load(f'{name}/fr_exc.npy')
	fr_inh = np.load(f'{name}/fr_inh.npy')
	
	ax1[i].plot(firing_rate_exc_t[trans:]/second, fr_exc[trans:], color='C3', label='E')
	ax1[i].plot(firing_rate_inh_t[trans:]/second, fr_inh[trans:], color='C0', label='I')
	ax1[i].set_ylabel('rate (Hz)')
	ax1[i].set_xlabel('time (s)')
	ax1[i].set_title(f'Input rate {v_in}')
	ax1[i].set_xlim([0.25,2.4])
	ax1[i].grid(linestyle='dotted')


## Average populations activity vs v_iv

## Average spiking activity
v_in_range = [47.7, 54.0, 58.8, 64.0, 74.0, 84.0]
Delta_list, Ratio_list = [], []
fr_exc_list, fr_inh_list = [], []

fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=(12,5),
							num=f'Average FR vs v_in g={g}')
for i,v_in in enumerate(v_in_range):
	name = name_folder+str(v_in)
	name_nostrong = 'EI_net_noSTP_balancenoSTP/Network_pe_v_in'+str(v_in)
	name_g025 = 'EI_net_noSTP/Network_pe_v_in'+str(v_in)

	# fr_exc_g025 = np.load(f'{name_g025}/fr_exc.npy')
	# fr_inh_g025 = np.load(f'{name_g025}/fr_inh.npy')
	fr_exc = np.load(f'{name}/fr_exc.npy')
	fr_inh = np.load(f'{name}/fr_inh.npy')

	Delta_fr = fr_inh.mean()-fr_exc.mean()
	Ratio_fr = fr_inh.mean()/fr_exc.mean()
	Delta_list.append(Delta_fr )
	fr_exc_list.append(fr_exc.mean())
	fr_inh_list.append(fr_inh.mean())

	# if v_in==47.7 or v_in==58.8:
	# 	fr_exc_nostrong = np.load(f'{name_nostrong}/fr_exc.npy')
	# 	fr_inh_nostrong = np.load(f'{name_nostrong}/fr_inh.npy')	
	# 	ax2[0].plot(v_in, fr_exc_nostrong.mean(), ls='', marker="o", color='C3')
	# 	ax2[0].plot(v_in, fr_inh_nostrong.mean(), ls='', marker="s", color='C0')
	# ax2[0].plot(v_in, fr_exc_g025.mean(), ls='', marker="o", color='C4')
	# ax2[0].plot(v_in, fr_inh_g025.mean(), ls='', marker="s", color='C5')
	
	ax2[1].plot(v_in, fr_inh.mean()/fr_exc.mean(), ls='', marker='o', color='C0')
	ax2[1].set_xlabel(r'$\nu_{in}$ (Hz)')
	ax2[1].set_ylabel(r'$I_{FR}$/$E_{FR}$')
	ax2[1].grid(linestyle='dotted')

ax2[0].plot(v_in_range, fr_exc_list, ls='', marker="$\u25EF$", color='C3', label='E')
ax2[0].plot(v_in_range, fr_inh_list, ls='', marker="$\u25A1$", color='C0', label='I')
ax2[0].set_xlabel(r'$\nu_{in}$ (Hz)')
ax2[0].set_ylabel('Mean FR (Hz)')
ax2[0].grid(linestyle='dotted')
ax2[0].legend()

print(f'Delta: {Delta_list}')
print(f'Delta mean: {np.array(Delta_list).mean()}')

## All g in a plot
v_in_range = [47.7, 54.0, 58.8, 64.0, 74.0, 84.0]
fr_exc_list_5, fr_inh_list_5 = [], []
fr_exc_list_2, fr_inh_list_2 = [], []
fr_exc_list_7, fr_inh_list_7 = [], []

for v_in in v_in_range:
	name2 = f'EI_net_noSTP_ext/EI_net_strong_I_input_g2.5_alpha{alpha}/Network_pe_v_in'+str(v_in)
	name5 = f'EI_net_noSTP_ext/EI_net_strong_I_input_g5.0_alpha{alpha}/Network_pe_v_in'+str(v_in)
	name7 = f'EI_net_noSTP_ext/EI_net_strong_I_input_g7.5_alpha{alpha}/Network_pe_v_in'+str(v_in)

	fr_exc_g2 = np.load(f'{name2}/fr_exc.npy').mean()
	fr_inh_g2 = np.load(f'{name2}/fr_inh.npy').mean()
	fr_exc_g5 = np.load(f'{name5}/fr_exc.npy').mean()
	fr_inh_g5 = np.load(f'{name5}/fr_inh.npy').mean()
	fr_exc_g7 = np.load(f'{name7}/fr_exc.npy').mean()
	fr_inh_g7 = np.load(f'{name7}/fr_inh.npy').mean()

	fr_exc_list_2.append(fr_exc_g2)
	fr_inh_list_2.append(fr_inh_g2)
	fr_exc_list_5.append(fr_exc_g5)
	fr_inh_list_5.append(fr_inh_g5)
	fr_exc_list_7.append(fr_exc_g7)
	fr_inh_list_7.append(fr_inh_g7)

fig3, ax3 = plt.subplots(nrows=1, ncols=1,
							num=f'Compare Average FR vs v_in all g and alpha{alpha}')

ax3.set_title(r'$\alpha=$'+f' {alpha}')						
ax3.plot(v_in_range, fr_exc_list_2, marker="$\u25EF$", color='C1', label='g=2.5')
ax3.plot(v_in_range, fr_inh_list_2, marker="$\u25A1$", color='C1')
ax3.plot(v_in_range, fr_exc_list_5,ls='--', marker="$\u25EF$", color='C2', label='g=5.0')
ax3.plot(v_in_range, fr_inh_list_5,ls='--', marker="$\u25A1$", color='C2')
ax3.plot(v_in_range, fr_exc_list_7,ls='dotted', marker="$\u25EF$", color='C4', label='g=7.5')
ax3.plot(v_in_range, fr_inh_list_7,ls='dotted', marker="$\u25A1$", color='C4')
ax3.set_xlabel(r'$\nu_{in}$ (Hz)')
ax3.set_ylabel('Mean FR (Hz)')
ax3.grid(linestyle='dotted')
ax3.legend()


plt.show()