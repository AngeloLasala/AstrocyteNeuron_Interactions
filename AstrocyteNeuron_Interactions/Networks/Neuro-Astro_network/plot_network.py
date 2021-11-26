"""
load variable and make intersting plot:
-Raster plot
-Variable dynamics
"""
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *

N_e = 3200
N_i = 800
N_a = 3200
C_Theta = 0.5*umolar

name=f'prova_con0.5_100_ph'

duration = np.load(f'{name}/duration.npy')*second

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

# I_stimulus = np.load(f'{name}/neurons_mon.I_stimulus.npy')
mon_v = np.load(f'{name}/neurons_mon.v.npy')
mon_g_e = np.load(f'{name}/neurons_mon.g_e.npy')
mon_g_i = np.load(f'{name}/neurons_mon.g_i.npy')
mon_t = np.load(f'{name}/neurons_mon.t.npy')

astro_connected = np.load(f'{name}/ecs_astro_to_syn.i.npy')
syn_connected = np.load(f'{name}/ecs_astro_to_syn.j.npy')


# for usefull connection plot
syn_connected_num = [len(astro_connected[astro_connected==i]) for i in np.arange(N_a)]

astro_connected_unique = np.unique(astro_connected)
print(f'connected astr: {astro_connected_unique.shape[0]} on 3200')

release_connected_astro = [np.where(astro_i==i)[0] for i in astro_connected_unique]
gliorelease_conn = np.array(t_astro[release_connected_astro][:,0])

print(f'gliorelease connected astro mean = {gliorelease_conn.mean():.2f}')
print(f'gliorelease connected astro std = {gliorelease_conn.std():.2f}')

# from raster plot and histogram of connected astrocyte emerge a second
# gruop of connected astrocyte with different gliorelease timing
gliorelease_second = np.unique(np.array([i for i in t_astro if i<2.75 and i>2.65]))
print(gliorelease_second)
gliorelease_second_pos =[]
for j in gliorelease_second:
    for i in np.where(t_astro==j)[0]: 
        gliorelease_second_pos.append(i)
# synaptically connected astrocytes fire in [2.6-2.8]s
astro_connected_second = np.array(gliorelease_second_pos)



# print(astro_i[:10])
# print(np.where(astro_i==0))
## PLOTS ##############################################

fig1, ax1 = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios': [3, 1]},
                        figsize=(12, 14), num=f'Raster plot file:{name}')
step = 1
ax1[0].plot(t_exc[exc_neurons_i%step==0]/ms, 
         exc_neurons_i[exc_neurons_i%step==0], '|', color='C3')
ax1[0].plot(t_inh[inh_neurons_i%step==0]/ms, 
         inh_neurons_i[inh_neurons_i%step==0]+N_e, '|', color='C0',)
ax1[0].plot(t_astro[astro_i%step==0]/ms, 
         astro_i[astro_i%step==0]+(N_e+N_i),'|' , color='green')
ax1[0].set_ylabel('cell index')

hist_step = 6
bin_size = (duration/ms)/((duration/ms)//hist_step)*ms

spk_count, bin_edges = np.histogram(np.r_[t_exc/ms,t_inh/ms], 
                                    int(duration/ms)//hist_step)
rate = double(spk_count)/(N_e+N_i)/bin_size
ax1[1].plot(bin_edges[:-1], rate, '-', color='k')
ax1[1].set_ylabel('rate (Hz)')
ax1[1].set_xlabel('time (ms)')
ax1[1].grid(linestyle='dotted')

fig2, ax2 = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(13, 9), 
                        num=f'astrocyte dynamics file:{name}')
index_plot_list = astro_connected_unique[540:560] #only syn connected astrocytes
print(index_plot_list)
for index_plot in index_plot_list:
    ax2[0].plot(t[:], Y_S[index_plot]/umolar, color='C3')
    ax2[0].set_ylabel(r'$Y_S$ ($\mu$M)')
    ax2[0].grid(linestyle='dotted')

    ax2[1].plot(t[:], Gamma_A[index_plot], color='C7')
    ax2[1].set_ylabel(r'$\Gamma_A$ ')
    ax2[1].grid(linestyle='dotted')

    ax2[2].plot(t[:], I[index_plot]/umolar, color='C0')
    ax2[2].set_ylabel(r'$I$ ($\mu$M)')
    ax2[2].grid(linestyle='dotted')

    ax2[3].plot(t[:], C[index_plot]/umolar, color='red')
    ax2[3].set_ylabel(r'$Ca^{2\plus}$ ($\mu$M)')
    ax2[3].plot(t[:], np.full(t.shape[0], C_Theta/umolar), ls='dashed', color='black')
    ax2[3].grid(linestyle='dotted')

fig3, ax3 = plt.subplots(nrows=1, ncols=1, 
                        num=f'gliorelease hist - connected astro file:{name}')
ax3.hist(gliorelease_conn, bins=20)

fig4, ax4 = plt.subplots(nrows=3, ncols=1, sharex=True, 
                        num=f'Neuronal variable dynamics')
ax4[0].plot(mon_t, mon_v[95]/mV)
ax4[1].plot(mon_t, mon_g_e[95]/nS)
ax4[2].plot(mon_t, mon_g_i[95]/nS)

plt.show()

