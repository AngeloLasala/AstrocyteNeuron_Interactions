"""
load network variables and makes some analysis and intersting plot:
- Raster plot
- Variable dynamics
- GRE distribuction 

Note: To obtain usefull information about network dynamics it is very usefull
run 'connectivity_analysis.py' to know advanced information of connectivity
"""
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
from network_analysis import transient

parser = argparse.ArgumentParser(description='Advanced connectivity connection')
parser.add_argument('file', type=str, help="file's name of network in 'Neuro_Astro_network' folder")
args = parser.parse_args()

## Load variables ######################################################################################
name=args.file

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
astro_to_syn_i = np.load(f'{name}/ecs_astro_to_syn.i.npy')
astro_to_syn_j = np.load(f'{name}/ecs_astro_to_syn.j.npy')
syn_to_astro_i = np.load(f'{name}/ecs_syn_to_astro.i.npy')
syn_to_astro_j = np.load(f'{name}/ecs_syn_to_astro.j.npy')

N_e = 3200
N_i = 800
N_a = 3200
C_Theta = 0.5*umolar
#######################################################################################################

## Analysis ##############################################################################
# Astro information 
# indeces of connected and free astrocites with synapses
astro_indeces = np.arange(N_a)
connected_astro = np.unique(astro_to_syn_i)
free_astro = astro_indeces[np.isin(astro_indeces,np.unique(astro_to_syn_i))==False]
print(f'connected astr: {len(connected_astro)} on 3200')

release_connected_astro = [np.where(astro_i==i)[0] for i in connected_astro]
gliorelease_conn = np.array(t_astro[release_connected_astro][:,0])

print(f'gliorelease connected astro mean = {gliorelease_conn.mean():.2f}')
print(f'gliorelease connected astro std = {gliorelease_conn.std():.2f}')

# # from raster plot and histogram of connected astrocyte emerge a second
# # gruop of connected astrocyte with different gliorelease timing
# print(t_astro)
# print(astro_i)
# gliorelease_second = np.unique(np.array([i for i in t_astro if i<2.54 and i>2.44]))
# print(gliorelease_second)
# t_astro_second = [np.where(t_astro==j) for j in gliorelease_second]
# gliorelease_second_pos =[]
# for j in gliorelease_second:
#     for i in np.where(t_astro==j)[0]: 
#         gliorelease_second_pos.append(i)
# # synaptically connected astrocytes fire in [2.6-2.8]s
# astro_connected_second = np.array(gliorelease_second_pos)
# print(f'astro firing second spot: {astro_connected_second}')


## PLOTS ######################################################################################
# transient time
trans = transient(t*second, 300)
print(f'transient: {trans}')

fig1, ax1 = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios': [3, 1]},
                        figsize=(12, 14), num=f'Raster plot')
step = 4
ax1[0].plot(t_exc[exc_neurons_i%step==0]/ms, 
         exc_neurons_i[exc_neurons_i%step==0], '|', color='C3')
ax1[0].plot(t_inh[inh_neurons_i%step==0]/ms, 
         inh_neurons_i[inh_neurons_i%step==0]+N_e, '|', color='C0',)
ax1[0].plot(t_astro[astro_i%step==0]/ms, 
         astro_i[astro_i%step==0]+(N_e+N_i),'|' , color='green')
ax1[0].set_ylabel('cell index')

hist_step = 5
bin_size = (duration/ms)/((duration/ms)//hist_step)*ms
print(bin_size)
spk_count, bin_edges = np.histogram(np.r_[t_exc/ms,t_inh/ms], 
                                    int(duration/ms)//hist_step)
rate = double(spk_count)/(N_e+N_i)/bin_size
ax1[1].plot(bin_edges[:-1], rate, '-', color='k')
ax1[1].set_ylabel('rate (Hz)')
ax1[1].set_xlabel('time (ms)')
ax1[1].grid(linestyle='dotted')

plt.savefig(name+f'Raster plot.png')

fig2, ax2 = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(13, 9), 
                        num=f'astrocyte dynamics')
con_index = connected_astro[0:1] # synaptically connected astrocytes
free_index = free_astro[0:1]     # not synaptically connected astrocytes

ax2[0].plot(t[trans:], Y_S[con_index][0,trans:]/umolar, color='C3', label='synaptically connected')
ax2[0].plot(t[trans:], Y_S[free_index][0,trans:]/umolar, color='C3', ls='dashed', label='free')
ax2[0].set_ylabel(r'$Y_S$ ($\mu$M)')
ax2[0].grid(linestyle='dotted')
ax2[0].legend()

ax2[1].plot(t[trans:], Gamma_A[con_index][0,trans:], color='C7', label='synaptically connected')
ax2[1].plot(t[trans:], Gamma_A[free_index][0,trans:], color='C7', ls='dashed', label='free')
ax2[1].set_ylabel(r'$\Gamma_A$ ')
ax2[1].grid(linestyle='dotted')
ax2[1].legend()

ax2[2].plot(t[trans:], I[con_index][0,trans:]/umolar, color='C0', label='synaptically connected')
ax2[2].plot(t[trans:], I[free_index][0,trans:]/umolar, color='C0', ls='dashed', label='free')
ax2[2].set_ylabel(r'$I$ ($\mu$M)')
ax2[2].grid(linestyle='dotted')
ax2[2].legend()

ax2[3].plot(t[trans:], C[con_index][0,trans:]/umolar, color='red', label='synaptically connected')
ax2[3].plot(t[trans:], C[free_index][0,trans:]/umolar, color='red', ls='dashed', label='free')
ax2[3].set_ylabel(r'$Ca^{2\plus}$ ($\mu$M)')
ax2[3].set_xlabel('time (s)')
ax2[3].plot(t[trans:], np.full(t[trans:].shape[0], C_Theta/umolar), ls='dashed', color='black')
ax2[3].grid(linestyle='dotted')
ax2[3].legend()

plt.savefig(name+f'astrocyte dynamics.png')

fig3, ax3 = plt.subplots(nrows=1, ncols=1, 
                        num=f'gliorelease hist - connected astro')
ax3.hist(gliorelease_conn, bins=20)
ax3.set_xlabel('time (s)')

plt.savefig(name+f'gliorelease hist - connected astro.png')

fig4, ax4 = plt.subplots(nrows=3, ncols=1, sharex=True, 
                        num=f'Neuronal variable dynamics')
ax4[0].plot(mon_t[trans:], mon_v[0,trans:]/mV)
ax4[0].set_ylabel('v (mV)')
ax4[1].plot(mon_t[trans:], mon_g_e[0,trans:]/nS)
ax4[1].set_ylabel(r'$g_e$ (nS)')
ax4[2].plot(mon_t[trans:], mon_g_i[0,trans:]/nS)
ax4[2].set_ylabel(r'$g_i$ (nS)')
ax4[2].set_xlabel('time (s)')

plt.savefig(name+f'Neuronal variable dynamics.png')

plt.show()

