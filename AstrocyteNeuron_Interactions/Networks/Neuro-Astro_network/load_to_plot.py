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

parser = argparse.ArgumentParser(description='plot module for NA network')
parser.add_argument("-Ne", type=int, help="excitatory neurons")
parser.add_argument("-Ni", type=int, help="inhibitory neurons")
parser.add_argument("-Na", type=int, help="astrocytes")

args = parser.parse_args()
N_e = args.Ne
N_i = args.Ni
N_a = args.Na
C_Theta = 0.5*umolar

name=f'Network:Ne={N_e}_Ni={N_i}_Na={N_a}'+'_prof'

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



## PLOTS ##############################################

fig1, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(12, 14), num=f'Raster plot: Ne={N_e}_Ni={N_i}_Na={N_a}')
step = 10
ax1.plot(t_exc[exc_neurons_i%step==0], 
         exc_neurons_i[exc_neurons_i%step==0], '|', color='C3')
ax1.plot(t_inh[inh_neurons_i%step==0], 
         inh_neurons_i[inh_neurons_i%step==0]+N_e, '|', color='C0',)
ax1.plot(t_astro[astro_i%step==0], 
         astro_i[astro_i%step==0]+(N_e+N_i),'|' , color='green')
ax1.set_xlabel('time (s)')
ax1.set_ylabel('cell index')

fig2, ax2 = plt.subplots(nrows=7, ncols=1, sharex=True, figsize=(14, 14), num='astrocyte dynamics')
index_plot_list = [5400-4000]
for index_plot in index_plot_list:
    ax2[0].plot(t[:], Y_S[index_plot]/umolar, color='C3')
    ax2[0].set_ylabel(r'$Y_S$ ($\mu$M)')
    ax2[0].grid(linestyle='dotted')

    ax2[1].plot(t[:], Gamma_A[index_plot], color='C7')
    ax2[1].set_ylabel(r'$\Gamma_A$ ')
    ax2[1].grid(linestyle='dotted')

    ax2[2].plot(t[:], I[index_plot]/umolar, color='C5')
    ax2[2].set_ylabel(r'$I$ ($\mu$M)')
    ax2[2].grid(linestyle='dotted')

    ax2[3].plot(t[:], C[index_plot]/umolar, color='red')
    ax2[3].set_ylabel(r'$Ca^{2\plus}$ ($\mu$M)')
    ax2[3].plot(t[:], np.full(t.shape[0], C_Theta/umolar), ls='dashed', color='black')
    ax2[3].grid(linestyle='dotted')

    ax2[4].plot(t[:], h[index_plot], color='C6')
    ax2[4].set_ylabel(r'$h$')
    ax2[4].grid(linestyle='dotted')

    ax2[5].plot(t[:], G_A[index_plot], color='C7')
    ax2[5].set_ylabel(r'$G_A$')
    ax2[5].grid(linestyle='dotted')

    ax2[6].plot(t[:], x_A[index_plot], color='C8')
    ax2[6].set_ylabel(r'$x_A$')
    ax2[6].grid(linestyle='dotted')

plt.show()