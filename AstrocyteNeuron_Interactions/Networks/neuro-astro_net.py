"""
Coupling neurons and astrocytes network

Randomly connected COBA network  with excitatory synapses modulated
by release-increasing gliotransmission from a connected network of astrocytes.

- "Modelling neuro-glia interactions with the Brian2 simulator" Stimberg et al (2017)
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
from AstrocyteNeuron_Interactions.Brian2_utils.connectivity import connectivity_plot,connectivity_EIring
from AstrocyteNeuron_Interactions import makedir
## PARAMETERS ###################################################################
# --  General parameters --
N_e = 3200                    # Number of excitatory neurons
N_i = 800                     # Number of inhibitory neurons
N_a = 3200                    # Number of astrocytes

# -- Some metrics parameters needed to establish proper connections --
size = 3.75*mmeter           # Length and width of the square lattice
distance = 50*umeter         # Distance between neurons

# -- Neuron --
E_l = -60*mV                 # Leak reversal potential
g_l = 9.99*nS                # Leak conductance
E_e = 0*mV                   # Excitatory synaptic reversal potential
E_i = -80*mV                 # Inhibitory synaptic reversal potential
C_m = 198*pF                 # Membrane capacitance
tau_e = 5*ms                 # Excitatory synaptic time constant
tau_i = 10*ms                # Inhibitory synaptic time constant
tau_r = 5*ms                 # Refractory period
I_ex = 100*pA                # External current
V_th = -50*mV                # Firing threshold
V_r = E_l                    # Reset potential

# -- Synapse --
rho_c = 0.005                # Synaptic vesicle-to-extracellular space volume ratio
Y_T = 500.*mmolar            # Total vesicular neurotransmitter concentration
Omega_c = 40/second          # Neurotransmitter clearance rate
U_0__star = 0.6              # Resting synaptic release probability
Omega_f = 3.33/second        # Synaptic facilitation rate
Omega_d = 2.0/second         # Synaptic depression rate
w_e = 0.05*nS                # Excitatory synaptic conductance
w_i = 1.0*nS                 # Inhibitory synaptic conductance
# - Presynaptic receptors
O_G = 1.5/umolar/second      # Agonist binding (activating) rate
Omega_G = 0.5/(60*second)    # Agonist release (deactivating) rate

# -- Astrocyte --
# CICR
O_P = 0.9*umolar/second      # Maximal Ca^2+ uptake rate by SERCAs
K_P = 0.05*umolar            # Ca2+ affinity of SERCAs
C_T = 2*umolar               # Total cell free Ca^2+ content
rho_A = 0.18                 # ER-to-cytoplasm volume ratio
Omega_C = 6/second           # Maximal rate of Ca^2+ release by IP_3Rs
Omega_L = 0.1/second         # Maximal rate of Ca^2+ leak from the ER
d_1 = 0.13*umolar            # IP_3 binding affinity
d_2 = 1.05*umolar            # Ca^2+ inactivation dissociation constant
O_2 = 0.2/umolar/second      # IP_3R binding rate for Ca^2+ inhibition
d_3 = 0.9434*umolar          # IP_3 dissociation constant
d_5 = 0.08*umolar            # Ca^2+ activation dissociation constant
#  IP_3 production
# Agonist-dependent IP_3 production
O_beta = 0.5*umolar/second   # Maximal rate of IP_3 production by PLCbeta
O_N = 0.3/umolar/second      # Agonist binding rate
Omega_N = 0.5/second         # Maximal inactivation rate
K_KC = 0.5*umolar            # Ca^2+ affinity of PKC
zeta = 10                    # Maximal reduction of receptor affinity by PKC
# Endogenous IP3 production
O_delta = 1.2*umolar/second  # Maximal rate of IP_3 production by PLCdelta
kappa_delta = 1.5*umolar     # Inhibition constant of PLC_delta by IP_3
K_delta = 0.1*umolar         # Ca^2+ affinity of PLCdelta
# IP_3 degradation
Omega_5P = 0.05/second       # Maximal rate of IP_3 degradation by IP-5P
K_D = 0.7*umolar             # Ca^2+ affinity of IP3-3K
K_3K = 1.0*umolar            # IP_3 affinity of IP_3-3K
O_3K = 4.5*umolar/second     # Maximal rate of IP_3 degradation by IP_3-3K
# IP_3 diffusion (astrocyte coupling)
F = 0.09*umolar/second       # GJC IP_3 permeability
I_Theta = 0.3*umolar         # Threshold gradient for IP_3 diffusion
omega_I = 0.05*umolar        # Scaling factor of diffusion
# Gliotransmitter release and time course
C_Theta = 0.5*umolar         # Ca^2+ threshold for exocytosis
Omega_A = 0.6/second         # Gliotransmitter recycling rate
U_A = 0.6                    # Gliotransmitter release probability
G_T = 200*mmolar             # Total vesicular gliotransmitter concentration
rho_e = 6.5e-4               # Astrocytic vesicle-to-extracellular volume ratio
Omega_e = 60/second          # Gliotransmitter clearance rate
alpha = 0.0                  # Gliotransmission nature
#################################################################################

## TIME PARAMETERS ##############################################################
defaultclock.dt = 0.1*ms
seed(28371)  # to get identical figures for repeated runs

dt_stim = 2*second
stimulus = TimedArray([1.0,1.2,1.0,1.0], dt=dt_stim)
duration = 4*dt_stim
#################################################################################

## NETWORK #####################################################################
## NEURONS 
neuron_eqs = """
# Neurons dynamics
dv/dt = (g_l*(E_l-v) + g_e*(E_e-v) + g_i*(E_i-v) + I_ex*stimulus(t))/C_m : volt (unless refractory)
dg_e/dt = -g_e/tau_e : siemens  # post-synaptic excitatory conductance
dg_i/dt = -g_i/tau_i : siemens  # post-synaptic inhibitory conductance

# Neuron position in space
x : meter (constant)
y : meter (constant)
"""

neurons = NeuronGroup(N_e+N_i, model=neuron_eqs, method='euler',
                    threshold='v>V_th', reset='v=V_r', refractory='tau_r')

exc_neurons = neurons[:N_e]
inh_neurons = neurons[N_e:]

# Arrange excitatory neurons in a grid
N_rows_exc = int(sqrt(N_e))
N_cols_exc = N_e/N_rows_exc
grid_dist = (size / N_cols_exc)
#square grid
xx = np.arange(N_rows_exc)
yy = np.arange(N_cols_exc)
XX,YY = np.meshgrid(xx,yy)

exc_neurons.x = XX.flatten()[:N_e]*grid_dist
exc_neurons.y = YY.flatten()[:N_e]*grid_dist
# exc_neurons.x = '(i // N_rows_exc)*grid_dist - N_rows_exc/2.0*grid_dist'
# exc_neurons.y = '(i % N_rows_exc)*grid_dist - N_cols_exc/2.0*grid_dist'

# Random initial membrane potential values and conductances
neurons.v = 'E_l + rand()*(V_th-E_l)'
neurons.g_e = 'rand()*w_e'
neurons.g_i = 'rand()*w_i'

# SYNAPSE
#Synapses
syn_model = """
du_S/dt = -Omega_f * u_S                           : 1 (event-driven)
dx_S/dt = Omega_d * (1-x_S)                        : 1 (event-driven)
dY_S/dt = -Omega_c*Y_S                             : mmolar (clock-driven)
dGamma_S/dt = O_G*G_A*(1-Gamma_S)-Omega_G*Gamma_S  : 1 (clock-driven)
G_A                                                : mmolar
r_S                                                : 1
U_0                                                : 1
# which astrocyte covers this synapse ?
astrocyte_index : integer (constant)
"""

syn_action = """
U_0 = (1 - Gamma_S) * U_0__star + alpha * Gamma_S
u_S += U_0 * (1 - u_S)
r_S = u_S * x_S
x_S -= r_S
Y_S += rho_c * Y_T * r_S
"""

exc_act="g_e_post+=w_e*r_S"
inh_act="g_i_post+=w_i*r_S"

exc_syn = Synapses(exc_neurons, neurons, model= syn_model, on_pre=syn_action+exc_act, method='linear')
exc_syn.connect(True, p=0.05)
exc_syn.x_S = 1.0

inh_syn = Synapses(inh_neurons, neurons, model= syn_model, on_pre=syn_action+inh_act, method='linear')
inh_syn.connect(True, p=0.2)
inh_syn.x_S = 1.0

# Connect excitatory synapses to an astrocyte depending on the position of the
# post-synaptic neuron
N_rows_astro = int(sqrt(N_a))
N_cols_astro = N_a/N_rows_astro
grid_dist = (size / N_rows_astro)
exc_syn.astrocyte_index = ('int(x_post/grid_dist) + '
                            'N_cols_astro*int(y_post/grid_dist)')

# ASTROCYTE
astro_eqs = """
# Fraction of activated astrocyte receptors (1):
dGamma_A/dt = O_N * Y_S * (1 - clip(Gamma_A,0,1)) -
            Omega_N*(1 + zeta * C/(C + K_KC)) * clip(Gamma_A,0,1) : 1

# IP_3 dynamics (1)
dI/dt = J_beta + J_delta - J_3K - J_5P + J_coupling              : mmolar

J_beta = O_beta * Gamma_A                                        : mmolar/second
J_delta = O_delta/(1 + I/kappa_delta) * C**2/(C**2 + K_delta**2) : mmolar/second
J_3K = O_3K * C**4/(C**4 + K_D**4) * I/(I + K_3K)                : mmolar/second
J_5P = Omega_5P*I                                                : mmolar/second
# Diffusion between astrocytes (1):
J_coupling                                                       : mmolar/second

# Calcium dynamics (2):
dC/dt = J_r + J_l - J_p                                   : mmolar
dh/dt = (h_inf - h)/tau_h                                 : 1

J_r = (Omega_C * m_inf**3 * h**3) * (C_T - (1 + rho_A)*C) : mmolar/second
J_l = Omega_L * (C_T - (1 + rho_A)*C)                     : mmolar/second
J_p = O_P * C**2/(C**2 + K_P**2)                          : mmolar/second
m_inf = I/(I + d_1) * C/(C + d_5)                         : 1
h_inf = Q_2/(Q_2 + C)                                     : 1
tau_h = 1/(O_2 * (Q_2 + C))                               : second
Q_2 = d_2 * (I + d_1)/(I + d_3)                           : mmolar

# Fraction of gliotransmitter resources available for release (1):
dx_A/dt = Omega_A * (1 - x_A) : 1
# gliotransmitter concentration in the extracellular space (1):
dG_A/dt = -Omega_e*G_A        : mmolar

# Neurotransmitter concentration in the extracellular space:
Y_S                           : mmolar

# The astrocyte position in space
x : meter (constant)
y : meter (constant)
"""

astro_release = """
G_A += rho_e*G_T*U_A*x_A
x_A -= U_A * x_A
"""

astrocyte = NeuronGroup(N_a, astro_eqs, 
                        threshold='C>C_Theta', refractory='C>C_Theta', reset=astro_release,
                        method='rk4', dt=1e-2*second)

# Arrange excitatory neurons in a grid
#square grid
x_astro = np.arange(N_rows_astro)
y_astro = np.arange(N_cols_astro)
XX_A,YY_A = np.meshgrid(x_astro,y_astro)

astrocyte.x = XX_A.flatten()[:N_a]*grid_dist
astrocyte.y = YY_A.flatten()[:N_a]*grid_dist
# astrocyte.x = '(i // N_rows_astro)*grid_dist - N_rows_astro/2.0*grid_dist'
# astrocyte.y = '(i % N_rows_astro)*grid_dist - N_cols_astro/2.0*grid_dist'


astrocyte.C =0.01*umolar
astrocyte.h = 0.9
astrocyte.I = 0.01*umolar
astrocyte.x_A = 1.0

# bidirectional connection beetwith astrocyte and excitatory synapses
# based on postsynaptic neurons position
# ASTRO TO EXC_SYNAPSES
ecs_astro_to_syn = Synapses(astrocyte, exc_syn, 'G_A_post = G_A_pre : mmolar (summed)')
ecs_astro_to_syn.connect('i == astrocyte_index_post')
print('Astro to Syn conn')
print(ecs_astro_to_syn.i[:])
print(ecs_astro_to_syn.j[:])

#EXC_SYNAPSES TO ASTRO
ecs_syn_to_astro = Synapses(exc_syn, astrocyte, 'Y_S_post = Y_S_pre/N_incoming : mmolar (summed)')
ecs_syn_to_astro.connect('astrocyte_index_pre == j')

# Diffusion between astrocytes
astro_to_astro_eqs = """
delta_I = I_post - I_pre            : mmolar
J_coupling_post = -(1 + tanh((abs(delta_I) - I_Theta)/omega_I))*
                sign(delta_I)*F/2 : mmolar/second (summed)
"""
astro_to_astro = Synapses(astrocyte, astrocyte,
                        model=astro_to_astro_eqs)
# Connect to all astrocytes less than 75um away
# (about 4 connections per astrocyte)
astro_to_astro.connect('i != j and '
                        'sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < 75*um')

##########################################################################################

## MOMITOR ###############################################################################
spikes_exc_mon = SpikeMonitor(exc_neurons)
spikes_inh_mon = SpikeMonitor(inh_neurons)
astro_mon = SpikeMonitor(astrocyte)
var_astro_mon = StateMonitor(astrocyte, ['C','I','h','Gamma_A','Y_S','G_A','x_A'], record=True)
###########################################################################################

## RUN and NETWORK INFORMATION ###################################################################
run(duration, report='text')
print(exc_syn)

print('\n NETWORK INFORMATION')
print(f'excitatory neurons = {N_e}')
print(f'inhibitory neurons = {N_i}')
print(f'excitatory synapses = {len(exc_syn.i)}')
print(f'inhibitory synapses = {len(inh_syn.i)}')
print('_______________\n')
print(f'astrocytes = {N_a}')
print(f'syn to astro connection = {len(ecs_syn_to_astro.i)}')
print(f'astro to syn connection = {len(ecs_astro_to_syn.i)}\n')
print('_______________\n')
print('Spatial arrangement')
print(f'neurons grid:   {N_rows_exc}x{N_rows_exc} dist={grid_dist/umetre} um')
print(f'astrocyte grid: {N_rows_astro}x{N_rows_astro} dist={grid_dist/umetre} um\n')
##################################################################################################

## SAVE IMPORTANT VALUES #########################################################################
name = f'Neuro-Astro_network/Network:Ne={N_e}_Ni={N_i}_Na={N_a}_mygrid'
makedir.smart_makedir(name)

# Duration
np.save(f'{name}/duration',duration)

# Raster plot
np.save(f'{name}/spikes_exc_mon.t',spikes_exc_mon.t)
np.save(f'{name}/spikes_exc_mon.i',spikes_exc_mon.i)
np.save(f'{name}/spikes_inh_mon.t',spikes_inh_mon.t)
np.save(f'{name}/spikes_inh_mon.i',spikes_inh_mon.i)
np.save(f'{name}/astro_mon.t',astro_mon.t)
np.save(f'{name}/astro_mon.i',astro_mon.i)

# Astrocte variables dynamics
np.save(f'{name}/var_astro_mon.t',var_astro_mon.t)
np.save(f'{name}/var_astro_mon.Y_S',var_astro_mon.Y_S)
np.save(f'{name}/var_astro_mon.Gamma_A',var_astro_mon.Gamma_A)
np.save(f'{name}/var_astro_mon.I',var_astro_mon.I)
np.save(f'{name}/var_astro_mon.C',var_astro_mon.C)
np.save(f'{name}/var_astro_mon.h',var_astro_mon.h)
np.save(f'{name}/var_astro_mon.x_A',var_astro_mon.x_A)
np.save(f'{name}/var_astro_mon.G_A',var_astro_mon.G_A)

# Network Structure
with open(f"Neuro-Astro_network/Network:Ne={N_e}_Ni={N_i}_Na={N_a}/network_structure.txt",
         'w', encoding='utf-8') as file:
        file.write(f"""NETWORK INFORMATION \n
excitatory neurons = {N_e}
inhibitory neurons = {N_i}
excitatory synapses = {len(exc_syn.i)}
inhibitory synapses = {len(inh_syn.i)}
________________________________________\n
astrocytes = {N_a}
syn to astro connection = {len(ecs_syn_to_astro.i)}
astro to syn connection = {len(ecs_astro_to_syn.i)}
___________________________________________\n
Spatial arrangement
neurons grid:   {N_rows_exc}x{N_rows_exc} dist={grid_dist/umetre} um
astrocyte grid: {N_rows_astro}x{N_rows_astro} dist={grid_dist/umetre} um""")
###################################################################################################

## PLOTS #########################################################################################
fig1, ax1 = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios': [3, 1]},
                        figsize=(12, 14), num=f'Raster plot: Ne={N_e} Ni={N_i}, Na={N_a}')
step = 1
ax1[0].plot(spikes_exc_mon.t[np.array(spikes_exc_mon.i)%step==0]/ms, 
            spikes_exc_mon.i[np.array(spikes_exc_mon.i)%step==0], '|', color='C3')
ax1[0].plot(spikes_inh_mon.t[np.array(spikes_inh_mon.i)%step==0]/ms, 
            spikes_inh_mon.i[np.array(spikes_inh_mon.i)%step==0]+N_e, '|', color='C0',)
ax1[0].plot(astro_mon.t[np.array(astro_mon.i)%step==0]/ms, 
            astro_mon.i[np.array(astro_mon.i)%step==0]+(N_e+N_i),'|' , color='green')
ax1[0].set_xlabel('time (s)')
ax1[0].set_ylabel('cell index')

hist_step = 1
bin_size = (duration/ms)/((duration/ms)//hist_step)*ms
spk_count, bin_edges = np.histogram(np.r_[spikes_exc_mon.t/ms,spikes_inh_mon.t/ms], 
                                    int(duration/ms)//hist_step)
rate = double(spk_count)/(N_e+N_i)/bin_size
ax1[1].plot(bin_edges[:-1], rate, '-', color='k')
ax1[1].set_ylabel('rate (Hz)')
ax1[1].set_xlabel('time (ms)')
ax1[1].grid(linestyle='dotted')

fig2, ax2 = plt.subplots(nrows=7, ncols=1, sharex=True, figsize=(14, 14), num='astrocyte dynamics')
index_plot = 0
ax2[0].plot(var_astro_mon.t[:], var_astro_mon.Y_S[index_plot]/umolar, color='C3')
ax2[0].set_ylabel(r'$Y_S$ ($\mu$M)')
ax2[0].grid(linestyle='dotted')

ax2[1].plot(var_astro_mon.t[:], var_astro_mon.Gamma_A[index_plot], color='C7')
ax2[1].set_ylabel(r'$\Gamma_A$ ')
ax2[1].grid(linestyle='dotted')

ax2[2].plot(var_astro_mon.t[:], var_astro_mon.I[index_plot]/umolar, color='C5')
ax2[2].set_ylabel(r'$I$ ($\mu$M)')
ax2[2].grid(linestyle='dotted')

ax2[3].plot(var_astro_mon.t[:], var_astro_mon.C[index_plot]/umolar, color='red')
ax2[3].set_ylabel(r'$Ca^{2\plus}$ ($\mu$M)')
ax2[3].axhline(C_Theta/umolar,0,duration/second, ls='dashed', color='black')
ax2[3].grid(linestyle='dotted')

ax2[4].plot(var_astro_mon.t[:], var_astro_mon.h[index_plot], color='C6')
ax2[4].set_ylabel(r'$h$')
ax2[4].grid(linestyle='dotted')

ax2[5].plot(var_astro_mon.t[:], var_astro_mon.G_A[index_plot], color='C7')
ax2[5].set_ylabel(r'$G_A$')
ax2[5].grid(linestyle='dotted')

ax2[6].plot(var_astro_mon.t[:], var_astro_mon.x_A[index_plot], color='C8')
ax2[6].set_ylabel(r'$x_A$')
ax2[6].grid(linestyle='dotted')

# connectivity_EIring(exc_syn, inh_syn, size=15, lw=0.5, split=False)
# connectivity_EIring(ecs_astro_to_syn, ecs_syn_to_astro, size=15, lw=0.5, split=False)
# connectivity_plot(exc_syn, source='Exc', target='Exc+Inh', color_s='red', color_t='indigo', size=10, name='exc syn')
# Connectivity_plot(inh_syn, source='Inh', target='Exc+Inh', color_s='C0', color_t='indigo', size=10)
# connectivity_plot(ecs_astro_to_syn, source='Astro', target='Exc syn',   
#                   color_s='green', color_t='red', size=15, lw=0.5, name='stro_to_syn')
# connectivity_plot(ecs_syn_to_astro, source='Exc syn', target='Astro',  
#                   color_s='red', color_t='green', size=15, lw=0.5, name='syn_to_astro')

# plt.figure(num='N_e grid')
# plt.scatter(exc_neurons.x/mmeter, exc_neurons.y/mmeter)
# plt.scatter(exc_syn.x_pre/mmeter, exc_syn.y_pre/mmetre, label='pre')
# plt.legend()

# plt.figure(num='N_e grid_1')
# plt.scatter(exc_neurons.x/mmeter, exc_neurons.y/mmeter)
# plt.scatter(exc_syn.x_post/mmeter, exc_syn.y_post/mmetre, label='post')
# plt.legend()

# plt.figure(num='Astro grid')
# plt.scatter(astrocyte.x/mmeter, astrocyte.y/mmeter)
# plt.legend()

plt.show()
