"""
Local circuit: 160 exc synapses inpigging single naive neurons to moinitor 
post synamptic conductance time course.
Neuron, synaptic and astrocyte parameters are equal to ones presente in
network.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from brian2 import *
from AstrocyteNeuron_Interactions import makedir

set_device('cpp_standalone', directory=None)  # Use fast "C++ standalone mode"

def jumps(conductance):
	"""
	Conductance jumps due to presynaptic action. The type of synaptic 
	(exc or inh) depends on input: g_i stands for presinaptic inhibitory 
	and g_e stands for excitatory.

	Parameters
	----------
	conductance: 1d array of conductance variable, for istance g_e_STP[index,trans:]

	Returns
	-------
	jump: array. Conductance jumps
	"""
	max_loc=argrelextrema(conductance, np.greater)
	min_loc=argrelextrema(conductance, np.less)

	if len(max_loc[0]) > len(min_loc[0]):
		jump = conductance[max_loc[0][:-1]] - conductance[min_loc]

	elif len(max_loc[0]) < len(min_loc[0]):
		jump = conductance[max_loc] - conductance[min_loc[0][:-1]]

	else:
		jump = conductance[max_loc] - conductance[min_loc]

	return jump

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Tripartite synapses')
	parser.add_argument('rate_in', type=float, help='presynaptic input rate: 0.12, 2.09, 7.7 Hz ' )
	parser.add_argument('-p', action='store_true', help="show paramount plots, default=False")
	args = parser.parse_args()

	## PARAMETERS ###################################################################
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
	defaultclock.dt = 0.5*ms
	duration = 250*second
	seed(28371)  # to get identical figures for repeated runs
	#################################################################################

	## SYNAPSES
	syn_model = """
	# Neurotrasmitter
	dY_S/dt = -Omega_c*Y_S                             : mmolar (clock-driven)

	# Fraction of activated astrocyte
	dGamma_S/dt = O_G*G_A*(1-Gamma_S)-Omega_G*Gamma_S  : 1 (clock-driven)
	G_A : mmolar

	# Synaptic variable
	du_S/dt = -Omega_f * u_S : 1 (clock-driven)
	dx_S/dt = Omega_d * (1-x_S) : 1 (clock-driven)
	r_S : 1
	U_0 : 1
	"""

	action="""
	U_0 = (1 - Gamma_S) * U_0__star + alpha * Gamma_S
	u_S += U_0*(1-u_S)
	r_S = u_S*x_S
	x_S -= r_S
	Y_S += rho_c * Y_T * r_S
	"""

	exc="g_e_post+=w_e*r_S"
	inh="g_i_post+=w_i*r_S"

	## Astrocyte
	astro_eqs = """
    # Fraction of activated astrocyte receptors (1):
    dGamma_A/dt = O_N * Y_S * (1 - clip(Gamma_A,0,1)) -
                Omega_N*(1 + zeta * C/(C + K_KC)) * clip(Gamma_A,0,1) : 1

    # IP_3 dynamics (1)
    dI/dt = J_beta + J_delta - J_3K - J_5P                           : mmolar

    J_beta = O_beta * Gamma_A                                        : mmolar/second
    J_delta = O_delta/(1 + I/kappa_delta) * C**2/(C**2 + K_delta**2) : mmolar/second
    J_3K = O_3K * C**4/(C**4 + K_D**4) * I/(I + K_3K)                : mmolar/second
    J_5P = Omega_5P*I                                                : mmolar/second

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
    """

	astro_release = """
    G_A += rho_e*G_T*U_A*x_A
    x_A -= U_A * x_A
    """
	## Model 
	N_exc = 160
	N_post_syn= 1
	rate_in = args.rate_in*Hz  #[0.12, 2.09, 7.7]

	pre_neurons = PoissonGroup(N_exc, rates=rate_in)
	post_neurons = NeuronGroup(2*N_post_syn, model="dg_e/dt = -g_e/tau_e : siemens # post-synaptic excitatory conductance",
					method='rk4')

	synapses = Synapses(pre_neurons, post_neurons, model=syn_model, on_pre=action+exc, method='linear')
	synapses.connect(i=np.arange(N_exc, dtype=int64), j=np.zeros(N_exc, dtype=int64)) 
	synapses.connect(i=np.arange(N_exc, dtype=int64), j=np.ones(N_exc, dtype=int64)) 
	synapses.x_S = 1.0


	astrocyte = NeuronGroup(N_exc, model=astro_eqs, method='rk4', dt=1e-2*second,
	                        threshold='C>C_Theta', refractory='C>C_Theta', reset=astro_release)
	astrocyte.x_A = 1.0
	astrocyte.h = 0.9
	astrocyte.I = 0.01*umolar
	astrocyte.C = 0.01*umolar

	# ASTRO TO EXC_SYNAPSES
	ecs_astro_to_syn = Synapses(astrocyte, synapses, 'G_A_post = G_A_pre : mmolar (summed)')
	ecs_astro_to_syn.connect(j='i if j<N_exc')

	# EXC_SYNAPSES TO ASTRO
	ecs_syn_to_astro = Synapses(synapses, astrocyte, 'Y_S_post = Y_S_pre : mmolar (summed)')
	ecs_syn_to_astro.connect(j='i if i<N_exc')

	#Monitor
	pre_AP = SpikeMonitor(pre_neurons)
	syn_mon = StateMonitor(synapses, ['u_S','x_S','U_0', 'r_S'], record=np.arange(N_exc), when='after_synapses')
	post_mon = StateMonitor(post_neurons, 'g_e', record=np.arange(2*N_post_syn))
	astro_mon = StateMonitor(astrocyte, ['Y_S','I','C'], record=np.arange(N_exc))
	GRE = SpikeMonitor(astrocyte)

	run(duration, report='text')
	trans = 50000

	r_S_mean = jumps(post_mon.g_e[0,trans:]).mean()/w_e
	r_S_mean_no = jumps(post_mon.g_e[1,trans:]).mean()/w_e
	print(f'rate = {rate_in/Hz:.2f} Hz')
	print(f'CLOSED  - r_S mean={r_S_mean:.4f}')
	print(f'NO GLIO - r_S mean={r_S_mean_no:.4f}')
	print('______________________________________\n')
	print(f'CLOSED  - g_e mean = {post_mon.g_e[0,trans:].mean()/nS:.5f} (nS) ')
	print(f'NO GLIO - g_e mean = {post_mon.g_e[1,trans:].mean()/nS:.5f} (nS)')
	print(f'mean Y_S view by astrocte: {astro_mon.Y_S[:].mean()/umolar:.5f} +- {astro_mon.Y_S[:].std()/umolar:.5f} uM')
	## Plots #########################################################################################
	if args.p:
		fig1 = plt.figure(figsize=(16,6),
						 num=f'Local_circuit - postsynaptic exc conductance v_in={rate_in/Hz:.2f} Hz')

		gs = fig1.add_gridspec(2,3)
		ax1_1 = fig1.add_subplot(gs[0, 0])
		ax1_2 = fig1.add_subplot(gs[1, 0])
		ax1_3 = fig1.add_subplot(gs[:, 1])
		ax1_4 = fig1.add_subplot(gs[0, 2])
		ax1_5 = fig1.add_subplot(gs[1, 2])

		setting = {0:['C6','closed loop'], 1:['black', 'no gliotrasmission']}
		ax1_1.set_title('Postsynaptic excitatory conductance')
		for i, axis in enumerate([ax1_1, ax1_2]):
			axis.plot(post_mon.t[trans:]/second, post_mon.g_e[i,trans:]/nS, color=setting[i][0], label=setting[i][1])
			# axis.axhline(2, 0, duration/second, ls='dashed', color='C0')
			axis.set_ylabel(r'$g_e$ (nS)')
			axis.grid(linestyle='dotted')
			axis.legend()
		ax1_2.set_xlabel('time (s)')

		ax1_3.set_title('GRE raster plot')
		ax1_3.scatter(GRE.t[:]/second, GRE.i[:], marker='|')
		ax1_3.set_xlabel('time (s)')

		ax1_4.set_title('Astrocyte')
		ax1_4.plot(astro_mon.t[:]/second, astro_mon.I[79]/umolar, color='C0')
		ax1_4.set_ylabel(r'I ($\mu M$)')
		ax1_4.grid(linestyle='dotted')
		
		ax1_5.plot(astro_mon.t[:]/second, astro_mon.C[79]/umolar, color='C3')
		ax1_5.axhline(C_Theta/umolar,0,duration/second, ls='dashed', color='black')
		ax1_5.set_ylabel(r'C ($\mu M$)')
		ax1_5.set_xlabel('time (s)')
		ax1_5.grid(linestyle='dotted')
		


	device.delete()
	plt.show()


