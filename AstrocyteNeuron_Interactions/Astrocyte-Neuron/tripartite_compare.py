"""
Extracellular neurotransmitter concentration (averaged across 500 synapses) for 
costant input stimulus nu_in.
Neuronal, synaptic and astrocyte parameters are equal to ones presente in
network.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
import makedir

set_device('cpp_standalone', directory=None)  # Use fast "C++ standalone mode"

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Tripartite synapses')
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
	O_G = 0.9/umolar/second      # Agonist binding (activating) rate
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
	duration = 20*second
	seed(145624)
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

	N_syn = 160                # Total number of synapses
	N_a = 2                    # Total number of astrocyte

	rate_in = 3.5*Hz             # Rate of presynaptic neurons
	pre_neurons = PoissonGroup(N_syn, rates=rate_in)
	post_neurons = NeuronGroup(N_syn, model="dg_e/dt = -g_e/tau_e : siemens # post-synaptic excitatory conductance",
								method='rk4')

	synapses = Synapses(pre_neurons, post_neurons, model=syn_model, on_pre=action+exc, method='linear')
	synapses.connect(j='i')   # closed-loop 
	synapses.connect(j='i')   # open-loop 
	synapses.connect(j='i')   # no gliotrasmission
	synapses.x_S = 1.0

	astrocyte = NeuronGroup(N_a * N_syn, model=astro_eqs, method='rk2', dt=1e-2*second,
                            threshold='C>C_Theta', refractory='C>C_Theta', reset=astro_release)
	astrocyte.x_A = 1.0
	astrocyte.h = 0.9
	astrocyte.I = 0.01*umolar
	astrocyte.C = 0.01*umolar

	# ASTRO TO EXC_SYNAPSES
	ecs_astro_to_syn = Synapses(astrocyte, synapses, 'G_A_post = G_A_pre : mmolar (summed)')
	ecs_astro_to_syn.connect(j='i if j<2*N_syn')

	# EXC_SYNAPSES TO ASTRO
	ecs_syn_to_astro = Synapses(synapses, astrocyte,
								'Y_S_post = Y_S_pre : mmolar (summed)')
	ecs_syn_to_astro.connect(j='i if i<N_syn')

	## Monitor
	pre_AP = SpikeMonitor(pre_neurons)
	GRE = SpikeMonitor(astrocyte)
	syn_mon = StateMonitor(synapses, ['Gamma_S', 'r_S', 'Y_S'], record=np.arange(N_syn*(N_a+1)), when='after_synapses')
	
	run(duration, report='text')
	print(syn_mon.Y_S[:].shape)
	print(syn_mon.Y_S[:].mean(axis=0))
	print(syn_mon.Y_S[2*N_syn:].mean(axis=0).mean())
	print(syn_mon.Y_S[2*N_syn:].mean(axis=0).std())
				
	## Plots #########################################################################################
	if args.p:
		print('start plot')
		plt.rc('font', size=13)
		plt.rc('legend', fontsize=10)
		fig1 = plt.figure(figsize=(13.5,6),
						  num=f'Compare: noglio, open and close, nu_i={rate_in/Hz} Hz', tight_layout=True)

		gs = fig1.add_gridspec(2,2)
		ax1_1 = fig1.add_subplot(gs[0, 0])
		ax1_2 = fig1.add_subplot(gs[1, 0])
		ax1_3 = fig1.add_subplot(gs[:, 1])

		# ax1_1.set_title(r'$\nu_{in}=$'+f'{rate_in/Hz} Hz')
		ax1_1.plot(syn_mon.t[:]/second, syn_mon.Y_S[2*N_syn:].mean(axis=0)/umolar, color='black', label='no gliotrasmission')
		ax1_1.set_ylabel(r'$\bar{Y_S}$ ($\mu$M)')
		ax1_1.grid(linestyle='dotted')
		ax1_1.legend()

		# ax1_2.axhline(syn_mon.Y_S[2*N_syn:].mean()/umolar,0,duration/second, ls='dashed', color='black', label='mean noglio')
		# ax1_2.plot(syn_mon.t[:]/second, syn_mon.Y_S[2*N_syn:].mean(axis=0)/umolar, alpha=0.5, color='black')
		ax1_2.plot(syn_mon.t[:]/second, syn_mon.Y_S[N_syn:2*N_syn].mean(axis=0)/umolar, color='C2', label='open-loop')
		ax1_2.plot(syn_mon.t[:]/second, syn_mon.Y_S[:N_syn].mean(axis=0)/umolar, color='C6', label='closed-loop')
		ax1_2.set_ylabel(r'$\bar{Y_S}$ ($\mu$M)')
		ax1_2.set_xlabel(r'time ($\rm{s}$)')
		ax1_2.grid(linestyle='dotted')
		ax1_2.legend()

		ax1_3.scatter(GRE.t[:][GRE.i[:]<N_syn]/second, GRE.i[:][GRE.i[:]<N_syn], marker='|', color='C6')
		ax1_3.scatter(GRE.t[:][GRE.i[:]>N_syn]/second, GRE.i[:][GRE.i[:]>N_syn], marker='|', color='C2')
		ax1_3.set_ylabel('astrocyte index')
		ax1_3.set_xlabel(r'time ($\rm{s}$)')
		
		fig2, ax2 = plt.subplots(nrows=1, ncols=1,  tight_layout=True, figsize=(5.5,3), 
							   num=f'open-, closed- astro dynamics, O_G={O_G*umolar*second:.4f} Hz_MT')

		# ax2[0].plot(syn_mon.t[:]/second, syn_mon.Gamma_S[0], color='C6', label='closed-loop')
		# ax2[0].plot(syn_mon.t[:]/second, syn_mon.Gamma_S[161], color='C2', label='open-loop')
		# ax2[0].set_ylabel(r'$\Gamma_S$')
		# ax2[0].grid(linestyle='dotted')

		# ax2.axhline(syn_mon.Y_S[2*N_syn:].mean()/umolar,0,duration/second, ls='dashed', color='black', label='mean noglio')
		ax2.plot(syn_mon.t[:]/second, syn_mon.Y_S[N_syn:2*N_syn].mean(axis=0)/umolar, color='C2', label='open-loop')
		ax2.plot(syn_mon.t[:]/second, syn_mon.Y_S[:N_syn].mean(axis=0)/umolar, color='C6', label='closed-loop')
		ax2.set_ylabel(r'$\bar{Y_S}$  ($\mu$M)')
		ax2.set_xlabel('time (s)')
		ax2.grid(linestyle='dotted')
		# ax2[1].legend()
		print('end plot')

	device.delete()
	plt.show()
	


