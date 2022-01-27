"""
Modulation of neurotrasmitter release by astrocity-neuron interaction.
neuronal, synaptic and astrocyte parameters are equal to ones presente in
network.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
from AstrocyteNeuron_Interactions import makedir

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Tripartite synapses')
	parser.add_argument('-p', action='store_true', help="show paramount plots, default=False")
	parser.add_argument('-by_hand', action='store_true', help="show paramount plots, default=False")
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
	defaultclock.dt = 0.05*ms
	duration = 2*second
	seed(28371)  # to get identical figures for repeated runs
	#################################################################################

	# Gliotrasmitter modulation
	# I want study only the SYNAPTIC variables, theremore there are
	# only synapses and astrocity group

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

	## NETWORK
	spikes = [16, 50, 100, 150, 200,
			300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400]*ms

	slow_spikes = np.arange(200,750,50)
	fast_spikes = np.arange(1000,1250,10)
	fast2_spikes = np.arange(1600,1725,0.1)

	spikes = slow_spikes.tolist() + fast_spikes.tolist()
	spikes = spikes*ms
	if args.by_hand :
		pre_neurons = SpikeGeneratorGroup(1, np.zeros(len(spikes)), spikes)
	else:
		pre_neurons = PoissonGroup(1, rates=3.5*Hz)

	post_neurons = NeuronGroup(2, model="dg_e/dt = -g_e/tau_e : siemens # post-synaptic excitatory conductance",
					method='rk4')

	synapses = Synapses(pre_neurons, post_neurons, model=syn_model, on_pre=action+exc, method='linear')
	synapses.connect(i=[0,0], j=[0,1]) 
	synapses.x_S = 1.0

	astrocyte = NeuronGroup(1, model=astro_eqs, method='rk4',
                        threshold='C>C_Theta', refractory='C>C_Theta', reset=astro_release)
	astrocyte.x_A = 1.0
	astrocyte.h = 0.9
	astrocyte.I = 0.4*umolar
	astrocyte.C = 0.4*umolar

	# ASTRO TO EXC_SYNAPSES
	ecs_astro_to_syn = Synapses(astrocyte, synapses, 'G_A_post = G_A_pre : mmolar (summed)')
	ecs_astro_to_syn.connect(i=0, j=1)

	# Synaptic to Astrocyte connection
	# ecs_syn_to_astro = Synapses(synapses, astrocyte,
	# 							'Y_S_post = Y_S_pre : mmolar (summed)')
	# ecs_syn_to_astro.connect(i=1, j=0)

	#Monitor
	pre_AP = SpikeMonitor(pre_neurons)
	syn_mon = StateMonitor(synapses, ['u_S','x_S','U_0', 'r_S', 'Y_S','Gamma_S'], record=True, when='after_synapses')
	astro_mon = SpikeMonitor(astrocyte)
	astro_var = StateMonitor(astrocyte, ['Y_S','Gamma_A','I','C','G_A'], record=True)
	post_mon = StateMonitor(post_neurons, 'g_e', record=True)

	net = Network(pre_neurons, synapses, astrocyte, post_neurons, ecs_astro_to_syn, 
						pre_AP, syn_mon, astro_mon, astro_var, post_mon )		
	net.store()
	fig4, ax4 = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(12,7),
							num='Facilitation whit respect to G_T' )
	
	ii = 0
	for G_T in [400,300,200]*mmolar:
		seed(1995235)
		ii +=1

		net.restore()		
		net.run(duration, report='text')
		print(syn_mon.u_S)
		print(astro_mon.t[:])

		## Facilitation - Paired Pulse Ratio (PPR)
		spk_position = []
		for spk in pre_AP.t[:]/ms:
			spk_position.append(np.where(syn_mon.t[:]/ms == spk)[0][0])
		
		r_S_noglio = syn_mon.r_S[0][spk_position]
		r_S_glio = syn_mon.r_S[1][spk_position]
		
		PPR_noglio = r_S_noglio[1:]/r_S_noglio[:-1]
		PPR_glio = r_S_glio[1:]/r_S_glio[:-1]
		print(PPR_noglio)
		print(PPR_glio)

		ax4[0].plot(syn_mon.t[:]/ms, syn_mon.U_0[1], label=r'$G_T$='f'{G_T/mmolar} mM')
		ax4[0].grid(linestyle="dotted")
		ax4[0].set_ylabel(r'$u_0$')
		ax4[0].legend()

		ax4[1].plot(syn_mon.t[:]/ms, syn_mon.Y_S[0]/umolar, color='black', alpha=0.1)
		ax4[1].plot(syn_mon.t[:]/ms, syn_mon.Y_S[1]/umolar)
		ax4[1].set_ylabel(r'$Y_S$ ($\mu$M)')
		ax4[1].set_xlabel('time (ms)')
		ax4[1].grid(linestyle="dotted")

		for t, PPR in zip(syn_mon.t[spk_position][1:]/ms, PPR_noglio):
			if PPR>1: color='C0'
			else: color='C1'
			ax4[2].scatter(t, 0, c=color, marker='|')

		for t, PPR in zip(syn_mon.t[spk_position][1:]/ms, PPR_glio):
			if PPR>1: color='C0'
			else: color='C1'
			ax4[2].scatter(t, ii, c=color, marker='|')
				

		print(syn_mon.u_S)
		print(astro_mon.t[:])
	
				
	## Plots #########################################################################################
	if args.p:
		fig1, ax1 = plt.subplots(nrows=5, ncols=1, sharex=True, figsize=(14,8), gridspec_kw={'height_ratios': [0.2,0.8,0.8,1,0.8]},
								num= 'OPEN LOOP - Modulation of synaptic release by GRE')

		ax1[0].set_title('Spikes (presynaptic)')
		for spk in pre_AP.t[:]/ms:
			ax1[0].axvline(x=spk, ymin=0.00, ymax=2.00, color='C0')

		ax1[1].set_title('Synaptic variables')
		ax1[1].plot(syn_mon.t[:]/ms, syn_mon.u_S[0], color='C3', label='STP')
		ax1[1].plot(syn_mon.t[:]/ms, syn_mon.u_S[1], color='C2', label='STP + glio')
		ax1[1].set_ylabel(r'$u_S$')
		ax1[1].grid(linestyle='dotted')
		ax1[1].legend()

		ax1[2].plot(syn_mon.t[:]/ms, syn_mon.x_S[0], color='C3', label='STP')
		ax1[2].plot(syn_mon.t[:]/ms, syn_mon.x_S[1], color='C2', label='STP + glio')
		ax1[2].set_ylabel(r'$x_S$')
		ax1[2].grid(linestyle='dotted')

		ax1[3].plot(syn_mon.t[:]/ms, syn_mon.Y_S[0]/umolar, color='C3', label='STP')
		ax1[3].plot(syn_mon.t[:]/ms, syn_mon.Y_S[1]/umolar, color='C2', label='STP + glio')
		ax1[3].set_ylabel(r'$Y_S$ ($\mu$M)')
		ax1[3].grid(linestyle='dotted')

		# spk_position = []
		# for spk in pre_AP.t[:]/ms:
		# 	spk_position.append(np.where(syn_mon.t[:]/ms == spk)[0][0])

		# ax1[2].vlines(syn_mon.t[spk_position]/ms, np.zeros(len(spk_position)),
		# 			  syn_mon.r_S[0][spk_position],color='C3')
		# ax1[2].vlines(syn_mon.t[spk_position]/ms, np.zeros(len(spk_position)),
		# 			  syn_mon.r_S[1][spk_position],color='C2')


		ax1[4].set_title("Postsynaptic conductance")
		ax1[4].plot(post_mon.t[:]/ms, post_mon.g_e[0]/nS, color='C3', label='STP')
		ax1[4].plot(post_mon.t[:]/ms, post_mon.g_e[1]/nS, color='C2', label='STP + glio')
		ax1[4].set_ylabel(r'$g_e$ (nS)')
		ax1[4].grid(linestyle='dotted')

		fig2, ax2 = plt.subplots(nrows=3, ncols=1, sharex=True,
								num="Release-descrising effect")

		ax2[0].plot(syn_mon.t[:]/ms, syn_mon.Gamma_S[0], color='C3', label='STP')
		ax2[0].plot(syn_mon.t[:]/ms, syn_mon.Gamma_S[1], color='C2', label='STP + glio')
		ax2[0].set_ylabel(r"$\Gamma_S$")
		ax2[0].grid(linestyle="dotted")
		ax2[0].legend()

		ax2[1].plot(syn_mon.t[:]/ms, syn_mon.U_0[0], color='C3')
		ax2[1].plot(syn_mon.t[:]/ms, syn_mon.U_0[1], color='C2')
		ax2[1].set_ylabel(r'$u_0$')
		ax2[1].grid(linestyle="dotted")
		
		ax2[2].plot(syn_mon.t[:]/ms, syn_mon.U_0[1], color='C2')
		ax2[2].set_ylabel(r'$u_0$')
		ax2[2].set_xlabel('time (ms)')
		ax2[2].grid(linestyle="dotted")

		fig3, ax3 = plt.subplots(nrows=5, ncols=1, sharex=True,
								num="Astrocite dynamics - Open Loop")

		ax3[0].plot(astro_var.t[:]/ms, astro_var.Y_S[0]/umolar, color='C3')
		ax3[0].set_ylabel(r'$Y_S$ ($\mu$M)')
		ax3[0].grid(linestyle='dotted')

		ax3[1].plot(astro_var.t[:]/ms, astro_var.Gamma_A[0], color='C4')
		ax3[1].set_ylabel(r'$\Gamma_A$')
		ax3[1].grid(linestyle='dotted')

		ax3[2].plot(astro_var.t[:]/ms, astro_var.I[0]/umolar, color='C0')
		ax3[2].set_ylabel(r'I ($\mu$M)')
		ax3[2].grid(linestyle='dotted')

		ax3[3].plot(astro_var.t[:]/ms, astro_var.C[0]/umolar, color='C3')
		ax3[3].set_ylabel(r'C ($\mu$M)')
		ax3[3].axhline(C_Theta/umolar,0,duration/second, ls='dashed', color='black')
		ax3[3].grid(linestyle='dotted')

		ax3[4].plot(astro_var.t[:]/ms, astro_var.G_A[0]/umolar, color='C5')
		ax3[4].set_ylabel(r'$G_A$ ($\mu$M)')
		ax3[4].set_xlabel('time (ms)')
		ax3[4].grid(linestyle='dotted')
		

		plt.show()


# ax1[2].set_title('Astrocite variables')
# ax1[2].plot(astro_var.t[:]/ms, astro_var.G_A[0]/umolar, color='C4')
# ax1[2].grid(linestyle='dotted')
# ax1[2].set_ylabel(r'$G_A$ ($\mu$M)')

# ax1[3].plot(astro_var.t[:]/ms, astro_var.C[0]/umolar, color='C3')
# ax1[3].set_ylabel(r'$Ca^{2\plus}$ ($\mu$M)')
# ax1[3].axhline(C_Theta/umolar,0,duration/ms, ls='dashed', color='black')
# ax1[3].grid(linestyle='dotted')

