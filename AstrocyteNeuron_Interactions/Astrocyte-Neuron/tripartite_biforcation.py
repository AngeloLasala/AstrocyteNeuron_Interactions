"""
Tripartite synapses is one of the main protagonist into NG networks dynamics.
To understand it role in dynamical behaviore is crucial to elucidate its dinamics in terms of
biforcation terms.

Steady state variable are here monitored to compute the biforcation analysis with the goal to
underline possible type of astrocite modulation and the time course od neurotrasmitters.

Note: For the plot, put same dt in th monitor to have congurent images. For monitor it is does not need
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
import makedir

set_device('cpp_standalone', directory=None)  # Use fast "C++ standalone mode"

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Tripartite synapses')
	parser.add_argument('modulation', type=str, help='type of astromodulation: A or F')
	parser.add_argument('-astro', action='store_false', help="""biforcation only for tripartite, else only 
														for astro. default=True""")
	parser.add_argument('-p', action='store_true', help="show paramount plots, default=False")
	args = parser.parse_args()

	mod = {'A':[0.5, 1.2], 'F':[3.2, 0.6]}
	## PARAMETERS ###################################################################
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
	O_G = 1.0/umolar/second      # Agonist binding (activating) rate
	Omega_G = 0.008/second   # Agonist release (deactivating) rate

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
	O_beta = mod[args.modulation][0]*umolar/second   # Maximal rate of IP_3 production by PLCbeta
	O_N = 0.3/umolar/second      # Agonist binding rate
	Omega_N = 0.5/second         # Maximal inactivation rate
	K_KC = 0.5*umolar            # Ca^2+ affinity of PKC
	zeta = 10                    # Maximal reduction of receptor affinity by PKC
	# Endogenous IP3 production
	O_delta = mod[args.modulation][1]*umolar/second  # Maximal rate of IP_3 production by PLCdelta
	kappa_delta = 1.5*umolar     # Inhibition constant of PLC_delta by IP_3
	K_delta = 0.1*umolar         # Ca^2+ affinity of PLCdelta
	# IP_3 degradation
	Omega_5P = 0.05/second       # Maximal rate of IP_3 degradation by IP-5P
	K_D = 0.7*umolar             # Ca^2+ affinity of IP3-3K
	K_3K = 1.0*umolar            # IP_3 affinity of IP_3-3K
	O_3K = 4.5*umolar/second     # Maximal rate of IP_3 degradation by IP_3-3K

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
	duration = 500*second
	#################################################################################

	## SYNAPSES
	syn_model = """
	# Neurotrasmitter
	dY_S/dt = -Omega_c*Y_S                             : mmolar (clock-driven)

	# Fraction of activated astrocyte
	dGamma_S/dt = O_G*G_A*(1-Gamma_S)-Omega_G*Gamma_S  : 1 (clock-driven)
	G_A : mmolar

	# Synaptic variable
	du_S/dt = -Omega_f * u_S : 1 (event-driven)
	dx_S/dt = Omega_d * (1-x_S) : 1 (event-driven)
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

	# exc="g_e_post+=w_e*r_S"
	# inh="g_i_post+=w_i*r_S"

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

	N_syn = 10                # Total number of synapses
	N_a = 1                    # Total number of astrocyte

	rate_start = 0.3
	rate_stop = 3.5
	rate_in = np.linspace(rate_start, rate_stop, N_syn)*Hz        # Rate of presynaptic neurons
	# Generate costant firing rate by simple group of neurons
	model = """
	dv/dt = (rate_p*2*pi)*cos(rate_p*2*pi*t) : 1
	rate_p : Hz
	"""
	pre_neurons = NeuronGroup(N_syn, model=model, threshold='v>0.5', refractory='v>0.5', method='rk4')
	pre_neurons.rate_p = rate_in

	# pre_neurons = PoissonGroup(N_syn, rates=rate_in)

	post_neurons = NeuronGroup(N_syn, model="")

	synapses = Synapses(pre_neurons, post_neurons, model=syn_model, on_pre=action, method='linear')
	synapses.connect(j='i')   # closed-loop 
	synapses.connect(j='i')   # no gliotrasmission
	synapses.x_S = 1.0

	astrocyte = NeuronGroup(N_a*N_syn, model=astro_eqs, method='rk4',
                            threshold='C>C_Theta', refractory='C>C_Theta', reset=astro_release)
	astrocyte.x_A = 1.0
	astrocyte.h = 0.9
	astrocyte.I = 0.01*umolar
	astrocyte.C = 0.01*umolar


	# # EXC_SYNAPSES TO ASTRO
	ecs_syn_to_astro = Synapses(synapses, astrocyte, 'Y_S_post = Y_S_pre : mmolar (summed)')
	ecs_syn_to_astro.connect(j='i if i<N_syn')   #closed-loop

	# # ASTRO TO EXC_SYNAPSES
	if args.astro:
		ecs_astro_to_syn = Synapses(astrocyte, synapses, 'G_A_post = G_A_pre : mmolar (summed)')
		ecs_astro_to_syn.connect(j='i if i<N_syn')                      #closed-loop
	
	#Monitor
	syn_mon = StateMonitor(synapses, ['Y_S','Gamma_S','U_0','u_S','x_S','r_S'], record=np.arange(N_syn*(N_a+1)), when='after_synapses' )
	astro_mon = SpikeMonitor(astrocyte)
	spike_mon = SpikeMonitor(pre_neurons)
	astro_var = StateMonitor(astrocyte, ['Y_S', 'G_A','Gamma_A','I','C'], record=True)

	run(duration, report='text')

	## Save variable ##################################################################################
	# name = f'Tripartite_synapses/Biforcation/O_beta{O_beta/(umolar/second):.2f}_O_delta{O_delta/(umolar/second):.1f}_long'
	# makedir.smart_makedir(name)

	# ## rate array
	# np.save(f'{name}/duration', duration)
	# np.save(f'{name}/rate_start', rate_start)
	# np.save(f'{name}/rate_stop', rate_stop)
	# np.save(f'{name}/N', N_syn)

	# # Synapses variable
	# np.save(f'{name}/Y_S', syn_mon.Y_S[:])
	# np.save(f'{name}/Gamma_S', syn_mon.Gamma_S[:])
	# np.save(f'{name}/r_S', syn_mon.r_S[:])

	# # Astrocytic variable
	# np.save(f'{name}/C', astro_var.C[:])
	# np.save(f'{name}/I', astro_var.I[:])
	# np.save(f'{name}/Gamma_A', astro_var.Gamma_A[:])
	# np.save(f'{name}/GRE_t', astro_mon.t[:])
	# np.save(f'{name}/GRE_i', astro_mon.i[:])

	## Plots #########################################################################################
	# Astro variable
	trans = 50000   #trans*dt=50000*0.5*ms=25 s
	trans = 350000   #trans*dt_sam = 40000*10*ms= 400 s

	if args.p:
		index = 1
		fig4, ax4 = plt.subplots(nrows=4, ncols=1, figsize=(13,7), sharex=True,
								num=f"Astrocite dynamics - rate={rate_in[index]/Hz:.2f} Hz")

		ax4[0].plot(syn_mon.t[trans:], syn_mon.Y_S[index,trans:]/umolar, label='Y_S')
		ax4[0].plot(astro_var.t[trans:]/second, astro_var.Y_S[index,trans:]/umolar, color='C1')
		ax4[0].set_ylabel(r'$Y_S$ ($\mu$M)')
		ax4[0].grid(linestyle='dotted')

		ax4[1].plot(astro_var.t[trans:]/second, astro_var.Gamma_A[index,trans:], color='C7')
		ax4[1].set_ylabel(r'$\Gamma_A$')
		ax4[1].grid(linestyle='dotted')

		ax4[2].plot(astro_var.t[trans:]/second, astro_var.I[index,trans:]/umolar, color='C0')
		ax4[2].set_ylabel(r'I ($\mu$M)')
		ax4[2].grid(linestyle='dotted')

		# ax4[3].scatter(spike_mon.t[:], spike_mon.i[:]+1.0, marker='|')
		ax4[3].plot(astro_var.t[trans:]/second, astro_var.C[index,trans:]/umolar, color='C3')
		# ax4[2].plot(astro_var.t[:]/second, astro_var.C[1]/umolar, color='C3')
		ax4[3].set_ylabel(r'C ($\mu$M)')
		# ax4[3].axhline(C_Theta/umolar,0,duration/second, ls='dashed', color='black')
		ax4[3].grid(linestyle='dotted')
		ax4[3].set_xlabel('time '+r'($\rm{s}$)')

		fig5, ax5 = plt.subplots(nrows=1, ncols=1, 
								num='Charateristic curve nu_A vs nu_S')
		#GRE event rate (count/duration)
		GRE_rates = []
		for ii in range(N_syn):
			GRE_rate = len(astro_mon.t[astro_mon.i==ii])/duration
			GRE_rates.append(GRE_rate)

		ax5.plot(rate_in/Hz, GRE_rates/Hz, color='C1')
		ax5.set_xscale('log')
		ax5.set_xlabel(r'$\nu_S$ (Hz)')
		ax5.set_ylabel(r'$\nu_A$ (Hz)')
		ax5.grid(linestyle='dotted')

		fig3, ax3 = plt.subplots(nrows=3, ncols=1, figsize=(13,7), sharex=True,
								num=f"synaptic variable - rate={rate_in[index]/Hz:.2f} Hz")

		ax3[0].plot(syn_mon.t[trans:], syn_mon.u_S[index,trans:], label='u_S')
		ax3[0].plot(syn_mon.t[trans:], syn_mon.x_S[index,trans:], label='x_S')
		ax3[1].plot(syn_mon.t[trans:], syn_mon.Gamma_S[index,trans:], label='Gamma_S')
		ax3[2].plot(syn_mon.t[trans:], syn_mon.Y_S[index,trans:], label='Y_S')


		fig2, ax2 = plt.subplots(nrows=3, ncols=1, figsize=(13,6), sharex=True,
								num=f"Gliorelease event")
		for axis in ax2:
			axis.grid(linestyle='dotted')

		ax2[0].plot(astro_var.t[10000:50000]/second, astro_var.C[index,10000:50000]/umolar, color='C3')
		ax2[0].axhline(C_Theta/umolar,0,duration/second, ls='dashed', color='black')
		ax2[0].set_ylabel(r'C ($\mu$M)')
		
		
		ax2[1].plot(astro_var.t[10000:50000]/second, astro_var.G_A[index,10000:50000]/umolar, color='C5')
		ax2[1].set_ylabel(r'$G_A$ ($\mu$M)')

		ax2[2].plot(syn_mon.t[10000:50000], syn_mon.Gamma_S[index,10000:50000], color='C9')
		ax2[2].set_ylabel(r'$\Gamma_A$')
		ax2[2].set_xlabel('time '+r'($\rm{s}$)')


		

	device.delete()
	plt.show()
	
	


