"""
Coupling neurons and astrocytes network

Randomly connected COBA network with excitatory synapses modulated
by release-increasing gliotransmission from a connected network of astrocytes.

- "Modelling neuro-glia interactions with the Brian2 simulator" Stimberg et al (2017)
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
import constant_NG as k_NG 
import makedir

# set_device('cpp_standalone', directory=None) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neuron-Glia (NG) network, find NG_netwrok parameters in "constant_NG.py"')
    parser.add_argument("rate_in", type=float, help="value of external input rate expressed in Hz")
    parser.add_argument("-linear", action='store_false', help="""Square grid for neurons and astrocyte, if -linear there is 
                                                             no spatial arrangement""")
    parser.add_argument("-trial", type=int, help="number of trial, defual=0", default=0)
    parser.add_argument("-cp", action='store_true', help="Connectivity plots, default=False")
    parser.add_argument("-p", action="store_true", help="Show all the plots, Default=False")
    args = parser.parse_args()

    ## PARAMETERS ###################################################################
    # --  General parameters --
    N_e = k_NG.N_e                 # Number of excitatory neurons
    N_i = k_NG.N_i                    # Number of inhibitory neurons
    N_a = k_NG.N_a                   # Number of astrocytes

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
    O_beta = 3.2*umolar/second   # Maximal rate of IP_3 production by PLCbeta
    O_N = 0.3/umolar/second      # Agonist binding rate
    Omega_N = 0.5/second         # Maximal inactivation rate
    K_KC = 0.5*umolar            # Ca^2+ affinity of PKC
    zeta = 10                    # Maximal reduction of receptor affinity by PKC
    # Endogenous IP3 production
    O_delta = 0.6*umolar/second  # Maximal rate of IP_3 production by PLCdelta
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
    defaultclock.dt = k_NG.dt*ms
    duration = k_NG.duration*second
    seed(19958)  # to get identical figures for repeated runs
    #################################################################################

    ## NETWORK #####################################################################
    ## NEURONS 
    neuron_eqs = """
    # External input from external synapses
    I_syn_ext = w_ext * (E_e-v) * X_ext : ampere
    w_ext : siemens                  # external conductance

    # LIF model with exponential synapses
    dv/dt = (g_l*(E_l-v) + g_e*(E_e-v) + g_i*(E_i-v) + I_syn_ext)/C_m : volt (unless refractory)
    dg_e/dt = -g_e/tau_e : siemens   # post-synaptic excitatory conductance
    dg_i/dt = -g_i/tau_i : siemens   # post-synaptic inhibitory conductance
    dX_ext/dt = -X_ext/tau_e :  1    # post-synaptic external input

    I_exc = abs(g_e*(E_e-v)) : ampere
    I_inh = abs(g_i*(E_i-v)) : ampere

    LFP = (abs(g_e*(E_e-v)) + abs(g_i*(E_i-v)) + abs(I_syn_ext))/g_l : volt

    # Neuron position in space
    x : 1 (constant)
    y : 1 (constant)
    """
    neurons = NeuronGroup(N_e+N_i, model=neuron_eqs, method='rk2',
                        threshold='v>V_th', reset='v=V_r', refractory='tau_r')
   
    exc_neurons = neurons[:N_e]
    inh_neurons = neurons[N_e:]

    # Arrange excitatory neurons in a grid
    N_rows_exc = int(sqrt(N_e+N_i))
    N_cols_exc = ((N_e+N_i)//N_rows_exc)
    grid_dist = (size / N_cols_exc)
    #square grid
    if args.linear:
        xx = np.arange(N_rows_exc)
        yy = np.arange(N_cols_exc)
        XX,YY = np.meshgrid(xx,yy)

        neurons.x = XX.flatten()[:(N_e+N_i)]*grid_dist
        neurons.y = YY.flatten()[:(N_e+N_i)]*grid_dist
    else:
        # exc_neurons.x = '(i // N_rows_exc)*grid_dist - N_rows_exc/2.0*grid_dist'
        # exc_neurons.y = '(i % N_rows_exc)*grid_dist - N_cols_exc/2.0*grid_dist'
        neurons.x = 'i'

    # Random initial membrane potential values and conductances
    neurons.v = 'E_l + rand()*(V_th-E_l)'
    neurons.g_e = 'rand()*w_e'
    neurons.g_i = 'rand()*w_i'

     # External input - Poisson heterogeneity 
    # rate_in values are comtuted from characteristic curve v_out vs v_poisson
    # this values are stored in a dictionary where the keys are the I_ex
    # I_ex = [100,105,110,115,120]
    # 's' define the ext-inh connection, "fast spiking inh": s = 1.3
    rate_in = args.rate_in*Hz
    print(rate_in)
    poisson = PoissonInput(neurons, 'X_ext', 160 , rate=rate_in, weight='1')
    s = k_NG.s
    exc_neurons.w_ext = w_e
    inh_neurons.w_ext = s*w_e 
    
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

    # Synaptic connection: it is introduced a balance parameter to define recurrent connection 
    # seed is put before synaptic connection to give same networks structure
    g = k_NG.g 
    p_e = k_NG.p_e
    p_i = p_e/g
    
    seed(15325)
    exc_syn = Synapses(exc_neurons, neurons, model= syn_model, on_pre=syn_action+exc_act, method='linear')
    exc_syn.connect(True, p=p_e)
    exc_syn.x_S = 1.0

    inh_syn = Synapses(inh_neurons, neurons, model=syn_model, on_pre=syn_action+inh_act, method='linear')
    inh_syn.connect(True, p=p_i)
    inh_syn.x_S = 1.0

   

    # Connect excitatory synapses to an astrocyte depending on the position of the
    # post-synaptic neuron
    N_rows_astro = int(sqrt(N_a))
    N_cols_astro = (N_a//N_rows_astro)
    grid_dist_astro = grid_dist
    exc_syn.astrocyte_index = 'x_post'
    # exc_syn.astrocyte_index = exc_syn.j[:]

    # ASTROCYTE
    # Astrocytic dynamics start at time t0 = start_astro
    # TimeedArray is creadted to stating vectot field after start
    t_0_astro = k_NG.t_0_astro*second
    A_auxiliar = [0,1]
    A_rest = [1,1]
    for i_windows in range(k_NG.windows):
        A_auxiliar += A_rest
    start_astro = TimedArray(A_auxiliar, dt=t_0_astro)

    astro_eqs = """
    # Fraction of activated astrocyte receptors (1):
    dGamma_A/dt = start_astro(t)*(O_N * Y_S * (1 - clip(Gamma_A,0,1)) -
                Omega_N*(1 + zeta * C/(C + K_KC)) * clip(Gamma_A,0,1)) : 1

    # IP_3 dynamics (1)
    dI/dt = start_astro(t)*(J_beta + J_delta - J_3K - J_5P)               : mmolar

    J_beta = O_beta * Gamma_A                                        : mmolar/second
    J_delta = O_delta/(1 + I/kappa_delta) * C**2/(C**2 + K_delta**2) : mmolar/second
    J_3K = O_3K * C**4/(C**4 + K_D**4) * I/(I + K_3K)                : mmolar/second
    J_5P = Omega_5P*I                                                : mmolar/second
    # Diffusion between astrocytes (1):
    # J_coupling                                                       : mmolar/second

    # Calcium dynamics (2):
    dC/dt = start_astro(t)*(J_r + J_l - J_p)                                   : mmolar
    dh/dt = start_astro(t)*((h_inf - h)/tau_h)                                 : 1

    J_r = (Omega_C * m_inf**3 * h**3) * (C_T - (1 + rho_A)*C) : mmolar/second
    J_l = Omega_L * (C_T - (1 + rho_A)*C)                     : mmolar/second
    J_p = O_P * C**2/(C**2 + K_P**2)                          : mmolar/second
    m_inf = I/(I + d_1) * C/(C + d_5)                         : 1
    h_inf = Q_2/(Q_2 + C)                                     : 1
    tau_h = 1/(O_2 * (Q_2 + C))                               : second
    Q_2 = d_2 * (I + d_1)/(I + d_3)                           : mmolar

    # Fraction of gliotransmitter resources available for release (1):
    dx_A/dt = start_astro(t)*(Omega_A * (1 - x_A)) : 1
    # gliotransmitter concentration in the extracellular space (1):
    dG_A/dt = start_astro(t)*(-Omega_e*G_A)        : mmolar

    # Neurotransmitter concentration in the extracellular space:
    Y_S                           : mmolar

    # The astrocyte position in space
    x : 1 (constant)
    y : 1 (constant)
    """

    astro_release = """
    G_A += rho_e*G_T*U_A*x_A
    x_A -= U_A * x_A
    """

    astrocyte = NeuronGroup(N_a, astro_eqs, 
                            threshold='C>C_Theta', refractory='C>C_Theta', reset=astro_release,
                            dt=1e-2*second, method='rk2')

    # Arrange excitatory neurons in a grid
    #square grid
    if args.linear:
        x_astro = np.arange(N_rows_astro)
        y_astro = np.arange(N_cols_astro)
        XX_A,YY_A = np.meshgrid(x_astro,y_astro)

        astrocyte.x = XX_A.flatten()[:N_a]*grid_dist_astro
        astrocyte.y = YY_A.flatten()[:N_a]*grid_dist_astro
    else:
        # astrocyte.x = '(i // N_rows_astro_astro)*grid_dist_astro - N_rows_astro/2.0*grid_dist_astro'
        # astrocyte.y = '(i % N_rows_astro)*grid_dist_astro - N_cols_astro/2.0*grid_dist_astro'
        astrocyte.x = 'i'

    astrocyte.C ="0.005*umolar + rand()*(0.015-0.005)*umolar"
    astrocyte.h = "0.85 + rand()*(0.95-0.85)"
    astrocyte.I = "0.005*umolar + rand()*(0.015-0.005)*umolar"
    astrocyte.x_A = 1.0

    # bidirectional connection beetwith astrocyte and excitatory synapses
    # based on postsynaptic neurons position
    # ASTRO TO EXC_SYNAPSES
    ecs_astro_to_syn = Synapses(astrocyte, exc_syn, 'G_A_post = G_A_pre : mmolar (summed)')
    ecs_astro_to_syn.connect('i == astrocyte_index_post')
  

    #EXC_SYNAPSES TO ASTRO
    ecs_syn_to_astro = Synapses(exc_syn, astrocyte, 'Y_S_post = Y_S_pre/N_incoming : mmolar (summed)')
    ecs_syn_to_astro.connect('astrocyte_index_pre == j')
    

    # Diffusion between astrocytes
    # astro_to_astro_eqs = """
    # delta_I = I_post - I_pre            : mmolar
    # J_coupling_post = -(1 + tanh((abs(delta_I) - I_Theta)/omega_I))*
    #                 sign(delta_I)*F/2 : mmolar/second (summed)
    # """
    # astro_to_astro = Synapses(astrocyte, astrocyte,
    #                         model=astro_to_astro_eqs)
    # # Connect to all astrocytes less than 75um away
    # # (about 4 connections per astrocyte)
    # astro_to_astro.connect('i != j and '
    #                         'sqrt((x_pre-x_post)**2+(y_pre-y_post)**2) < 75*um')

    ##########################################################################################

    ## MOMITOR ###############################################################################
    for enu, windows in enumerate(range(k_NG.windows)):
        print(f'run number {enu+1}')
        spikes_exc_mon = SpikeMonitor(exc_neurons)
        spikes_inh_mon = SpikeMonitor(inh_neurons)
        astro_mon = SpikeMonitor(astrocyte)

        firing_rate_exc = PopulationRateMonitor(exc_neurons)
        firing_rate_inh = PopulationRateMonitor(inh_neurons)
        firing_rate = PopulationRateMonitor(neurons)
        
        neurons_mon = StateMonitor(neurons, ['v','g_e','g_i','I_exc', 'I_inh', 'I_syn_ext'], 
                                record=[i for i in range(200)] + [i for i in range(N_e,N_e+200)], dt= k_NG.dt_sam*ms)
        mon_LFP = StateMonitor(exc_neurons, 'LFP', record=[i for i in range(1000)], dt= k_NG.dt_sam*ms)
        var_astro_mon = StateMonitor(astrocyte, ['C','I','h','Gamma_A','Y_S','G_A','x_A'], record=[i for i in range(100)])
        ###########################################################################################

        ## RUN and NETWORK INFORMATION ###################################################################
        print('============================')
        print(f'we : {w_e}')
        print(f'g : {g}')
        print(f's : {s}')
        print(f'dt : {k_NG.dt*ms}')
        print(f'dt_sam : {k_NG.dt_sam*ms}')
        print('============================')
        run(duration, report='text', report_period=120*second)
        # print(np.unique(ecs_astro_to_syn.i[:]).shape)
        # print(np.unique(exc_syn.astrocyte_index[:]))
        # print(np.unique(exc_syn.astrocyte_index[:]).shape)        
    

        ##################################################################################################

        ## SAVE IMPORTANT VALUES #########################################################################
        if args.linear: 
            grid_name='mygrid'
        else: 
            grid_name='nogrid'
        
        name = f'Neuro_Glia_network/NG_network_rate_in{rate_in/Hz:.1f}_ph_'+grid_name+f'_g{g}_s{s}_we_{w_e/nS:.2f}_running_F_{O_beta/(umolar/second):.1f}_{O_delta/(umolar/second):.1f}_fixed_0.05_h0.5/trial_{args.trial}/time_windows_{enu+1}'
        makedir.smart_makedir(name)

        # Duration
        np.save(f'{name}/duration',duration)
        np.save(f'{name}/rate_in',rate_in)

        # Raster plot
        np.save(f'{name}/spikes_exc_mon.t',spikes_exc_mon.t)
        np.save(f'{name}/spikes_exc_mon.i',spikes_exc_mon.i)
        np.save(f'{name}/spikes_inh_mon.t',spikes_inh_mon.t)
        np.save(f'{name}/spikes_inh_mon.i',spikes_inh_mon.i)
        np.save(f'{name}/astro_mon.t',astro_mon.t)
        np.save(f'{name}/astro_mon.i',astro_mon.i)

        # Neurons variables
        np.save(f'{name}/neurons_mon.v',neurons_mon.v)
        # np.save(f'{name}/neurons_mon.g_e',neurons_mon.g_e)
        # np.save(f'{name}/neurons_mon.g_i',neurons_mon.g_i)
        np.save(f'{name}/neurons_mon.I_exc',neurons_mon.I_exc)
        np.save(f'{name}/neurons_mon.I_inh',neurons_mon.I_inh)
        np.save(f'{name}/neurons_mon.I_syn_ext',neurons_mon.I_syn_ext)
        np.save(f'{name}/neurons_mon.t',neurons_mon.t)
        np.save(f'{name}/firing_rate_exc.t',firing_rate_exc.t)
        np.save(f'{name}/firing_rate_exc.rate',firing_rate_exc.rate)
        # np.save(f'{name}/firing_rate_inh.t',firing_rate_inh.t)
        np.save(f'{name}/firing_rate_inh.rate',firing_rate_inh.rate)
        # np.save(f'{name}/firing_rate.t',firing_rate.t)
        np.save(f'{name}/firing_rate.rate',firing_rate.rate)

        # LFP 
        np.save(f'{name}/mon_LFP.LFP',mon_LFP.LFP)

        # Astrocte variables 
        np.save(f'{name}/var_astro_mon.t',var_astro_mon.t)
        np.save(f'{name}/var_astro_mon.Y_S',var_astro_mon.Y_S)
        np.save(f'{name}/var_astro_mon.Gamma_A',var_astro_mon.Gamma_A)
        np.save(f'{name}/var_astro_mon.I',var_astro_mon.I)
        np.save(f'{name}/var_astro_mon.C',var_astro_mon.C)
        np.save(f'{name}/var_astro_mon.h',var_astro_mon.h)
        np.save(f'{name}/var_astro_mon.x_A',var_astro_mon.x_A)
        np.save(f'{name}/var_astro_mon.G_A',var_astro_mon.G_A)

        # Connection
        # np.save(f'{name}/exc_syn.i',exc_syn.i)
        # np.save(f'{name}/exc_syn.j',exc_syn.j)
        # np.save(f'{name}/inh_syn.i',inh_syn.i)
        # np.save(f'{name}/inh_syn.j',inh_syn.j)
        # np.save(f'{name}/ecs_astro_to_syn.i',ecs_astro_to_syn.i)
        # np.save(f'{name}/ecs_astro_to_syn.j',ecs_astro_to_syn.j)
        # np.save(f'{name}/ecs_syn_to_astro.i',ecs_syn_to_astro.i)
        # np.save(f'{name}/ecs_syn_to_astro.j',ecs_syn_to_astro.j)
        # np.save(f'{name}/astro_to_astro.i',astro_to_astro.i)
        # np.save(f'{name}/astro_to_astro.j',astro_to_astro.j)

        # Network Structure
        with open(f"{name}/network_structure.txt",
                'w', encoding='utf-8') as file:
                file.write(f"""NETWORK INFORMATION \n
        TIME SIMULATION
        dt = {defaultclock.dt/ms} ms
        duration = {duration/second} s

        PARAMETERS
        O_beta = {O_beta}
        O_delta = {O_delta}

        STRUCTURE PARAMETERS
        g = {g}
        p_e = {p_e}
        p_i = {p_i}
        s = {s}
        dt : {k_NG.dt} ms')
        dt_sam : {k_NG.dt_sam} ms')
        rate_in = {rate_in/Hz} Hz (single external neuron)
        rate_in = {(rate_in/Hz) * 160} Hz(total external neurons)

        RECURRENT ELEMENT
        excitatory neurons = {N_e}
        inhibitory neurons = {N_i}
        excitatory synapses = {len(exc_syn.i)}
        inhibitory synapses = {len(inh_syn.i)}
        ________________________________________\n
        astrocytes = {N_a}
        syn to astro connection = {len(ecs_syn_to_astro.i)}
        astro to syn connection = {len(ecs_astro_to_syn.i)}
        ___________________________________________\n""")
    ###################################################################################################

    # ## PLOTS #########################################################################################
    if args.p:
        fig1, ax1 = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios': [3, 1]},
                                figsize=(12, 14), num=f'NG_network_{rate_in/Hz:.1f}_ph_'+grid_name+f'_run {enu+1}')
        step = 4
        ax1[0].plot(spikes_exc_mon.t[np.array(spikes_exc_mon.i)%step==0]/second, 
                    spikes_exc_mon.i[np.array(spikes_exc_mon.i)%step==0], '|', color='C3')
        ax1[0].plot(spikes_inh_mon.t[np.array(spikes_inh_mon.i)%step==0]/second, 
                    spikes_inh_mon.i[np.array(spikes_inh_mon.i)%step==0]+N_e, '|', color='C0',)
        ax1[0].plot(astro_mon.t[np.array(astro_mon.i)%step==0]/second, 
                    astro_mon.i[np.array(astro_mon.i)%step==0]+(N_e+N_i),'|' , color='green')
        ax1[0].set_xlabel('time (s)')
        ax1[0].set_ylabel('cell index')

        ax1[1].plot(firing_rate.t[:]/second, firing_rate.smooth_rate(window='gaussian', width=1*ms), color='k')
        ax1[1].grid(linestyle='dotted')
        ax1[1].set_ylabel('rate (Hz)')
        ax1[1].set_xlabel('time (ms)')

        # hist_step = 20
        # bin_size = (duration/ms)/((duration/ms)//hist_step)*ms
        # print(f'bin size for hist: {bin_size}')
        # spk_count, bin_edges = np.histogram(np.r_[spikes_exc_mon.t/ms,spikes_inh_mon.t/ms], 
        #                                     int(duration/ms)//hist_step)
        # rate = double(spk_count)/(N_e+N_i)/bin_size
        # ax1[1].plot(bin_edges[:-1], rate, '-', color='k')
        # ax1[1].set_ylabel('rate (Hz)')
        # ax1[1].set_xlabel('time (ms)')
        # ax1[1].grid(linestyle='dotted')

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

        fig3, ax3 = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True,
                                    num='Physical space', figsize=(15,7))

        ax3[0].title.set_text('Neurons grid')
        ax3[0].scatter(exc_neurons.x/mmeter, exc_neurons.y/mmeter, marker='o', color='red')
        ax3[0].scatter(inh_neurons.x/mmeter, inh_neurons.y/mmeter, marker='o', color='C0')
        ax3[0].set_xlabel(r'x ($\mu$m)')
        ax3[0].set_ylabel(r'y ($\mu$m)')

        ax3[1].title.set_text('Astrocytes grid')
        ax3[1].scatter(astrocyte.x/mmeter, astrocyte.y/mmeter, marker='o', color='green')
        ax3[1].set_xlabel(r'x ($\mu$m)')
        ax3[1].set_ylabel(r'y ($\mu$m)')

        fig4, ax4 = plt.subplots(nrows=2, ncols=1, 
                                num='Population firing rate')

        fr_exc = firing_rate_exc.smooth_rate(window='gaussian', width=1*ms)
        fr_inh = firing_rate_inh.smooth_rate(window='gaussian', width=1*ms)

        ax4[0].plot(firing_rate_exc.t[:]/second, fr_exc[:], color='C3')
        ax4[1].plot(firing_rate_inh.t[:]/second, fr_inh[:], color='C0')
        ax4[0].set_ylabel('rate (Hz)')
        ax4[1].set_ylabel('rate (Hz)')
        ax4[0].grid(linestyle='dotted')
        ax4[1].grid(linestyle='dotted')

        fig5, ax5 = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(14,6),
    							num=f'Impinging postsynaptic current: g:{g} s{s} we{w_e}')
        ax5[0].plot(neurons_mon.t[:]/second, neurons_mon.I_exc[0,:]/pA, color='C3')
        ax5[0].set_ylabel(r'$I_{exc}^{rec}$ (pA)')
        ax5[0].grid(linestyle='dotted')

        ax5[1].plot(neurons_mon.t[:]/second, neurons_mon.I_inh[0,:]/pA, color='C0')
        ax5[1].set_ylabel(r'$I_{inh}^{rec}$ (pA)')
        ax5[1].grid(linestyle='dotted')

        ax5[2].plot(neurons_mon.t[:]/second, neurons_mon.I_syn_ext[0,:]/pA, color='C1')
        ax5[2].set_ylabel(r'$I_{ext}$ (pA)')
        ax5[2].set_xlabel('time (s)')
        ax5[2].grid(linestyle='dotted')


        ## Connectivity plot
        if args.cp:
            connectivity_plot(exc_syn, source='Exc', target='Exc+Inh',  name='Exitatory connection',
                            color_s='red', color_t='indigo', size=5, lw=0.3)
            connectivity_plot(inh_syn, source='Inh', target='Exc+Inh', name='Inhibitory connection',
                            color_s='C0', color_t='indigo', size=5, lw=0.3)                
            connectivity_plot(ecs_astro_to_syn, source='Astro', target='Exc syn',name='Astro_to_syn',   
                            color_s='green', color_t='red', size=5, lw=0.3)
            connectivity_plot(ecs_syn_to_astro, source='Exc syn', target='Astro', name='Syn_to_Astro',  
                            color_s='red', color_t='green', size=5, lw=0.3)
            connectivity_plot(astro_to_astro, source='Astro', target='Astro', name='Astro_to_Astro',
                            color_s='green', color_t='green', size=5, lw=0.3)
    
    
    del spikes_exc_mon, spikes_inh_mon, astro_mon
    del firing_rate_exc, firing_rate_inh, firing_rate
    del neurons_mon, mon_LFP, var_astro_mon
    # # device.delete()
    plt.show()
