"""
Neuronal network simulation using Brian 2

- "Modeling euron-glia interaction with Brian 2 simulator", Stimberg et al (2017)
"""
import matplotlib.pyplot as plt
from brian2 import *
from AstrocyteNeuron_Interactions.Brian2_tutorial.connectivity import Connectivity_plot

# Parameters
N_e = 16
N_i = 4
duration = 1.0*second  # Total simulation time

#Neurons parameters
E_l = -60*mV           # Leak reversal potential
g_l = 9.99*nS          # Leak conductance
E_e = 0*mV             # Excitatory synaptic reversal potential
E_i = -80*mV           # Inhibitory synaptic reversal potential
C_m = 198*pF           # Membrane capacitance
tau_e = 5*ms           # Excitatory synaptic time constant
tau_i = 10*ms          # Inhibitory synaptic time constant
tau_r = 5*ms           # Refractory period
I_ex = 95*pA          # External current
V_th = -50*mV          # Firing threshold
V_r = E_l              # Reset potential 

#Synapse parameters
w_e = 0.05*nS          # Excitatory synaptic conductance
w_i = 1.0*nS           # Inhibitory synaptic conductance
U_0 = 0.6              # Synaptic release probability at rest
Omega_d = 2.0/second   # Synaptic depression rate
Omega_f = 3.33/second  # Synaptic facilitation rate

#Neurons
neuron_eqs = """
dv/dt = (g_l*(E_l-v) + g_e*(E_e-v) + g_i*(E_i-v) + I_ex)/C_m : volt (unless refractory)
dg_e/dt = -g_e/tau_e : siemens
dg_i/dt = -g_i/tau_i : siemens
"""
Np = 1
poisson = PoissonGroup(Np, rates=30.0*Hz)
neurons = NeuronGroup(N_e+N_i, model=neuron_eqs, method='rk4',
                     threshold='v>V_th', reset='v=V_r', refractory='tau_r')

neurons.v = 'E_l + rand()*V_th-E_l'
neurons.g_e = 'rand()*w_e'
neurons.g_i = 'rand()*w_i'

exc_neurons = neurons[:N_e]
inh_neurons = neurons[N_e:]

syn_model = """
du_s/dt = -Omega_f * u_s : 1 (event-driven)
dx_s/dt = Omega_d * (1-x_s) : 1 (event-driven)
"""

action="""
u_s += U_0*(1-u_s)
r_s = u_s*x_s
x_s -= r_s"""

exc="""
g_e+=w_e*r_s*150
"""

inh="""
g_i+=w_i*r_s
"""

synapse = Synapses(poisson, neurons, model= syn_model, on_pre=action+exc)
exc_syn = Synapses(exc_neurons, neurons, model= syn_model, on_pre=action+exc)
inh_syn = Synapses(inh_neurons, neurons, model= syn_model, on_pre=action+inh)
synapse.connect(condition='j<3')
exc_syn.connect(i=0, j=[6,7,8,9,10,17])
inh_syn.connect(i=1, j=5)

synapse.x_s = 1
exc_syn.x_s = 1
inh_syn.x_s = 1

state_exc_mon = StateMonitor(neurons, ['v', 'g_e', 'g_i'], record=True)
syn_mon = StateMonitor(synapse, ['u_s','x_s'], record=True)
spikes_mon = SpikeMonitor(neurons)
spikes_exc_mon = SpikeMonitor(exc_neurons)
spikes_inh_mon = SpikeMonitor(inh_neurons)

run(duration)
print(spikes_mon.count[:])


#Plots
fig1 = plt.figure(figsize=(10,10))
ax11 = fig1.add_subplot(2,2,1)
ax12 = fig1.add_subplot(2,2,2)
ax13 = fig1.add_subplot(2,2,3)

ax11.plot(state_exc_mon.t/ms, state_exc_mon.v[0], label='neuron 0')
# ax11.plot(state_exc_mon.t/ms, state_exc_mon.v[5], label='neuron 5')
ax11.plot(state_exc_mon.t/ms, state_exc_mon.v[5], label='neuron 5')
ax11.plot(state_exc_mon.t/ms, state_exc_mon.v[10], label='neuron 10')
ax11.set_xlabel('time (ms)')
ax11.set_ylabel('Membran potential (V)')
ax11.set_title('Neurons dynamics')
ax11.grid(linestyle='dotted')
ax11.legend()


ax12.plot(state_exc_mon.t/ms, state_exc_mon.g_i[0], color='red', label='0 inh g')
ax12.plot(state_exc_mon.t/ms, state_exc_mon.g_e[0], color='green', label='0 exc g')
ax12.set_xlabel('time (ms)')
ax12.set_ylabel('Conductance (S)')
ax12.set_title('Conductance dynamics')
ax12.grid(linestyle='dotted')
ax12.legend()

ax13.plot(syn_mon.t/ms, syn_mon.u_s[0], label=r'$u_s$', color='C1')
ax13.plot(syn_mon.t/ms, syn_mon.x_s[0], label=r'$x_s$', color='C4')
ax13.set_xlabel('time (ms)')
ax13.set_ylabel(r'$u_s$, $x_s$')
ax13.set_title('Synapse dynamics')
ax13.grid(linestyle='dotted')
ax13.legend()


fig2 = plt.figure(figsize=(5,5))
ax21 = fig2.add_subplot(1,1,1)

ax21.scatter(spikes_exc_mon.t[:], spikes_exc_mon.i[:], color='C3', marker='|')
ax21.scatter(spikes_inh_mon.t[:], spikes_inh_mon.i[:]+N_e, color='C0', marker='|')
ax21.set_xlabel('time (ms)')
ax21.set_ylabel('neuron index')
ax21.set_title('Raster plot')



plt.show()

