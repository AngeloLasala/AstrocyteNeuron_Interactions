"""
Module of network parameters of Neuro-glia network that could be changes over simulation
"""
# NETWORK SIZE
# Note: both neurons and astrocyte arranged on square lattice. To have nice connection one-to-one
# total number of neurons (N_e+N_i) must be egual to astrocytes (N_a) AND must be a pefect square (sqrt(N_a+N_i) = int)
# example: 
# grd size = 70*70
# N_e = 3920
# N_i = 980
# N_a = 4900
# FOr "-lenear" no spatial arrangement is used to built syn to astro connection and so there are no limitation

N_e = 3200
N_i = 800
N_a = 4000

# TIME CONSTANT
dt = 0.1 #ms            #0.05ms
duration = 10          #s   
trans_time = 300 #ms    #300ms

# Simulation, expecialy for log time one, is divide in different and 
# consecutive sub-simulation of total duration equals to 'duration'. Thus,
# total time simulation is duration*windows.
# Note: at the end of each sub-simulation, all variable are saved in specific
# folder, all 'Monitor group are deleted' such that the RAM is clear and 
# next sub-simulation starts. 
windows = 10                


# index: excitatory neurron index used to monitor variable
index = 400

# NETWORK STRUCTURE
# how much the ext-inh synaptic strength is highr
# than ext-exc synaptic strength
s = 1.00

# Recurrent balance degrees: g = pe/pi, IMPORTANT: I always fix pe
# g = 5: recurrent balance
# g < 5: inhibitory prevails
# g > 5: excitatory prevails 
g = 0.25 
p_e = 0.05  

