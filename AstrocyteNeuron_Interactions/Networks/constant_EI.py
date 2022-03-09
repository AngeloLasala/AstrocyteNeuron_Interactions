"""
Module of network parameters that could be changes over simulation
"""
# TIME CONSTANT
dt = 0.05 #ms            #0.05ms
duration = 2.3 #s    #2.3s
trans_time = 300 #ms          #300ms

# index: excitatory neurron index used to monitor variable
index = 400

# NETWORK STRUCTURE
# how much the ext-inh synaptic strength is highr
#  than ext-exc synaptic strength
alpha = 1.0

# Recurrent balance degrees: g = pe/pi, IMPORTANT: I always fix pe
# g = 5: recurrent balance
# g < 5: inhibitory prevails
# g > 5: excitatory prevails 
g = 0.25   
p_e = 0.05  

