"""
Network analysis based on informations come from multiple run
of 'plot_network.py' file
"""
import matplotlib.pyplot as plt

# onset of GRE dependent on Amplitude (I_ext)
# the input data come from 'mygrid' and 'profgrid' folder
I_ex = [100.0,105.0,110.0,115.0,120.0]
timing = {100.0:2.26, 105.0:2.18, 110.0:2.13, 115.0:2.11, 120.0:2.09}


## Plots ##############################################################################
fig1, ax1 = plt.subplots(nrows=1, ncols=1, num='GRE based on I_ex Amplitude')

for i in I_ex:
	ax1.plot(i, timing[i], ls='', marker='o', markersize=5, color='C2')
ax1.set_xlabel(r'$I_{ex}$ (pA)')
ax1.set_ylabel(r'GRE time (s)')

plt.show()