"""
From data in "Network_pe_v_##.#_fIcurve" returns characteristic v_in/v_out curve 
of E-I network for STP synapeses and inh neurons received strong external input.
Equivalent to plot_fr.py
"""
import argparse
import numpy as np 
import matplotlib.pyplot as plt
from brian2 import *

parser = argparse.ArgumentParser(description='f-I curve with STP for different value of g')
parser.add_argument('g', type=float, help="g value, degrees of balance")
parser.add_argument('s', type=float, help="s value, external-inh factor")
parser.add_argument('we', type=float, help="we value, excitatory synaptic strengh")
args = parser.parse_args()

g = args.g
s = args.s
we = args.we

fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=(12,5.5),
						 num=f'v_in-v_exc characteristic curve g:{g} s:{s} we:{we:.2f}')
ax2[0].set_title('Average Population Faring Rate')
ax2[0].set_ylabel(r'$\nu_{S}^{rec}$ (Hz)')
ax2[0].set_xlabel(r'$\nu_{in}$ (Hz)')
ax2[0].grid(linestyle='dotted')

ax2[1].set_title('I/E Firing rate Ratio')
ax2[1].set_ylabel(r'$I_{fr}$/$E_{fr}$')
ax2[1].set_xlabel(r'$\nu_{in}$ (Hz)')
ax2[1].grid(linestyle='dotted')

rates_in =[40.0,45.0,50.0,55.0,60.0,65.0,70.0]
for rate_in in rates_in:
	name_folder = f"EI_net_STP/Network_pe_g{g}_s{s}_we{we}/v_in{rate_in}/f-I_curve"
	## LOAD VARIABLES ###################################################################
	duration = np.load(f'{name_folder}/duration.npy')*second

	# Excitatory neurons variable
	pop_exc = np.load(f'{name_folder}/fr_exc.npy')
	# Inhibitory neurons variable
	pop_inh = np.load(f'{name_folder}/fr_inh.npy')

	if rate_in == 40.0: 
		ax2[0].scatter(rate_in, pop_exc.mean(), marker='o', color='C3',label='Exc')
		ax2[0].scatter(rate_in, pop_inh.mean(), marker='s', color='C0',label='Inh')
	else:
		ax2[0].scatter(rate_in, pop_exc.mean(), marker='o', color='C3')
		ax2[0].scatter(rate_in, pop_inh.mean(), marker='s', color='C0')

	ax2[1].scatter(rate_in, pop_inh.mean()/pop_exc.mean(), marker='d', color='k')

	ax2[0].legend()

plt.savefig(f'EI_net_STP/Network_pe_g{g}_s{s}_we{we}/v_in-v_exc characteristic curve g:{g} s:{s} we:{we:.2f}.png')
plt.show()




