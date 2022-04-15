"""
Power spectrum analysis for EI network in the precence of STP for population
firing rate and LFP. The spectrum is averaged over different trial
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="""Firirng rate and LFP spectrum analysis of EI network with STP
												Select the network with parameters g, s and input v_in""")
parser.add_argument('g', type=float, help="g value, degrees of balance")
parser.add_argument('s', type=float, help="s value, external-inh factor")
parser.add_argument('v_in', type=float, help="input frequency value")
args = parser.parse_args()

g = args.g
s = args.s
we = 0.05
rate_in = args.v_in

## Load data #############################################################################
name = f'EI_net_STP/Network_pe_g{g}_s{s}_we{we}/v_in{rate_in}/data'

freq_fr = np.load(name+f'/trial-1'+'/freq_fr.npy')
freq_LFP = np.load(name+f'/trial-1'+'/freq_LFP.npy')

spectrum_fr_list = [np.load(name+f'/{trl}'+'/spectrum_fr.npy') for trl in os.listdir(name)]
spectrum_LFP_list = [np.load(name+f'/{trl}'+'/spectrum_LFP.npy') for trl in os.listdir(name)]
###########################################################################################

## Analisis ###############################################################################
spectrum_fr = np.array(spectrum_fr_list).mean(axis=0)
spectrum_LFP = np.array(spectrum_LFP_list).mean(axis=0)

##Plots
fig3, ax3 = plt.subplots(nrows=1, ncols=2, figsize=(12,6),
								num=f"Power Spectrum - {rate_in} - over {len(os.listdir(name))} Trials")

ax3[0].set_title('Population Firing Rate')
ax3[0].plot(freq_fr, spectrum_fr, color='k')
# ax3[0].plot(freq_fr, 1/((freq_fr)**2), lw=0.7, alpha=0.9)
ax3[0].set_xlabel('frequency (Hz)')
ax3[0].set_xlim([-10,500])
# ax3[0].set_ylim([-0,0.02])
ax3[0].grid(linestyle='dotted')

ax3[1].set_title('LFP')
ax3[1].plot(freq_LFP, spectrum_LFP, color='C5')
ax3[1].plot(freq_LFP, 1/(freq_LFP**2), lw=0.7, alpha=0.9)
ax3[1].set_xscale('log')
ax3[1].set_yscale('log')
ax3[1].set_xlabel('frequency (Hz)')
ax3[1].grid(linestyle='dotted')
plt.savefig(name+'/'+f"Power Spectrum - {rate_in} - over {len(os.listdir(name))} Trials.png")
plt.show()