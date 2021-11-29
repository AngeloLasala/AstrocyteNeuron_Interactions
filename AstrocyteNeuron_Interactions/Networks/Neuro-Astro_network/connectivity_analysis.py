"""
Gives more information about connectivity in the Neuron-Glia network
"""
import argparse
import numpy as np 
import matplotlib.pyplot as plt
from brian2 import *

def neurons_postsynapses(index, syn_i, syn_j):
	"""
	Select a postsynaptic neurons and return the ideces of
	presynaptic neurons (ingoing synapses of lesected neuron).

	Parameters
	----------
	index : integer
			neuron's index

	syn_i : numpy array 
			indeces of presynapstic neurons. 'See brian2.synapses.synapses.Synapses.i'

	syn_j : numpy array
			indeces of presynapstic neurons. 'See brian2.synapses.synapses.Synapses.j'

	Returns
	-------
	post_syn : list
		post synapses indeces of selected neuron
	"""
	indeces = np.where(syn_j==index)
	outgoing_n = syn_i[indeces]
	return outgoing_n

def from_astro_to_neuron(astro_index, astro_to_syn_i, astro_to_syn_j, syn_j):
	"""
	Return neurno index of astrocyte connection 

	Parameters
	----------
	astro_index : integer
				 astro index

	astro_to_syn_i : numpy array
					indeces of connected astrocyte in asto_to_sun connection. See 'brian2.synapses.synapses.Synapses.i'

	astro_to_syn_j : numpy array
					indeces of connected synapses in asto_to_sun connection. See brian2.synapses.synapses.Synapses.i'

	syn_j : numpy array 
			indeces of presynapstic neurons. 'See brian2.synapses.synapses.Synapses.j'
			
	Returns
	-------
	neurons_index: numpy array
		neuron indeces connected with astrocyte
	"""
	index = astro_index
	astro_position = np.where(astro_to_syn_i==index)

	synapses_con_indeces = astro_to_syn_j[astro_position]
	neurons_index = np.unique(syn_j[synapses_con_indeces])
	return neurons_index
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Advanced connectivity connection')
	parser.add_argument('file', type=str, help="file's name of network in 'Neuro_Astro_network' folder")
	parser.add_argument('-n', type=int, default=0, help='Neuron index, defalut=0')
	parser.add_argument('-a', type=int, default=0, help='Astrocyte index, defalut=0')
	args = parser.parse_args()

	N_e = 3200
	N_i = 800
	N_a = 3200

	name=args.file

	## Load parameters
	exc_i = np.load(f'{name}/exc_syn.i.npy')
	exc_j = np.load(f'{name}/exc_syn.j.npy')
	inh_i = np.load(f'{name}/inh_syn.i.npy')+N_e
	inh_j = np.load(f'{name}/inh_syn.j.npy')
	astro_to_syn_i = np.load(f'{name}/ecs_astro_to_syn.i.npy')
	astro_to_syn_j = np.load(f'{name}/ecs_astro_to_syn.j.npy')
	syn_to_astro_i = np.load(f'{name}/ecs_syn_to_astro.i.npy')
	syn_to_astro_j = np.load(f'{name}/ecs_syn_to_astro.j.npy')
	###########################################################################################
	
	neuron_index = args.n
	if neuron_index < N_e : neuro_type = 'exitatory'
	if neuron_index >= N_e: neuro_type = 'inhibitory'

	print(f'NEURON: {neuron_index} ({neuro_type})')
	print(f'===============================')
	print('SYNAPSES')
	print(f'exc_syn outgoing: {len(exc_i[exc_i==neuron_index])}')
	print(f'exc_syn ingoing: {len(exc_j[exc_j==neuron_index])}')
	print(f'inh_syn outgoing: {len(inh_i[inh_i==neuron_index])}')
	print(f'inh_syn ingoing: {len(inh_j[inh_j==neuron_index])}')
	print('')

	print(f'neuron {neuron_index} is postsynaptic')
	print(f'exc presynaptic: {neurons_postsynapses(neuron_index, exc_i, exc_j)}')
	print(f'inh presynaptic: {neurons_postsynapses(neuron_index, inh_i, inh_j)}')
	
	# print("""the connection between astro e exc_syn is specified on basis 
	# of the postaynaptic neuron index(position)""")
	print(f'total exc syn: {len(exc_i)}')
	print(f'total inh syn: {len(inh_i)}')
	print(f'total trip syn: {len(syn_to_astro_i)}')
	print(f'% tri on exc: {100*len(syn_to_astro_i)/(len(exc_i)):.1f}%')
	print(f'% tri on total: {100*len(syn_to_astro_i)/(len(exc_i)+len(inh_i)):.1f}%')
	print('')
	
	# synapses indeces with {neuron_index} as postsynapitic button
	syn_indeces = np.where(exc_j==neuron_index)[0]
	print(f'synapses with {neuron_index} as postsynapitic button:')
	print(syn_indeces)
	# check if synapses has a connection with an astrocyte
	check=np.isin(syn_indeces,syn_to_astro_i)
	for num,k in enumerate(syn_indeces):
		if check[num] : tripartite = f', is a triparte syn with astrocyte {syn_to_astro_j[np.where(syn_to_astro_i==k)]} '+' \U0001F534'+'\U0001F7E2'
		else : tripartite = f', is a simple syn'+ ' \U0001F534'
		print(f'the exc_syn {k} connects pre {exc_i[k]} with post {exc_j[k]}'+tripartite)
	print('=============================================================================')
	print('ASTROCYTE')
	astro_index = args.a
	print(f'astrocyte: {astro_index}')
	print(f'connected syn: {len(astro_to_syn_j[np.where(astro_to_syn_i==astro_index)])}')
	n = from_astro_to_neuron(astro_index, astro_to_syn_i, astro_to_syn_j, exc_j)
	print(f'connected neur: {len(n)} {n}')

	neur = [len(from_astro_to_neuron(k, astro_to_syn_i, astro_to_syn_j, exc_j)) 
			for k in np.arange(N_a)]
			
	## Plots 
	fig1, ax1 = plt.subplots(nrows=1, ncols=1, num='astrocyte connection distibuction')
	ax1.bar(np.arange(N_a),neur)
	ax1.set_xlabel('astrocyte index')
	ax1.set_ylabel('number of connected neurons')

	plt.show()
