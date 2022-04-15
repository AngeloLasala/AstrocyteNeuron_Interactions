"""
From data in "Network_pe_v_##.#_fIcurve" returns characteristic v_in/v_out curve 
of E-I network for STP synapeses and inh neurons received strong external input.
Equivalent to plot_fr.py
"""
import argparse
import numpy as np 
import matplotlib.pyplot as plt
from brian2 import *

def CV_population(v_population):
	"""
	Coeficient of variation for each neuron in a given population

	Parameters
	----------
	v_population : 2d array
					Membran potential for each neurons into a population
					size = (N_x, duration/dt)

	Returns
	-------
	CV_list : 1d array
			array of CV
	"""
	v_population = np.abs(v_population)
	standard_dev_v = np.std(v_population, axis=1)
	mean_v = np.mean(v_population, axis=1)
	return standard_dev_v/mean_v

def crosscorr(x, y, max_lag, bootstrap_test=False, color='k'):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    cross_corr = []
    for d in np.arange(0,max_lag,50):
        cc = 0
        for i in range(len(x)-d):
            cc += (x[i] - x_mean) * (y[i+d] - y_mean)
        cc = cc / np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
        cross_corr.append(cc)

    if bootstrap_test:
        cross_corr_s = []
        for i in range(100):
            xs, ys = x, y
            np.random.shuffle(xs)
            np.random.shuffle(ys)
            xs_mean = np.mean(xs)
            ys_mean = np.mean(ys)
            cross_corr_s_i = []
            for d in range(max_lag):
                cc = 0
                for i in range(len(x)-d):
                    cc += (xs[i] - xs_mean) * (ys[i+d] - ys_mean)
                cc = cc / np.sqrt(np.sum((xs - xs_mean)**2) * np.sum((ys- ys_mean)**2))
                cross_corr_s_i.append(cc)
            cross_corr_s.append(cross_corr_s_i)
        meancc = np.mean(np.array(cross_corr_s), axis=0)
        stdcc = np.std(np.array(cross_corr_s), axis=0)
        plt.plot(meancc - 3*stdcc, 'crimson', lw=0.5)
        plt.plot(meancc, 'crimson', lw=0.5)
        plt.plot(meancc + 3*stdcc, 'crimson', lw=0.5)
        plt.fill_between(np.arange(0, max_lag), meancc + 3*stdcc, meancc - 3*stdcc, color='crimson', alpha=0.6)
    return cross_corr


def neurons_firing(t_spikes, neurons_i, time_start, time_stop):
    """
    Firing rate of single neurons
    Distribuction of neurons spikes activity

    Parameters
    ----------
    t_spikes : array
            spiking time of all neurons (comes from SpikesMonitor group of Brian2),
            there are expressed in second. For istance:
            monitor = SpikesMonitor(neurons)
            t_spikes = monitor.t

    neurons_i : array
            indexes array come from SpikesMonitor group of Brian2, for istance
            monitor = SpikesMonitor(neurons)
            neurons_i = monitor.i
    
    time_start : float
                start of time window, second
    
    stop_start : float
                stop of time window, second

    Returns
    -------
    neurons_fr : array
                list of neuron's firing rate in time_start-time_stop windows
    """
    
    # indeces: indeces of firing neurons
    indeces = np.unique(neurons_i)
    time = (time_stop-time_start)*second

    neurons_fr = []
    for ind in indeces:
        spikes = [spk for spk in t_spikes[neurons_i==ind] if (spk > time_start and spk < time_stop)]
        firing_rate = len(spikes)/time
        neurons_fr.append(firing_rate)

    return neurons_fr

def max_firing_rate(neurons_fr, k=2):
	"""
	Return neurons ideces with greater firing rate
	"""
	neurons_fr = np.array(neurons_fr)
	max_firing_rate = []
	for times in range(k):
		max_value = np.max(neurons_fr)
		index = np.where(neurons_fr == max_value)[0][0]
		max_firing_rate.append(index)
		neurons_fr = np.delete(neurons_fr, index)
	return np.array(max_firing_rate)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='f-I curve with STP for different value of g')
	parser.add_argument('rate_in', type=float, help="rate-in , input firing rate")
	parser.add_argument('g', type=float, help="g value, degrees of balance")
	parser.add_argument('s', type=float, help="s value, external-inh factor")
	args = parser.parse_args()

	rate_in = args.rate_in
	g = args.g
	s = args.s
	we = 0.05

	## Coeficient of variation and Cross correlation
	type_connection = ['f-I_curve', 'f-I_curve_no_connection']
	fig1, ax1 = plt.subplots(nrows=1, ncols=3, figsize=(17,5), 
							num=f'Cross-Correlation g {g} - rate_in {rate_in}')
 	
	for types in type_connection:
		name_folder = f"EI_net_STP/Network_pe_g{g}_s{s}_we{we}/v_in{rate_in}/{types}"
		if types == type_connection[0] : ttt = 'STP'
		if types == type_connection[1] : ttt = 'no connection'

		v_exc = np.load(f'{name_folder}/v_exc.npy')
		v_inh = np.load(f'{name_folder}/v_inh.npy')

		CV_exc = CV_population(v_exc)
		CV_inh = CV_population(v_inh)
		
		## Cross Correlation
		spk_exc_t = np.load(f'{name_folder}/spikes_exc_mon_t.npy')
		spk_exc_i = np.load(f'{name_folder}/spikes_exc_mon_i.npy')
		spk_inh_t = np.load(f'{name_folder}/spikes_inh_mon_t.npy')
		spk_inh_i = np.load(f'{name_folder}/spikes_inh_mon_i.npy')

		exc_neuron_fr = np.array(neurons_firing(spk_exc_t, spk_exc_i, 0.5, 3.0))
		inh_neuron_fr = np.array(neurons_firing(spk_inh_t, spk_inh_i, 0.5, 3.0))

		max_exc_neurons = max_firing_rate(exc_neuron_fr, k=2)
		max_inh_neurons = max_firing_rate(inh_neuron_fr, k=2)
		
		cc_exc_to_exc = crosscorr(v_exc[max_exc_neurons[0]], v_exc[max_exc_neurons[1]], 5000)
		cc_inh_to_inh = crosscorr(v_inh[max_inh_neurons[0]], v_inh[max_inh_neurons[1]], 5000)
		cc_exc_to_inh = crosscorr(v_exc[max_exc_neurons[0]], v_inh[max_inh_neurons[0]], 5000)
		
		ax1[0].set_title(f'cross-correlation E-E')
		ax1[0].plot(cc_exc_to_exc, label=ttt)
		ax1[1].set_title(f'cross-correlation I-I')
		ax1[1].plot(cc_inh_to_inh, label=ttt)
		ax1[2].set_title(f'cross-correlation E-I')
		ax1[2].plot(cc_exc_to_inh, label=ttt)
		
		for axes in [ax1[0], ax1[1], ax1[2]]:	
			axes.set_xlabel('Lags - (i*50*0.05 ms)')
			axes.grid(linestyle='dotted')
			axes.legend()
		
		print(f'CV POPULATION - {ttt}')
		print(f'exc : {CV_exc.mean()}')
		print(f'inh : {CV_inh.mean()}')
		print()

	plt.figure('Max neurons')
	t = np.linspace(0, 3, len(v_exc[0]))
	plt.plot(t, v_exc[max_exc_neurons[0]])
	plt.plot(t, v_exc[max_exc_neurons[1]])
	plt.show()
