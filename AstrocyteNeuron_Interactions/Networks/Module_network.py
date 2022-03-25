"""
Module. Usefull function to plot and analyse data about 
Neuro-Glia network.
"""
import numpy as np
from brian2 import *

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


def transient(t, transient):
	"""
	Return a time array without transient time, usefull to avoid 
	transient regime of dynamics.

	Parameters
	----------
	t : <class 'brian2.units.fundamentalunits.Quantity'>
		time array 
	
	transient : float
				transient time expressed in millisecond

	Returns
	-------
	transient_position: integer
						position of transient value in time array

	"""
	time = t/ms
	time_trans = time[time>transient]
	transient_position = len(time)-len(time_trans)
	return transient_position


def smoothing_b(x, window='gaussian', width=None, ddt=defaultclock.dt):
	"""
	Return a smooth version of signal x(t).

	Reference: https://brian2.readthedocs.io/en/2.5.0.1/_modules/brian2/monitors/ratemonitor.html

	Parameters
	----------
	x : ndarray
		input signal 

	window : str
		The window to use for smoothing. It is a string to chose a
		predefined window(``'flat'`` for a rectangular, and ``'gaussian'``
		for a Gaussian-shaped window). In this case the width of the window
		is determined by the ``width`` argument. Note that for the Gaussian
		window, the ``width`` parameter specifies the standard deviation of
		the Gaussian, the width of the actual window is ``4*width + ddt``
		(rounded to the nearest dt). For the flat window, the width is
		rounded to the nearest odd multiple of dt to avoid shifting the rate
		in time.

	width : `Quantity`, optional
		The width of the ``window`` in seconds (for a predefined window).

	ddt : `Quantity`, optional
		sampling sepatation 

	Returns
	-------
	x(t) : `Quantity`
		x(t), smoothed with the given window. Note that
		the values are smoothed and not re-binned, i.e. the length of the
		returned array is the same as the length of input array
		and can be plotted against the same time 't'.
	"""
	if window == 'gaussian':
		width_dt = int(np.round(2*width / ddt))
		# Rounding only for the size of the window, not for the standard
		# deviation of the Gaussian
		window = np.exp(-np.arange(-width_dt,
								width_dt + 1)**2 *
						1. / (2 * (width/ddt) ** 2))
	elif window == 'flat':
		width_dt = int(width / 2 / ddt)*2 + 1
		used_width = width_dt * ddt
		if abs(used_width - width) > 1e-6*ddt:
			logger.info(f'width adjusted from {width} to {used_width}',
						'adjusted_width', once=True)
		window = np.ones(width_dt)
	else:
		raise NotImplementedError(f'Unknown pre-defined window "{window}"')

	return np.convolve(x, window * 1. / sum(window), mode='same')
