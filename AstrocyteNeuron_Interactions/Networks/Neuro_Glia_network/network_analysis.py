"""
Network analysis based on informations come from multiple run
of 'plot_network.py' file

As module contain some usefull fuction to deal with dynamical variables
"""
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *

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

def selected_window(v, start, stop, duration=12):
	"""
	Given time evolution of variable 'v' returns its time evolution 
	over selected window 
	
	Parameters
	----------
	v: array
		Time evoulution of varible over a proprer integration steps

	start: float
		initial time of selected window expressed in second (ms)

	stop: float
		final time of selected window expressed in second (ms)

	duration: <class 'brian2.units.fundamentalunits.Quantity'> (optional)
		total time simulation. Default=12 second

	Returns
	-------
	v_window: array
		time evolution of variable over selected window
	"""

	dt = duration/len(v)              # expressed in second
	start_position = int(start/dt)
	stop_position = int(stop/dt)   

	v_window = v[start_position:stop_position] 
	return v_window


if __name__ == "__main__":
	# onset of GRE dependent on Amplitude (I_ext)
	# the input data come from 'mygrid' and 'profgrid' folder
	I_ex = [100.0,105.0,110.0,115.0,120.0]
	timing = {100.0:2.26, 105.0:2.18, 110.0:2.13, 115.0:2.11, 120.0:2.09}



	## Plots ##############################################################################
	fig1, ax1 = plt.subplots(nrows=1, ncols=1, num='GRE based on I_ex Amplitude')

	for i in I_ex:
		ax1.plot(i, timing[i], ls='', marker='o', markersize=6, color='C2')
	ax1.set_xlabel(r'$I_{ex}$ (pA)')
	ax1.set_ylabel(r'GRE time (s)')
	ax1.grid(linestyle='dotted')

	plt.show()