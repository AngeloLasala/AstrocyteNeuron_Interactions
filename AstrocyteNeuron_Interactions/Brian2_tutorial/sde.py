"""
How resolve stochastic differential equations with Brian 2

Wiener and OU processes are simulated
"""
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *

# Wiener process

dt = 0.01*second
np.random.seed(156478)
D = 1/second
k = 0.5/second
X0 = 0

G = NeuronGroup(10, 'dX/dt = D*xi*second**0.5: 1',
                dt=dt, method='milstein')
G.X = X0
mon_G = StateMonitor(G, 'X', record=True)
net_G = Network(G, mon_G)
net_G.run(60*second)

O = NeuronGroup(10, 'dX/dt = -k*X + D*xi*second**0.5: 1',
                dt=dt, method='milstein')
O.X = X0
mon_O = StateMonitor(O, 'X', record=True)
net_O = Network(O, mon_O)
net_O.run(60*second)


fig1 = plt.figure(num= 'Wiener and OU process', figsize=(12,10))
ax11 = fig1.add_subplot(2,1,1)
ax12 = fig1.add_subplot(2,1,2)

for i in range(10):
    ax11.plot(mon_G.t[:], mon_G.X[i])
ax11.grid(linestyle='dotted')

for i in range(10):
    ax12.plot(mon_O.t[:], mon_O.X[i])
ax12.grid(linestyle='dotted')

plt.show()
