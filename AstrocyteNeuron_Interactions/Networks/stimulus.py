import numpy as np
import matplotlib.pyplot as plt
# from brian2 import *

t = np.arange(0,200,0.5)
s = [int((i % (50))<20) for i in t]

plt.figure()
plt.plot(t,s)
plt.show()