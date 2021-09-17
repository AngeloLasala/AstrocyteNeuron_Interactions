import matplotlib.pyplot as plt
from brian2 import *

if __name__ == "__main__":

    #Parameters
    tau = 1.0*second
    eps = 0.05*second
    a = 0.5

    astro_eqs = """
    dC/dt = (C*(C-a)*(1-C)-h)/eps : 1
    dh/dt = (C-h-b)/tau : 1

    b : 1

    """

    Astrocyte = NeuronGroup(1, astro_eqs, method='rk4')
    Astrocyte.b = 0.5


    M_Astro = StateMonitor(Astrocyte, ['C','h'], record=True)

    run(5*second)
    print(M_Astro.C)

    fig = plt.figure(figsize=(12,5))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    ax1.plot(M_Astro.t/second, M_Astro.C[0], label=f'C')
    ax1.plot(M_Astro.t/second, M_Astro.h[0], label=f'h')
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('C,h (mmolar)')
    ax1.legend()

    ax2.plot(M_Astro.C[0],M_Astro.h[0])
    ax2.set_xlabel('C (mmolar)')
    ax2.set_ylabel('h (mmolar)')

    plt.show()