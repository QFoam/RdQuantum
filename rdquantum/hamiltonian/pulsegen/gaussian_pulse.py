import numpy as np
def GaussianPulse(amp, gate_time, tau):
    def pulse (t, args):
        # t = t % gate_time
        t0 = gate_time / 2
        a = 0.5 * np.exp(- t0**2 / tau**2)
        return amp * (np.exp(-0.5 * (t-t0)**2 / tau**2) - a) / (1-a)
    return pulse