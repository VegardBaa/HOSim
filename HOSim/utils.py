import numpy as np
import matplotlib.pyplot as plt

def get_initial_condition(params):
    Hs = params["Hs"]
    Tp = params["Tp"]
    gamma = params["gamma"]
    g = params["gravity"]
    length = params["length"]
    modes = params["modes"]

    k0 = 2 * np.pi / length
    k = np.arange(1, modes+1)*k0
    w = np.sqrt(k * g)

    T1 = 0.834 * Tp
    T2 = Tp / 1.408

    sigma = 0.07 * (w <= 5.24/T1) + 0.09 * (w > 5.24/T1)
    gamma = 3.3
    Sw = (155 * Hs**2 / T1**4 / w**5 *
        np.exp(-944 / (T1*w)**4) *
        gamma ** np.exp(-((0.191*w*T1 - 1)/(np.sqrt(2)*sigma))**2))
    Sk = Sw * g / (2 * np.sqrt(k * g))

    phases = np.exp(np.random.rand(modes)*2*np.pi*1.j)
    eta_hat = np.sqrt(Sk * k0 * 0.5) * phases * 2 * modes
    phi_hat = eta_hat * np.exp(-1.j * np.pi / 2) * np.sqrt(g / k)

    eta_hat = np.insert(eta_hat, 0, 0)
    phi_hat = np.insert(phi_hat, 0, 0)

    return np.concatenate((eta_hat, phi_hat))