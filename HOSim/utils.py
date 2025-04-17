import numpy as np
import matplotlib.pyplot as plt

def get_initial_condition(params):
    Hs = params["Hs"]
    Tp = params["Tp"]
    gamma = params["gamma"]
    g = params["gravity"]
    length = params["length"]
    modes = params["modes"]

    fp = 1 / Tp
    k0 = 2 * np.pi / length

    f = np.arange(1, modes+1)*k0
    sigma = 0.07 * (f <= fp) + 0.09 * (f > fp)
    Sf = (g**2 / (2 * np.pi)**4) * f**(-5) * \
        np.exp(-1.25 * (fp / f)**4) * \
        gamma**np.exp(-((f / fp - 1)**2) / (2 * sigma**2))

    alpha = Hs**2 / 16 / np.trapezoid(Sf, f)
    Sf = alpha * Sf

    f = np.insert(f, 0, 0)
    Sf = np.insert(Sf, 0, 0)

    Sf_max = alpha * g**2 * (2 * np.pi)**(-4) * fp**(-5) * np.exp(-5/4) * gamma

    assert Sf[-1]/Sf_max <= 0.01, "Not enough modes to represent seastate!"

    phases = np.exp(np.random.rand(modes+1)*2*np.pi*1.j)
    eta_hat = np.sqrt(Sf * k0 * 0.5) * phases * 2 * modes
    phi_hat = eta_hat[1:] * np.exp(-1.j * np.pi / 2) * np.sqrt(g / f[1:])
    phi_hat = np.insert(phi_hat, 0, 0)

    return np.concatenate((eta_hat, phi_hat))