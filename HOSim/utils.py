import numpy as np
import matplotlib.pyplot as plt

def get_initial_condition(params):
    Hs = params["Hs"]
    Tp = params["Tp"]
    gamma = params["gamma"]
    g = params["gravity"]
    length = params["length"]
    modes = params["modes"]
    mHOS = params["mHOS"]

    k0 = 2 * np.pi / length
    T1 = 0.834 * Tp

    if params["2d"]:
        beta = params["beta"]

        kx = np.linspace(-modes, modes-1, 2*modes, dtype=np.float32) * k0
        KX, KY = np.meshgrid(kx, kx)
        theta = np.atan2(KY, np.abs(KX))
        K = np.sqrt(KX**2 + KY**2)
        K[modes, modes] = 1
        W = np.sqrt(K * g)

        sigma = 0.07 * (W <= 5.24/T1) + 0.09 * (W > 5.24/T1)
        SW = (155 * Hs**2 / T1**4 / W**5 *
            np.exp(-944 / (T1*W)**4) *
            gamma ** np.exp(-((0.191*W*T1 - 1)/(np.sqrt(2)*sigma))**2))
        SK = SW * g / (2 * np.sqrt(K * g))

        GT = 1 / beta * np.cos(2 * np.pi * (theta) / (4 * beta))**2
        GT[(theta < -beta) | (theta > beta)] = 0

        spec = SK * GT / K
        spec[modes, modes] = 0

        spec = spec / np.trapezoid(np.trapezoid(spec, kx, axis=0), kx) * 2 * Hs**2 / 16

        phases = np.zeros((2*modes, 2*modes), dtype=np.complex64)
        phases[:modes+1, :modes+1] = np.exp(1.j*2*np.pi*np.random.rand(modes+1, modes+1))
        phases[1:modes, modes+1:] = np.exp(1.j*2*np.pi*np.random.rand(modes-1, modes-1))
        phases[modes+1:, 1:modes] = np.conj(phases[1:modes, modes+1:][::-1, ::-1])
        phases[0, 0] = 1
        phases[modes, 0] = 1
        phases[0, modes] = 1
        phases[modes, modes] = 1
        phases[modes:, modes:] = np.conj(phases[1:modes+1, 1:modes+1][::-1, ::-1])
        phases[modes+1:, 0] = np.conj(phases[1:modes, 0][::-1])
        phases[0, modes+1:] = np.conj(phases[0, 1:modes][::-1])

        SIGN = np.fft.fftshift(np.sign(KX))
        eta_hat = np.sqrt(np.fft.fftshift(spec)*k0*k0*0.5) * phases * 2 * modes * 2 * modes
        phi_hat = eta_hat * np.exp(-1.j * SIGN * np.pi / 2) * np.sqrt(g / np.fft.fftshift(K))

        index = (np.linspace(-modes, modes-1, 2*modes) < modes * 2 / (mHOS + 1)) & (np.linspace(-modes, modes-1, 2*modes) > -modes * 2 / (mHOS + 1))
        eta_hat[index, index] = 0
        phi_hat[index, index] = 0

        eta_hat = np.fft.rfft2(np.real(np.fft.ifft2(eta_hat)))
        phi_hat = np.fft.rfft2(np.real(np.fft.ifft2(phi_hat)))

        return np.stack([eta_hat, phi_hat])
    else:
        k = np.arange(1, modes+1)*k0
        w = np.sqrt(k * g)

        sigma = 0.07 * (w <= 5.24/T1) + 0.09 * (w > 5.24/T1)
        Sw = (155 * Hs**2 / T1**4 / w**5 *
            np.exp(-944 / (T1*w)**4) *
            gamma ** np.exp(-((0.191*w*T1 - 1)/(np.sqrt(2)*sigma))**2))
        Sk = Sw * g / (2 * np.sqrt(k * g))

        phases = np.exp(np.random.rand(modes)*2*np.pi*1.j)
        eta_hat = np.sqrt(Sk * k0 * 0.5) * phases * 2 * modes
        phi_hat = eta_hat * np.exp(-1.j * np.pi / 2) * np.sqrt(g / k)

        alias_mask = np.arange(modes+1) < modes * 2 / (mHOS + 1)

        eta_hat = np.insert(eta_hat, 0, 0) * alias_mask
        phi_hat = np.insert(phi_hat, 0, 0) * alias_mask

        return np.concatenate((eta_hat, phi_hat))
    