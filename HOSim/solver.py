import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import factorial
jax.config.update("jax_enable_x64", True)

import time

import matplotlib.pyplot as plt
from numpy.fft import irfft, rfft

from . import utils
from . import io_utils

def f(t, y, modes, g, k0, mHOS, Ta):
    eta_hat = y[:modes+1]
    phi_hat = y[modes+1:]

    freqs = jnp.arange(modes+1) * k0

    phi_hat_pert = jnp.zeros((mHOS, modes+1), dtype=jnp.complex128).at[0].set(phi_hat)
    W = jnp.zeros((mHOS, 2*modes), dtype=jnp.float64).at[0].set(jnp.fft.irfft(phi_hat_pert[0] * freqs))

    eta = jnp.fft.irfft(eta_hat)

    factorials = jnp.array([factorial(n) for n in range(mHOS+1)])
    for m in range(1, mHOS):
        for n in range(1, m+1):
            phi_hat_pert = phi_hat_pert.at[m].add(
                -jnp.fft.rfft((eta ** n / factorials[n]) * jnp.fft.irfft(phi_hat_pert[m-n] * freqs**n))
            )

        for n in range(m+1):
            W = W.at[m].add((eta ** n / factorials[n]) * jnp.fft.irfft(phi_hat_pert[m-n] * freqs**(n+1)))

    deta = jnp.array(W[0], dtype=jnp.float64)
    dphi = jnp.array(-g * eta, dtype=jnp.float64)

    relax = 1 - jnp.exp(- (t / Ta) ** 4)

    dphideta = jnp.fft.irfft(1j * freqs * eta_hat) * jnp.fft.irfft(1j * freqs * phi_hat)
    dphidphi_square = jnp.fft.irfft(1j * freqs * phi_hat) ** 2
    detadeta_square = jnp.fft.irfft(1j * freqs * eta_hat) ** 2

    if mHOS > 1:
        deta += relax * -dphideta
        dphi += relax * -0.5 * dphidphi_square

    for m in range(2, mHOS+1):
        W_m_2 = jnp.zeros(2*modes, jnp.float64)
        for n in range(1, m):
            W_m_2 += W[n-1] * W[m-n-1]

        deta += relax * W[m-1]
        dphi += relax * 0.5 * W_m_2

    for m in range(3, mHOS+1):
        deta += relax * W[m-2-1] * detadeta_square

    for m in range(4, mHOS+1):
        W_m_2 = jnp.zeros(2*modes, dtype=jnp.float64)
        for n in range(1, m-2):
            W_m_2 += W[n-1] * W[m-n-1-2] * detadeta_square

        dphi += relax * 0.5 * W_m_2

    alias_mask = jnp.arange(modes+1) < modes * 2 / (mHOS + 1) * 0.9
    return jnp.concatenate((jnp.fft.rfft(deta) * alias_mask, jnp.fft.rfft(dphi) * alias_mask))

def f2d(t, y, modes, g, k0, mHOS, Ta):
    eta_hat = y[0]
    phi_hat = y[1]
    eta = jnp.fft.irfft2(eta_hat)
    # Static arrays:
    ky = jnp.arange(0, modes+1, dtype=eta.dtype)*k0
    kx = jnp.concatenate((ky, -ky[1:-1][::-1]))
    KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
    K = jnp.sqrt(KX**2 + KY**2)

    factorials = factorial(jnp.arange(mHOS + 1, dtype=eta.dtype))

    eta_hat = y[0]
    phi_hat = y[1]

    phi_hat_pert = jnp.zeros((mHOS, 2*modes, modes+1), dtype=y.dtype).at[0].set(phi_hat)
    W = jnp.zeros((mHOS, 2*modes, 2*modes), dtype=eta.dtype).at[0].set(jnp.fft.irfft2(phi_hat_pert[0] * K))

    eta = jnp.fft.irfft2(eta_hat)

    for m in range(1, mHOS):
        for n in range(1, m+1):
            phi_hat_pert = phi_hat_pert.at[m].add(
                -jnp.fft.rfft2(eta ** n / factorials[n] * jnp.fft.irfft2(phi_hat_pert[m-n]*K**n))
            )
        for n in range(0, m+1):
            W = W.at[m].add((eta ** n / factorials[n]) * jnp.fft.irfft2(phi_hat_pert[m-n] * K**(n+1)))

    dphideta = jnp.fft.irfft2(1j * KX * eta_hat) * jnp.fft.irfft2(1j * KX * phi_hat) + jnp.fft.irfft2(1j * KY * eta_hat) * jnp.fft.irfft2(1j * KY * phi_hat) 
    dphidphi_square = jnp.fft.irfft2(1j * KX * phi_hat) ** 2 + jnp.fft.irfft2(1j * KY * phi_hat) ** 2
    detadeta_square = jnp.fft.irfft2(1j * KX * eta_hat) ** 2 + jnp.fft.irfft2(1j * KY * eta_hat) ** 2

    deta = jnp.array(W[0], dtype=eta.dtype)
    dphi = jnp.array(-g * eta, dtype=eta.dtype)

    relax = 1 - jnp.exp(- (t / Ta) ** 4)

    if mHOS > 1:
        deta += relax * -dphideta
        dphi += relax * -0.5 * dphidphi_square

    for m in range(2, mHOS+1):
        W_m_2 = jnp.zeros((2*modes, 2*modes), eta.dtype)
        for n in range(1, m):
            W_m_2 += W[n-1] * W[m-n-1]

        deta += relax * W[m-1]
        dphi += relax * 0.5 * W_m_2

    for m in range(3, mHOS+1):
        deta += relax * W[m-2-1] * detadeta_square

    for m in range(4, mHOS+1):
        W_m_2 = jnp.zeros((2*modes, 2*modes), dtype=eta.dtype)
        for n in range(1, m-2):
            W_m_2 += W[n-1] * W[m-n-1-2] * detadeta_square

        dphi += relax * 0.5 * W_m_2

    index = modes * 2 // (mHOS + 1)
    alias_mask = jnp.ones((2*modes, modes+1), dtype=jnp.bool)
    alias_mask = alias_mask.at[index+1:-index, :].set(False)
    alias_mask = alias_mask.at[:, index+1:].set(False)

    return jnp.stack([jnp.fft.rfft2(deta)*alias_mask, jnp.fft.rfft2(dphi)*alias_mask])

def rk4_step(t, y, h, modes, g, k0, mHOS, Ta, f_jit):
    k1 = h * f_jit(t, y, modes, g, k0, mHOS, Ta)
    k2 = h * f_jit(t + 0.5*h, y + 0.5*k1, modes, g, k0, mHOS, Ta)
    k3 = h * f_jit(t + 0.5*h, y + 0.5*k2, modes, g, k0, mHOS, Ta)
    k4 = h * f_jit(t + h, y + k3, modes, g, k0, mHOS, Ta)
    
    y_next = y + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
    
    return y_next

def run_simulation(params, counter):
    y = jnp.asarray(utils.get_initial_condition(params), dtype=jnp.complex64)
    step_size = params["step_size"]
    steps = int(np.ceil(params["time"] / step_size))
    f_jit = None
    if params["2d"]:
        f_jit = jax.jit(f2d, static_argnums=(2, 3, 4, 5, 6))
    else:
        f_jit = jax.jit(f, static_argnums=(2, 3, 4, 5, 6))
    rk4_step_jit = jax.jit(rk4_step, static_argnums=(2, 3, 4, 5, 6, 7, 8))

    g = params["gravity"]
    k0 = 2 * np.pi / params["length"]
    mHOS = params["mHOS"]
    Ta = params["Ta"]
    modes = params["modes"]

    output_interval = params["output_interval"]
    result = None
    if params["2d"]:
        result = np.zeros((steps // output_interval + 1, 2, 2*modes, modes+1), dtype=np.complex64)
    else:
        result = np.zeros((steps // output_interval + 1, 2*(modes+1)), dtype=np.complex64)

    start_time = time.time()
    t = 0
    for i in range(steps):
        if i % output_interval == 0:
            result[i // output_interval] = y

        y = rk4_step_jit(t, y, step_size, modes, g, k0, mHOS, Ta, f_jit)
        t += step_size

        if (i % (steps // 10) == 0) and (i != 0):
            with counter.get_lock():
                counter.value += 1

    result[-1] = y
    with counter.get_lock():
        counter.value += 1

    io_utils.save_results(params, result)