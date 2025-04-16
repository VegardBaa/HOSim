import jax
import jax.numpy as jnp
from jax.scipy.special import factorial
jax.config.update("jax_enable_x64", True)

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

    relax = 1 - jnp.exp(- (t / Ta) ** nRelax)

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

    deta_hat = jnp.fft.rfft(deta)
    dphi_hat = jnp.fft.rfft(dphi)

    alias_mask = jnp.arange(modes+1) < modes * 2 / (mHOS + 1) + 1
    return jnp.concatenate((jnp.fft.rfft(deta) * alias_mask, jnp.fft.rfft(dphi) * alias_mask))

def rk4_step(t, y, h, modes, g, k0, mHOS, Ta):
    k1 = h * f_jit(t, y, modes, g, k0, mHOS, nRelax, Ta)
    k2 = h * f_jit(t + 0.5*h, y + 0.5*k1, modes, g, k0, mHOS, nRelax, Ta)
    k3 = h * f_jit(t + 0.5*h, y + 0.5*k2, modes, g, k0, mHOS, nRelax, Ta)
    k4 = h * f_jit(t + h, y + k3, modes, g, k0, mHOS, nRelax, Ta)
    
    y_next = y + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
    t_next = t + h
    
    return y_next

def simulation(steps, step_size, y0, g, k0, mHOS, nRelax, Ta):
    y = y0
    t = 0
    for i in range(steps):
        y = rk4_step_jit(t, y, step_size, g, k0, mHOS, nRelax, Ta)
        t += step_size

    return y