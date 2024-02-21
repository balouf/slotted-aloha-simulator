import numpy as np
from scipy.optimize import fsolve
from numba import njit


def mean_field(p0=1/8, alpha=.5, n=2, c_max=40):
    """
    Asymptotic mean fields values.

    Parameters
    ----------
    p0: :class:`float`

    alpha
    n
    c_max

    Returns
    -------

    """
    def noise_f(s):
        return (1-(1-p0*(1-s/alpha)/(1-s))**(n-1)) - s
    noise = fsolve(noise_f, [alpha])
    states = (1-noise/alpha)*(noise/alpha)**np.arange(c_max)
    good = n*p0*(1-noise/alpha)
    occ = (1-(1-p0*(1-noise/alpha)/(1-noise))**n)
    eff = 1-noise
    return states, occ, good, eff


@njit
def dynamic_mean_field(p0=1/8, alpha=.5, n=2, c_max=40, t_sim=10):
    c = np.zeros(c_max)
    c[0] = 1
    speeds = p0*alpha**np.arange(c_max)
    states = np.zeros((c_max, t_sim+1))
    occ = np.zeros(t_sim+1)
    good = np.zeros(t_sim+1)
    eff = np.zeros(t_sim+1)
    for t in range(t_sim+1):
        s_t = np.zeros(c_max)
        for _ in range(2**t):
            s = 1 - (1-c@speeds)**(n-1)
            c[1:] = c[1:]*(1-speeds[1:]) + s*c[:-1]*speeds[:-1]
            c[0] = max(0, 1-np.sum(c[1:]))
            s_t += c
        s_t /= 2**t
        x = s_t@speeds
        occ[t] = 1-(1-x)**n
        good[t] = n * x * (1-x)**(n-1)
        eff[t] = (1-x)**(n-1)
        states[:, t] = s_t
    return states, occ, good, eff


@njit
def aloha_run(p0=1 / 8, alpha=1 / 2, n=64, tsim=20, cmax=60):
    state = np.zeros(n, dtype=int64)
    speeds = p0 * alpha ** np.arange(cmax)
    dist = np.zeros((cmax, tsim + 1), dtype=int64)
    emissions = np.zeros(tsim + 1, dtype=int64)
    occupancy = np.zeros(tsim + 1, dtype=int64)
    goodput = np.zeros(tsim + 1, dtype=int64)
    for t in range(tsim + 1):
        o = 0
        g = 0
        e = 0
        d_t = np.zeros(cmax, dtype=int64)

        for _ in range(2 ** t):
            emission = False
            reception = True
            for i, s in enumerate(state):
                d_t[s] += 1
                if speeds[s] > np.random.rand():
                    e += 1
                    if emission:
                        reception = False
                    else:
                        emission = True
                        emitter = i
                    state[i] += 1
            if emission:
                o += 1
                if reception:
                    state[emitter] = 0
                    g += 1
        occupancy[t] = o
        emissions[t] = e
        goodput[t] = g
        dist[:, t] = d_t
    return dist, occupancy, goodput, emissions


@njit
def multi_aloha_run(p0=1 / 8, alpha=1 / 2, n=64, tsim=20, cmax=60, m=10):
    dist, occupancy, goodput, emissions = aloha_run(p0=p0, alpha=alpha, n=n, tsim=tsim, cmax=cmax)
    for i in range(m):
        for _ in range(2 ** (m - i)):
            d, o, g, e = aloha_run(p0=p0, alpha=alpha, n=n, tsim=i, cmax=cmax)
            dist[:, :(i + 1)] += d
            occupancy[:(i + 1)] += o
            goodput[:(i + 1)] += g
            emissions[:(i + 1)] += e
    #     print(goodput)
    #     print(emissions)
    dist = dist.astype(float64)
    occupancy = occupancy.astype(float64)
    goodput = goodput.astype(float64)
    emissions = emissions.astype(float64)
    for i in range(m):
        dist[:, i] /= 2 ** (m - i + 1) - 1
        occupancy[i] /= 2 ** (m - i + 1) - 1
        goodput[i] /= 2 ** (m - i + 1) - 1
        emissions[i] /= 2 ** (m - i + 1) - 1
    #     print(goodput)
    #     print(emissions)
    efficiency = goodput / emissions
    for i in range(tsim + 1):
        occupancy[i] /= 2 ** i
        goodput[i] /= 2 ** i
        dist[:, i] = dist[:, i] / n / 2 ** i
    #     print(goodput)
    #     print(emissions)
    return dist, occupancy, goodput, efficiency
