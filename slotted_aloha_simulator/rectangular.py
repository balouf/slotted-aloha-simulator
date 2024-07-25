from  numba import njit
import numpy as np


@njit
def core_rec(c_min=3, c_max=7, n=64, t_sim=20):
    base = 2**c_min
    state = np.zeros(n, dtype=np.int64)
    countdown = np.floor(base*np.random.rand(n)).astype(np.int64)
    states = np.zeros((c_max + 1, t_sim + 1), dtype=np.int64)
    emissions = np.zeros(t_sim + 1, dtype=np.int64)
    occupancy = np.zeros(t_sim + 1, dtype=np.int64)
    goodput = np.zeros(t_sim + 1, dtype=np.int64)
    for t in range(t_sim + 1):
        o = 0
        g = 0
        e = 0
        emitter = 0
        d_t = np.zeros(c_max + 1, dtype=np.int64)

        for _ in range(2 ** t):
            emission = False
            reception = True
            for i, s in enumerate(state):
                d_t[s] += 1
                if countdown[i] == 0:
                    e += 1
                    if emission:
                        reception = False
                    else:
                        emission = True
                        emitter = i
                    if s < c_max:
                        s += 1
                        state[i] = s
                    countdown[i] = int(2**(c_min+s)*np.random.rand())
                else:
                    countdown[i] -= 1

            if emission:
                o += 1
                if reception:
                    state[emitter] = 0
                    countdown[emitter] = countdown[emitter] % base
                    g += 1
        occupancy[t] = o
        emissions[t] = e
        goodput[t] = g
        states[:, t] = d_t
    return states, occupancy, goodput, emissions

@njit
def rec_run(c_min=3, c_max=7, n=64, t_sim=20, m=10, seed=None):
    if seed is not None:
        np.random.seed(seed)
    # Main run
    dist, occupancy, goodput, emissions = core_rec(c_min=c_min, c_max=c_max, n=n, t_sim=t_sim)
    # Extra short runs
    for i in range(m):
        for _ in range(2 ** (m - i)):
            d, o, g, e = core_rec(c_min=c_min, c_max=c_max, n=n, t_sim=i)
            dist[:, :(i + 1)] += d
            occupancy[:(i + 1)] += o
            goodput[:(i + 1)] += g
            emissions[:(i + 1)] += e
    # extra runs normalization
    dist = dist.astype(np.float64)
    occupancy = occupancy.astype(np.float64)
    goodput = goodput.astype(np.float64)
    emissions = emissions.astype(np.float64)
    for i in range(m):
        dist[:, i] /= 2 ** (m - i + 1) - 1
        occupancy[i] /= 2 ** (m - i + 1) - 1
        goodput[i] /= 2 ** (m - i + 1) - 1
        emissions[i] /= 2 ** (m - i + 1) - 1
    # Counter conversion
    efficiency = goodput / emissions
    for i in range(t_sim + 1):
        occupancy[i] /= 2 ** i
        goodput[i] /= 2 ** i
        dist[:, i] = dist[:, i] / n / 2 ** i
    return dist, occupancy, goodput, efficiency
