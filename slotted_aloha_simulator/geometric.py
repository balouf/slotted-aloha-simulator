import numpy as np
from numba import njit


@njit
def core_aloha_run(p0=1/8, alpha=1/2, n=64, c_max=60, t_sim=20):
    """
    Core Aloha simulator.

    Parameters
    ----------
    p0: :class:`float`, optional
        Default emission probability.
    alpha: :class:`float`, optional
        Back-off ratio.
    n: :class:`int`, optional
        Number of stations.
    c_max: :class:`int`, optional
        Upper bound on state value.
    t_sim: :class:`int`, optional
        Time range (exponential).

    Returns
    -------
    states: :class:`~numpy.ndarray`
        States (counter, not normalized)
    occupancy: :class:`~numpy.ndarray`
        Busy slots (counter, not normalized)
    goodput: :class:`~numpy.ndarray`
        Useful slots (counter, not normalized)
    emissions: :class:`~numpy.ndarray`
        Number of emissions (counter, not normalized)
    """
    current_state = np.zeros(n, dtype=np.int64)
    speeds = p0 * alpha ** np.arange(c_max + 1)
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
            for i, s in enumerate(current_state):
                d_t[s] += 1
                if speeds[s] > np.random.rand():
                    e += 1
                    if emission:
                        reception = False
                    else:
                        emission = True
                        emitter = i
                    if s < c_max:
                        current_state[i] += 1
            if emission:
                o += 1
                if reception:
                    current_state[emitter] = 0
                    g += 1
        occupancy[t] = o
        emissions[t] = e
        goodput[t] = g
        states[:, t] = d_t
    return states, occupancy, goodput, emissions


@njit
def aloha_run(p0=1/8, alpha=1/2, n=2, t_sim=20, c_max=60, m=10, seed=None):
    """
    Simulation.

    Parameters
    ----------
    p0: :class:`float`, optional
        Default emission probability.
    alpha: :class:`float`, optional
        Back-off ratio.
    n: :class:`int`, optional
        Number of stations.
    c_max: :class:`int`, optional
        Upper bound on state value.
    t_sim: :class:`int`, optional
        Time range (exponential).
    m: :class:`int`, optional
        Minimal number of samples per time bucket (exponential).
    seed: :class:`int`, optional
        Seed

    Returns
    -------
    states: :class:`~numpy.ndarray`
        State distribution
    occupancy: :class:`~numpy.ndarray`
        Proportion of busy slots
    goodput: :class:`~numpy.ndarray`
        Proportion of useful slots
    efficiency: `~numpy.ndarray`
        Proportion of useful emissions

    Examples
    --------

    >>> s, o, g, e = aloha_run(n=2, t_sim=10, m=5)
    >>> s, o, g, e = aloha_run(n=2, t_sim=14, m=10, seed=42)
    >>> s[:4, :4].round(4)
    array([[1.    , 0.9765, 0.942 , 0.8958],
           [0.    , 0.0235, 0.058 , 0.1022],
           [0.    , 0.    , 0.    , 0.002 ],
           [0.    , 0.    , 0.    , 0.    ]])
    >>> e[:4].round(4)
    array([0.8504, 0.8697, 0.8799, 0.8807])
    >>> s, o, g, e = aloha_run(n=1000, t_sim=14, m=10, seed=42)
    >>> s[:4, :4].round(4)
    array([[1.    , 0.8207, 0.5545, 0.2581],
           [0.    , 0.1754, 0.3908, 0.511 ],
           [0.    , 0.0039, 0.0528, 0.2057],
           [0.    , 0.    , 0.0018, 0.0242]])
    >>> e[:4]
    array([0., 0., 0., 0.])
    """
    if seed is not None:
        np.random.seed(seed)
    # Main run
    dist, occupancy, goodput, emissions = core_aloha_run(p0=p0, alpha=alpha, n=n, t_sim=t_sim, c_max=c_max)
    # Extra short runs
    for i in range(m):
        for _ in range(2 ** (m - i)):
            d, o, g, e = core_aloha_run(p0=p0, alpha=alpha, n=n, t_sim=i, c_max=c_max)
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
