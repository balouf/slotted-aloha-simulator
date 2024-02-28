import numpy as np
from scipy.optimize import fsolve
from numba import njit, int64, float64


def mean_field(p0=1/8, alpha=.5, n=2, c_max=40):
    """
    Asymptotic mean fields values.

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

    Returns
    -------
    states: :class:`~numpy.ndarray`
        State distribution
    occupancy: :class:`float`
        Proportion of busy slots
    goodput: :class:`float`
        Proportion of useful slots
    efficiency: :class:`float`
        Proportion of useful emissions

    Examples
    --------

    >>> s, o, g, e = mean_field()
    >>> s[:4].round(4)
    array([0.7808, 0.1712, 0.0375, 0.0082])
    >>> round(e, 4)
    0.8904
    >>> s, o, g, e = mean_field(n=1000)
    >>> s[:4].round(4)
    array([0.0028, 0.0028, 0.0028, 0.0027])
    >>> round(e, 4)
    0.5014
    """
    # Solve noise equation
    def noise_f(s):
        return (1-(1-p0*(1-s/alpha)/(1-s))**(n-1)) - s
    noise = float(fsolve(noise_f, np.array([alpha])))
    # Geometric state distribution
    states = (1-noise/alpha)*(noise/alpha)**np.arange(c_max)
    # Goodput: emission prob is (1-s/a)sigma p0 s^i = p0*(1-s/alpha)/(1-s)
    goodput = n * p0 * (1 - noise / alpha)
    # Occupancy : not everyone quiet
    occupancy = (1-(1-p0*(1-noise/alpha)/(1-noise))**n)
    # Efficiency: no noise from others
    efficiency = 1 - noise
    return states, occupancy, goodput, efficiency


@njit
def dynamic_mean_field(p0=1/8, alpha=.5, n=2, c_max=40, t_sim=10):
    """
    Dynamic mean fields values.

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
        State distribution
    occupancy: :class:`~numpy.ndarray`
        Proportion of busy slots
    goodput: :class:`~numpy.ndarray`
        Proportion of useful slots
    efficiency: `~numpy.ndarray`
        Proportion of useful emissions

    Examples
    --------

    >>> s, o, g, e = dynamic_mean_field()
    >>> s[:4, :4].round(4)
    array([[1.000e+00, 9.772e-01, 9.392e-01, 8.861e-01],
           [0.000e+00, 2.280e-02, 5.990e-02, 1.095e-01],
           [0.000e+00, 1.000e-04, 9.000e-04, 4.300e-03],
           [0.000e+00, 0.000e+00, 0.000e+00, 1.000e-04]])
    >>> e[:4].round(4)
    array([0.875 , 0.8764, 0.8788, 0.8823])
    >>> s, o, g, e = dynamic_mean_field(n=1000)
    >>> s[:4, :4].round(4)
    array([[1.    , 0.8203, 0.5545, 0.2578],
           [0.    , 0.1758, 0.3909, 0.5112],
           [0.    , 0.0039, 0.0528, 0.2059],
           [0.    , 0.    , 0.0019, 0.0241]])
    >>> e[:4].round(4)
    array([0., 0., 0., 0.])
    """
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
            s_t += c
            s = 1 - (1-c@speeds)**(n-1)
            c[1:] = c[1:]*(1-speeds[1:]) + s*c[:-1]*speeds[:-1]
            c[0] = max(0, 1-np.sum(c[1:]))
        s_t /= 2**t
        x = s_t@speeds
        occ[t] = 1-(1-x)**n
        good[t] = n * x * (1-x)**(n-1)
        eff[t] = (1-x)**(n-1)
        states[:, t] = s_t
    return states, occ, good, eff


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
    current_state = np.zeros(n, dtype=int64)
    speeds = p0 * alpha ** np.arange(c_max)
    states = np.zeros((c_max, t_sim + 1), dtype=int64)
    emissions = np.zeros(t_sim + 1, dtype=int64)
    occupancy = np.zeros(t_sim + 1, dtype=int64)
    goodput = np.zeros(t_sim + 1, dtype=int64)
    for t in range(t_sim + 1):
        o = 0
        g = 0
        e = 0
        emitter = 0
        d_t = np.zeros(c_max, dtype=int64)

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
    dist = dist.astype(float64)
    occupancy = occupancy.astype(float64)
    goodput = goodput.astype(float64)
    emissions = emissions.astype(float64)
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
