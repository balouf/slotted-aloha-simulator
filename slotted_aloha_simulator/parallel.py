from joblib import Parallel, delayed

from slotted_aloha_simulator.slotted_aloha_simulator import Aloha


def compute(parameters):
    """
    Simple wrapper for launching a single simulation

    Parameters
    ----------
    parameters: :class:`dict`
        Parameters for the aloha simulation

    Returns
    -------
    :class:`dict`
        The input parameters (for traceability) plus a 'results' key that contain the simulation measurements.
    """
    aloha = Aloha(**parameters)
    aloha()
    return {'results': aloha.res_, **parameters}


def parallel_compute(parameters_list, n_jobs=-1):
    """
    Parameters
    ----------
    parameters_list: :class:`list` of :class:`dict`
        Simulation parameters.
    n_jobs: :class:`int`, default=-1
        Number of workers to spawn, joblib style (-1 means all CPUs).

    Returns
    -------
    :class:`list` of :class:`dict`
        For each setting, input parameters with an extra 'results' key.

    Examples
    --------

    >>> p_values = [1/8, 1/4, 1/2, 2/3]
    >>> p_list = [{'p0': p, 'n': 4, 't_sim': 10, 'seed': 42} for p in p_values]
    >>> data = parallel_compute(p_list, n_jobs=len(p_values))
    >>> for dat in data:
    ...     print(round(dat['p0'], 4))
    ...     mf = dat['results']['mf']
    ...     print([mf[k][3].round(4) for k in ['occupancy', 'goodput', 'efficiency']])
    ...     sim = dat['results']['simulation']
    ...     print([sim[k][3].round(4) for k in ['occupancy', 'goodput', 'efficiency']])
    0.125
    [0.3642, 0.3049, 0.712]
    [0.3632, 0.3039, 0.7118]
    0.25
    [0.5215, 0.3873, 0.5753]
    [0.5176, 0.3804, 0.5627]
    0.5
    [0.6501, 0.4202, 0.4549]
    [0.6466, 0.423, 0.4637]
    0.6667
    [0.6929, 0.4217, 0.4125]
    [0.702, 0.4377, 0.4302]
    """
    return Parallel(n_jobs=n_jobs)(delayed(compute)(p) for p in parameters_list)

