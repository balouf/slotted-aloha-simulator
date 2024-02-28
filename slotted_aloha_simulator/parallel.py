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
    ...     print(dat['p0'])
    ...     mf = dat['results']['mf']
    ...     print([mf[k][3] for k in ['occupancy', 'goodput', 'efficiency']])
    ...     sim = dat['results']['simulation']
    ...     print([sim[k][3] for k in ['occupancy', 'goodput', 'efficiency']])
    0.125
    [0.36420070073707755, 0.30486864402911484, 0.7120164602702012]
    [0.36323529411764705, 0.30392156862745096, 0.7118254879448909]
    0.25
    [0.5215184798200845, 0.3872977788242961, 0.5753059648859896]
    [0.5176470588235295, 0.3803921568627451, 0.5627266134880348]
    0.5
    [0.6501379554140254, 0.42017983281634214, 0.4549070027900601]
    [0.6465686274509804, 0.4230392156862745, 0.46372917786136486]
    0.6666666666666666
    [0.69289320276714, 0.4217366387626196, 0.412540956923515]
    [0.7019607843137254, 0.4377450980392157, 0.4301541425818883]
    """
    return Parallel(n_jobs=n_jobs)(delayed(compute)(p) for p in parameters_list)

