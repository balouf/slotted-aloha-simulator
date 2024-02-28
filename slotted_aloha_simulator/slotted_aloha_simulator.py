from slotted_aloha_simulator.jit import mean_field, dynamic_mean_field
from slotted_aloha_simulator.jit import aloha_run


class Aloha:
    """
    Top-level class of the package.

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
        Time range in epochs (exponential).
    m: :class:`int`, optional
        Minimal number of samples per epoch (exponential).
    seed: :class:`int`, optional
        Seed


    Examples
    --------

    Initiate an Aloha simulator:

    >>> aloha = Aloha(p0=1/2, alpha=1/2, n=4, c_max=20, t_sim=10, m=8, seed=42)

    Launch simulation + approximations

    >>> aloha()

    Asymptotic results (from approximation)

    >>> mfa = aloha.res_['mf_asymptotic']
    >>> [mfa[k] for k in ['occupancy', 'goodput', 'efficiency']]
    [0.5000000000000001, 0.3784142300054423, 0.5946035575013606]

    Results at epoch 3 (from approximation)

    >>> mf = aloha.res_['mf']
    >>> [mf[k][3] for k in ['occupancy', 'goodput', 'efficiency']]
    [0.6501379554140254, 0.42017983281634214, 0.4549070027900601]

    Results at epoch 3 (from simulations)

    >>> sim = aloha.res_['simulation']
    >>> [sim[k][3] for k in ['occupancy', 'goodput', 'efficiency']]
    [0.6666666666666666, 0.42658730158730157, 0.44698544698544695]
    """
    def __init__(self, p0=1/8, alpha=1/2, n=2, c_max=40, t_sim=20, m=10, seed=None):
        self.p0 = p0
        self.alpha = alpha
        self.n = n
        self.c_max = c_max
        self.t_sim = t_sim
        self.m = m
        self.seed = seed
        self.res_ = dict()

    def __call__(self):
        """
        All-in-one computation.

        Returns
        -------
        None
        """
        self.mean_field()
        self.dynamic_mean_field()
        self.simulation()

    def mean_field(self):
        """
        Compute asymptotic approximation.

        Returns
        -------
        None
        """
        s, o, g, e = mean_field(p0=self.p0, alpha=self.alpha, n=self.n, c_max=self.c_max)
        self.res_['mf_asymptotic'] = {'state_distribution': s, 'occupancy': o, 'goodput': g, 'efficiency': e}

    def dynamic_mean_field(self):
        """
        Compute per-epoch approximations.

        Returns
        -------
        None
        """
        s, o, g, e = dynamic_mean_field(p0=self.p0, alpha=self.alpha, n=self.n, c_max=self.c_max, t_sim=self.t_sim)
        self.res_['mf'] = {'state_distribution': s, 'occupancy': o, 'goodput': g, 'efficiency': e}

    def simulation(self):
        """
        Compute per-epoch simulations.

        Returns
        -------
        None
        """
        s, o, g, e = aloha_run(p0=self.p0, alpha=self.alpha, n=self.n, c_max=self.c_max,
                               t_sim=self.t_sim, m=self.m, seed=self.seed)
        self.res_['simulation'] = {'state_distribution': s, 'occupancy': o, 'goodput': g, 'efficiency': e}
