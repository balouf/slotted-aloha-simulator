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
        Time range (exponential).
    m: :class:`int`, optional
        Minimal number of samples per time bucket (exponential).
    seed: :class:`int`, optional
        Seed
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
        self.mean_field()
        self.dynamic_mean_field()
        self.simulation()

    def mean_field(self):
        s, o, g, e = mean_field(p0=self.p0, alpha=self.alpha, n=self.n, c_max=self.c_max)
        self.res_['mf_asymptotic'] = {'state_distribution': s, 'occupancy': o, 'goodput': g, 'efficiency': e}

    def dynamic_mean_field(self):
        s, o, g, e = dynamic_mean_field(p0=self.p0, alpha=self.alpha, n=self.n, c_max=self.c_max, t_sim=self.t_sim)
        self.res_['mf'] = {'state_distribution': s, 'occupancy': o, 'goodput': g, 'efficiency': e}

    def simulation(self):
        s, o, g, e = aloha_run(p0=self.p0, alpha=self.alpha, n=self.n, c_max=self.c_max,
                               t_sim=self.t_sim, m=self.m, seed=self.seed)
        self.res_['simulation'] = {'state_distribution': s, 'occupancy': o, 'goodput': g, 'efficiency': e}
