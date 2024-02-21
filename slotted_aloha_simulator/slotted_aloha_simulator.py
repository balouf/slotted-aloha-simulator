class Aloha:
    def __init__(self, p0=1/8, alpha=1/2, n=2, c_max=40, t_sim=20, m=10):
        self.p0 = p0
        self.alpha = alpha
        self.n = n
        self.c_max = c_max
        self.t_sim = t_sim
        self.m = m

    def mean_field(self):
        s, g, e = mean_field(p0=self.p0, alpha=self.alpha, n=self.n, c_max=self.c_max)
        return {'state_distribution': s, 'goodput': g, 'efficiency': e}

    def dynamic_mean_field(self):
        s, g, e = dynamic_mean_field(p0=self.p0, alpha=self.alpha, n=self.n, t_sim=self.t_sim, c_max=self.c_max)
        return {'state_distribution': s, 'goodput': g, 'efficiency': e}


