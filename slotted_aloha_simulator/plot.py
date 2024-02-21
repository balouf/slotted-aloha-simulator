from matplotlib import pylab as plb  # type: ignore

colors = plb.rcParams['axes.prop_cycle'].by_key()['color']

sorts = ['corsort_delta_max_rho', 'mergesort_bottom_up', 'ford_johnson_rho',
         'mergesort_top_down_rho', 'mergesort_top_down', 'mergesort_bottom_up_rho',
         'quicksort', 'heapsort']

color_dict = {k: v for k, v in zip(sorts, colors)}
color_dict['ford_johnson'] = color_dict['ford_johnson_rho']


p0=1/8
n=8
tsim=24
aloha = Aloha(p0=p0, n=n, t_sim=tsim)
dyn = aloha.dynamic_mean_field()
m_dist, m_good, m_eff = dyn['state_distribution'], dyn['goodput'], dyn['efficiency']
dist, occ, good, eff = multi_aloha_run(n=n, tsim=tsim,m=16)
dyn = aloha.mean_field()
ad, ag, ae = dyn['state_distribution'], dyn['goodput'], dyn['efficiency']


from matplotlib import pyplot as plt
plt.plot(m_eff, '-', color='red', label='Eff MF')
plt.plot(eff, 'x',color='red', label='Eff')
plt.plot([0, tsim], [ae, ae], '-.',color='red', label='as')
plt.plot(m_good, '-', color='black', label='G MF')
plt.plot(good, 'x',color='black', label='G')
plt.plot([0, tsim], [ag, ag], '-.',color='black', label='as')
plt.legend()
plt.ylim([0, 1])
plt.xlim([0, tsim])
plt.show()
