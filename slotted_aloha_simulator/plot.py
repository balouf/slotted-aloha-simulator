import tikzplotlib
from matplotlib import pyplot as plt
from matplotlib import pylab as plb  # type: ignore
import numpy as np


color_list = plb.rcParams['axes.prop_cycle'].by_key()['color']

p_labels = {'n': 'N', 'p0': 'p_0', 'alpha': '\\alpha', 't': 'T'}


def locate_result(data, parameters):
    """
    Parameters
    ----------
    data: :class:`list` of `dict`
        List of results from parallelized computation.
    parameters: :class:`dict`
        Parameters to fit

    Returns
    -------
    :class:`dict`
        First result that fits the parameters.
    """
    for d in data:
        for k, v in parameters.items():
            if d[k] != v:
                break
        else:
            return d


def tikzplotlib_fix_ncols(obj):
    """
    Workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    if hasattr(obj, "_unscaled_dash_pattern"):
        obj._us_dashSeq = obj._unscaled_dash_pattern[1]
        obj._us_dashOffset = obj._unscaled_dash_pattern[0]
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


def evolution_plot(data, params, legend=True, export=None, keys=None, names=None):
    """
    Parameters
    ----------
    data: :class:`list` of `dict`
        List of results from parallelized computation.
    params: :class:`dict`
        Parameters to fit
    legend: :class:`bool`, default=True
        Draw legend
    export: :class:`str`, optional
        Save the figure as export.tex in pgfplots format
    keys: :class:`list`, optional
        Sub-list of occupancy, goodput, efficiency to display.
    names: :class:`list`, optional
        Display names for the keys.

    Returns
    -------
    None
    """
    if keys is None:
        keys = ['occupancy', 'goodput', 'efficiency']
    if names is None:
        names = [s.title() for s in keys]
    labels = {k: n for k, n in zip(keys, names)}
    colors = {k: c for k, c in zip(keys, color_list)}
    types = ['simulation', 'mf', 'mf_asymptotic']
    styles = {k: s for k, s in zip(types, ['x', '-', ':'])}
    style_labels = {k: s for k, s in zip(types, ['Simulation', 'Approx.', 'Approx. ($t\\rightarrow\\infty)$'])}
    dat = locate_result(data, params)
    if dat is None:
        return None
    t_sim = dat['t_sim']
    res = dat['results']
    fig = plt.figure()
    for k in keys:
        for t in types[:-1]:
            plt.plot(res[t][k], styles[t], color=colors[k])
        t = 'mf_asymptotic'
        a = res[t][k]
        plt.plot([0, t_sim], [a, a], styles[t], color=colors[k])
    plt.xlim([0, t_sim])
    plt.ylim([0, 1])
    plt.xlabel('Epoch $T$')
    title = ', '.join([f"{p_labels[k]}={v}" for k, v in params.items()])
    plt.title(f"${title}$")
    if legend:
        for t in styles:
            plt.plot([-1], styles[t], color='black', label=style_labels[t])
        for k in keys:
            plt.plot([-1], color=colors[k], label=labels[k])
        plt.legend(ncol=3)
    if export is not None:
        tikzplotlib_fix_ncols(fig)
        tikzplotlib.save(f'{export}.tex',table_row_sep='\\\\', encoding='utf8')
    plt.show()


def distribution_plot(data, params, legend=True, export=None, keys=None, ym=1e-8, xm=30):
    """
    Parameters
    ----------
    data: :class:`list` of `dict`
        List of results from parallelized computation.
    params: :class:`dict`
        Parameters to fit
    legend: :class:`bool`, default=True
        Draw legend
    export: :class:`str`, optional
        Save the figure as export.tex in pgfplots format
    keys: :class:`list`, optional
        List of Epochs to display
    ym: :class:`float`, optional
        Lower y-bound
    xm: :class:`int`, optional
        Higher x-bound

    Returns
    -------
    None
    """
    dat = locate_result(data, params)
    if dat is None:
        return None
    res = dat['results']
    cm, tm = res['mf']['state_distribution'].shape
    if keys is None:
        keys = [t for t in range(10, tm, 10)]
    colors = {k: c for k, c in zip(keys, color_list)}

    type_style = {'simulation': 'x', 'mf': '-'}
    type_label = {'simulation': 'Simulation', 'mf': 'Approx.'}

    fig = plt.figure()
    x = np.array([t for t in range(cm)])

    for t in keys:
        for k in ['simulation', 'mf']:
            y = res[k]['state_distribution'][:, t]
            plt.semilogy(x[y > 0], y[y > 0], type_style[k], color=colors[t])
    plt.xlim([0, xm])
    plt.ylim([ym, 1])
    plt.xlabel('State')
    plt.ylabel('Probability')
    title = ', '.join([f"{p_labels[k]}={v}" for k, v in params.items()])
    plt.title(f"${title}$")

    if legend:
        for k in type_style:
            plt.plot([2], type_style[k], color='black', label=type_label[k])
        for t in keys:
            plt.plot([2], color=colors[t], label=f"T={t}")

    y = res['mf_asymptotic']['state_distribution']
    acolor = color_list[len(keys)]
    plt.semilogy(y, ':', color=acolor, label='Approx. ($t\\rightarrow\\infty)$')

    if legend:
        plt.legend(ncol=3)

    if export is not None:
        tikzplotlib_fix_ncols(fig)
        tikzplotlib.save(f'{export}.tex', table_row_sep='\\\\', encoding='utf8')
    plt.show()


def live_plot(data, params, c=4, legend=True, export=None, keys=None):
    """
    Parameters
    ----------
    data: :class:`list` of `dict`
        List of results from parallelized computation.
    params: :class:`dict`
        Parameters to fit
    c: :class:`int`
        Max state value considered as *alive*.
    legend: :class:`bool`, default=True
        Draw legend.
    export: :class:`str`, optional
        Save the figure as export.tex in pgfplots format.
    keys: :class:`list`, optional
        List of Epochs to display.

    Returns
    -------
    None
    """
    live_data = [d for d in data if all(d[k] == v for k, v in params.items())]
    n = np.array([d['n'] for d in live_data])

    cm, tm = live_data[0]['results']['mf']['state_distribution'].shape
    if keys is None:
        keys = [t for t in range(10, tm, 10)]
    colors = {k: c for k, c in zip(keys, color_list)}

    type_style = {'simulation': 'x', 'mf': '-'}
    type_label = {'simulation': 'Simulation', 'mf': 'Approx.'}

    fig = plt.figure()

    for t in keys:
        for k in type_style:
            y = [np.sum(d['results'][k]['state_distribution'][:(c + 1), t]) for d in live_data]
            plt.loglog(n, np.array(y) * n, type_style[k], color=colors[t])

    if legend:
        for k in type_style:
            plt.plot([2], type_style[k], color='black', label=type_label[k])
        for t in keys:
            plt.plot([2], color=colors[t], label=f"T={t}")

    y = [np.sum(d['results']['mf_asymptotic']['state_distribution'][:(c + 1)]) for d in live_data]
    acolor = color_list[len(keys)]
    plt.loglog(n, np.array(y) * n, ':', color=acolor, label='Approx. ($t\\rightarrow\\infty)$')

    if legend:
        plt.legend(ncol=3)
    plt.xlim([n[0], n[-1]])
    plt.ylim([1, None])
    plt.xlabel('N')
    plt.ylabel('Neff')
    plt.title(f'$C={c}$')
    if export is not None:
        tikzplotlib_fix_ncols(fig)
        tikzplotlib.save(f'{export}.tex', table_row_sep='\\\\', encoding='utf8')
    plt.show()
