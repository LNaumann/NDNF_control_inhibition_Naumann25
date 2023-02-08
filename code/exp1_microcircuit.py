"""
Experiments 1: NDNF interneurons as part of the microcircuit.
- paired recordings to show slow NDNF-mediated inhibition and unidirectional SOM-NDNF inhibition
- 'imaging' of SOM boutons shows presynaptic inhibition mediated by NDNF interneurons
- layer-specificity: only the outputs of SOMs in layer 1 are affected, not their outputs in deeper layers
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('pretty')
import model_base as mb
from experiments import get_null_ff_input_arrays, get_model_colours

cPC, cPV, cSOM, cNDNF, cVIP, cpi = get_model_colours()

DPI = 150

def fig1_paired_recordings_invitro(dur=800, dt=1, w_hetero=False, mean_pop=True, pre_inh=True, noise=0, save=False):
    """
    Run experiment of paired "in vitro" recordings and plot.
    - stimulate NDNF and SOM, record from PC, SOM and NDNF
    - "in vitro" means all cells have 0 baseline activity
    - paired recordings means the ingoing currents to each cell
    :param dur:  length of experiment
    :param dt:   time step
    """

    # simulation paramters
    nt = int(dur / dt)
    t = np.arange(0, dur, dt)
    t0 = 100
    amp = 3

    # create figure
    dpi = 300 if save else DPI
    fig, ax = plt.subplots(3, 2, figsize=(3, 2.5), dpi=dpi, sharex=True, sharey='row',
                           gridspec_kw={'right': 0.95, 'bottom': 0.21, 'left': 0.15})

    # get default parameters
    N_cells, w_mean, conn_prob, bg_inputs, taus = mb.get_default_params(flag_mean_pop=mean_pop)

    # increase NDNF->dendrite inhibition so the effect is comparable to SOM->dendrite (visually)
    w_mean['DN'] = 1.

    # stimulate SOM and NDNF, respectively
    for i, cell in enumerate(['S', 'N']):

        # array of FF input
        xFF = get_null_ff_input_arrays(nt, N_cells)
        xFF[cell][:, :] = amp * np.tile(np.exp(-(t - t0) / 50) * np.heaviside(t - t0, 1), (N_cells[cell], 1)).T
        # xFF[cell][50:150, :] = 2

        # create model and run
        model = mb.NetworkModel(N_cells, w_mean, conn_prob, taus, bg_inputs, wED=1, flag_w_hetero=w_hetero,
                                flag_pre_inh=pre_inh, gamma=1)
        t, rE, rD, rS, rN, rP, rV, p, cGABA, other = model.run(dur, xFF, dt=dt, init_noise=0, noise=noise, 
                                                               rE0=0, rD0=1, rN0=0, rS0=0, rP0=0, monitor_currents=True)
        # note: dendritic activity is set to 1 so that the inhibition by SOM and NDNF shows in the soma

        # plotting and labels
        # ax[0, i].plot(t[1:], other['curr_rE'], c=cPC, alpha=0.1)
        # ax[1, i].plot(t[1:], other['curr_rS'], c=cSOM, alpha=0.1)
        # ax[2, i].plot(t[1:], other['curr_rN'], c=cNDNF, alpha=0.1)
        ax[0, i].plot(t[1:], np.mean(other['curr_rE'], axis=1), c=cPC)
        ax[1, i].plot(t[1:], np.mean(other['curr_rS'], axis=1), c=cSOM)
        ax[2, i].plot(t[1:], np.mean(other['curr_rN'], axis=1), c=cNDNF)
        ax[2, i].set(xlabel='time (ms)', ylim=[-0.1, 3])

    ax[0, 0].set(ylim=[-1, 1], ylabel='curr. (au)')
    ax[1, 0].set(ylim=[-3, 3], ylabel='curr. (au)', yticks=[-2, 0, 2])
    ax[2, 0].set(ylim=[-3, 3], ylabel='curr. (au)', yticks=[-2, 0, 2])

    if save:
        fig.savefig('../results/figs/cosyne-collection/exp1-1_paired-recordings.png', dpi=300)
        plt.close(fig)


fig1_paired_recordings_invitro(mean_pop=True, w_hetero=True, noise=0, save=True)
plt.show()
