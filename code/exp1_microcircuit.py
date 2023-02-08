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
from experiments import get_null_ff_input_arrays, get_model_colours, plot_violin

cPC, cPV, cSOM, cNDNF, cVIP, cpi = get_model_colours()

DPI = 150

def exp101_paired_recordings_invitro(dur=800, dt=1, w_hetero=False, mean_pop=True, pre_inh=True, noise=0, save=False):
    """
    Run experiment of paired "in vitro" recordings and plot.
    - stimulate NDNF and SOM, record from PC, SOM and NDNF
    - "in vitro" means all cells have 0 baseline activity
    - paired recordings means the ingoing currents to each cell

    Parameters
    - dur: length of experiment
    - dt: time step
    - ...
    """
    # TODO: complete list of parameters

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


def exp102_preinh_bouton_imaging(dur=2000, ts=200, te=500, dt=1, stim_NDNF=1.5, w_hetero=True, mean_pop=False, pre_inh=True, noise=0.2, save=False):
    """
    Experiment: Image SOM bouton in response to stimulation of NDNF interneurons.

    Parameters
    - dur: duration of experiment (ms)
    - ts: start of NDNF stimulation
    - te: end of NDNF stimulation
    - dt: integration time step (ms)
    - stim_NDNF: strength of NDNF stimulation
    - noise: level of white noise added to neural activity
    - flag_w_hetero: whether to add heterogeneity to weight matrices
    - mean_pop: if true, simulate only one neuron (mean) per population
    """

    # define parameter dictionaries
    N_cells, w_mean, conn_prob, bg_inputs, taus = mb.get_default_params(flag_mean_pop=mean_pop)

    # instantiate model
    model = mb.NetworkModel(N_cells, w_mean, conn_prob, taus, bg_inputs, wED=1, flag_w_hetero=w_hetero,
                           flag_pre_inh=pre_inh)

    # simulation paramters
    nt = int(dur/dt)

    # generate inputs
    xFF = get_null_ff_input_arrays(nt, N_cells)
    xFF['N'][ts:te] = stim_NDNF

    # run model
    t, rE, rD, rS, rN, rP, rV, p, cGABA, other = model.run(dur, xFF, init_noise=0.1, noise=noise, dt=dt, monitor_boutons=True,
                                                           monitor_currents=True)
    # TODO: add init_noise as parameter to the function?

    # plotting
    # --------
    # 3 different plots here: an overview plot, bouton imaging + quantification and only bouton imaging

    # plot 1: response of all neurons to NDNF stimulation
    dpi = 300 if save else DPI
    fig, ax = plt.subplots(6, 1, figsize=(4, 5), dpi=dpi, sharex=True)
    ax[0].plot(t, rE, c='C3', alpha=0.5)
    ax[1].plot(t, rD, c='k', alpha=0.5)
    ax[2].plot(t, rS, c='C0', alpha=0.5)
    ax[3].plot(t, rN, c='C1', alpha=0.5)
    ax[4].plot(t, rP, c='darkblue', alpha=0.5)
    ax[5].plot(t, p, c='C2', alpha=1)
    for i, label in enumerate(['PC', 'dend.', 'SOM', 'NDNF', 'PV']):
        ax[i].set(ylabel=label, ylim=[0, 3])
    ax[5].set(ylabel='p', ylim=[0, 1], xlabel='time (ms)')

    # plot 2: monitoring SOM boutons
    fig2, ax2 = plt.subplots(1, 1, figsize=(2.5, 2), dpi=dpi, gridspec_kw={'left':0.25, 'right':0.9, 'bottom':0.2})
    boutons = np.array(other['boutons_SOM'])
    boutons_nonzero = boutons[:, np.mean(boutons, axis=0) > 0]
    cm = ax2.pcolormesh(boutons_nonzero.T, cmap='Blues', vmin=0) #, vmax=0.15)
    plt.colorbar(cm, ticks=[0, 0.5])
    ax2.set(xlabel='time (ms)', ylabel='# bouton', yticks=[0, (len(boutons_nonzero.T)//50)*50], xticks=[0, 1000, 2000])

    # plot 3: quantification of monitoring SOM boutons
    fig3, ax3 = plt.subplots(1, 1, figsize=(2.5, 2), dpi=dpi, gridspec_kw={'left': 0.25, 'right':0.9, 'bottom':0.2})
    boutons_NDNFact = np.mean(boutons_nonzero[ts:te+200], axis=0)
    boutons_cntrl = np.mean(boutons_nonzero[0:ts], axis=0)
    plot_violin(ax3, 0, boutons_cntrl, color=cSOM)
    plot_violin(ax3, 1, boutons_NDNFact, color='#E9B86F')
    ax3.set(xlim=[-0.5, 1.5], xticks=[0, 1], xticklabels=['ctrl', 'NDNF act.'], ylim=[0, 0.6], yticks=[0, 0.5],
            ylabel='SOM bouton act.')
    
    if save:
        fig.savefig('../results/figs/cosyne-collection/exp1-2_preinh-allneurons.png', dpi=300)
        fig2.savefig('../results/figs/cosyne-collection/exp1-2_preinh-boutons-all.png', dpi=300)
        fig3.savefig('../results/figs/cosyne-collection/exp1-2_preinh-boutons-violin.png', dpi=300)
        [plt.close(ff) for ff in [fig, fig2, fig3]]


if __name__ in "__main__":


    # exp101_paired_recordings_invitro(mean_pop=True, w_hetero=True, noise=0, save=True)

    exp102_preinh_bouton_imaging(save=True)

    plt.show()
