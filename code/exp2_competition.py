"""
Experiments 2: Competition between NDNF interneurons and SOM outputs in layer 1.
- some of this is already illustrated in exp1_microcircuit
- in addition we here show that NDNF INs and SOM outputs can form a bistable mutual inhibition motif
- this bistable mutual inhibition can function as a switch, allowing to switch between SOM- and NDNF-mediated inhibition
- this switching is particularly relevant if NDNF and SOM carry distinct information (top-down vs bottom-up)
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('pretty')
import model_base as mb
from experiments import get_null_ff_input_arrays, get_model_colours, plot_violin

cPC, cPV, cSOM, cNDNF, cVIP, cpi = get_model_colours()

DPI = 150


def ex202_mutual_inhibition_switch(noise=0.0, wNS=1.4, w_hetero=False, mean_pop=True, pre_inh=True, reduced=False, save=False):
    """
    Experiment: Switch between NDNDF and SOM-dominated dendritic inhibition. Network is in bistable mututal inhibition
                regime. Activate and inactive NDNF interneurons to create switching.
    - noise: level of white noise added to neural activity
    - flag_w_hetero: whether to add heterogeneity to weight matrices
    - save: if it's a string, name of the saved file, else if False nothing is saved
    """

    # define parameter dictionaries
    N_cells, w_mean, conn_prob, bg_inputs, taus = mb.get_default_params(flag_mean_pop=mean_pop)

    # increase SOM to NDNF inhibition to get bistable regime
    w_mean['NS'] = wNS

    # increase NDNF-dendrite inhibition s.t. mean PC rate doesn't change when dendritic inhibition changes
    w_mean['DN'] = 0.7 

    if reduced:  # remove 
        w_mean['EP'], w_mean['PE'], w_mean['SE'] = 0, 0, 0

    # instantiate model
    model = mb.NetworkModel(N_cells, w_mean, conn_prob, taus, bg_inputs, wED=1, flag_w_hetero=w_hetero,
                           flag_pre_inh=pre_inh)

    # simulation paramters
    dur = 12000
    dt = 1
    nt = int(dur/dt)

    # generate inputs
    t_act_s, t_act_e = 2000, 3000
    t_inact_s, t_inact_e = 6000, 7000
    xFF = get_null_ff_input_arrays(nt, N_cells)
    xFF['N'][t_act_s:t_act_e] = 1.
    xFF['N'][t_inact_s:t_inact_e] = -1

    # add time-varying inputs to SOM (and NDNF)
    tt = np.arange(0, dur/1000, dt/1000)
    sine1 = np.sin(2*np.pi*tt*1)
    sine2 = np.sin(2*np.pi*tt*4)
    amp_sine = 0.5
    # xFF['N'][:,:] += amp_sine*np.tile(sine1, [N_cells['N'], 1]).T
    xFF['S'][:,:] += amp_sine*np.tile(sine2, [N_cells['S'], 1]).T

    # run model
    t, rE, rD, rS, rN, rP, rV, p, cGABA, other = model.run(dur, xFF, dt=dt, p0=0.5, init_noise=0, calc_bg_input=True,
                                                           monitor_dend_inh=True, noise=noise)

    # plotting
    dpi = 300 if save else DPI
    fig, ax = plt.subplots(3, 1, figsize=(3, 3.5), dpi=dpi, sharex=True,
                           gridspec_kw={'left': 0.15, 'bottom': 0.15, 'top': 0.95, 'right': 0.95,
                                        'height_ratios': [1, 1, 0.5]})
    ax[1].plot(t/1000, rN, c=cNDNF, alpha=1, lw=1, label='NDNF')
    ax[1].plot(t/1000, rS, c=cSOM, alpha=1, lw=1, label='SOM')
    ax[1].legend(loc='best', frameon=False)
    ax[0].plot(t/1000, np.mean(np.array(other['dend_inh_NDNF']), axis=1), c=cNDNF, ls='-', lw=1)
    ax[0].plot(t/1000, np.mean(np.array(other['dend_inh_SOM']), axis=1), c=cSOM, ls='-', lw=1)
    ax[2].plot(t/1000, p, c=cpi, alpha=1)

    # labels etc
    ax[0].set(ylabel='dend. inh. (au)', ylim=[-0.1, 2], yticks=[0, 1])
    ax[1].set(ylabel='activity (au)', ylim=[-0.1, 3.5], yticks=[0, 2])
    ax[2].set(ylabel='release', ylim=[-0.05, 1.05], yticks=[0, 1], xlabel='time (s)')#, xticks=[0, 1, 2, 3])

    fig2, ax2 = plt.subplots(4, 1, figsize=(3, 3.5), dpi=dpi, sharex=True, sharey=False,
                             gridspec_kw={'left': 0.15, 'bottom': 0.15, 'top': 0.95, 'right': 0.95})
    # ax2[1].plot(t/1000, sine2/4+1, alpha=1, c='darkturquoise', lw=1)
    ax2[1].plot(t/1000, rE, alpha=1, c=cPC, lw=1)
    ax2[0].plot(t/1000, rD, alpha=1, c='k', lw=1)
    ax2[2].plot(t/1000, rP, alpha=1, c=cPV, lw=1)
    # plot correlation of 
    mean_rE = np.mean(rE, axis=1)
    wbin = 500
    corrs = np.array([np.corrcoef(mean_rE[i*wbin:(i+1)*wbin], sine2[i*wbin:(i+1)*wbin])[0, 1] for i in range(nt//wbin)])
    corrs2 = np.array([np.corrcoef(mean_rE[i*wbin:(i+1)*wbin], sine1[i*wbin:(i+1)*wbin])[0, 1] for i in range(nt//wbin)])
    ax2[3].plot(np.arange(0, nt, wbin)/1000, corrs, '.-', c='mediumturquoise', ms=5)
    ax2[3].plot(np.arange(0, nt, wbin)/1000, corrs2, '.-', c='goldenrod', ms=5)
    [ax2[ii].set(ylim=[0, 2]) for ii in range(3)]
    ax2[0].set(ylabel='dend act.')
    ax2[1].set(ylabel='PC rate')
    ax2[2].set(ylabel='PV rate')
    ax2[3].set(ylim=[-1, 1], ylabel='corr inp-rE', xlabel='time (s)')



    # saving (optional)
    if save:
        fig.savefig('../results/figs/cosyne-collection/exp2-2_switch_sine_reduced.png', dpi=300)
        fig2.savefig('../results/figs/cosyne-collection/exp2-2_switch_sine_reduced_PCcorr.png', dpi=300)
        plt.close(fig)
        plt.close(fig2)


if __name__ in "__main__":


    ex202_mutual_inhibition_switch(mean_pop=True, noise=0, w_hetero=False, wNS=1.2, reduced=False, save=False)

    plt.show()