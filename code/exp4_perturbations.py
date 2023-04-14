"""
Experiments 4: Perturbation experiments.
- 
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('poster')
import matplotlib as mpl
lw = mpl.rcParams['lines.linewidth']
import seaborn as sns
import model_base as mb
from experiments import get_null_ff_input_arrays, get_model_colours, plot_violin

cPC, cPV, cSOM, cNDNF, cVIP, cpi = get_model_colours()

DPI = 150


def exp401_perturbations(mean_pop=True, w_hetero=False, reduced=False, pre_inh=False, noise=0.0, save=False):
    """
    Experiment1: activate and inactive different cell types and check effect on all other cells. Plots big fig array of
    results.
    """

    # simulation parameters
    dur = 1300
    dt = 1
    nt = int(dur/dt)

    # activation/inactivation parameters
    ts, te = 300, 400  # start and end point of activation
    I_activate = 1  # -1 for inactivation

    # get default parameters
    N_cells, w_mean, conn_prob, bg_inputs, taus = mb.get_default_params(flag_mean_pop=mean_pop)
    wED = 1

    if reduced:
        w_mean.update(dict(PE=0, SE=0, DE=0))

    # default params
    # w_mean.update(dict(DS=1))

    # disinhibitory params
    w_mean_d = w_mean.copy()
    w_mean_d.update(dict(DN=1.5, PN=0.5) ) #, PN=0.1, PS=0.5, PE=1))
    # dict(NS=1, DS=0.5, DN=1.5, SE=0.5, NN=0, PS=0.5, PN=0.5, PP=0, PE=0.7, EP=1, DE=0, VS=0, SV=0))
    # dict(NS=0.7, DS=0.5, DN=0.4, SE=0.8, NN=0.2, PS=0.8, PN=0.5, PP=0.1, PE=1, EP=0.5, DE=0, VS=0.5, SV=0.5)
    
    # parameters can be adapted: e.g.
    # w_mean['PN'] = 1

    # print stuff about weights (derived from math, indicate effect of SOM input on PC and PV)
    # ToDo: remove in the future
    # print(w_mean[''])

    gamma = w_mean['EP']*w_mean['PS'] - wED*w_mean['DS']
    print(w_mean['EP']*w_mean['PS'], wED*w_mean['DS'], 'gamma=', gamma)
    # print(1+w_mean['EP']*w_mean['PE'], (w_mean['EP']*w_mean['PS']-wED*w_mean['DS'])*w_mean['SE'])
    print(w_mean['PE'], w_mean['PS']*w_mean['SE'])
    print((w_mean['PE']-w_mean['PS']*w_mean['SE'])*gamma/(1+w_mean['EP']*w_mean['PE']-gamma), w_mean['PS'] )

    # create empty figure
    dpi = 300 if save else DPI
    fig, ax = plt.subplots(4, 2, figsize=(6, 5), dpi=dpi, sharex=True, sharey='row',
                           gridspec_kw={'right': 0.95, 'left': 0.2, 'bottom': 0.2, 'top': 0.95, 'height_ratios':[1, 2, 2, 2]})

    # colors
    colsGray = sns.color_palette(f"light:darkgray", n_colors=3)[1:]
    colsSOM = sns.color_palette(f"light:{cSOM}", n_colors=3)[1:]
    colsNDNF = sns.color_palette(f"light:{cNDNF}", n_colors=3)[1:]
    colsPC = sns.color_palette(f"light:{cPC}", n_colors=3)[1:]
    colsPV = sns.color_palette(f"light:{cPV}", n_colors=3)[1:]

    # Activation/ Inactivation of different cell types
    # for i, cell in enumerate(['S']):
    cell = 'N'
    for i, wparam in enumerate([w_mean, w_mean_d]):
        for j, sdur in enumerate([100, 700]):
            
            # initialise model
            model = mb.NetworkModel(N_cells, wparam, conn_prob, taus, bg_inputs, wED=wED,
                            flag_w_hetero=w_hetero, flag_pre_inh=pre_inh, flag_with_VIP=False, flag_with_NDNF=True)

            # create FF inputs (i.e. stimulation)
            xFF = get_null_ff_input_arrays(nt, N_cells)
            xFF[cell][ts:ts+sdur, :] = I_activate  # N_cells[cell]//2

            # run network
            t, rE, rD, rS, rN, rP, rV, p, cGABA, other = model.run(dur, xFF, dt=dt, init_noise=0, noise=noise)

            # plot
            alpha = 1
            ax[0, i].plot(t/1000, np.mean(xFF['N'], axis=1), c=colsGray[j], alpha=alpha, zorder=-j, lw=lw)
            ax[1, i].plot(t/1000, np.mean(rN, axis=1), c=colsNDNF[j], alpha=alpha, zorder=-j, lw=lw)
            ax[2, i].plot(t/1000, np.mean(rP, axis=1), c=colsPV[j], alpha=alpha, zorder=-j, lw=lw)
            ax[3, i].plot(t/1000, np.mean(rE, axis=1), c=colsPC[j], alpha=alpha, zorder=-j, lw=lw)
            # ax[4, i].plot(t, rD, c='k', alpha=alpha)
            # ax[5, i].plot(t, rV, c='C4', alpha=0.5)
            # ax[0, i].set(title='act. '+cell)
            ax[-1, i].set(xlabel='time (s)', xticks=[0, 1])  #, ylim=[0, 1])

    # add labels for rows
    for j, name in enumerate(['input', 'NDNF', 'PV', 'PC']):
        ax[j, 0].set( ylim=[0, 2.5], yticks=[]) #, yticks=[0, 2]) #ylabel=name,
        # ax[j, 0].spines['left'].set_visible(False)
        # ax[j, 1].spines['left'].set_visible(False)
        for i in [0, 1]:
            ax[j, i].axis('off')

    # for i in [0, 1]:
    #     ax[0, i].spines['bottom'].set_visible(False)
    #     ax[1, i].spines['bottom'].set_visible(False)
    #     ax[2, i].spines['bottom'].set_visible(False)

    ax[0, 0].set(ylim=[-0.1, 1.2])
    ax[0, 1].set(ylim=[-0.1, 1.2])

    if save:
        fig.savefig('../results/figs/cosyne-collection/exp4-1_perturb_responses.png', dpi=300)
        plt.close(fig)


def exp402_perturb_vary_params(mean_pop=True, w_hetero=False, reduced=False, pre_inh=False, noise=0.0, save=False):
    """
    Experiment2: vary wDN and wPN for long and short stimulus and check effect on PV and PC
    """

    # simulation parameters
    dur = 1000
    dt = 1
    nt = int(dur/dt)

    # activation/inactivation parameters
    ts, te = 300, 400  # start and end point of activation
    I_activate = 1  # -1 for inactivation

    # get default parameters
    N_cells, w_mean, conn_prob, bg_inputs, taus = mb.get_default_params(flag_mean_pop=mean_pop)
    wED = 1

    # increase SOM-dendrite inhibition
    # w_mean.update(dict(DS=1))

    if reduced:
        w_mean.update(dict(PE=0, SE=0, DE=0))

    wDNs = np.arange(0.1, 1.51, 0.2)
    wPNs = np.arange(0.1, 1.51, 0.2)#(0, 2, 0.2)
    durs = np.array([100, 700])

    # empty arrays for storage
    rE_record = np.zeros((len(wDNs), len(wPNs)))
    rP_record = np.zeros((len(wDNs), len(wPNs)))

    # dict(NS=1, DS=0.5, DN=1.5, SE=0.5, NN=0, PS=0.5, PN=0.5, PP=0, PE=0.7, EP=1, DE=0, VS=0, SV=0))
    # dict(NS=0.7, DS=0.5, DN=0.4, SE=0.8, NN=0.2, PS=0.8, PN=0.5, PP=0.1, PE=1, EP=0.5, DE=0, VS=0.5, SV=0.5)
    
    # parameters can be adapted: e.g.
    # w_mean['PN'] = 1

    # create empty figure
    dpi = 300 if save else DPI
    fig, ax = plt.subplots(2, 2, figsize=(6, 5), dpi=dpi, sharex=True, sharey='row',
                           gridspec_kw={'right': 0.95, 'left': 0.2, 'bottom': 0.2})

    # colors
    # colsNDNF = sns.color_palette(f"light:{cNDNF}", n_colors=3)[1:]
    colsPC = sns.color_palette(f"light:{cPC}", n_colors=len(durs)+1)[1:]
    colsPV = sns.color_palette(f"light:{cPV}", n_colors=len(durs)+1)[1:]

    # Activation/ Inactivation of different cell types
    # for i, cell in enumerate(['S']):
    cell = 'N'
    for k, sdur in enumerate(durs):

        # create FF inputs (i.e. stimulation)
        xFF = get_null_ff_input_arrays(nt, N_cells)
        xFF[cell][ts:ts+sdur, :] = I_activate

        for j, wPN in enumerate(wPNs):
            for i, wDN in enumerate(wDNs):

                # adapt parameters
                wparam = w_mean.copy()
                wparam.update(dict(DN=wDN, PN=wPN))

                # initialise model
                model = mb.NetworkModel(N_cells, wparam, conn_prob, taus, bg_inputs, wED=wED,
                                       flag_w_hetero=w_hetero, flag_pre_inh=pre_inh, flag_with_VIP=False, flag_with_NDNF=True)

                # run network
                t, rE, rD, rS, rN, rP, rV, p, cGABA, other = model.run(dur, xFF, dt=dt, init_noise=0, noise=noise)

                # save
                rE_record[i, j] = np.mean(rE[ts:ts+sdur]-np.mean(rE[:ts]))
                rP_record[i, j] = np.mean(rP[ts:ts+sdur]-np.mean(rP[:ts]))

        ishow = 2
        jshow = 2
        ax[0, 0].plot(wDNs, rE_record[:, jshow], '.-', c=colsPC[k], lw=lw, ms=4*lw)
        ax[1, 0].plot(wDNs, rP_record[:, jshow], '.-', c=colsPV[k], lw=lw, ms=4*lw)

        ax[0, 1].plot(wPNs, rE_record[ishow, :], '.-', c=colsPC[k], lw=lw, ms=4*lw)
        ax[1, 1].plot(wPNs, rP_record[ishow, :], '.-', c=colsPV[k], lw=lw, ms=4*lw)

    for cc in [0, 1]:
        ax[0, cc].hlines(0, 0, 1.5, color='k', lw=2, ls='--', zorder=-1)
        ax[1, cc].hlines(0, 0, 1.5, color='k', lw=2, ls='--', zorder=-1)

    # for rr in [0, 1]:
        # ax[rr, 0].vlines(wDNs[ishow], 0.2, 1.3, )

    ax[0, 0].set(ylim=[-0.6, 0.4], ylabel=r'$\Delta$ PC act.')
    ax[0, 1].set(ylim=[-0.6, 0.4])
    ax[1, 0].set(ylim=[-0.9, 0.1], xlabel='NDNF-dend.', ylabel=r'$\Delta$ PV act.')
    ax[1, 1].set(ylim=[-0.9, 0.1], xlabel='NDNF-PV')

        # cm1 = ax[0, k].pcolormesh(rE_record, cmap=sns.blend_palette(['blue', 'white', 'red'], as_cmap=True), vmin=0.5, vmax=1.5)
        # cm2 = ax[1, k].pcolormesh(rP_record, cmap=sns.blend_palette(['blue', 'white', 'red'], as_cmap=True), vmin=0.5, vmax=1.5)
    # plt.colorbar(cm1, ax=ax[0, 1])
    # plt.colorbar(cm2, ax=ax[1, 1])

    if save:
        fig.savefig('../results/figs/cosyne-collection/exp4-2_perturb_vary-weights.png', dpi=300)
        plt.close(fig)


if __name__ in "__main__":

    # exp401_perturbations(save=True, reduced=False, pre_inh=True, mean_pop=False, w_hetero=True, noise=0.1)
    exp402_perturb_vary_params(reduced=False, pre_inh=True, mean_pop=False, w_hetero=True, noise=0.1, save=True)

    plt.show()