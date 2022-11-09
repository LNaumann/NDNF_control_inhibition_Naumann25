"""
Experiments on the network model.
"""

import numpy as np
import matplotlib.pyplot as plt
from model_base import get_default_params, NetworkModel
plt.style.use('pretty')

# ToDo: make experiments classes?

# colours
cPC = '#B83D49'
cPV = '#345377'
cSOM = '#5282BA'
cNDNF = '#E18E69'
cVIP = '#D1BECF'

def ex_activation_inactivation():
    """
    Experiment: activate and inactive different cell types and check effect on all other cells. Plots big fig array of
    results. Mostly for rough inspection of the model
    """

    # simulation parameters
    dur = 700
    dt = 1
    nt = int(dur/dt)

    # activation/inactivation parameters
    ts, te = 300, 450  # start and end point of activation
    I_activate = 1  # -1 for inactivation

    # get default parameters
    N_cells, w_mean, conn_prob, bg_inputs, taus = get_default_params()
    wED = 1

    # parameters can be adapted: e.g.
    # w_mean['PN'] = 1

    # initialise model
    model = NetworkModel(N_cells, w_mean, conn_prob, taus, bg_inputs, wED=wED, flag_SOM_ad=False,
                         flag_w_hetero=False, flag_pre_inh=False, flag_with_VIP=True, flag_with_NDNF=False)

    # print stuff about weights (derived from math, indicate effect of SOM input on PC and PV)
    # ToDo: remove in the future
    gamma = w_mean['EP']*w_mean['PS'] - wED*w_mean['DS']
    print(w_mean['EP']*w_mean['PS'], wED*w_mean['DS'], 'gamma=', gamma)
    # print(1+w_mean['EP']*w_mean['PE'], (w_mean['EP']*w_mean['PS']-wED*w_mean['DS'])*w_mean['SE'])
    print(w_mean['PE'], w_mean['PS']*w_mean['SE'])
    print((w_mean['PE']-w_mean['PS']*w_mean['SE'])*gamma/(1+w_mean['EP']*w_mean['PE']-gamma), w_mean['PS'] )

    # create empty figure
    fig1, ax1 = plt.subplots(6, 6, figsize=(6, 5), dpi=150, sharex=True, sharey='row', gridspec_kw={'right': 0.95})

    # Activation/ Inactivation of different cell types
    for i, cell in enumerate(['E', 'D', 'S', 'N', 'P', 'V']):

        # create FF inputs (i.e. stimulation)
        xFF = get_null_ff_input_arrays(nt, N_cells)
        xFF[cell][ts:te, :] = I_activate  # N_cells[cell]//2

        # run network
        t, rE, rD, rS, rN, rP, rV, p, other = model.run(dur, xFF, dt=dt, init_noise=0)

        # plot
        ax1[0, i].plot(t, rE, c='C3', alpha=0.5)
        ax1[1, i].plot(t, rD, c='darkred', alpha=0.5)
        ax1[2, i].plot(t, rS, c='C0', alpha=0.5)
        ax1[3, i].plot(t, rN, c='C1', alpha=0.5)
        ax1[4, i].plot(t, rP, c='darkblue', alpha=0.5)
        ax1[5, i].plot(t, rV, c='C4', alpha=0.5)
        ax1[0, i].set(title='act. '+cell)
        ax1[-1, i].set(xlabel='time (ms)')  #, ylim=[0, 1])

    # add labels for rows
    for j, name in enumerate(['PC', 'dend', 'SOM', 'NDNF', 'PV', 'VIP']):
        ax1[j, 0].set(ylabel=name, ylim=[0, 2.5])


def fig1_paired_recordings_invitro(dur=300, dt=1):
    """
    Run experiment of paired "in vitro" recordings and plot.
    :param dur:  length of experiment
    :param dt:   time step
    """

    # simulation paramters
    nt = int(dur / dt)
    t = np.arange(0, dur, dt)
    t0 = 50

    # create figure
    fig, ax = plt.subplots(2, 2, figsize=(3, 1.7), dpi=400, sharex=True, sharey='row',
                           gridspec_kw={'right': 0.95, 'bottom': 0.21, 'left': 0.15})

    # get default parameters
    N_cells, w_mean, conn_prob, bg_inputs, taus = get_default_params()

    # stimulate SOM and NDNF, respectively
    for i, cell in enumerate(['S', 'N']):
        # array of FF input
        xFF = get_null_ff_input_arrays(nt, N_cells)
        xFF[cell][:, :] = 2 * np.tile(np.exp(-(t - t0) / 50) * np.heaviside(t - t0, 1), (N_cells[cell], 1)).T
        # create model and run
        model = NetworkModel(N_cells, w_mean, conn_prob, taus, bg_inputs, wED=1, flag_SOM_ad=False,
                             flag_w_hetero=False, flag_pre_inh=False, flag_with_VIP=False, flag_with_NDNF=True)
        t, rE, rD, rS, rN, rP, rV, p, other = model.run(dur, xFF, dt=dt, init_noise=0, monitor_currents=True,
                                                        rE0=0, rD0=0, rN0=0, rS0=0, rP0=0)

        # plotting and labels
        ax[0, i].plot(t[1:], other['curr_rS'], c=cSOM)
        ax[1, i].plot(t[1:], other['curr_rN'], c=cNDNF)
        ax[1, i].set(xlabel='time (ms)', ylim=[-0.1, 3])
    ax[0, 0].set(ylim=[-2, 2], ylabel='curr. (au)')
    ax[1, 0].set(ylim=[-2, 2], ylabel='curr. (au)')


def fig1_activation(I_activate=1, dur=1000, ts=400, te=600, dt=1):
    """
    Hacky function to test activation of SOM/NDNF/VIP in networks with NDNFs and VIPs, respectively. Plots first draft
    for different panels in Figure 1.

    :param I_activate:  activation input
    :param dur:         duration of experiment (in ms)
    :param ts:          start time of stimulation
    :param te:          end time of stimulation
    :param dt:          time step
    """

    nt = int(dur/dt)

    # i) default, with NDNF
    fig1, ax1 = plt.subplots(4, 2, figsize=(2, 2), dpi=300, sharex=True, sharey='row', gridspec_kw={'right': 0.95})
    N_cells, w_mean, conn_prob, bg_inputs, taus = get_default_params()
    for i, cell in enumerate(['S', 'N']):
        xFF = get_null_ff_input_arrays(nt, N_cells)
        xFF[cell][ts:te, :] = I_activate
        model = NetworkModel(N_cells, w_mean, conn_prob, taus, bg_inputs, wED=1, flag_SOM_ad=False,
                             flag_w_hetero=False, flag_pre_inh=False, flag_with_VIP=False, flag_with_NDNF=True)
        t, rE, rD, rS, rN, rP, rV, p, other = model.run(dur, xFF, dt=dt, init_noise=0)
        ax1[0, i].plot(t, rE, c=cPC, alpha=0.5)
        ax1[1, i].plot(t, rP, c=cPV, alpha=0.5)
        ax1[2, i].plot(t, rS, c=cSOM, alpha=0.5)
        ax1[3, i].plot(t, rN, c=cNDNF, alpha=0.5)
        for j in range(4):
            ax1[j, i].axis('off')

    # ii) default, with VIP
    fig2, ax2 = plt.subplots(4, 2, figsize=(2, 2), dpi=300, sharex=True, sharey='row', gridspec_kw={'right': 0.95})
    N_cells, w_mean, conn_prob, bg_inputs, taus = get_default_params()
    for i, cell in enumerate(['S', 'V']):

        xFF = get_null_ff_input_arrays(nt, N_cells)
        xFF[cell][ts:te, :] = I_activate
        model = NetworkModel(N_cells, w_mean, conn_prob, taus, bg_inputs, wED=1, flag_SOM_ad=False,
                             flag_w_hetero=False, flag_pre_inh=False, flag_with_VIP=True, flag_with_NDNF=False)
        t, rE, rD, rS, rN, rP, rV, p, other = model.run(dur, xFF, dt=dt, init_noise=0)

        ax2[0, i].plot(t, rE, c=cPC, alpha=0.5)
        ax2[1, i].plot(t, rP, c=cPV, alpha=0.5)
        ax2[2, i].plot(t, rS, c=cSOM, alpha=0.5)
        ax2[3, i].plot(t, rV, c=cVIP, alpha=0.5)
        for j in range(4):
            ax2[j, i].axis('off')

    # iii) stim SOM, different settings
    fig2, ax3 = plt.subplots(4, 2, figsize=(2, 2), dpi=300, sharex=True, sharey='row', gridspec_kw={'right': 0.95})
    N_cells, w_mean, conn_prob, bg_inputs, taus = get_default_params()

    # a: disinhibition-dominated
    w_mean = dict(NS=1, DS=0.5, DN=1.5, SE=0.5, NN=0, PS=0.5, PN=0.5, PP=0, PE=0.7, EP=1, DE=0, VS=0, SV=0)
    xFF = get_null_ff_input_arrays(nt, N_cells)
    xFF['S'][ts:te, :] = I_activate
    model = NetworkModel(N_cells, w_mean, conn_prob, taus, bg_inputs, wED=1, flag_SOM_ad=False,
                         flag_w_hetero=False, flag_pre_inh=False, flag_with_VIP=False, flag_with_NDNF=True)
    t, rE, rD, rS, rN, rP, rV, p, other = model.run(dur, xFF, dt=dt, init_noise=0)
    ax3[0, 0].plot(t, rE, c=cPC, alpha=0.5)
    ax3[1, 0].plot(t, rP, c=cPV, alpha=0.5)
    ax3[2, 0].plot(t, rS, c=cSOM, alpha=0.5)
    ax3[3, 0].plot(t, rN, c=cNDNF, alpha=0.5)

    # b: same but NDNF inactive
    model = NetworkModel(N_cells, w_mean, conn_prob, taus, bg_inputs, wED=1, flag_SOM_ad=False,
                         flag_w_hetero=False, flag_pre_inh=False, flag_with_VIP=False, flag_with_NDNF=True)
    t, rE, rD, rS, rN, rP, rV, p, other = model.run(dur, xFF, dt=dt, init_noise=0, rN0=0)
    ax3[0, 1].plot(t, rE, c=cPC, alpha=0.5)
    ax3[1, 1].plot(t, rP, c=cPV, alpha=0.5)
    ax3[2, 1].plot(t, rS, c=cSOM, alpha=0.5)
    ax3[3, 1].plot(t, rN, c=cNDNF, alpha=0.5)

    for j in range(4):
        ax3[j, 0].axis('off')
        ax3[j, 1].axis('off')


def fig1_weights_role(I_activate=1, dur=1000, ts=400, te=600, dt=1):
    """
    Hacky function to plot effect of different weights on the responses to simulated optogenetic experiments.
    Plots figure panel draft for fig 1.

    :param I_activate:  activation input
    :param dur:         duration of experiment (in ms)
    :param ts:          start time of stimulation
    :param te:          end time of stimulation
    :param dt:          time step
    """

    nt = int(dur / dt)

    # create figure
    fig, ax = plt.subplots(1, 2, figsize=(2.8, 1.5), dpi=400, sharex=False, sharey='row',
                           gridspec_kw={'right': 0.95, 'left':0.16, 'bottom':0.25})

    # use paramterisation from disinhibition-dominated regime (overwrite w_mean)
    N_cells, w_mean, conn_prob, bg_inputs, taus = get_default_params()
    w_mean_df = dict(NS=1, DS=0.5, DN=1.5, SE=0.5, NN=0, PS=0.5, PN=0.5, PP=0, PE=0.7, EP=1, DE=0, VS=0, SV=0)
    w_mean = w_mean_df.copy()

    # create input (stimulation of SOMs)
    xFF = get_null_ff_input_arrays(nt, N_cells)
    xFF['S'][ts:te, :] = I_activate

    # vary wPN
    change_rE_wPN = []
    change_rP_wPN = []
    wPN_range = np.arange(0, 2.1, 0.1)
    for wPN in wPN_range:
        w_mean['PN'] = wPN
        model = NetworkModel(N_cells, w_mean, conn_prob, taus, bg_inputs, wED=1, flag_SOM_ad=False,
                            flag_w_hetero=False, flag_pre_inh=False, flag_with_VIP=False, flag_with_NDNF=True)
        t, rE, rD, rS, rN, rP, rV, p, other = model.run(dur, xFF, dt=dt, init_noise=0)
        change_rE_wPN.append(np.mean(rE[ts:te])/np.mean(rE[:te-ts]))
        change_rP_wPN.append(np.mean(rP[ts:te])/np.mean(rP[:te-ts]))
    ax[0].plot(wPN_range, change_rE_wPN, cPC)
    ax[0].plot(wPN_range, change_rP_wPN, cPV)

    # vary wDN
    w_mean = w_mean_df.copy()
    change_rE_wDN = []
    change_rP_wDN = []
    wDN_range = np.arange(0, 2.1, 0.1)
    for wDN in wDN_range:
        w_mean['DN'] = wDN
        model = NetworkModel(N_cells, w_mean, conn_prob, taus, bg_inputs, wED=1, flag_SOM_ad=False,
                            flag_w_hetero=False, flag_pre_inh=False, flag_with_VIP=False, flag_with_NDNF=True)
        t, rE, rD, rS, rN, rP, rV, p, other = model.run(dur, xFF, dt=dt, init_noise=0)
        change_rE_wDN.append(np.mean(rE[ts:te])/np.mean(rE[:te-ts]))
        change_rP_wDN.append(np.mean(rP[ts:te])/np.mean(rP[:te-ts]))
    ax[1].plot(wDN_range, change_rE_wDN, cPC)
    ax[1].plot(wDN_range, change_rP_wDN, cPV)

    # pretty up the plot
    ax[0].hlines(1, 0, 2, ls='--', color='silver', lw=1, zorder=-1)
    ax[1].hlines(1, 0, 2, ls='--', color='silver', lw=1, zorder=-1)
    ax[0].set(xlabel='NDNF->PV', ylabel='rate change (rel)', ylim=[0.5, 1.6], yticks=[0.5, 1, 1.5])
    ax[1].set(xlabel='NDNF->dendrite')


def ex_bouton_imaging(dur=1000, ts=300, te=400, stim_NDNF=2):
    # ToDo: document!

    # define parameter dictionaries
    N_cells, w_mean, conn_prob, bg_inputs, taus = get_default_params()

    # instantiate model
    model = NetworkModel(N_cells, w_mean, conn_prob, taus, bg_inputs, wED=0.7, flag_SOM_ad=False, flag_w_hetero=True,
                         flag_pre_inh=True)

    # simulation paramters
    dt = 1
    nt = int(dur/dt)

    # generate inputs
    xFF = get_null_ff_input_arrays(nt, N_cells)
    xFF['N'][ts:te] = stim_NDNF

    # run model
    t, rE, rD, rS, rN, rP, rV, p, other = model.run(dur, xFF, init_noise=0.1, noise=0.2, dt=dt, monitor_boutons=True,
                                                    monitor_currents=True, calc_bg_input=True)

    # plotting
    fig, ax = plt.subplots(6, 1, figsize=(4, 5), dpi=150, sharex=True)
    ax[0].plot(t, rE, c='C3', alpha=0.5)
    ax[1].plot(t, rD, c='k', alpha=0.5)
    ax[2].plot(t, rS, c='C0', alpha=0.5)
    ax[3].plot(t, rN, c='C1', alpha=0.5)
    ax[4].plot(t, rP, c='darkblue', alpha=0.5)
    ax[5].plot(t, p, c='C2', alpha=1)

    # label stuff
    for i, label in enumerate(['PC', 'dend.', 'SOM', 'NDNF', 'PV']):
        ax[i].set(ylabel=label, ylim=[0, 3])
    ax[5].set(ylabel='p', ylim=[0, 1], xlabel='time (ms)')

    fig2, ax2 = plt.subplots(1, 1, figsize=(2, 1.1), dpi=300, gridspec_kw={'left':0.3, 'right':0.9, 'bottom':0.35})
    boutons = np.array(other['boutons_SOM'])
    boutons_nonzero = boutons[:, np.mean(boutons, axis=0) > 0]
    cm = ax2.pcolormesh(boutons_nonzero.T, cmap='Blues', vmin=0, vmax=0.15)
    plt.colorbar(cm, ticks=[0, 0.1])
    ax2.set(xlabel='time (ms)', ylabel='# bouton', yticks=[0, 400], xticks=[0, 1000])


def get_null_ff_input_arrays(nt, N_cells):
    """
    Generate empty arrays for feedforward input.
    :param nt:      Number of timesteps
    :param Ncells:  Dictionary of cell numbers (soma E, dendrite E, SOMs S, NDNFs N, PVs P)
    :return:        Dictionary with an empty array of size nt x #cell for each neuron type
    """

    xFF_null = dict()
    for key in N_cells.keys():
        xFF_null[key] = np.zeros((nt, N_cells[key]))

    return xFF_null


if __name__ in "__main__":

    # run different experiments; comment in or out to run only some of them

    # ex_activation_inactivation()
    # fig1_paired_recordings_invitro()
    # fig1_activation()
    # fig1_weights_role()
    ex_bouton_imaging()
    plt.show()