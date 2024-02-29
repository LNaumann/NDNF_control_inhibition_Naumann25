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
cpi = '#A7C274'


def exp_unused_preinh_bouton_imaging(dur=2000, ts=200, te=500, dt=1, stim_NDNF=1.5, w_hetero=True, mean_pop=False, pre_inh=True, noise=0.2, save=False):
    """
    Experiment2: Image SOM bouton in response to stimulation of NDNF interneurons.

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
    ax[4].plot(t, rE, c=cPC, alpha=0.5)
    ax[3].plot(t, rD, c='k', alpha=0.5)
    ax[1].plot(t, rS, c=cSOM, alpha=0.5)
    ax[0].plot(t, rN, c=cNDNF, alpha=0.5)
    ax[2].plot(t, rP, c=cPV, alpha=0.5)
    ax[5].plot(t, p, c=cpi, alpha=1)
    for i, label in enumerate(['PC', 'dend.', 'SOM', 'NDNF', 'PV']):
        ax[i].set(ylabel=label, ylim=[0, 3])
    ax[5].set(ylabel='p', ylim=[0, 1], xlabel='time (s)')

    # plot 2: monitoring SOM boutons
    fig2, ax2 = plt.subplots(1, 1, figsize=(2.2, 1.6), dpi=dpi, gridspec_kw={'left':0.22, 'right':0.9, 'bottom':0.23})
    boutons = np.array(other['boutons_SOM'])
    boutons_nonzero = boutons[:, np.mean(boutons, axis=0) > 0]
    cm = ax2.pcolormesh(boutons_nonzero.T, cmap='Blues', vmin=0) #, vmax=0.15)
    cb = plt.colorbar(cm, ticks=[0, 0.5])
    cb.set_label('act. (au)', rotation=0, labelpad=-30, y=1.15)
    # cb.ax.get_yaxis().labelpad = 15
    ax2.set(xlabel='time (ms)', ylabel='# bouton (SOM out)', yticks=[0, (len(boutons_nonzero.T)//50)*50],
            xticks=[0, 1000, 2000], xticklabels=[0, 1, 2])

    # plot 3: quantification of monitoring SOM boutons
    fig3, ax3 = plt.subplots(1, 1, figsize=(2.5, 2), dpi=dpi, gridspec_kw={'left': 0.25, 'right':0.9, 'bottom':0.2})
    boutons_NDNFact = np.mean(boutons_nonzero[ts:te+200], axis=0)
    boutons_cntrl = np.mean(boutons_nonzero[0:ts], axis=0)
    plot_violin(ax3, 0, boutons_cntrl, color=cSOM)
    plot_violin(ax3, 1, boutons_NDNFact, color='#E9B86F')
    ax3.set(xlim=[-0.5, 1.5], xticks=[0, 1], xticklabels=['ctrl', 'NDNF act.'], ylim=[0, 0.6], yticks=[0, 0.5],
            ylabel='SOM bouton act.')
    
    if save:
        # fig.savefig('../results/figs/cosyne-collection/exp1-2_preinh-allneurons.png', dpi=300)
        fig2.savefig('../results/figs/Naumann23_draft1/exp1-2_preinh-boutons-all.pdf', dpi=300)
        # fig3.savefig('../results/figs/cosyne-collection/exp1-2_preinh-boutons-violin.png', dpi=300)
        [plt.close(ff) for ff in [fig, fig2, fig3]]


def exp_unused_amplifcation_ndnf_inhibition(dur=1500, dt=1, w_hetero=False, mean_pop=True, noise=0.0, pre_inh=True, save=False):
    """
    Experiment3b: Vary input to NDNF interneurons and pre inh strength, check how this affects NDNF inhibition to dendrite.

    Parameters
    - dur: duration of experiment (ms)
    - dt: integration time step (ms)
    - noise: level of white noise added to neural activity
    - flag_w_hetero: whether to add heterogeneity to weight matrices
    - save: if it's a string, name of the saved file, else if False nothing is saved
    """

    # extract number of timesteps
    nt = int(dur / dt)

    # get default parameters
    N_cells, w_mean, conn_prob, bg_inputs, taus = mb.get_default_params(flag_mean_pop=mean_pop)

    # instantiate and run model
    model = mb.NetworkModel(N_cells, w_mean, conn_prob, taus, bg_inputs, wED=1, flag_w_hetero=w_hetero,
                            flag_pre_inh=pre_inh)

    # array for varying NDNF input
    ndnf_input = np.arange(-1, 1, 0.05)
    betas = np.linspace(0, 0.5, 2, endpoint=True)

    # empty arrays for recording stuff
    # rS_inh_record = np.zeros((len(ndnf_input), len(betas)))
    rN_inh_record = np.zeros((len(ndnf_input), len(betas)))

    # set up figure
    dpi = 300 if save else DPI
    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=dpi, gridspec_kw={'left': 0.22, 'bottom': 0.25, 'top': 0.95,
                                                                           'right': 0.95}, sharex=True)
    cols = sns.color_palette(f"dark:{cpi}", n_colors=len(betas))

    for j, bb in enumerate(betas):

        print(f"beta: {bb:1.1f}")

        model.b = bb

        for i, I_activate in enumerate(ndnf_input):

            # create input (stimulation of NDNF)
            xFF = get_null_ff_input_arrays(nt, N_cells)
            xFF['N'][:, :] = I_activate

            t, rE, rD, rS, rN, rP, rV, p, cGABA, other = model.run(dur, xFF, dt=dt, init_noise=0, monitor_dend_inh=True,
                                                               noise=noise)
            # TODO: add init_noise?

            # save stuff
            # rS_inh_record[i, j] = np.mean(np.array(other['dend_inh_SOM'][-1]))
            # rN_inh_record[i, j] = np.mean(np.array(other['dend_inh_NDNF'][-1]))
            rN_inh_record[i, j] = np.mean(rN[-1])
        print(model.Xbg)


        ax.plot(ndnf_input, rN_inh_record[:, j], c=cols[j], ls='-', label=f"{bb:1.1f}", lw=lw)
    
    ax.set(xlabel=r'$\Delta$ NDNF input', xticks=[-1, 0, 1], xlim=[-1, 1], ylim=[-0.1, 2.5], yticks=[0, 1, 2], ylabel='NDNF act. (au)')
    ax.legend(['no pre. inh.', 'pre. inh.'], loc=(0.05, 0.7))

    # saving
    if save:
        fig.savefig('../results/figs/cosyne-collection/exp1-3c_amplification_NDNFinh', dpi=300)
        plt.close(fig)


def exp_unused_motifs_SOM_PC(dur=2000, dt=1, w_hetero=False, mean_pop=True, pre_inh=True, noise=0, save=False):
    """
    Experiment4a: Measure inhibition from SOM to PC with active and inactive NDNF (i.e. with and without pre inh).

    Parameters
    - dur: length of experiment
    - dt: time step
    - ...
    """
    # TODO: complete list of parameters

    # simulation paramters
    nt = int(dur / dt)
    t = np.arange(0, dur, dt)
    t0 = 1100
    amp = 3

    # create figure
    dpi = 300 if save else DPI
    fig, ax = plt.subplots(1, 1, figsize=(1.2, 1), dpi=dpi, gridspec_kw={'right': 0.95, 'bottom': 0.1, 'left': 0.1, 'top': 0.95})
    
    # get default parameters
    N_cells, w_mean, conn_prob, bg_inputs, taus = mb.get_default_params(flag_mean_pop=mean_pop)

    # labels
    labelz = ['SOM', 'NDNF']

    # stimulate SOM and NDNF, respectively

    # array of FF input
    xFF = get_null_ff_input_arrays(nt, N_cells)
    xFF['S'][:, :] = amp * np.tile(np.exp(-(t - t0) / 50) * np.heaviside(t - t0, 1), (N_cells['S'], 1)).T
    # xFF[cell][50:150, :] = 2

    # create model and run
    model = mb.NetworkModel(N_cells, w_mean, conn_prob, taus, bg_inputs, wED=1, flag_w_hetero=w_hetero,
                            flag_pre_inh=pre_inh, gamma=1)
    
    t, rE, rD, rS, rN, rP, rV, p, cGABA, other = model.run(dur, xFF, dt=dt, init_noise=0, noise=noise, monitor_currents=True,
                                                            rP0=0, rN0=0, rE0=0, rS0=0, rV0=0,rD0=1,
                                                            calc_bg_input=True)
    print('bg inputs:', model.Xbg)

    # now run again but NDNFs are more active
    xFF['N'] += 1.5
    t, rE2, rD2, rS2, rN2, rP2, rV2, p2, cGABA2, other2 = model.run(dur, xFF, dt=dt, init_noise=0, noise=noise, monitor_currents=True,
                                                                    rP0=0, rN0=0, rE0=0, rS0=0, rV0=0,rD0=1,
                                                                    calc_bg_input=False)
    
    # note: dendritic activity is set to 1 so that the inhibition by SOM soma, all other neurons inactive by default

    curr_rE = np.mean(other['curr_rE'], axis=1)
    curr_rE2 = np.mean(other2['curr_rE'], axis=1)
    ts = 1000
    tplot = (t[ts+1:]-ts)/1000
    ax.plot(tplot, (curr_rE-np.mean(curr_rE[1000:1100]))[ts:], c=cPC, lw=1)
    ax.plot(tplot, (curr_rE2-np.mean(curr_rE2[1000:1100]))[ts:], c='#D29FA3', lw=1)
    ax.axis('off')

    # add scale bars    
    ax.add_patch(plt.Rectangle((0.8, -0.1), 0.2, 0.02, facecolor='k', edgecolor='k', lw=0))
    ax.text(0.9, -0.15, '200ms', ha='center', va='top', color='k', fontsize=8)  
    ax.add_patch(plt.Rectangle((0.05, -0.6), 0.02, 0.2, facecolor='k', edgecolor='k', lw=0))
    ax.text(0.035, -0.5, '0.2', ha='right', va='center', color='k', fontsize=8)  

    # add SOM activation patch
    ax.add_patch(plt.Rectangle((0.1, 0.05), 0.1, 0.05, facecolor=cSOM, lw=0, alpha=0.5))

    # plot activity of all cell types
    fig2, ax2 = plt.subplots(6, 1, dpi=150, figsize=(4, 4), sharex=True)
    ax2[0].plot(t/1000, np.mean(rE2, axis=1), c=cPC, lw=1)
    ax2[1].plot(t/1000, np.mean(rD2, axis=1), c='k', lw=1)
    ax2[2].plot(t/1000, np.mean(rS2, axis=1), c=cSOM, lw=1)
    ax2[3].plot(t/1000, np.mean(rN2, axis=1), c=cNDNF, lw=1)
    ax2[4].plot(t/1000, np.mean(rP2, axis=1), c=cPV, lw=1)
    ax2[5].plot(t/1000, p2, c=cpi, lw=1)

    if save:
        fig.savefig('../results/figs/Naumann23_draft1/exp1-4_motif_SOM-PC.pdf', dpi=300)
        plt.close(fig)
        plt.close(fig2)


def exp_unused_motifs_SOM_NDNF(dur=2000, dt=1, w_hetero=False, mean_pop=True, pre_inh=True, noise=0, save=False):
    """
    Experiment4: Experiment4a: Measure inhibition from SOM to NDNF with active and inactive NDNF (i.e. with and without pre inh).

    Parameters
    - dur: length of experiment
    - dt: time step
    - ...
    """
    # TODO: complete list of parameters

    # simulation paramters
    nt = int(dur / dt)
    t = np.arange(0, dur, dt)
    t0 = 1100
    amp = 3

    # create figure
    dpi = 300 if save else DPI
    fig, ax = plt.subplots(1, 1, figsize=(1.2, 1), dpi=dpi, gridspec_kw={'right': 0.95, 'bottom': 0.1, 'left': 0.1, 'top': 0.95})
    
    # get default parameters
    N_cells, w_mean, conn_prob, bg_inputs, taus = mb.get_default_params(flag_mean_pop=mean_pop)

    # labels
    labelz = ['SOM', 'NDNF']

    # stimulate SOM and NDNF, respectively

    # array of FF input
    xFF = get_null_ff_input_arrays(nt, N_cells)
    xFF['S'][:, :] = amp * np.tile(np.exp(-(t - t0) / 50) * np.heaviside(t - t0, 1), (N_cells['S'], 1)).T
    # xFF[cell][50:150, :] = 2

    # create model and run
    model = mb.NetworkModel(N_cells, w_mean, conn_prob, taus, bg_inputs, wED=1, flag_w_hetero=w_hetero,
                            flag_pre_inh=pre_inh, gamma=1)
    
    t, rE, rD, rS, rN, rP, rV, p, cGABA, other = model.run(dur, xFF, dt=dt, init_noise=0, noise=noise, monitor_currents=True,
                                                            rP0=0, rN0=0, rE0=0, rS0=0, rV0=0,rD0=1,
                                                            calc_bg_input=True)
    print('bg inputs:', model.Xbg)

    # now run again but NDNFs are more active
    xFF['N'] += 1.5
    t, rE2, rD2, rS2, rN2, rP2, rV2, p2, cGABA2, other2 = model.run(dur, xFF, dt=dt, init_noise=0, noise=noise, monitor_currents=True,
                                                                    rP0=0, rN0=0, rE0=0, rS0=0, rV0=0,rD0=1,
                                                                    calc_bg_input=False)
    
    # note: dendritic activity is set to 1 so that the inhibition by SOM soma, all other neurons inactive by default

    curr_rN = np.mean(other['curr_rN'], axis=1)
    curr_rN2 = np.mean(other2['curr_rN'], axis=1)
    ts = 1000
    tplot = (t[ts+1:]-ts)/1000
    ax.plot(tplot, (curr_rN-np.mean(curr_rN[1000:1100]))[ts:], c=cNDNF, lw=1)
    ax.plot(tplot, (curr_rN2-np.mean(curr_rN2[1000:1100]))[ts:], c='#EAC8B7', lw=1)
    ax.axis('off')

    # add scale bars    
    ax.add_patch(plt.Rectangle((0.8, -0.15), 0.2, 0.03, facecolor='k', edgecolor='k', lw=0))
    ax.text(0.9, -0.22, '200ms', ha='center', va='top', color='k', fontsize=8)  
    ax.add_patch(plt.Rectangle((0.05, -0.9), 0.02, 0.3, facecolor='k', edgecolor='k', lw=0))
    ax.text(0.035, -0.75, '0.3', ha='right', va='center', color='k', fontsize=8)  

    # add SOM activation patch
    ax.add_patch(plt.Rectangle((0.1, 0.09), 0.1, 0.1, facecolor=cSOM, lw=0, alpha=0.5))

    # plot activity of all cell types
    fig2, ax2 = plt.subplots(6, 1, dpi=150, figsize=(4, 4), sharex=True)
    ax2[0].plot(t/1000, np.mean(rE2, axis=1), c=cPC, lw=1)
    ax2[1].plot(t/1000, np.mean(rD2, axis=1), c='k', lw=1)
    ax2[2].plot(t/1000, np.mean(rS2, axis=1), c=cSOM, lw=1)
    ax2[3].plot(t/1000, np.mean(rN2, axis=1), c=cNDNF, lw=1)
    ax2[4].plot(t/1000, np.mean(rP2, axis=1), c=cPV, lw=1)
    ax2[4].plot(t/1000, np.mean(rV2, axis=1), c=cVIP, lw=1)
    ax2[5].plot(t/1000, p2, c=cpi, lw=1)

    if save:
        fig.savefig('../results/figs/Naumann23_draft1/exp1-4_motif_SOM-NDNF.pdf', dpi=300)
        plt.close(fig)
        plt.close(fig2)


def exp_unused_signaltransmission_pathways_NDNF(noise=0.0, w_hetero=False, mean_pop=True, pre_inh=True, reduced=False, save=False):
    """
    Experiment 3a: Provide sine stimulus to NDNF INs, then vary the NDNF-PV inhibition and check what's represented in PC rate.
    Depending on the balance if NDNF inhibition and disinhibition via PV, the sine is represented pos or neg in the PCs.
    To account for delays introduced by slow timescale of NDNF and GABA spillover, the sine is shifted to assess its contribution.
    - same arguements as functions above
    """

    # define parameter dictionaries
    N_cells, w_mean, conn_prob, bg_inputs, taus = mb.get_default_params(flag_mean_pop=mean_pop)

    if reduced:  # remove 
        w_mean['EP'], w_mean['PE'], w_mean['SE'] = 0, 0, 0

    # simulation paramters
    dur = 5000
    dt = 1
    nt = int(dur/dt)

    # generate inputs
    tt = np.arange(0, dur, dt)/1000
    sine1 = np.sin(2*np.pi*tt*1)  # slow sine (1Hz)
    sine2 = np.sin(2*np.pi*tt*4)  # fast sine (4Hz)
    amp_sine = 1

    xFF = get_null_ff_input_arrays(nt, N_cells)
    xFF['N'][:,:] = amp_sine*np.tile(sine1, [N_cells['N'], 1]).T

    # list of weights to test
    wPN_list = np.arange(0, 1.3, 0.2)

    signal_corr = np.zeros((len(wPN_list), 5)) # empty array for storage
    shifts = np.array([1, 50, 100, 150, 200])

    fig, ax = plt.subplots(1, 2, figsize=(4.5, 2), dpi=150, gridspec_kw={'left':0.1, 'bottom': 0.2, 'right': 0.95,
                                                                        'wspace': 0.5})
    cols = sns.color_palette("dark:salmon", n_colors=len(wPN_list))
    cols_shift = sns.color_palette("dark:gold", n_colors=len(shifts))

    for i, wPN in enumerate(wPN_list):

        w_mean['PN'] = wPN
        w_mean['DN'] = 0.7

        # taus['N'] = 10
        # w_mean['PE'], w_mean['SE'], w_mean['PP'] = 0, 0, 0

        # instantiate model
        model = mb.NetworkModel(N_cells, w_mean, conn_prob, taus, bg_inputs, wED=1, flag_w_hetero=w_hetero,
                            flag_pre_inh=pre_inh)#, tauG=10)

        # run model
        t, rE, rD, rS, rN, rP, rV, p, cGABA, other = model.run(dur, xFF, dt=dt, p0=0.5, init_noise=0, calc_bg_input=True,
                                                               monitor_dend_inh=True, noise=noise)
        
        for j, shift in enumerate(shifts):
            betas = quantify_signals([sine1[1000:-shift]], np.mean(rE, axis=1)[1000+shift:])
            signal_corr[i, j] = betas[0]

        ax[0].plot(t, rE, c=cols[i])
    
    # ax[0].plot(t+shift, sine1/4+1, c='goldenrod', lw=1)
    [ax[1].plot(wPN_list*w_mean['EP']/w_mean['DN'], signal_corr[:,j], c=cols_shift[j]) for j in range(len(shifts))]
    ax[1].set(xlabel='NDNF disinh/inh (wPN*wEP/wDN)', ylabel='signal strength')
    ax[0].set(xlabel='time (ms)', ylabel='PC rate')


def exp_unused_signaltransmission_pathways_SOM(noise=0.0, w_hetero=False, mean_pop=True, pre_inh=True, reduced=False, save=False):
    """
    Experiment 3a: Provide sine stimulus to SOM INs, then vary the NDNF-PV inhibition and check what's represented in PC rate.
    Depending on the balance if SOM inhibition and disinhibition via PV, the sine is represented pos or neg in the PCs.
    To account for delays introduced by slow integration of SOMs, the sine is shifted to assess its contribution.
    - same arguements as functions above
    """

    # define parameter dictionaries
    N_cells, w_mean, conn_prob, bg_inputs, taus = mb.get_default_params(flag_mean_pop=mean_pop)

    if reduced:  # remove 
        w_mean['EP'], w_mean['PE'], w_mean['SE'] = 0, 0, 0

    # w_mean['PE'], w_mean['SE'] = 0, 0

    # simulation paramters
    dur = 2000
    dt = 1
    nt = int(dur/dt)

    # generate inputs
    tt = np.arange(0, dur, dt)/1000
    sine1 = np.sin(2*np.pi*tt*4)  # slow sine (1Hz)
    sine2 = np.sin(2*np.pi*tt*4)  # fast sine (4Hz)
    amp_sine = 1

    xFF = get_null_ff_input_arrays(nt, N_cells)
    xFF['S'][:,:] = amp_sine*np.tile(sine1, [N_cells['S'], 1]).T

    # list of weights to test
    wPS_list = np.arange(0, 1.7, 0.2)

    signal_corr = np.zeros((len(wPS_list), 5)) # empty array for storage
    shifts = np.array([1, 20, 50]) # 100, 150, 200])

    fig, ax = plt.subplots(1, 2, figsize=(4.5, 2), dpi=150, gridspec_kw={'left':0.1, 'bottom': 0.2, 'right': 0.95,
                                                                        'wspace': 0.5})
    cols = sns.color_palette("dark:mediumturquoise", n_colors=len(wPS_list))
    cols_shift = sns.color_palette("dark:gold", n_colors=len(shifts))

    for i, wPS in enumerate(wPS_list):

        w_mean['PS'] = wPS
        # w_mean['DN'] = 0.7

        # taus['N'] = 10
        # w_mean['PE'], w_mean['SE'], w_mean['PP'] = 0, 0, 0

        # instantiate model
        model = mb.NetworkModel(N_cells, w_mean, conn_prob, taus, bg_inputs, wED=1, flag_w_hetero=w_hetero,
                            flag_pre_inh=pre_inh)#, tauG=10)

        # run model
        t, rE, rD, rS, rN, rP, rV, p, cGABA, other = model.run(dur, xFF, dt=dt, p0=0.5, init_noise=0, calc_bg_input=True,
                                                               monitor_dend_inh=True, noise=noise)
        
        for j, shift in enumerate(shifts):
            betas = quantify_signals([sine1[1000:-shift]], np.mean(rE, axis=1)[1000+shift:])
            signal_corr[i, j] = betas[0]

        ax[0].plot(t, rE, c=cols[i])
    
    [ax[1].plot(wPS_list*w_mean['EP']/w_mean['DS'], signal_corr[:,j], c=cols_shift[j]) for j in range(len(shifts))]
    ax[1].set(xlabel='SOM disinh/inh (wPS*wEP/wDS)', ylabel='signal strength')
    ax[0].set(xlabel='time (ms)', ylabel='PC rate')


def quantify_signals(signals, rate, bias=False):
    """ Quantify the signal using the regressors beta of a linear regression of the signals onto the rate."""

    X = np.array(signals).reshape((len(signals), -1)).T
    y = rate
    if bias:
        X = np.concatenate((np.ones((len(rate), 1)), X), axis=1)
        return (np.linalg.inv(X.T@X)@(X.T@y))[1:]
    else:
        return np.linalg.inv(X.T@X)@(X.T@y)


def exp_unused_frequency_preference(noise=0.0, w_hetero=False, mean_pop=True, pre_inh=True, save=False, plot_supp=False):
    """
    Experiment 1: Test frequency transmission of NDNF and SOM to PC.
    """

    freqs = np.array([0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15])
    betas = np.array([0.5])

    # define parameter dictionaries
    N_cells, w_mean, conn_prob, bg_inputs, taus = mb.get_default_params(flag_mean_pop=mean_pop)

    # w_mean['DS'] = 1
    # w_mean['DN'] = 1

    # instantiate model
    model = mb.NetworkModel(N_cells, w_mean, conn_prob, taus, bg_inputs, wED=1, flag_w_hetero=w_hetero, w_std_rel=0.01,
                            flag_pre_inh=pre_inh)
    
    # simulation paramters
    dur = 10000
    dt = 1
    nt = int(dur/dt)

    # empty arrays for storage of PC signal amplitudes
    amplitudes_n = np.zeros(len(freqs))
    amplitudes_s = np.zeros(len(freqs))

    # amplitude of sine input
    amp = 1.5

    # set up figure
    dpi = 300 if save else DPI
    fig, ax = plt.subplots(1, 1, dpi=dpi, figsize=(1.8, 1.15), gridspec_kw={'left': 0.25, 'bottom': 0.33, 'hspace':0.2, 'right':0.95})
    cols_s = sns.color_palette(f"light:{cSOM}", n_colors=len(betas)+1)[1:]
    cols_n = sns.color_palette(f"light:{cNDNF}", n_colors=len(betas)+1)[1:]

    # loop over frequencies and presynpatic inhibition strengths (TODO: remove loop over betas)
    for j, bb in enumerate(betas):
        print(f"pre inh beta = {bb:1.1f}")

        for i, ff in enumerate(freqs):

            model.b = bb

            # make input
            xff_null = np.zeros(nt)  
            xff_sine = amp*(make_sine(nt, ff)-0.5)

            # simulate with sine input to SOM
            xFF = get_null_ff_input_arrays(nt, N_cells)
            xFF['S'] = np.tile(xff_sine, [N_cells['S'], 1]).T
            xFF['N'] = np.tile(xff_null, [N_cells['N'], 1]).T
            t, rE1, rD1, rS1, rN1, rP1, rV1, p1, cGABA1, other1 = model.run(dur, xFF, dt=dt, calc_bg_input=True, init_noise=0, noise=noise)

            # simulate with sine input to NDNF
            xFF = get_null_ff_input_arrays(nt, N_cells)
            xFF['S'] = np.tile(xff_null, [N_cells['S'], 1]).T
            xFF['N'] = np.tile(xff_sine, [N_cells['N'], 1]).T
            t, rE2, rD2, rS2, rN2, rP2, rV2, p2, cGABA2, other2 = model.run(dur, xFF, dt=dt, calc_bg_input=True, init_noise=0, noise=noise)
 
            # calculate signal amplitudes
            amplitudes_s[i] = signal_amplitude(np.mean(rE1, axis=1), tstart=int(1000*dt))
            amplitudes_n[i] = signal_amplitude(np.mean(rE2, axis=1), tstart=int(1000*dt))

            if i in [0, 3, 5, 7]:
                fig1, ax1 = plt.subplots(2, 2, sharex=True, sharey=True)
                ax1[0, 0].plot(t, rE1, cPC)
                ax1[0, 0].plot(t, rD1, 'k')
                ax1[1, 0].plot(t, rS1, cSOM)
                ax1[1, 0].plot(t, rP1, cPV)
                ax1[1, 0].plot(t, rN1, cNDNF)
                ax1[0, 1].plot(t, rE2, cPC)
                ax1[0, 1].plot(t, rD2, 'k')
                ax1[1, 1].plot(t, rS2, cSOM)
                ax1[1, 1].plot(t, rP2, cPV)
                ax1[1, 1].plot(t, rN2, cNDNF)
                ax1[0, 0].set(xlim=[0, 2000], ylim=[0, 2])
                # ax1.plot(t, cGABA2, 'C2')

        ax.plot(freqs, amplitudes_s/amplitudes_s.max(), '.-', c=cols_s[j], ms=4*lw, label=f"b={bb:1.1f}")
        ax.plot(freqs, amplitudes_n/amplitudes_n.max(), '.-', c=cols_n[j], ms=4*lw, label=f"b={bb:1.1f}")


    ax.set(ylim=[0, 1.1], ylabel='signal (norm)', xlabel='stimulus freq. (Hz)', yticks=[0, 1], xticks=[0, 5, 10, 15])

    if save:
        fig.savefig('../results/figs/Naumann23_draft1/exp3-1_freq-pref.pdf', dpi=300)
        plt.close(fig)


def exp_unused_transient_effects(mean_pop=True, w_hetero=False, pre_inh=True, noise=0.0, reduced=False, save=False):
    """
    Experiment4: test if circuit can have transient behaviour. When NDNFs (+others) receive some input,
                 there is a transient window of dynamics before NDNFs change the ciruit dynamics.
    - give some longer input to NDNFs
    - test circuit response/dynamics for some input stimulus early and late during NDNF input
    """

    # define parameter dictionaries
    N_cells, w_mean, conn_prob, bg_inputs, taus = mb.get_default_params(flag_mean_pop=mean_pop)

    if reduced: # remove recurrence from model for reduced variant 
        w_mean['EP'], w_mean['PE'], w_mean['SE'] = 0, 0, 0

    # increase NDNF-dendrite inhibition
    w_mean['DS'] = 0.8 # so SOM-dend inhibition dominates over SOM-NDNF-dend disinhibition

    # instantiate model
    model = mb.NetworkModel(N_cells, w_mean, conn_prob, taus, bg_inputs, wED=1, flag_w_hetero=w_hetero,
                            flag_pre_inh=pre_inh)

    # simulation paramters
    dur = 5000
    dt = 1
    nt = int(dur/dt)

    # stimulation
    xFF = get_null_ff_input_arrays(nt, N_cells)
    xFF['N'][1000:4000] = 0.6  # long constant stimulus to NDNF
    xsine = np.sin(np.arange(0, 0.5, 0.001)*5*np.pi*2)*1  # 500ms long sine wave
    ts1, te1, ts2, te2 = 1000, 1500, 3500, 4000
    xFF['S'][ts1:te1] = np.tile(xsine, [N_cells['S'], 1]).T  # early sine input to SOM
    xFF['S'][ts2:te2] = np.tile(xsine, [N_cells['S'], 1]).T  # late sine input to SOM

    # simulate
    t, rE, rD, rS, rN, rP, rV, p, cGABA, other = model.run(dur, xFF, dt=dt, p0=0.5, noise=noise)

    # print(np.corrcoef(np.mean(rE[ts1:te1], axis=1), xsine)[0, 1])
    # print(np.corrcoef(np.mean(rE[ts2:te2], axis=1), xsine)[0, 1])

    sh = 35 # shift signals relative to rate to account for delay from integration

    # plot
    dpi = 300 if save else 120
    fig, ax = plt.subplots(5, 1, dpi=dpi, gridspec_kw=dict(hspace=0.4, wspace=0.5, right=0.95, top=0.95, bottom=0.13, left=0.2),
                                 figsize=(6, 6.22))
    # cols = sns.color_palette("flare", n_colors=len(stim_durs))
    ax[4].plot(t/1000, rE, c=cPC, lw=2)
    ax[2].plot(t/1000, rD, c='k', lw=2)
    ax[1].plot(t/1000, rS, c=cSOM, lw=2)
    ax[0].plot(t/1000, rN, c=cNDNF, lw=2)
    ax[3].plot(t/1000, rP, c=cPV, lw=2)
    # ax[5].plot(t/1000, p, c=cpi)
    # ax[0].hlines(2.4, 1, 4, color=cNDNF, alpha=0.2, lw=5)
    # ax[6].plot(t, p, c='C2', alpha=1)
    for i, label in enumerate(['NDNF', 'SOM', 'dend.', 'PV']):
        ax[i].set(ylabel=label, ylim=[0, 2.5], yticks=[0, 2], xticks=[])
        ax[i].spines['bottom'].set_visible(False)
    ax[-1].set(xlabel='time (s)', ylim=[0, 2.5], yticks=[0, 2], ylabel='PC', xticks=[0, 2, 4])

    ax[4].plot(np.arange(ts1+sh,te1+sh)/1000, xsine/3+2.1, c=cSOM, lw=2, ls='-')
    ax[4].plot(np.arange(ts2+sh,te2+sh)/1000, xsine/3+2.1, c=cSOM, lw=2, ls='-')
    # ax[1].plot(np.arange(ts1-600,te1-600)/1000, xsine/5+2, c=cSOM, lw=3, ls='-', alpha=0.2)
    # ax[1].plot(np.arange(ts2,te2)/1000, xsine/5+2.3, c=cSOM, lw=1, ls='-', alpha=0.2)

    if save:
        fig.savefig('../results/figs/cosyne-collection/exp3-4_transient-effect.png', dpi=300)
        plt.close(fig)


def signal_amplitude(x, tstart=500):
    """
    Calculate the amplitude of a signal x by computing the mean difference between the peak and valley.

    Parameters:
    - x: signal
    - tstart: time point to start analysis

    Returns:
    - mean amplitude
    """

    x_use = x[tstart:]

    ind_peak = detect_peaks(x_use, mpd=5)
    ind_valley = detect_peaks(x_use, mpd=5, valley=True)
    n_vp = np.min([len(ind_peak), len(ind_valley)])

    ampls = x_use[ind_peak[:n_vp]]-x_use[ind_valley[:n_vp]]
    return np.mean(ampls)


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


##########################################################
# Older experiments for initial inspection of the model  #
##########################################################

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


def fig1_weights_role(I_activate=1, dur=1000, ts=400, te=600, dt=1, save=False, noise=0.0):
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
    dpi = 400 if save else 200
    fig, ax = plt.subplots(1, 2, figsize=(3.4, 1.6), dpi=400, sharex=False, sharey='row',
                           gridspec_kw={'right': 0.98, 'left':0.15, 'bottom':0.25})

    # use paramterisation from disinhibition-dominated regime (overwrite w_mean)
    N_cells, w_mean, conn_prob, bg_inputs, taus = get_default_params()
    # w_mean_df = dict(NS=0.7, DS=0.5, DN=1.5, SE=0.5, NN=0.2, PS=0.8, PN=0.5, PP=0.1, PE=1, EP=0.5, DE=0, VS=0, SV=0)
    w_mean_df = dict(NS=0.6, DS=0.3, DN=1.5, SE=0.5, NN=0.2, PS=0.8, PN=1, PP=0.1, PE=1, EP=0.5, DE=0, VS=0, SV=0)
    # w_mean_df = dict(NS=1, DS=0.5, DN=1.5, SE=0.5, NN=0, PS=0.5, PN=0.5, PP=0, PE=0.7, EP=1, DE=0, VS=0, SV=0)
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
                             flag_w_hetero=True, flag_pre_inh=True, flag_with_VIP=False, flag_with_NDNF=True)
        t, rE, rD, rS, rN, rP, rV, p, other = model.run(dur, xFF, dt=dt, init_noise=0, noise=noise)
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

    if save:
        plt.savefig('../results/figs/tmp/'+save, dpi=400)


def ex_perturb_circuit(save=False, I_activate=1, dur=1500, ts=600, te=800, dt=1, noise=0.0):
    """
    Experiment: stimulate SOM for the default parameters and the disinhibition-dominated regime
    """

    # number of timesteps
    nt = int(dur/dt)

    # get default parameters and weights for disinhibition-dominated regime
    N_cells, w_mean, conn_prob, bg_inputs, taus = get_default_params()
    w_mean['DS'] = 1.2
    w_mean_disinh = dict(NS=0.6, DS=0.3, DN=1.5, SE=0.5, NN=0.2, PS=0.8, PN=1, PP=0.1, PE=1, EP=0.5, DE=0, VS=0, SV=0)

    # create stimulus array
    xFF = get_null_ff_input_arrays(nt, N_cells)
    xFF['S'][ts:te, :] = I_activate

    # set up models
    model = NetworkModel(N_cells, w_mean, conn_prob, taus, bg_inputs, wED=1, flag_w_hetero=True, flag_pre_inh=True)
    model_disinh = NetworkModel(N_cells, w_mean_disinh, conn_prob, taus, bg_inputs, wED=1, flag_w_hetero=True,
                                flag_pre_inh=True)

    # run and plot for default and disinhibition-dominated model
    dpi = 400 if save else 200
    fig, ax = plt.subplots(4, 2, dpi=dpi, figsize=(2, 1.7), sharex=True, gridspec_kw={'top':0.95, 'bottom': 0.05,
                                                                                      'left': 0.05})
    for i, mod in enumerate([model, model_disinh]):
        t, rE, rD, rS, rN, rP, rV, p, other = mod.run(dur, xFF, dt=dt, init_noise=0, noise=noise)

        ax[0, i].plot(t, rN, c=cNDNF, alpha=0.5, lw=0.8)
        ax[1, i].plot(t, rS, c=cSOM, alpha=0.5, lw=0.8)
        ax[2, i].plot(t, rP, c=cPV, alpha=0.5, lw=0.8)
        ax[3, i].plot(t, rE, c=cPC, alpha=0.5, lw=0.8)

        for axx in ax[:, i]:
            axx.set(ylim=[-0.1, 2.3])
            axx.axis('off')

    if save:
        plt.savefig('../results/figs/tmp/'+save, dpi=400)


def ex_bouton_imaging(dur=1000, ts=300, te=400, dt=1, stim_NDNF=2, noise=0.0, flag_w_hetero=False):
    """
    Experiment: Image SOM bouton in response to stimulation of NDNF interneurons.
    - dur: duration of experiment (ms)
    - ts: start of NDNF stimulation
    - te: end of NDNF stimulation
    - dt: integration time step (ms)
    - stim_NDNF: strength of NDNF stimulation
    - noise: level of white noise added to neural activity
    - flag_w_hetero: whether to add heterogeneity to weight matrices
    """

    # define parameter dictionaries
    N_cells, w_mean, conn_prob, bg_inputs, taus = get_default_params()

    # instantiate model
    model = NetworkModel(N_cells, w_mean, conn_prob, taus, bg_inputs, wED=1, flag_w_hetero=flag_w_hetero,
                         flag_pre_inh=True)

    # simulation paramters
    nt = int(dur/dt)

    # generate inputs
    xFF = get_null_ff_input_arrays(nt, N_cells)
    xFF['N'][ts:te] = stim_NDNF

    # run model
    t, rE, rD, rS, rN, rP, rV, p, cGABA, other = model.run(dur, xFF, init_noise=0.1, noise=noise, dt=dt, monitor_boutons=True,
                                                    monitor_currents=True, calc_bg_input=True)

    # plotting
    # --------
    # 3 different plots here: an overview plot, bouton imaging + quantification and only bouton imaging
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

    fig2, ax2 = plt.subplots(1, 1, figsize=(3, 2), dpi=300, gridspec_kw={'left':0.3, 'right':0.9, 'bottom':0.35})
    boutons = np.array(other['boutons_SOM'])
    boutons_nonzero = boutons[:, np.mean(boutons, axis=0) > 0]
    cm = ax2.pcolormesh(boutons_nonzero.T, cmap='Blues', vmin=0, vmax=0.15)
    plt.colorbar(cm, ticks=[0, 0.1])
    ax2.set(xlabel='time (ms)', ylabel='# bouton', yticks=[0, 400], xticks=[0, 1000])

    boutons_NDNFact = np.mean(boutons_nonzero[ts:te], axis=0)
    boutons_cntrl = np.mean(boutons_nonzero[0:ts], axis=0)

    fig3, ax3 = plt.subplots(1, 1, figsize=(2, 1.5), dpi=300, gridspec_kw={'left': 0.25, 'right':0.9, 'bottom':0.15})
    plot_violin(ax3, 0, boutons_cntrl, color=cSOM)
    plot_violin(ax3, 1, boutons_NDNFact, color='#E9B86F')

    # vl2 = ax3.violinplot(boutons_NDNFact, positions=[1])
    # for pc in vl1['bodies']:
    #     pc.set_facecolor(cSOM)
    #     pc.set_edgecolor(cSOM)
    # for pc in vl2['bodies']:
    #     pc.set_facecolor('#E9B86F')
    #     pc.set_edgecolor('#E9B86F')
    ax3.set(xlim=[-0.5, 1.5], xticks=[0, 1], xticklabels=['ctrl', 'NDNF act.'], ylim=[0, 0.12], yticks=[0, 0.1],
            ylabel='SOM bouton act.')


def ex_layer_specific_inhibition(dur=1000, dt=1, noise=0.0, flag_w_hetero=True, save=False):
    """
    Experiment: Vary input to NDNF interneurons, monitor NDNF- and SOM-mediated dendritic inhibition and their activity.
    - dur: duration of experiment (ms)
    - dt: integration time step (ms)
    - noise: level of white noise added to neural activity
    - flag_w_hetero: whether to add heterogeneity to weight matrices
    - save: if it's a string, name of the saved file, else if False nothing is saved
    """

    # extract number of timesteps
    nt = int(dur / dt)

    # get default parameters
    N_cells, w_mean, conn_prob, bg_inputs, taus = get_default_params()

    # array for varying NDNF input
    ndnf_input = np.arange(-1, 1, 0.05)

    # empty arrays for recording stuff
    rS_inh_record = []
    rN_inh_record = []
    rS_record = []
    rN_record = []

    for i, I_activate in enumerate(ndnf_input):

        # create input (stimulation of NDNF)
        xFF = get_null_ff_input_arrays(nt, N_cells)
        xFF['N'][:, :] = I_activate

        # instantiate and run model
        model = NetworkModel(N_cells, w_mean, conn_prob, taus, bg_inputs, wED=1, flag_w_hetero=flag_w_hetero,
                             flag_pre_inh=True)
        t, rE, rD, rS, rN, rP, rV, p, cGABA, other = model.run(dur, xFF, dt=dt, init_noise=0, monitor_dend_inh=True,
                                                        noise=noise)

        # save stuff
        rS_record.append(rS[-1])
        rN_record.append(rN[-1])
        rS_inh_record.append(np.mean(np.array(other['dend_inh_SOM'][-1])))
        rN_inh_record.append(np.mean(np.array(other['dend_inh_NDNF'][-1])))

    # plotting
    dpi = 400 if save else 200
    fig, ax = plt.subplots(2, 1, figsize=(2.1, 2.8), dpi=dpi, gridspec_kw={'left': 0.25, 'bottom': 0.15, 'top': 0.95,
                                                                           'right': 0.95,
                                                                           'height_ratios': [1, 1]}, sharex=True)
    ax[0].plot(ndnf_input, rS_inh_record, c=cSOM, ls='--')
    ax[0].plot(ndnf_input, rN_inh_record, c=cNDNF, ls='--')
    ax[1].plot(ndnf_input, np.mean(np.array(rS_record), axis=1), c=cSOM)
    ax[1].plot(ndnf_input, np.mean(np.array(rN_record), axis=1), c=cNDNF)

    # labels etc
    ax[0].set(ylabel='dend. inhibition (au)', ylim=[-0.05, 1], yticks=[0, 1])
    ax[1].set(xlabel='input to NDNF (au)', ylabel='neural activity (au)', xlim=[-1, 1], ylim=[-0.1, 2.5],
              yticks=[0, 1, 2])

    # saving
    if save:
        plt.savefig('../results/figs/tmp/'+save, dpi=400)


def ex_switch_activity(noise=0.0, flag_w_hetero=True, save=False):
    """
    Experiment: Switch between NDNDF and SOM-dominated dendritic inhibition. Network is in bistable mututal inhibition
                regime. Activate and inactive NDNF interneurons to create switching.
    - noise: level of white noise added to neural activity
    - flag_w_hetero: whether to add heterogeneity to weight matrices
    - save: if it's a string, name of the saved file, else if False nothing is saved
    """

    # define parameter dictionaries
    N_cells, w_mean, conn_prob, bg_inputs, taus = get_default_params(flag_mean_pop=False)

    # increase SOM to NDNF inhibition to get bistable regime
    w_mean['NS'] = 1.4

    # instantiate model
    model = NetworkModel(N_cells, w_mean, conn_prob, taus, bg_inputs, wED=0.7, flag_w_hetero=flag_w_hetero,
                         flag_pre_inh=True)

    # simulation paramters
    dur = 3000
    dt = 1
    nt = int(dur/dt)

    # generate inputs
    t_act_s, t_act_e = 500, 1000
    t_inact_s, t_inact_e = 2000, 2500
    xFF = get_null_ff_input_arrays(nt, N_cells)
    xFF['N'][t_act_s:t_act_e] = 1.5
    xFF['N'][t_inact_s:t_inact_e] = -1.5

    # run model
    t, rE, rD, rS, rN, rP, rV, p, cGABA, other = model.run(dur, xFF, dt=dt, p0=0.5, init_noise=0, calc_bg_input=True,
                                                    monitor_dend_inh=True, noise=noise)

    # plotting
    dpi = 400 if save else 200
    fig, ax = plt.subplots(3, 1, figsize=(2.1, 2.8), dpi=dpi, sharex=True,
                           gridspec_kw={'left': 0.25, 'bottom': 0.15, 'top': 0.95, 'right': 0.95,
                                        'height_ratios': [1, 1, 0.5]})
    ax[1].plot(t/1000, rN, c=cNDNF, alpha=0.5)
    ax[1].plot(t/1000, rS, c=cSOM, alpha=0.5)
    ax[0].plot(t/1000, np.mean(np.array(other['dend_inh_NDNF']), axis=1), c=cNDNF, ls='--')
    ax[0].plot(t/1000, np.mean(np.array(other['dend_inh_SOM']), axis=1), c=cSOM, ls='--')
    ax[2].plot(t/1000, p, c=cpi, alpha=1)

    # labels etc
    ax[0].set(ylabel='dend. inh. (au)', ylim=[-0.1, 1.5], yticks=[0, 1])
    ax[1].set(ylabel='activity (au)', ylim=[-0.1, 3.5], yticks=[0, 2])
    ax[2].set(ylabel='release', ylim=[-0.05, 1.05], yticks=[0, 1], xlabel='time (s)', xticks=[0, 1, 2, 3])

    # saving (optional)
    if save:
        plt.savefig('../results/figs/tmp/'+save, dpi=400)


def plot_gfunc(b=0.5, save=False):
    """
    Plot presynaptic inhibition transfer function.
    - b: strength of presynaptic inhibition
    - save: if it's a string, name of the saved file, else if False nothing is saved

    """

    # get default parameters and instantiate model
    N_cells, w_mean, conn_prob, bg_inputs, taus = get_default_params()
    model = NetworkModel(N_cells, w_mean, conn_prob, taus, bg_inputs, flag_pre_inh=True, b=b)

    # get release probability for range of NDNF activity
    ndnf_act = np.arange(0, 2.5, 0.1)
    p = model.g_func(ndnf_act)

    # plotting
    fig, ax = plt.subplots(1, 1, figsize=(1.8, 1), dpi=400, gridspec_kw={'left': 0.25, 'bottom':0.4,'right':0.95})
    ax.plot(ndnf_act, p, c=cpi)
    ax.set(xlabel='NDNF activity (au)', ylabel='rel. factor', xlim=[0, 2.5], ylim=[-0.05, 1], xticks=[0, 1, 2],
           yticks=[0, 1])

    # saving (optional)
    if save:
        plt.savefig('../results/figs/tmp/'+save, dpi=400)


def plot_violin(ax, pos, data, color=None, showmeans=True):
    """
    Makes violin of data at x position pos in axis object ax.
    - data is an array of values
    - pos is a scalar
    - ax is an axis object

    Kwargs: color (if None default mpl is used) and whether to plot the mean
    """

    parts = ax.violinplot(data, positions=[pos], showmeans=showmeans, widths=0.6)
    if color:
        for pc in parts['bodies']:
            pc.set_color(color)
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
            vp = parts[partname]
            vp.set_edgecolor(color)
            vp.set_linewidth(1)



if __name__ in "__main__":

    # run different experiments; comment in or out to run only some of them

    # ex_activation_inactivation()
    # fig1_paired_recordings_invitro()
    # fig1_activation()
    # fig1_weights_role()
    # ex_bouton_imaging()

    # generating figures for cosyne abstract submission
    noise = 0.15
    # ex_layer_specific_inhibition(noise=noise, save=False) # 'fig2c.pdf', )
    ex_switch_activity(noise=noise, save=False) #'fig2d.pdf', )
    # plot_gfunc(save='fig2b.pdf')
    # ex_perturb_circuit(save=False)  #save='fig1b.pdf', noise=noise)
    # fig1_weights_role(save='fig1c.pdf', noise=noise)
    plt.show()