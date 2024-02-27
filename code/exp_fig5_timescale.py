"""
Experiments 3: Slow inhibition by NDNF interneurons preferentially transmits certain signals.
- 
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('pretty')
import matplotlib as mpl
lw = mpl.rcParams['lines.linewidth']
import seaborn as sns
import model_base as mb
from experiments import get_null_ff_input_arrays, get_model_colours, plot_violin
from scipy.fft import fft, fftfreq
from detect_peaks import detect_peaks
from exp_fig4_switching import quantify_signals

cPC, cPV, cSOM, cNDNF, cVIP, cpi = get_model_colours()

DPI = 150


def exp_fig5B_IPSC_timescale(dur=1000, dt=1, w_hetero=False, mean_pop=True, pre_inh=True, noise=0, save=False):
    """
    Paired "in vitro" recordings and plot. Stimulate NDNF and SOM, record the 'currents' to PC to show the slower IPSC
    elicted by NDNFs compared to SOMs. "in vitro" means all cells have 0 baseline activity. 

    Parameters
    - dur: length of experiment
    - dt: time step
    - w_hetero: whether to add heterogeneity to weight matrices
    - mean_pop: if true, simulate only one neuron (mean) per population
    - pre_inh: whether to include presynaptic inhibition
    - noise: level of white noise added to neural activity
    - save: if it's a string, name of the saved file, else if False nothing is saved
    """

    # simulation paramters
    nt = int(dur / dt)
    t = np.arange(0, dur, dt)
    t0 = 100
    amp = 3

    # create figure
    dpi = 300 if save else DPI
    
    fig, ax = plt.subplots(1, 1, figsize=(1.8, 1.15), dpi=dpi, sharex=True, sharey='row',
                           gridspec_kw={'right': 0.95, 'bottom': 0.33, 'left': 0.25})
    
    # not used: figure for plotting currents in SOM and NDNF
    # fig2, ax2 = plt.subplots(2, 2, figsize=(2, 1.5), dpi=dpi, sharex=True, sharey='row',
    #                     gridspec_kw={'right': 0.93, 'bottom': 0.24, 'left': 0.22, 'top': 0.95, 'wspace': 0.3})

    # get default parameters
    N_cells, w_mean, conn_prob, bg_inputs, taus = mb.get_default_params(flag_mean_pop=mean_pop)

    # labels
    labelz = ['SOM', 'NDNF']

    # stimulate SOM and NDNF, respectively
    for i, cell in enumerate(['S', 'N']):

        # array of FF input, instantaneous increase and exponential decay
        xFF = get_null_ff_input_arrays(nt, N_cells)
        xFF[cell][:, :] = amp * np.tile(np.exp(-(t - t0) / 50) * np.heaviside(t - t0, 1), (N_cells[cell], 1)).T

        # create model and run
        model = mb.NetworkModel(N_cells, w_mean, conn_prob, taus, bg_inputs, wED=1, flag_w_hetero=w_hetero,
                                flag_pre_inh=pre_inh, gamma=1)
        t, rE, rD, rS, rN, rP, rV, p, cGABA, other = model.run(dur, xFF, dt=dt, init_noise=0, noise=noise, 
                                                               rE0=0, rD0=1, rN0=0, rS0=0, rP0=0, monitor_currents=True)
        # note: dendritic activity is set to 1 so that the inhibition by SOM and NDNF shows in the soma

        # plotting and labels
        mean_act = np.mean(other['curr_rE'], axis=1)
        ax.plot(t[1:]/1000, -mean_act, c=eval('c'+labelz[i]), label=f'{labelz[i]} inh.', lw=1)

        # not used: plotting currents in SOM and NDNF
        # ax2[1, i].plot(t[1:], -np.mean(other['curr_rS'], axis=1), c=cSOM, lw=2)
        # ax2[0, i].plot(t[1:], -np.mean(other['curr_rN'], axis=1), c=cNDNF, lw=2)
        # ax2[1, i].set(xlabel='time (ms)', ylim=[-0.1, 3])

    ax.legend(loc=(0.35, 0.56), handlelength=1, frameon=False, fontsize=8)
    ax.set(xlabel='time (s)', ylim=[-0.05, 0.8], xticks=[0, 1], ylabel='PC curr. (au)')

    # not used: labels for currents in SOM and NDNF
    # ax2[0, 0].set(ylim=[-3, 3], ylabel='curr. (au)', yticks=[-2, 0, 2], xlim=[0, 500])
    # ax2[1, 0].set(ylim=[-3, 3], ylabel='curr. (au)', yticks=[-2, 0, 2], xlim=[0, 500])

    if save:
        fig.savefig('../results/figs/Naumann23_draft1/fig5B_IPSC_timescales.pdf', dpi=300)
        plt.close(fig)
        # not used: saving the second figure
        # fig2.savefig('../results/figs/Naumann23_draft1/exp1-1_paired-recordings1.pdf', dpi=300)


def exp_fig5CD_transient_signals(mean_pop=True, w_hetero=False, save=False, noise=0, plot_supp=False):
    """
    Study transmission of transient signals by NDNF and SOM. Stimulate NDNF and SOM with pulses of different length
    and record the change in PC activity. Perform the same experiment with and without presynaptic inhibition and
    with weak and strong NDNF-dendrite inhibition.

    Parameters:
    - mean_pop: if true, simulate only one neuron (mean) per population
    - w_hetero: whether to add heterogeneity to weight matrices
    - save: whether to save the figure
    - noise: level of white noise added to neural activity
    """

    # define parameter dictionaries
    N_cells, w_mean, conn_prob, bg_inputs, taus = mb.get_default_params(flag_mean_pop=mean_pop)

    # increase NDNF-dendrite inhibition
    wDNs = [w_mean['DN'], 0.8]      

    # simulation paramters
    dur = 5000
    dt = 1
    nt = int(dur/dt)
    stim_durs = (np.array([10, 20, 50, 100, 200, 500, 1000])*dt).astype(int)
    ts = int(1000*dt)
    amp = 1.5

    # empty arrays for storage of PC signal amplitudes
    deltaPC_stim_N = np.zeros(len(stim_durs))
    deltaPC_stim_S = np.zeros(len(stim_durs))

    i_show = 5

    # set up figures
    dpi = DPI if save else 200
    fig, ax = plt.subplots(2, 2, figsize=(2.75, 2.1), gridspec_kw=dict(right=0.97, top=0.95, bottom=0.27, left=0.2, wspace=0.15, hspace=0.15),
                             sharex=True, sharey=True, dpi=dpi)
    fig2, ax2 = plt.subplots(1, 1, dpi=dpi, figsize=(1.8, 1.15), gridspec_kw={'left': 0.25, 'bottom': 0.33, 'hspace':0.2, 'right':0.95})
    cols = sns.color_palette("flare", n_colors=len(stim_durs))
    cols_PC = ['#DB9EA4', cPC]

    if plot_supp:
        fig3, ax3 = plt.subplots(3, 3, figsize=(5, 5), dpi=dpi, sharex=True, sharey='row')
        fig4, ax4 = plt.subplots(3, 3, figsize=(5, 5), dpi=dpi, sharex=True, sharey='row')
        plot_count = 0
        i_supp_list = [3, 4, 5]

    # loop over presynaptic inhibition and NDNF-dendrite inhibition strength
    for k, pre_inh in enumerate([True, False]):

        print(f"pre inh = {pre_inh}")  # print progress

        for j, wDN in enumerate(wDNs):

            # change wDN parameter
            w_mean['DN'] = wDN
            model = mb.NetworkModel(N_cells, w_mean, conn_prob, taus, bg_inputs, wED=1, flag_w_hetero=w_hetero, flag_pre_inh=pre_inh)

            for i, sdur in enumerate(stim_durs):

                xff_null = np.zeros(nt)
                xff_stim = xff_null.copy()
                xff_stim[ts:ts+sdur] = amp
                
                # simulate with sine input to SOM
                xFF = get_null_ff_input_arrays(nt, N_cells)
                xFF['S'] = np.tile(xff_stim, [N_cells['S'], 1]).T
                xFF['N'] = np.tile(xff_null, [N_cells['N'], 1]).T
                t, rE1, rD1, rS1, rN1, rP1, rV1, p1, cGABA1, other1 = model.run(dur, xFF, dt=dt, calc_bg_input=True, noise=noise, monitor_dend_inh=True)

                # simulate with sine input to NDNF
                xFF = get_null_ff_input_arrays(nt, N_cells)
                xFF['S'] = np.tile(xff_null, [N_cells['S'], 1]).T
                xFF['N'] = np.tile(xff_stim, [N_cells['N'], 1]).T
                t, rE2, rD2, rS2, rN2, rP2, rV2, p2, cGABA2, other2 = model.run(dur, xFF, dt=dt, calc_bg_input=True, noise=noise, monitor_dend_inh=True)

                # compute change in PC activity
                bl1 = np.mean(rE1[500:ts])
                bl2 = np.mean(rE2[500:ts])
                deltaPC_stim_S[i] = (np.mean(rE1[ts:ts+sdur]-bl1))
                deltaPC_stim_N[i] = (np.mean(rE2[ts:ts+sdur]-bl2))

                if pre_inh and i==i_show:
                    # ax2.plot(t, np.mean(rE1, axis=1), cSOM)
                    ax2.plot(t/1000, np.mean(rE2, axis=1), color=cols_PC[j])

                if plot_supp and pre_inh and (i in i_supp_list):
                    axx = ax3 if j == 0 else ax4
                    axx[0, plot_count].plot(t/1000, np.mean(rE2, axis=1), c=cPC)
                    axx[1, plot_count].plot(t/1000, np.mean(rN2, axis=1), c=cNDNF)
                    axx[1, plot_count].plot(t/1000, np.mean(rS2, axis=1), c=cSOM)
                    axx[1, plot_count].plot(t/1000, np.mean(rP2, axis=1), c=cPV)
                    mean_inh_NDNF = np.mean(np.array(other2['dend_inh_NDNF']), axis=1)
                    mean_inh_SOM = np.mean(np.array(other2['dend_inh_SOM']), axis=1)
                    mean_inh_PV = np.mean(np.array(other2['soma_inh_PV']), axis=1)
                    axx[2, plot_count].plot(t/1000, mean_inh_NDNF-np.mean(mean_inh_NDNF[:ts]), c=cNDNF, lw=1, ls='--')
                    axx[2, plot_count].plot(t/1000, mean_inh_SOM-np.mean(mean_inh_SOM[:ts]), c=cSOM, lw=1, ls='--')
                    axx[2, plot_count].plot(t/1000, mean_inh_PV-np.mean(mean_inh_PV[:ts]), c=cPV, lw=1, ls='--')
                    axx[0, plot_count].fill_between([ts/1000, (ts+sdur)/1000], 0, 3, facecolor=cNDNF, alpha=0.15, zorder=-1)
                    axx[1, plot_count].fill_between([ts/1000, (ts+sdur)/1000], 0, 3, facecolor=cNDNF, alpha=0.15, zorder=-1)
                    axx[2, plot_count].fill_between([ts/1000, (ts+sdur)/1000], 3, 3, facecolor=cNDNF, alpha=0.15, zorder=-1)
                    axx[0, plot_count].set(xlim=[0, 2.5], ylim=[0, 1.5], yticks=[0, 1], title=f"{stim_durs[i]} ms")
                    axx[1, plot_count].set(xlim=[0, 2.5], ylim=[0, 3], yticks=[0, 1, 2, 3])
                    axx[2, plot_count].set(xlim=[0, 2.5], ylim=[-1.5, 1.5], yticks=[-1, 0, 1], xlabel='time (s)')
                    plot_count += 1
            if plot_supp:
                plot_count = 0
                axx[0, 0].set(ylabel='PC act. (au)')
                axx[1, 0].set(ylabel='IN act. (au)')
                axx[2, 0].set(ylabel=r'$\Delta$ inh.')
            
            # plotting
            ax[j, k].plot(np.arange(len(stim_durs)), deltaPC_stim_N, '.-', c=cNDNF, label='NDNF', lw=lw, ms=4*lw)
            ax[j, k].plot(np.arange(len(stim_durs)), deltaPC_stim_S, '.-', c=cSOM, label='SOM', lw=lw, ms=4*lw)
            # labels and formatting
            ax[j, k].hlines(0, 0, len(stim_durs)-1, ls='--', color='k', zorder=-1, lw=1)
            ax[j, k].set(xticks=np.arange(len(stim_durs)), ylim=[-0.55, 0.55], yticks=[0.5, 0, -0.5])
            ax[j, 0].set(ylabel=r'$\Delta$ PC act.')
        ax[1, k].set(xlabel='stim. dur. (ms)')
        ax[1, k].set_xticklabels(stim_durs, rotation=45, ha='right', rotation_mode="anchor")

        ax2.set(ylim=[0, 1.5], xlim=[0, 3], ylabel='PC act. (au)', xlabel='time (s)')
        # ax2.legend(['N>D weak', 'N>D strong'], loc='best', handlelength=1, frameon=False, fontsize=8)
        ax2.fill_between([ts/1000, (ts+stim_durs[i_show])/1000], 0, 1.5, facecolor=cNDNF, alpha=0.1, zorder=-1)




    # saving
    if save:
        fig.savefig('../results/figs/Naumann23_draft1/fig5D_sum_transient_input.pdf', dpi=300)
        fig2.savefig('../results/figs/Naumann23_draft1/fig5C_ex_transient_input.pdf', dpi=300)
        plt.close(fig)
        plt.close(fig2)
        if plot_supp:
            fig3.savefig('../results/figs/Naumann23_draft1/supps/fig5_supp1.pdf', dpi=300)
            fig4.savefig('../results/figs/Naumann23_draft1/supps/fig5_supp2.pdf', dpi=300)
            plt.close(fig3)
            plt.close(fig4)


def exp_fig5E_inh_change(mean_pop=True, w_hetero=False, pre_inh=True, save=False, noise=0, wDN=0.4):
    """
    Experiment2: Study transmission of transient signal by NDNF and SOM.
    """

    # define parameter dictionaries
    N_cells, w_mean, conn_prob, bg_inputs, taus = mb.get_default_params(flag_mean_pop=mean_pop)

    # increase NDNF-dendrite inhibition
    w_mean['DN'] = wDN

    # simulation paramters
    dur = 4000
    dt = 1
    nt = int(dur/dt)

    stim_durs = (np.array([100, 1000])*dt).astype(int)
    ts = int(1000*dt)
    amp = 1.5

    amplitudes_n = np.zeros(len(stim_durs))
    amplitudes_s = np.zeros(len(stim_durs))

    dpi = 300 if save else 300

    fig, ax = plt.subplots(1, 1, dpi=300, figsize=(1.5, 0.9), sharex=True, sharey=True, gridspec_kw=dict(left=0.25, right=0.97))

    # create model
    model = mb.NetworkModel(N_cells, w_mean, conn_prob, taus, bg_inputs, wED=1, flag_w_hetero=w_hetero, flag_pre_inh=pre_inh)

    alphas = [1, 1]

    # loop over stimulus durations
    for i, sdur in enumerate(stim_durs):

        xff_null = np.zeros(nt)
        xff_stim = xff_null.copy()
        xff_stim[ts:ts+sdur] = amp
        
        # simulate with input to SOM
        xFF = get_null_ff_input_arrays(nt, N_cells)
        xFF['S'] = np.tile(xff_stim, [N_cells['S'], 1]).T
        xFF['N'] = np.tile(xff_null, [N_cells['N'], 1]).T
        t, rE1, rD1, rS1, rN1, rP1, rV1, p1, cGABA1, other1 = model.run(dur, xFF, dt=dt, calc_bg_input=True, noise=noise, monitor_dend_inh=True)

        # simulate with input to NDNF
        xFF = get_null_ff_input_arrays(nt, N_cells)
        xFF['S'] = np.tile(xff_null, [N_cells['S'], 1]).T
        xFF['N'] = np.tile(xff_stim, [N_cells['N'], 1]).T
        t, rE2, rD2, rS2, rN2, rP2, rV2, p2, cGABA2, other2 = model.run(dur, xFF, dt=dt, calc_bg_input=True, noise=noise, monitor_dend_inh=True)

        # compute change in PC activity
        bl2 = np.mean(rE2[500:ts])
        amplitude = (np.mean(rE2[ts:ts+sdur]-bl2))
        print(f"stim dur = {sdur}, amplitude = {amplitude:1.3f}")

        # plot time course
        fig2, ax2 = plt.subplots(3, 1, figsize=(1.5, 1.5), gridspec_kw=dict(right=0.97, top=0.95, bottom=0.3, left=0.25, wspace=0.15, hspace=0.15, height_ratios=[1, 1, 1]),
                               sharex=True, sharey=False, dpi=dpi)
        ax2[0].plot(t/1000, np.mean(rN2, axis=1), c=cNDNF, lw=1)
        ax2[0].plot(t/1000, np.mean(rS2, axis=1), c=cSOM, lw=1)
        ax2[0].plot(t/1000, np.mean(rP2, axis=1), c=cPV, lw=1)
        mean_inh_NDNF = np.mean(np.array(other2['dend_inh_NDNF']), axis=1)
        mean_inh_SOM = np.mean(np.array(other2['dend_inh_SOM']), axis=1)
        mean_inh_PV = np.mean(np.array(other2['soma_inh_PV']), axis=1)
        ax2[1].plot(t/1000, mean_inh_NDNF-np.mean(mean_inh_NDNF[:ts]), c=cNDNF, lw=1, ls='--')
        ax2[1].plot(t/1000, mean_inh_SOM-np.mean(mean_inh_SOM[:ts]), c=cSOM, lw=1, ls='--')
        ax2[1].plot(t/1000, mean_inh_PV-np.mean(mean_inh_PV[:ts]), c=cPV, lw=1, ls='--')
        ax2[-1].plot(t/1000, np.mean(rE2, axis=1)-np.mean(rE2[:ts]), c=cPC, lw=1)
        ax2[-1].set(xlabel='time (s)', ylim=[-0.35, 0.35], yticks=[-0.2, 0, 0.2], ylabel=r'$\Delta$ PC act.')
        # ax[1, k].set_xticklabels(stim_durs, rotation=45, ha='right', rotation_mode="anchor")


        # plot change in dendritic and somatic inhibition
        dend_inh_SOM = np.array(other2['dend_inh_SOM']).mean(axis=1)
        dend_inh_NDNF = np.array(other2['dend_inh_NDNF']).mean(axis=1)
        soma_inh_PV = np.array(other2['soma_inh_PV']).mean(axis=1)
        ddi_SOM = np.mean(dend_inh_SOM[ts:ts+sdur])-np.mean(dend_inh_SOM[:ts])
        ddi_NDNF = np.mean(dend_inh_NDNF[ts:ts+sdur])-np.mean(dend_inh_NDNF[:ts])
        dsi_PV = np.mean(soma_inh_PV[ts:ts+sdur])-np.mean(soma_inh_PV[:ts])
        ax.bar(i*1.3-0.3, ddi_NDNF, facecolor='none', edgecolor=cNDNF, hatch='/////', width=0.2, label='NDNF' if i==0 else None)
        ax.bar(i*1.3-0.1, ddi_SOM, facecolor='none', edgecolor=cSOM, hatch='/////', width=0.2, label='SOM' if i==0 else None)
        ax.bar(i*1.3+0.1, dsi_PV, facecolor='none', edgecolor=cPV, hatch='/////', width=0.2, label='PV' if i==0 else None)
        ax.bar(i*1.3+0.3, ddi_SOM+ddi_NDNF+dsi_PV, facecolor='none', edgecolor='silver', hatch='/////', width=0.2)
        # labels and formatting
        ax.hlines(0, -0.5, 1.8, color='k', lw=1)
        ax.set(ylabel=r'$\Delta$ inh.', xticks=[], ylim=[-0.6, 1.1])
        ax.spines['bottom'].set_visible(False)

    if save:
        wDN_str = str(wDN).replace('.', 'p')
        # fig.savefig('../results/figs/Naumann23_draft1/exp3-2_transient-input.pdf', dpi=300)
        fig.savefig(f"../results/figs/Naumann23_draft1/exp3-3_transient-input_inh-change_{wDN_str}.pdf", dpi=300)
        plt.close(fig)


def make_sine(nt, freq, plot=False):
    """
    Make a sine wave.

    Parameters:
    - nt: number of time points
    - freq: frequency of the sine wave
    - plot: whether to plot the sine wave

    Returns:
    - sine wave as numpy array
    """
    t = np.arange(nt)/1000
    sine = (np.sin(2*np.pi*freq*t)+1)/2
    if plot:
        plt.figure(figsize=(3, 2), dpi=300)
        plt.plot(t, sine, lw=lw, c='k')
    return sine


def exp_old_frequency_preference(noise=0.0, w_hetero=False, mean_pop=True, pre_inh=True, save=False, plot_supp=False):
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


def exp_old_transient_effects(mean_pop=True, w_hetero=False, pre_inh=True, noise=0.0, reduced=False, save=False):
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
    # print(f"first signal: {quantify_signals([xsine], np.mean(rE[ts1+sh:te1+sh], axis=1), bias=True)[0]:1.3f}")
    # print(f"second signal: {quantify_signals([xsine], np.mean(rE[ts2+sh:te2+sh], axis=1), bias=True)[0]:1.3f}")

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


if __name__ in "__main__":

    SAVE = False

    # Figure 5: redistribution of inhibition in time
    # ----------------------------------------------

    # # Fig 5 B: timescale of inhibition (SOM vs NDNF) 
    # exp_fig5C_IPSC_timescale(mean_pop=False, w_hetero=True, noise=0.1, save=SAVE)

    # # Fig 5 C: frequency preference
    # exp301_frequency_preference(save=SAVE, mean_pop=False, w_hetero=True, noise=0.0, pre_inh=True, plot_supp=True)

    # # Fig 5 D&E: responses of PC to SOM/NDNF stimulation depend on parameters and stimulus duration
    # exp_fig5CD_transient_signals(save=SAVE, mean_pop=False, w_hetero=True, noise=0.1, plot_supp=True)
    exp_fig5E_inh_change(save=SAVE, mean_pop=False, w_hetero=True, noise=0.1, pre_inh=True, wDN=0.4)
    exp_fig5E_inh_change(save=SAVE, mean_pop=False, w_hetero=True, noise=0.1, pre_inh=True, wDN=0.8)

    # old stuff
    # exp304_transient_effects(reduced=False, pre_inh=True, mean_pop=False, w_hetero=True, noise=0.1, save=False)
    # make_sine(1000, 4, plot=True)

    plt.show()
