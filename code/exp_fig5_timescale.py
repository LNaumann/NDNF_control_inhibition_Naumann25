"""
Experiments for Figure 5: Redistribution of dendritic inhibition in time.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# optional custom style sheet
if 'pretty' in plt.style.available:
    plt.style.use('pretty')
lw = mpl.rcParams['lines.linewidth']

import model_base as mb
from helpers import get_null_ff_input_arrays, get_model_colours

# get model colours
cPC, cPV, cSOM, cNDNF, cVIP, cpi = get_model_colours()

# figure path and settings
FIG_PATH = '../results/figs/Naumann23_draft1/'
SUPP_PATH = '../results/figs/Naumann23_draft1/supps/'
DPI = 300


def exp_fig5B_IPSC_timescale(dur=1000, dt=1, w_hetero=True, mean_pop=False, pre_inh=True, noise=0.1, save=False):
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

    # get default parameters
    N_cells, w_mean, conn_prob, bg_inputs, taus = mb.get_default_params(flag_mean_pop=mean_pop)

    # labels
    labelz = ['SOM', 'NDNF']

    # stimulate SOM and NDNF, respectively
    print("Running model for IPSC timescale experiment...")
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

    ax.legend(loc=(0.35, 0.56), handlelength=1, frameon=False, fontsize=8)
    ax.set(xlabel='time (s)', ylim=[-0.05, 0.8], xticks=[0, 1], ylabel='PC curr. (au)')

    if save:
        fig.savefig(f'{FIG_PATH}exp_fig5B_IPSCs.pdf', dpi=300)
        plt.close(fig)


def exp_fig5CD_transient_signals(mean_pop=False, w_hetero=True, save=False, noise=0.1, plot_supp=False):
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
    cols_PC = ['#DB9EA4', cPC]

    if plot_supp:
        fig3, ax3 = plt.subplots(3, 3, figsize=(5, 5), dpi=dpi, sharex=True, sharey='row')
        fig4, ax4 = plt.subplots(3, 3, figsize=(5, 5), dpi=dpi, sharex=True, sharey='row')
        plot_count = 0
        i_supp_list = [3, 4, 5]

    # loop over presynaptic inhibition and NDNF-dendrite inhibition strength
    print("Running model for transient input experiment...")
    for k, pre_inh in enumerate([True, False]):

        print(f"\t - pre inh = {pre_inh}")  # print progress

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
        ax2.fill_between([ts/1000, (ts+stim_durs[i_show])/1000], 0, 1.5, facecolor=cNDNF, alpha=0.1, zorder=-1)

    # saving
    if save:
        fig.savefig(f'{FIG_PATH}exp_fig5D_sum_transient_input.pdf', dpi=300)
        fig2.savefig(f'{FIG_PATH}exp_fig5C_ex_transient_input.pdf', dpi=300)
        plt.close(fig)
        plt.close(fig2)
        if plot_supp:
            fig3.savefig(f'{SUPP_PATH}fig5_supp1.pdf', dpi=300)
            fig4.savefig(f'{SUPP_PATH}fig5_supp2.pdf', dpi=300)
            plt.close(fig3)
            plt.close(fig4)


def exp_fig5E_inh_change(mean_pop=False, w_hetero=True, pre_inh=True, noise=0.1, wDN=0.4, save=False):
    """
    Study the change in inhibition at PCs for different stimulus durations. Stimulate NDNF and SOM with pulses
    of different length and monitor the NDNF-dendrite, SOM-dendrite and PV-PC inhibition. Plot results.

    Parameters:
    - mean_pop: if true, simulate only one neuron (mean) per population
    - w_hetero: whether to add heterogeneity to weight matrices
    - pre_inh: whether to include presynaptic inhibition
    - noise: level of white noise added to neural activity
    - wDN: strength of NDNF-dendrite inhibition
    - save: whether to save the figure
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

    # set up figure
    dpi = 300 if save else 300
    fig, ax = plt.subplots(1, 1, dpi=300, figsize=(1.5, 0.9), sharex=True, sharey=True, gridspec_kw=dict(left=0.25, right=0.97))

    # create model
    model = mb.NetworkModel(N_cells, w_mean, conn_prob, taus, bg_inputs, wED=1, flag_w_hetero=w_hetero, flag_pre_inh=pre_inh)

    # loop over stimulus durations
    print("Running model for inhibition change experiment...")
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
        # bl2 = np.mean(rE2[500:ts])
        # amplitude = (np.mean(rE2[ts:ts+sdur]-bl2))
        # print(f"stim dur = {sdur}, amplitude = {amplitude:1.3f}")

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

    # save
    if save:
        wDN_str = str(wDN).replace('.', 'p')
        fig.savefig(f"{FIG_PATH}exp_fig5E_inh_change_{wDN_str}.pdf", dpi=300)
        plt.close(fig)


if __name__ in "__main__":

    SAVE = False
    plot_supps = False

    # Figure 5: redistribution of inhibition in time
    # ----------------------------------------------

    # # Fig 5 B: timescale of inhibition (SOM vs NDNF) 
    exp_fig5B_IPSC_timescale(save=SAVE)

    # # Fig 5 C, D, E: responses of PC to SOM/NDNF stimulation depend on parameters and stimulus duration
    exp_fig5CD_transient_signals(save=SAVE, plot_supp=plot_supps)
    exp_fig5E_inh_change(save=SAVE, wDN=0.4)
    exp_fig5E_inh_change(save=SAVE, wDN=0.8)

    plt.show()
