"""
Experiments for Figure 4: NDNF INs can act as a switch for dendritic inhibition.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import model_base as mb
from helpers import get_null_ff_input_arrays, get_model_colours

# optional custom style sheet
if 'pretty' in plt.style.available:
    plt.style.use('pretty')
lw = mpl.rcParams['lines.linewidth']

# get model colours
cPC, cPV, cSOM, cNDNF, cVIP, cpi = get_model_colours()

# figure path and settings
FIG_PATH = '../results/figs/Naumann23_draft1/'
SUPP_PATH = '../results/figs/Naumann23_draft1/supps/'
DPI = 300


def exp_fig3BC_bistability(noise=0.1, w_hetero=True, mean_pop=False, pre_inh=True, save=False, target_DN=False, target_VS=False):
    """
    Check for bistability within the SOM-NDNF mutual inhibition motif. NDNF INs receive brief positive or
    negative input pulses and the NDNF rate is monitored. Vary the pulse strength and SOM-NDNF inhibition.
    Plot results.

    Parameters:
    ----------
    - noise: level of white noise added to neural activity
    - w_hetero: whether to add heterogeneity to weight matrices
    - mean_pop: whether to use mean population parameters
    - pre_inh: whether to include presynaptic inhibition
    - save: if it's a string, name of the saved file, else if False nothing is saved
    - target_DN: whether to target NDNF-dendrite synapses with presynaptic inhibition
    - target_VS: whether to target VIP-SOM synapses with presynaptic inhibition
    """

    # define parameter dictionaries
    N_cells, w_mean, conn_prob, bg_inputs, taus = mb.get_default_params(flag_mean_pop=mean_pop)

    # increase NDNF-dendrite inhibition s.t. mean PC rate doesn't change when dendritic inhibition changes
    w_mean['DN'] = 0.6
    if target_DN:
        w_mean['DN'] = 0.8

    # simulation paramters
    dur = 8000
    dt = 1
    nt = int(dur/dt)

    # generate inputs
    t_act_s, t_act_e = 1000, 2000
    xFF = get_null_ff_input_arrays(nt, N_cells)

    # array of pulse strengths and SOM-NDNF inhibition to test
    stim_NDNF = np.arange(-1.1, 1.2, 0.2)
    vals_wNS = np.arange(0.5, 1.61, 0.1)

    # empty array for storage
    rNDNF = np.zeros((len(stim_NDNF), len(vals_wNS)))

    # loop over pulse strengths and SOM-NDNF inhibition
    print(f"Running model with varying SOM-NDNF inhibition and NDNF stimulation...")
    for i, wNS in enumerate(vals_wNS):

        print(f'\t - wNS={wNS:.2f}')

        w_mean['NS'] = wNS
        
        for j, stim in enumerate(stim_NDNF):

            xFF['N'][t_act_s:t_act_e] = stim

            # instantiate model
            model = mb.NetworkModel(N_cells, w_mean, conn_prob, taus, bg_inputs, wED=1, flag_w_hetero=w_hetero,
                                   flag_pre_inh=pre_inh, flag_p_on_DN=target_DN, flag_p_on_VS=target_VS)

            # run model
            t, rE, rD, rS, rN, rP, rV, p, cGABA, other = model.run(dur, xFF, dt=dt, calc_bg_input=True,
                                                                   monitor_dend_inh=True, noise=noise)
            
            # save mean NDNF rate a few seconds after the pulse
            rNDNF[j, i] = np.mean(rN[7000:8000, :])

    # Plotting
    # --------
    dpi = 300 if save else DPI
    fig, ax = plt.subplots(2, 1, figsize=(1.7, 2.8), dpi=dpi, gridspec_kw={'left':0.26, 'bottom': 0.13, 'top':0.97, 'right': 0.94, 'wspace': 0.5, 'hspace': 0.8})
    # plot NDNF rate for positive and negative pulse strength as a function of SOM-NDNF inhibition
    ax[0].plot(vals_wNS, rNDNF[-1, :], c='#E7A688', lw=2, ls='--', label='pos pulse')
    ax[0].plot(vals_wNS, rNDNF[0, :], c='#BB5525', lw=2, label='neg pulse', zorder=-1)
    # plot 2D heatmap of NDNF rate as a function of pulse strength and SOM-NDNF inhibition
    mp = ax[1].pcolormesh(vals_wNS, stim_NDNF, rNDNF, cmap='YlOrBr_r', vmin=0, vmax=3)
    # colorbar, labels etc
    cb = fig.colorbar(mp, ax=ax[1], ticks=[0, 1, 2, 3])
    cb.set_label('NDNF act.', rotation=0, labelpad=-15, y=1.17)
    ax[0].set(xlabel='SOM-NDNF inh.', ylabel='NDNF act.', ylim=[-0.1, 2.5], xticks=[0.5, 1, 1.5], yticks=[0, 1, 2])
    ax[1].set(xlabel='SOM-NDNF inh.', ylabel='pulse to NDNF', xticks=[0.5, 1, 1.5], yticks=[-1, 0, 1])

    # Saving
    # ------
    if save:
        pre_inh_str = '_with_pre_inh' if pre_inh else '_without_pre_inh'
        savename = f"{FIG_PATH}exp_fig4BC_bistability{pre_inh_str}.pdf" if not isinstance(save, str) else save
        fig.savefig(savename, dpi=300)
        plt.close(fig)


def exp_fig4DEFG_mutual_inhibition(w_hetero=True, mean_pop=False, pre_inh=True, save=False, noise=0.1, wNS=1.4, 
                                  flag_sine=False, stimup=0.5, stimdown=-0.5, target_DN=False, target_VS=False):
    """
    Check for bistability. NDNF INs receive a positive and a negative pulse. In a bistable regime, the NDNF rate
    is switched to higher or lower activity after the pulse. Plot IN rates and mean NDNF- and SOM-mediated dendritic
    inhibition. If flag_sine is True, add a time-varying input to the SOM INs and check the PC rate.

    Parameters:
    ----------
    - w_hetero: whether to add heterogeneity to weight matrices
    - mean_pop: whether to use mean population parameters
    - pre_inh: whether to include presynaptic inhibition
    - save: if it's a string, name of the saved file, else if False nothing is saved
    - noise: level of white noise added to neural activity
    - wNS: weight of NDNF to SOM inhibition
    - flag_sine: whether to add a time-varying input to the SOM INs
    - stimup: amplitude of positive pulse
    - stimdown: amplitude of negative pulse
    - target_ND: whether to target NDNF-dendrite synapses with presynaptic inhibition
    - target_VS: whether to target VIP-SOM synapses with presynaptic inhibition
    """

    # define parameter dictionaries
    N_cells, w_mean, conn_prob, bg_inputs, taus = mb.get_default_params(flag_mean_pop=mean_pop)

    # increase SOM to NDNF inhibition to get bistable regime
    w_mean['NS'] = wNS

    # increase NDNF-dendrite inhibition s.t. mean PC rate doesn't change when dendritic inhibition changes
    w_mean['DN'] = 0.6
    if target_DN:
        w_mean['DN'] = 0.8 # increase NDNF to dendrite inhibition further to compensate

    # instantiate model
    model = mb.NetworkModel(N_cells, w_mean, conn_prob, taus, bg_inputs, wED=1, flag_w_hetero=w_hetero,
                           flag_pre_inh=pre_inh, flag_p_on_DN=target_DN, flag_p_on_VS=target_VS)

    # simulation paramters
    dur = 10000
    dt = 1
    nt = int(dur/dt)

    # generate inputs
    t_act_s, t_act_e = 1000, 2000
    t_inact_s, t_inact_e = 5000, 6000
    xFF = get_null_ff_input_arrays(nt, N_cells)
    xFF['N'][t_act_s:t_act_e] = stimup
    xFF['N'][t_inact_s:t_inact_e] = stimdown

    if flag_sine:
        # add time-varying inputs to SOM (and NDNF)
        tt = np.arange(0, dur/1000+1, dt/1000)  # generate 1s longer to enable shifting when quantifying signal
        sine = np.sin(2*np.pi*tt*2)
        amp_sine = 0.5
        xFF['S'][:,:] += amp_sine*np.tile(sine[1000:], [N_cells['S'], 1]).T

    # run model
    print(f"Running model with wNS={wNS} and sine={flag_sine}...")
    t, rE, rD, rS, rN, rP, rV, p, cGABA, other = model.run(dur, xFF, dt=dt, calc_bg_input=True,
                                                           monitor_dend_inh=True, noise=noise)

    # Plotting
    # --------
    dpi = 300 if save else DPI

    if not flag_sine:
        fig, ax = plt.subplots(2, 1, figsize=(1.77, 1.6), dpi=dpi, sharex=True,
                            gridspec_kw={'left': 0.2, 'bottom': 0.25, 'top': 0.95, 'right': 0.95,
                                            'height_ratios': [1, 1]})
        # plot IN rates
        alpha = 0.5
        ax[0].plot(t/1000, rN, c=cNDNF, alpha=alpha, lw=1, label='NDNF')
        ax[0].plot(t/1000, rS, c=cSOM, alpha=alpha, lw=1, label='SOM')
        if target_VS:
            ax[0].plot(t/1000, rV, alpha=alpha, c=cVIP, lw=1)
        # plot mean NDNF- and SOM-mediated dendritic inhibition
        mean_dend_NDNF = np.mean(np.array(other['dend_inh_NDNF']), axis=1)
        mean_dend_SOM = np.mean(np.array(other['dend_inh_SOM']), axis=1)
        ax[1].plot(t/1000, mean_dend_NDNF, c=cNDNF, ls='--', lw=lw)
        ax[1].plot(t/1000, mean_dend_SOM, c=cSOM, ls='--', lw=lw)

        # labels etc
        ax[1].set(ylabel='dend inh', ylim=[-0.1, 2], yticks=[0, 2], xlabel='time (s)')
        ax[0].set(ylabel='IN act.', ylim=[-0.1, 3.5], yticks=[0, 2])

    elif flag_sine:
        # plot the same as above but with the PC rate as well
        fig, ax = plt.subplots(3, 1, figsize=(1.77, 1.8), dpi=dpi, sharex=True, sharey=False,
                                gridspec_kw={'left': 0.25, 'bottom': 0.22, 'top': 0.95, 'right': 0.95, 'hspace': 0.3})
        alpha = 0.5
        ax[0].plot(t/1000, rN, alpha=alpha, c=cNDNF, lw=1)
        ax[0].plot(t/1000, rS, alpha=alpha, c=cSOM, lw=1)
        ax[1].plot(t/1000, rD, alpha=alpha, c='gray', lw=1)
        ax[1].plot(t/1000, np.mean(rD, axis=1), alpha=1, c='k', lw=1)
        cPClight = sns.color_palette(f"light:{cPC}", n_colors=3)[1]
        ax[2].plot(t/1000, rE, alpha=alpha, c=cPClight, lw=1)
        ax[2].plot(t/1000, np.mean(rE, axis=1), alpha=1, c=cPC, lw=1)
        # labels etc
        [ax[ii].set(ylim=[0, 2]) for ii in range(3)]
        ax[0].set(ylabel='IN act.', ylim=[0, 3], yticks=[0, 2])
        ax[1].set(ylabel='dend.')
        ax[2].set(ylabel='PC act.')
        ax[-1].set(xlabel='time (s)')

        # plot a zoom-in of bottom from fig above:
        fig2, ax2 = plt.subplots(1, 2, figsize=(1.77, 0.45), dpi=dpi, sharey=True,
                                 gridspec_kw={'left': 0.25, 'bottom': 0., 'top': 1, 'right': 0.95, 'wspace': 0.4})
        zs1, ze1 = 3000, 5000
        zs2, ze2 = 8000, 10000
        # PC rates
        ax2[0].plot((t/1000)[zs1:ze1], rE[zs1:ze1], c=cPClight, lw=0.5, alpha=0.5)
        ax2[1].plot((t/1000)[zs2:ze2], rE[zs2:ze2], c=cPClight, lw=0.5, alpha=0.5)
        ax2[0].plot((t/1000)[zs1:ze1], np.mean(rE, axis=1)[zs1:ze1], c=cPC, lw=1)
        ax2[1].plot((t/1000)[zs2:ze2], np.mean(rE, axis=1)[zs2:ze2], c=cPC, lw=1)
        # sine curve
        ax2[0].plot((t/1000)[zs1:ze1], sine[zs1:ze1]/4+1, c='k', lw=1, ls='--')
        ax2[1].plot((t/1000)[zs2:ze2], sine[zs2:ze2]/4+1, c='k', lw=1, ls='--')
        # labels etc
        ax2[0].axis('off')
        ax2[1].axis('off')
        ax2[1].set(ylim=[0.6, 1.4])

        # plot the contribution of signals (correlation between SOM signal and PC rate)
        fig3, ax3 = plt.subplots(1, 1, figsize=(1.77, 0.8), dpi=dpi,
                                 gridspec_kw={'left': 0.3, 'bottom': 0.45, 'top': 0.92, 'right': 0.95})
        rEmu = np.mean(rE, axis=1)
        wbin = 1000 # bin width for quantification
        sine_shift = sine[1000-40:nt+1000-40] # shift sine to account for delay
        corr = np.zeros((nt//wbin, 2))
        # loop over bins and calculate correlation
        for ti in range(nt//wbin):
            tts, tte = ti*wbin, (ti+1)*wbin
            corr[ti] = np.corrcoef(sine_shift[tts:tte], rEmu[tts:tte])[0, :]
        # plot correlation
        ax3.plot((np.arange(0, nt, wbin)+wbin/2)/1000, corr[:, 1], '.-', c='k', ms=lw*3, lw=lw)
        ax3.hlines(0, 0, 10, color='silver', ls=':', lw=1, zorder=-1)
        # plot box between t_act_s and t_act_e
        ylow, yhigh = -1.2, 1.2
        ax3.fill_between([t_act_s/1000, t_act_e/1000], ylow, yhigh, facecolor=cNDNF, alpha=0.2, zorder=-1)
        ax3.fill_between([t_inact_s/1000, t_inact_e/1000], ylow, yhigh, facecolor=cNDNF, alpha=0.2, zorder=-1)
        # labeling
        ax3.set(xlabel='time (s)', ylabel='corr.', ylim=[ylow, yhigh], yticks=[-1, 0, 1])

    # Saving
    # ------
    if save:
        wNS_str = str(wNS).replace('.', 'p')
        if flag_sine:
            fig.savefig(f'{FIG_PATH}exp_fig4F_switch_sine_wNS-{wNS_str}_withPC.pdf', dpi=300)
            fig2.savefig(f'{FIG_PATH}exp_fig4F_switch_sine_zoom.pdf', dpi=300)
            fig3.savefig(f'{FIG_PATH}exp_fig4G_switch_sine_corr.pdf', dpi=300)
            plt.close(fig2)
            plt.close(fig3)
        else:
            savename = f'{FIG_PATH}exp_fig4DE_switch_wNS-{wNS_str}.pdf' if not isinstance(save, str) else save
            fig.savefig(savename, dpi=300)
        plt.close(fig)


if __name__ in "__main__":

    SAVE = False
    plot_supps = False
    
    # Fig 4: Mutual inhibition between NDNF and SOM
    # ---------------------------------------------
    
    #B&C: parameter sweep for quantification of switch regime
    exp_fig3BC_bistability(save=SAVE)

    # E: pulse input example (not bistable)
    exp_fig4DEFG_mutual_inhibition(wNS=0.7, save=SAVE)
    
    # D: pulse input example (bistable)
    exp_fig4DEFG_mutual_inhibition(wNS=1.2, save=SAVE)

    # F&G: switch with time-varying input to SOM
    exp_fig4DEFG_mutual_inhibition(wNS=1.2, flag_sine=True, save=SAVE)

    if plot_supps:

    # Supplementary Figures
    # ---------------------

        # Fig 3/4, Supp 1c/d: bistability with pre inh on NDNF-dendrite synapses
        exp_fig4DEFG_mutual_inhibition(wNS=1.2, target_DN=True, save=f'{SUPP_PATH}fig34_supp1c.pdf')
        exp_fig3BC_bistability(target_DN=True, save=f'{SUPP_PATH}fig34_supp1d.pdf')
        
        # Fig 3/4, Supp 2: bistability with pre in on SOM-VIP synapses
        exp_fig4DEFG_mutual_inhibition(wNS=1.2, stimdown=-0.4, target_VS=True, save=f'{SUPP_PATH}fig34_supp2c.pdf')
        exp_fig3BC_bistability(target_VS=True, save=f'{SUPP_PATH}fig34_supp2d.pdf')

    plt.show()