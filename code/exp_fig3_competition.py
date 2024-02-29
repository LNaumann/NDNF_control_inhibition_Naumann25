"""
Experiments for Figure 3: Competition between SOM- and NDNF-mediated dendritic inhibition.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

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


def exp_fig3top_vary_NDNF_input(dur=1500, dt=1, w_hetero=True, mean_pop=False, noise=0.1, pre_inh=True,
                                target_ND=False, target_VS=False, save=False):
    """
    Vary input to NDNF interneurons, monitor NDNF- and SOM-mediated dendritic inhibition and their activity.

    Parameters
    - dur: duration of experiment (ms)
    - dt: integration time step (ms)
    - w_hetero: whether to add heterogeneity to weight matrices
    - mean_pop: if true, simulate only one neuron (mean) per population
    - noise: level of white noise added to neural activity
    - pre_inh: whether to include presynaptic inhibition
    - save: if it's a string, name of the saved file, else if False nothing is saved
    - target_ND: whether to target NDNF->dendrite synapse with presynaptic inhibition
    - target_VS: whether to target SOM->VIP synapse with presynaptic inhibition
    """

    # extract number of timesteps
    nt = int(dur / dt)

    # get default parameters
    N_cells, w_mean, conn_prob, bg_inputs, taus = mb.get_default_params(flag_mean_pop=mean_pop)

    # array for varying NDNF input
    ndnf_input = np.arange(-1, 1, 0.05)
    ninput = len(ndnf_input)

    # empty arrays for recording stuff
    rS_inh_record = np.zeros(ninput)
    rN_inh_record = np.zeros(ninput)
    rS_record = np.zeros((ninput, N_cells['S']))
    rN_record = np.zeros((ninput, N_cells['N']))
    cGABA_record = np.zeros(ninput)
    p_record = np.zeros(ninput)

    print(f"Running model with varying NDNF and for pre. inh. = {pre_inh}...")
    for i, I_activate in enumerate(ndnf_input):

        # create input (stimulation of NDNF)
        xFF = get_null_ff_input_arrays(nt, N_cells)
        xFF['N'][:, :] = I_activate

        # instantiate and run model
        model = mb.NetworkModel(N_cells, w_mean, conn_prob, taus, bg_inputs, wED=1, flag_w_hetero=w_hetero,
                                flag_pre_inh=pre_inh, flag_p_on_DN=target_ND, flag_p_on_VS=target_VS)
        t, rE, rD, rS, rN, rP, rV, p, cGABA, other = model.run(dur, xFF, dt=dt, init_noise=0, monitor_dend_inh=True,
                                                               noise=noise)

        # save stuff
        rS_record[i] = rS[-1]
        rN_record[i] = rN[-1]
        rS_inh_record[i] = np.mean(np.array(other['dend_inh_SOM'][-1]))
        rN_inh_record[i] = np.mean(np.array(other['dend_inh_NDNF'][-1]))
        cGABA_record[i] = np.mean(cGABA[-1])
        p_record[i] = p[-1]     

    # Plotting
    # --------
    dpi = 300 if save else DPI
    fig, ax = plt.subplots(2, 1, figsize=(1.6, 2.5), dpi=dpi, gridspec_kw={'left': 0.22, 'bottom': 0.15, 'top': 0.95,
                                                                           'right': 0.95, 'hspace': 0.2,
                                                                           'height_ratios': [1, 1]}, sharex=True)
    # plot SOM and NDNF activity
    ax[0].plot(ndnf_input, np.mean(rS_record, axis=1), color=cSOM, lw=lw)
    ax[0].plot(ndnf_input, np.mean(rN_record, axis=1), color=cNDNF, lw=lw)
    ax[0].legend(['SOM', 'NDNF'], frameon=False, handlelength=1, loc=(0.05, 0.6), fontsize=8)
    # plot SOM and NDNF dendritic inhibition and sum (i.e. total dendritic inhibition)
    ax[1].plot(ndnf_input, rS_inh_record, c=cSOM, ls='--', lw=lw)
    ax[1].plot(ndnf_input, rN_inh_record, c=cNDNF, ls='--', lw=lw)
    ax[1].plot(ndnf_input, rS_inh_record+rN_inh_record, c='#978991', ls='-', lw=lw, zorder=-1)
    # labels etc
    ax[0].set(ylabel='activity (au)', xlim=[-1, 1], ylim=[-0.1, 2.5], yticks=[0, 1, 2])
    ax[1].set(ylabel='dend. inh. (au)', ylim=[-0.05, 1.1], yticks=[0, 1], xlabel=r'$\Delta$ NDNF input')

    # Saving
    # ------
    if save:
        pre_inh_str = '_with_pre_inh' if pre_inh else '_without_pre_inh'
        savename = f"{FIG_PATH}exp_fig3top_competition{pre_inh_str}.pdf" if not isinstance(save, str) else save 
        fig.savefig(savename, dpi=300)
        plt.close(fig)


def exp_fig3bottom_total_dendritic_inhibition(dur=1500, dt=1, w_hetero=True, mean_pop=False, noise=0.1, pre_inh=True, save=False):
    """
    Vary input to NDNF interneurons and NDNF->dendrite weight, check how this affects total dendritic inhibition.

    Parameters
    - dur: duration of experiment (ms)
    - dt: integration time step (ms)
    - w_hetero: whether to add heterogeneity to weight matrices
    - mean_pop: if true, simulate only one neuron (mean) per population
    - noise: level of white noise added to neural activity
    - pre_inh: whether to include presynaptic inhibition
    - save: if it's a string, name of the saved file, else if False nothing is saved
    """

    # extract number of timesteps
    nt = int(dur / dt)

    # get default parameters
    N_cells, w_mean, conn_prob, bg_inputs, taus = mb.get_default_params(flag_mean_pop=mean_pop)

    # array for varying NDNF input
    ndnf_input = np.arange(-1, 1.05, 0.1)
    weightsDN = np.arange(0, 0.9, 0.2)

    # empty arrays for recording stuff
    rS_inh_record = np.zeros((len(ndnf_input), len(weightsDN)))
    rN_inh_record = np.zeros((len(ndnf_input), len(weightsDN)))

    # set up figure
    dpi = 300 if save else DPI
    fig, ax = plt.subplots(1, 1, figsize=(1.6, 1.5), dpi=dpi, gridspec_kw={'left': 0.22, 'bottom': 0.25, 'top': 0.95,
                                                                           'right': 0.95}, sharex=True)
    cols = sns.color_palette(f"blend:{cSOM},{cNDNF}", n_colors=len(weightsDN))

    # loop over NDNF input and NDNF->dendrite weight, simulate, record and plot
    print(f"Running model with varying NDNF-dendrite weight and NDNF input for pre. inh. = {pre_inh}...")
    for j, wDN in enumerate(weightsDN):

        w_mean['DN'] = wDN
        print(f"\t - NDNF/SOM dendritic inh: {w_mean['DN']/w_mean['DS']:1.2f}")

        # instantiate and run model
        model = mb.NetworkModel(N_cells, w_mean, conn_prob, taus, bg_inputs, wED=1, flag_w_hetero=w_hetero,
                                flag_pre_inh=pre_inh)

        for i, I_activate in enumerate(ndnf_input):

            # create input (stimulation of NDNF)
            xFF = get_null_ff_input_arrays(nt, N_cells)
            xFF['N'][:, :] = I_activate

            # simulate
            t, rE, rD, rS, rN, rP, rV, p, cGABA, other = model.run(dur, xFF, dt=dt, init_noise=0, monitor_dend_inh=True,
                                                                   noise=noise, calc_bg_input=True)

            # save dendritic inhibition
            rS_inh_record[i, j] = np.mean(np.array(other['dend_inh_SOM'][-1]))
            rN_inh_record[i, j] = np.mean(np.array(other['dend_inh_NDNF'][-1]))

        # plot
        ax.plot(ndnf_input, rS_inh_record[:, j]+rN_inh_record[:, j], c=cols[j], ls='-', label=f"{wDN/w_mean['DS']:1.1f}", lw=lw)

    # labels
    ax.set(xlabel=r'$\Delta$ NDNF input', xticks=[-1, 0, 1], xlim=[-1, 1], ylim=[-0.1, 2], yticks=[0, 1, 2], ylabel=r'$\Sigma$ dend. inh.')

    # saving
    if save:
        pre_inh_str = '_with_pre_inh' if pre_inh else '_without_pre_inh'
        savename = f"{FIG_PATH}exp_fig3bottom_dendritic_inh{pre_inh_str}.pdf" if not isinstance(save, str) else save 
        fig.savefig(savename, dpi=300)
        plt.close(fig) 


if __name__ in "__main__":

    SAVE = False
    plot_supps = False

    # Figure 3: Competition for dendritic inhibition

    # Fig 3 (top): Layer-specificity of NDNF control (with & without pre inh)
    exp_fig3top_vary_NDNF_input(pre_inh=True, save=SAVE)
    exp_fig3top_vary_NDNF_input(pre_inh=False, save=SAVE)

    # Fig 3 (bottom): total dendritic inhibiion
    exp_fig3bottom_total_dendritic_inhibition(pre_inh=True, mean_pop=False, w_hetero=True, noise=0.1, save=SAVE)
    exp_fig3bottom_total_dendritic_inhibition(pre_inh=False, mean_pop=False, w_hetero=True, noise=0.1, save=SAVE)

    if plot_supps:

        # Supplementary figures
        # ---------------------

        # Fig 3/4, Supp 1b: competition with pre inh on NDNF-dendrite synapses
        exp_fig3top_vary_NDNF_input(pre_inh=True, target_ND=True, save=f"{SUPP_PATH}fig34_supp1b.pdf")

        # Fig 3/4, Supp 2: competition with pre in on SOM-VIP synapses
        exp_fig3top_vary_NDNF_input(pre_inh=True, target_VS=True, save=f"{SUPP_PATH}fig34_supp2b.pdf")

    plt.show()
 