import numpy as np
import matplotlib.pyplot as plt

# optional custom style sheet
if 'pretty' in plt.style.available:
    plt.style.use('pretty')

import model_base as mb
from helpers import get_model_colours, get_null_ff_input_arrays, slice_dict

# colours
cPC, cPV, cSOM, cNDNF, cVIP, cpi = get_model_colours()
cpred = '#832161'
csens = '#F5B656'

# figure path and settings
FIG_PATH = '../results/figs/Naumann23_draft1/'
SUPP_PATH = '../results/figs/Naumann23_draft1/supps/'
DPI = 300


def fig6_predictive_coding(mean_pop=False, w_hetero=True, pre_inh=True, with_NDNF=True, with_wPN=False, NDNF_get_P=False,
                           noise=0.1, NDNF_act_strength=1, rN0=4, b=0.15, plot_all_variables=False, save=False,
                           is_supp=False, plot_vary_NDNF_input=False):
    """
    Experiments to explore predictive coding microcircuit. Plots all panels for the paper (Fig. 6).

    Parameters:
    ----------
    - mean_pop:   bool, if True, model mean population activity, if False, model single cell activity 
    - w_hetero:   bool, if True, model heterogenous weights, else homogeneous weights
    - pre_inh:    bool, if True, model presynaptic inhibition, else not
    - with_NDNF:  bool, if True, include NDNF interneurons in the model, else not
    - with_wPN:   bool, if True, include nonzero NDNF-PV inhibition
    - NDNF_get_P: bool, if True, NDNFs receive prediction input
    - noise:      float, noise level in the model (std of added Gaussian white noise)
    - NDNF_act_strength:   float, additional feedforward input to NDNF interneurons
    - rN0:     float, initial value of NDNF activity
    - b:          float (0-1), presynaptic inhibition strength
    - plot_all_variables: bool, whether to plot all variables for all phases
    - save:       bool, whether to save figures
    - is_supp:    bool, if True, save figures in supplementary folder
    - plot_vary_NDNF_input: bool, if True, plot effect of varying NDNF input on mismatch responses
    """
    

    # define deafault parameter dictionaries
    N_cells, w_mean, conn_prob, bg_inputs, taus = mb.get_default_params(flag_mean_pop=mean_pop)

    # set parameters to get mismatch responses
    w_mean_update = dict(EP=2, DE=0.2, DS=1, PE=1.2, PP=0.4, PS=0.3, PV=0.15, SE=1, SV=0.5, VE=1, VS=1,
                         NS=0.5, DN=1.5, PN=0., VN=0.1)

    save_name_add = ''


    # Update parameters depending on model details
    # --------------------------------------------

    # parameters with NDNF>PV inhibition > 0
    if with_wPN:
        w_mean_update.update(dict(PN=0.1))
        save_name_add += '_with_wPN'

    # paramaters without presynaptic inhibition
    if not pre_inh:
        w_mean_update.update(dict(DS=1.6))
        save_name_add += '_no_pre_inh'

    # parameters when NDNFs get prediction (P)
    if NDNF_get_P:
        w_mean_update.update(dict(VN=0.3))
        save_name_add += '_NDNF_get_P'

    # parameters without NDNFs
    if not with_NDNF:
        bg_inputs['N'] = 0
        rN0 = 0
        pre_inh = False
        save_name_add += '_no_NDNF'

    # update mean weights using weight updates create above
    w_mean.update(w_mean_update)


    # Simulation parameters
    # ---------------------
    dur_stim = 2000  # duration of stimulus per phase (ms)
    buffer = 2000    # buffer time before and after stimulus (ms)
    dur = 2*buffer + dur_stim  # duration of one phase (ms)
    dur_tot = dur*3  # total duration (ms)
    dt = 1  # integration time step (ms), note: since dt=1ms, time index = time in ms
    t = np.arange(0, dur_tot, dt)  # time vector
    nt = len(t)  # number of time steps


    # Sensory and prediction input
    # ----------------------------
    amp_s, amp_p = 1, 1  # amplitude of sensory and prediction input
    sensory, prediction = get_s_and_p_inputs(amp_s, amp_p, dur_stim, buffer, nt)

    # set input of cells
    xFF = get_null_ff_input_arrays(nt, N_cells)
    xFF['E'] = np.tile(sensory, [N_cells['E'], 1]).T
    xFF['D'] = np.tile(prediction, [N_cells['D'], 1]).T
    xFF['P'] = np.tile(sensory, [N_cells['P'], 1]).T
    xFF['S'] = np.tile(sensory, [N_cells['S'], 1]).T
    xFF['V'] = np.tile(prediction, [N_cells['V'], 1]).T
    if NDNF_get_P:
        xFF['N'] = np.tile(prediction, [N_cells['N'], 1]).T

    # store baseline ff input to NDNFs (to be manipulated later)
    xFF_NDNF_bl = xFF['N']
    

    # Instantiate model
    # -----------------
    model = mb.NetworkModel(N_cells, w_mean.copy(), conn_prob, taus, bg_inputs.copy(), wED=1, flag_w_hetero=w_hetero, flag_pre_inh=pre_inh, flag_with_NDNF=with_NDNF,
                            w_std_rel=0.01, b=b)
    

    # Run simulation without manipulations
    # -------------------------------------
    print(f"Running predictive coding experiment without manipulations...")
    t, res_fp, res_op, res_up, bg_inputs_df = run_pc_phases(dur, model, xFF, rN0=rN0, p0=model.g_func(rN0), dt=dt, calc_bg_input=True, noise=noise)

    # plot mismatch responses
    plot_mismatch_responses(t, res_fp, res_op, res_up, prediction, sensory,
                            save=save, save_name='default'+save_name_add, is_supp=is_supp)
    # plot_changes_bars(t, res_fp, res_op, res_up, prediction, sensory, buffer, dur_stim)  # old
    if plot_all_variables:
        plot_all_variables_all_phases(t, res_fp, res_op, res_up, prediction, sensory)


    # Activate NDNF interneurons and simulate again, use calculated background inputs
    # -------------------------------------------------------------------------------
    xFF['N'] = xFF_NDNF_bl + NDNF_act_strength # additional feedforward input to NDNF interneurons
    rN0 = 6 if pre_inh else 4.8                # adapt initial value of NDNF activity
    if NDNF_get_P:
        rN0 = 5.
    # set background inputs to defaults, they are not recomputed
    model.Xbg = bg_inputs_df

    # run simulations
    print(f"Running predictive coding experiment with additional NDNF activation...")
    t, res_fp_act, res_op_act, res_up_act, _ = run_pc_phases(dur, model, xFF, rN0=rN0, p0=model.g_func(rN0), dt=dt, noise=noise,
                                                             calc_bg_input=False, scale_w_by_p=False)

    # plotting
    plot_mismatch_responses(t, res_fp_act, res_op_act, res_up_act, prediction, sensory,
                            save_name='actNDNF'+save_name_add, save=save, is_supp=is_supp)
    if plot_all_variables:
        plot_all_variables_all_phases(t, res_fp_act, res_op_act, res_up_act, prediction, sensory)


    # Change in inh and exc inputs to PCs
    # -----------------------------------
    fig, ax = plt.subplots(1, 2, dpi=DPI, figsize=(2.8, 1.), sharey=True,
                            gridspec_kw={'wspace': 0.1, 'right': 0.98, 'left': 0.15, 'bottom': 0.1, 'top': 0.95})
    for i, res in enumerate([res_fp, res_fp_act]):

        dend_inh_SOM = np.array(res['other']['dend_inh_SOM']).mean(axis=1)
        dend_inh_NDNF = np.array(res['other']['dend_inh_NDNF']).mean(axis=1)
        soma_inh_PV = np.array(res['other']['soma_inh_PV']).mean(axis=1)

        ddi_SOM = np.mean(dend_inh_SOM[buffer:buffer+dur_stim])-np.mean(dend_inh_SOM[:buffer])
        ddi_NDNF = np.mean(dend_inh_NDNF[buffer:buffer+dur_stim])-np.mean(dend_inh_NDNF[:buffer])
        dsi_PV = np.mean(soma_inh_PV[buffer:buffer+dur_stim])-np.mean(soma_inh_PV[:buffer])
        ax[i].bar(0-0.3, ddi_NDNF, facecolor='none', edgecolor=cNDNF, hatch='/////', width=0.2)
        ax[i].bar(-0.1, ddi_SOM, facecolor='none', edgecolor=cSOM, hatch='/////', width=0.2)
        ax[i].bar(0+0.1, ddi_SOM+ddi_NDNF, facecolor='none', edgecolor='silver', hatch='/////', width=0.2)
        ax[i].bar(0+0.3, prediction[buffer:buffer+dur_stim].mean(), facecolor='none', edgecolor=cpred, hatch='/////', width=0.2)

        ax[i].bar(0+0.7, dsi_PV, facecolor='none', edgecolor=cPV, hatch='/////', width=0.2)
        ax[i].bar(0+0.9, sensory[buffer:buffer+dur_stim].mean(), facecolor='none', edgecolor=csens, hatch='/////', width=0.2)

        ax[i].set(ylim=[-1.5, 2.5], xticks=[])
        ax[i].spines['bottom'].set_visible(False)
        ax[i].axhline(0, c='k', lw=1)
    ax[0].set(ylabel=r'$\Delta$ exc./inh.')

    # saving
    if save:
        file_path = SUPP_PATH if is_supp else FIG_PATH
        fig.savefig(file_path+'exp_fig6_mismatch_inh_exc'+save_name_add+'.pdf', dpi=300)
        plt.close(fig)


    # Fedback, mismatch and playback response for different levels of NDNF activation
    # (optional plotting)
    # -------------------------------------------------------------------------------
    if plot_vary_NDNF_input:

        # levels of NDNF activation
        ndnf_act_levels = np.arange(0, 1.6, 0.2)

        # empty arrays to store responses
        fb_response = np.zeros(len(ndnf_act_levels))
        mm_response = np.zeros(len(ndnf_act_levels))
        pb_response = np.zeros(len(ndnf_act_levels))

        # run simulation for different levels of NDNF activation
        print(f"Running simulation with varying NDNF input...")
        for j, ndnf_act in enumerate(ndnf_act_levels):
            print(f"\t - NDNF activation: {ndnf_act:1.1f}")
            xFF['N'] = xFF_NDNF_bl + ndnf_act
            t, res_fp, res_op, res_up, _ = run_pc_phases(dur, model, xFF, rN0=rN0, p0=model.g_func(rN0) , dt=dt, calc_bg_input=False, scale_w_by_p=False)
            fb_response[j] = np.mean(res_fp['rE'][buffer:buffer+dur_stim])-np.mean(res_fp['rE'][:buffer])
            mm_response[j] = np.mean(res_op['rE'][buffer:buffer+dur_stim])-np.mean(res_op['rE'][:buffer])
            pb_response[j]  = np.mean(res_up['rE'][buffer:buffer+dur_stim])-np.mean(res_up['rE'][:buffer])

        # plotting
        fig2, ax2 = plt.subplots(1, 1, dpi=DPI, figsize=(2.5, 1.), gridspec_kw={'right': 0.95, 'left': 0.2, 'bottom': 0.35, 'top': 0.94})
        ax2.plot(ndnf_act_levels, mm_response, c=cpred, lw=1, ls='-', marker='.', label='mismatch')
        ax2.plot(ndnf_act_levels, pb_response, c=csens, lw=1, ls='-', marker='.', label='playback')
        ax2.plot(ndnf_act_levels, fb_response, c='k', ls='-', marker='.', lw=1, label='feedback')
        ax2.set(xlabel='NDNF activation', ylabel=r'$\Delta$ PC act.', ylim=[-0.05, 0.7], yticks=[0, 0.5], xticks=[0, 1])

        # saving
        if save:
            file_path = SUPP_PATH if is_supp else FIG_PATH
            fig2.savefig(f"{file_path}exp_fig6_vary_NDNF_input{save_name_add}.pdf", dpi=300)
            plt.close(fig2)


def run_pc_phases(dur, model, xFF, rE0=1, rD0=0, rS0=4, rP0=4, rV0=4, rN0=4, p0=0.5, dt=1, calc_bg_input=True,
                  scale_w_by_p=True, p_scale=None, noise=0.1):
    """
    Run predictive coding experiment for a given circuit model and input. The input is split into three phases:
    - fp (fully predicted = feedback): prediction and sensory input
    - op (overpredicted = mismatch): only prediction input
    - up (underpredicted = playback): only sensory input

    Parameters:
    ----------
    - dur:          int, duration of one phase (ms)
    - model:        instance of NetworkModel as custuom python object
    - xFF:          dict, feedforward input to the model
    - rE0:          float, initial value of PC activity
    - rD0:          float, initial value of dendrite activity
    - rS0:          float, initial value of SOM activity
    - rP0:          float, initial value of PV activity
    - rV0:          float, initial value of VIP activity
    - rN0:          float, initial value of NDNF activity
    - p0:           float, initial value of release factor
    - dt:           int, integration time step (ms)
    - calc_bg_input: bool, if True, calculate background input to the model
    - scale_w_by_p: bool, if True, scale weights affected by pre. inh.
    - p_scale:      float, if not None, use this value to scale weights affected by pre. inh. (p_scale)
    - noise:        float, noise level in the model (std of added Gaussian white noise)

    Returns:
    -------
    - t:             array, time vector (s)
    - res_fp:        dict, results of the simulation for the fully predicted phase
    - res_op:        dict, results of the simulation for the overpredicted phase
    - res_up:        dict, results of the simulation for the underpredicted phase
    - bg_inputs_calc: dict, calculated background inputs to the model
    """

    # split FF input into three phases
    xFFfp = slice_dict(xFF, 0, dur)
    xFFop = slice_dict(xFF, dur, 2*dur)
    xFFup = slice_dict(xFF, 2*dur, 3*dur)

    # run simulation
    t, rEfp, rDfp, rSfp, rNfp, rPfp, rVfp, pfp, cGABAfp, otherfp = model.run(dur, xFFfp, dt=dt, rE0=rE0, rP0=rP0, rS0=rS0, rV0=rV0, rN0=rN0, rD0=rD0, p0=p0,
                                                                            calc_bg_input=calc_bg_input, scale_w_by_p=scale_w_by_p, init_noise=0, noise=noise,
                                                                            monitor_dend_inh=True, p_scale=p_scale)
    t, rEop, rDop, rSop, rNop, rPop, rVop, pop, cGABAop, otherop = model.run(dur, xFFop, dt=dt, rE0=rE0, rP0=rP0, rS0=rS0, rV0=rV0, rN0=rN0, rD0=rD0, p0=p0,
                                                                            calc_bg_input=calc_bg_input, scale_w_by_p=scale_w_by_p, init_noise=0, noise=noise,
                                                                            monitor_dend_inh=True, p_scale=p_scale)
    t, rEup, rDup, rSup, rNup, rPup, rVup, pup, cGABAup, otherup = model.run(dur, xFFup, dt=dt, rE0=rE0, rP0=rP0, rS0=rS0, rV0=rV0, rN0=rN0, rD0=rD0, p0=p0,
                                                                            calc_bg_input=calc_bg_input, scale_w_by_p=scale_w_by_p, init_noise=0, noise=noise,
                                                                            monitor_dend_inh=True, p_scale=p_scale)
    
    res_fp = dict(rE=rEfp, rD=rDfp, rS=rSfp, rN=rNfp, rP=rPfp, rV=rVfp, p=pfp, cGABA=cGABAfp, other=otherfp)
    res_op = dict(rE=rEop, rD=rDop, rS=rSop, rN=rNop, rP=rPop, rV=rVop, p=pop, cGABA=cGABAop, other=otherop)
    res_up = dict(rE=rEup, rD=rDup, rS=rSup, rN=rNup, rP=rPup, rV=rVup, p=pup, cGABA=cGABAup, other=otherup)

    bg_inputs_calc = model.Xbg

    return t/1000, res_fp, res_op, res_up, bg_inputs_calc


def get_s_and_p_inputs(amp_s, amp_p, dur_stim, buffer, nt):
    """
    Function to construct sensory and prediction input arrays. Input has three phases:
    - fp (fully predicted = feedback): prediction and sensory input
    - op (overpredicted = mismatch): only prediction input
    - up (underpredicted = playback): only sensory input
    
    Parameters:
    ----------
    - amp_s:    float, amplitude of sensory input
    - amp_p:    float, amplitude of prediction input
    - dur_stim: int, duration of stimulus per phase (ms)
    - buffer:   int, buffer time before and after stimulus (ms)
    - nt:       int, number of time steps

    Returns:
    -------
    - sensory:    array, sensory input (length nt)
    - prediction: array, prediction input (length nt)
    """

    fp_s = buffer
    op_s = fp_s+buffer*2+dur_stim
    up_s = op_s+buffer*2+dur_stim
    sensory = np.zeros(nt)
    sensory[fp_s:fp_s+dur_stim] = amp_s
    sensory[up_s:up_s+dur_stim] = amp_s
    prediction = np.zeros(nt)
    prediction[fp_s:fp_s+dur_stim] = amp_p
    prediction[op_s:op_s+dur_stim] = amp_p

    return sensory, prediction


def plot_all_variables_all_phases(t, res_fp, res_op, res_up, prediction, sensory):
    """
    Plot all variables for all phases of the experiment (fully predicted=feedback, overpredicted=mismatch,
    underpredicted=playback).

    Parameters:
    ----------
    - t:           array, time vector (s)
    - res_fp:      dict, results of the simulation for the fully predicted phase
    - res_op:      dict, results of the simulation for the overpredicted phase
    - res_up:      dict, results of the simulation for the underpredicted phase
    - prediction:  array, prediction input (length nt)
    - sensory:     array, sensory input (length nt)
    """

    # set up figure
    fig, ax = plt.subplots(6, 3, dpi=DPI, figsize=(4, 3), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1, 1, 0.6, 0.6]}, sharey='row')
    titles = ['P=S', 'P>S', 'P<S']

    dur = len(t)

    # loop over the three phases and plot all neuron activities
    for i, cond in enumerate(['fp', 'op', 'up']):
        res = eval('res_'+cond)
        ax[0, i].plot(t, res['rE'], c=cPC, alpha=0.5, lw=1)
        ax[1, i].plot(t, res['rD'], c='k')
        ax[2, i].plot(t, res['rP'], c=cPV)
        ax[2, i].plot(t, res['rV'], c='silver')
        ax[2, i].plot(t, res['rS'], c=cSOM)
        ax[2, i].plot(t, res['rN'], c=cNDNF)
        ax[3, i].plot(t, res['p'], c=cpi)
        ax[4, i].plot(t, prediction[i*dur:(i+1)*dur], c=cpred)
        ax[5, i].plot(t, sensory[i*dur:(i+1)*dur], c=csens)
        ax[0, i].set(title=titles[i])

    # labels
    ax[0, 0].set(ylabel='PC', ylim=[0, 2])
    ax[1, 0].set(ylabel='dend.', ylim=[0, 3])
    ax[2, 0].set(ylabel='INs', ylim=[0, 7])
    ax[3, 0].set(ylabel='p', ylim=[-0.05, 1.05])
    ax[4, 0].set(ylabel='sens.', ylim=[-0.05, 1.05])
    ax[5, 0].set(ylabel='pred.', xlabel='time', ylim=[-0.05, 1.05])


def plot_mismatch_responses(t, res_fp, res_op, res_up, prediction, sensory, save_name='default', save=False, is_supp=False):
    """
    Plot mismatch responses for the three phases of the experiment (fully predicted=feedback, overpredicted=mismatch,
    underpredicted=playback).

    Parameters:
    ----------
    - t:           array, time vector (s)
    - res_fp:      dict, results of the simulation for the fully predicted phase
    - res_op:      dict, results of the simulation for the overpredicted phase
    - res_up:      dict, results of the simulation for the underpredicted phase
    - prediction:  array, prediction input (length nt)
    - sensory:     array, sensory input (length nt)
    - save_name:   str, name experiment to save figure
    - save:        bool, whether to save figure
    - is_supp:     bool, if True, save figure in supplementary folder
    """

    fig, ax = plt.subplots(3, 3, figsize=(2.2, 1.8), dpi=DPI, gridspec_kw=dict(height_ratios=[1.5, 1.5, 1], wspace=0.15, hspace=0.35, bottom=0.1, right=0.95, top=0.95, left=0.2))

    dur = len(t)

    for i, cond in enumerate(['fp', 'op', 'up']):

        res = eval('res_'+cond)

        # create plot for figure
        LW = 1.
        ax[0, i].plot(t, np.mean(res['rE'], axis=1), c=cPC)
        ax[0, i].plot(t, np.mean(res['rD'], axis=1), c='k', ls='--', lw=1)
        ax[1, i].plot(t, np.mean(res['rV'], axis=1), c=cVIP, lw=LW)
        ax[1, i].plot(t, np.mean(res['rP'], axis=1), c=cPV, lw=LW)
        ax[1, i].plot(t, np.mean(res['rS'], axis=1), c=cSOM, lw=LW)
        ax[1, i].plot(t, np.mean(res['rN'], axis=1), c=cNDNF, lw=LW)


        # ax5[0, i].plot(t, np.mean(eval('rD'+cond), axis=1), c='k', lw=1, alpha=0.5)
        ax[2, i].plot(t, 1.3+prediction[i*dur:(i+1)*dur], c=cpred)
        ax[2, i].plot(t, sensory[i*dur:(i+1)*dur], c=csens)

        # limits
        ax[0, i].set(ylim=[-0.2, 2], xticks=[])
        ax[1, i].set(ylim=[0, 7], xticks=[], yticks=[0, 5])
        ax[2, i].set(ylim=[-0.8, 2.4], xticks=[], xlabel='', yticks=[])

        ax[2, i].spines['left'].set_visible(False)
        ax[2, i].spines['bottom'].set_visible(False)

        # remove axes
        for xi in range(2):
            ax[xi, i].spines['bottom'].set_visible(False)

        if i!= 0:
            for xx in ax[:, i]:
                xx.spines['left'].set_visible(False)
                xx.set(yticks=[])
        else:
            ax[0, i].set(ylabel='act. (au)')
            ax[1, i].set(ylabel='act. (au)')


    # add rectangle to axis
    ax[2, 2].add_patch(plt.Rectangle((2, -0.6), 2, 0.2, facecolor='k', edgecolor='k', lw=0))
    ax[2, 2].text(3, -0.7, '2s', ha='center', va='top', color='k', fontsize=8)  

    # saving
    if save:
        file_path = SUPP_PATH if is_supp else FIG_PATH
        fig.savefig(f"{file_path}exp_fig6_mismatch_{save_name}.pdf", dpi=300)
        plt.close(fig)


if __name__ in "__main__":

    SAVE = False
    plot_supps = False

    # Figure 6: predictive coding example
    # -----------------------------------
    fig6_predictive_coding(NDNF_act_strength=1, save=SAVE, plot_vary_NDNF_input=True)


    if plot_supps:

        # Supps to Fig 6
        # --------------

        # no presynaptic inhibition
        fig6_predictive_coding(NDNF_act_strength=1, save=SAVE, pre_inh=False, is_supp=True)

        # NDNFs get prediction
        fig6_predictive_coding(NDNF_act_strength=0.5, save=SAVE, NDNF_get_P=True, is_supp=True)

        # with NDNF-to-PV inhibition
        fig6_predictive_coding(NDNF_act_strength=1, save=SAVE, with_wPN=True, is_supp=True)

    plt.show()


