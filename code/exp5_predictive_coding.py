import numpy as np
import matplotlib.pyplot as plt
plt.style.use('pretty')

import model_base as mb
from experiments import get_model_colours
from experiments import get_null_ff_input_arrays

cPC, cPV, cSOM, cNDNF, cVIP, cpi = get_model_colours()
cpred = '#832161'
csens = '#F5B656'


DPI = 200


def slice_dict(dic, ts, te):

    dic_new = dict()
    for k in dic.keys():
        dic_new[k] = dic[k][ts:te]

    return dic_new


def exp501_predictive_coding(mean_pop=True, w_hetero=False, reduced=False, pre_inh=False, noise=0.0, save=False, with_NDNF=True, xFF_NDNF=0, rNinit=0, calc_bg_input=False,
                             b=0.12, scale_w_by_p=False, p_scale=None, NDNF_get_P=False):

    # define parameter dictionaries
    N_cells, w_mean, conn_prob, bg_inputs, taus = mb.get_default_params(flag_mean_pop=False)

    # set parameters to get negative prediction errors
    w_mean_update = dict(EP=2, DE=0.2, DS=1, PE=1.2, PP=0.4, PS=0.3, PV=0.15, SE=1, SV=0.5, VE=1, VS=1)

    if NDNF_get_P:
        # parameters for prediction error neurons when NDNFs get P
        # ---------------------------------------------------------
        w_mean_update.update(dict(NS=0.7, DN=0.8, PN=0.1, VN=0.1))
        # update parameters to get prediction errors with NDNF active
        # w_mean_update.update(dict(NS=0.5, DN=1, PN=0.1, VN=0.4, PS=0.27))  # parameters with wPN > 0 
        w_mean_update.update(dict(DN=1, PN=0)) # parameters with wPN = 0

        # background inputs adapted to default
        # bg_inputs = {'E': 9, 'D': 4., 'N': 2.8, 'S': 5.0, 'P': 6.2, 'V': 7.0}
        bg_inputs = {'E': 9, 'D': 8, 'N': 7.6, 'S': 5.0, 'P': 6.2, 'V': 7.4}

    else:
        # paremters for prediction error neurons when NDNFs don't get P
        # ------------------------------------------------------------
        w_mean_update.update(dict(NS=0.5, DN=1.5, PN=0., VN=0.1))  # parameters with wPN = 0

        # w_mean_update.update(dict(NS=0.5, DN=1, PN=0.1, VN=0.2, PS=0.25))  # parameters with wPN > 0 

        # background inputs adapted to default
        bg_inputs = {'E': 9, 'D': 10, 'N': 6.8, 'S': 5.0, 'P': 6.2, 'V': 7.4}

    # update mean weights
    w_mean.update(w_mean_update)

    # simulation parameters
    dur_stim = 2000
    buffer = 2000
    dur = 2*buffer + dur_stim
    dur_tot = dur*3
    dt = 1  # ms
    t = np.arange(0, dur_tot, dt)
    nt = len(t)

    # construct sensory and prediction input
    amp = 1
    sensory, prediction = get_s_and_p_inputs(amp, dur_stim, buffer, nt)

    # sensory += 2

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

    # give extra input to NDNF interneurons to get "prediction neurons" ( if input to NDNF is > 0)
    # xFF['N'] += xFF_NDNF

    # scale_w_by_p = False

    if not with_NDNF:
        bg_inputs['N'] = 0
        rN0 = 0
        pre_inh = False
        xFF['N'] = xFF_NDNF_bl
    else:
        rN0 = rNinit
        # calc_bg_input = False  # set to False to get prediction errors neurons with inactive NDNF and predictions with active NDNFs
        # xFF['N'] = xFF_NDNF_bl + xFF_NDNF
    
    # instantiate model
    model = mb.NetworkModel(N_cells, w_mean.copy(), conn_prob, taus, bg_inputs.copy(), wED=1, flag_w_hetero=w_hetero, flag_pre_inh=pre_inh, flag_with_NDNF=with_NDNF,
                            w_std_rel=0.01)
    if pre_inh:
        model.b = b
    p0 = model.g_func(rN0) #0.3
    print('pscale = ', p_scale if p_scale else model.g_func(rN0))

    # run simulation without manipulations
    t, res_fp, res_op, res_up, bg_inputs_df = run_pc_phases(dur, model, xFF, rN0=rN0, p0=p0, dt=dt, calc_bg_input=True)
    print(bg_inputs_df)

    plot_all_variables_all_phases(t, res_fp, res_op, res_up, prediction, sensory, buffer)
    plot_mismatch_responses(t, res_fp, res_op, res_up, prediction, sensory, buffer, save=save)
    plot_changes_bars(t, res_fp, res_op, res_up, prediction, sensory, buffer, dur_stim)

    # activate NDNF interneurons and simulate again, use calculated background inputs
    xFF_NDNF = 1
    rN0 = 6
    p0 = model.g_func(rN0)
    xFF['N'] = xFF_NDNF_bl + xFF_NDNF
    t, res_fp_act, res_op_act, res_up_act, _ = run_pc_phases(dur, model, xFF, rN0=rN0, p0=p0, dt=dt, calc_bg_input=False, scale_w_by_p=False)
    plot_all_variables_all_phases(t, res_fp_act, res_op_act, res_up_act, prediction, sensory, buffer)
    plot_mismatch_responses(t, res_fp_act, res_op_act, res_up_act, prediction, sensory, buffer, save_name='actNDNF', save=save)

    # inactivate NDNF interneurons and simulate again
    xFF_NDNF = -1
    rN0 = 2
    p0 = model.g_func(rN0)
    xFF['N'] = xFF_NDNF_bl + xFF_NDNF
    t, res_fp_inact, res_op_inact, res_up_inact, _ = run_pc_phases(dur, model, xFF, rN0=rN0, p0=p0, dt=dt, calc_bg_input=False, scale_w_by_p=False)
    plot_all_variables_all_phases(t, res_fp_inact, res_op_inact, res_up_inact, prediction, sensory, buffer)

    # plot change in inh and exc inputs to PCs
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

    # plot feedback, mismatch and playback response for different levels of NDNF activation

    ndnf_act_levels = np.arange(0, 1.6, 0.2)
    fb_response = np.zeros(len(ndnf_act_levels))
    mm_response = np.zeros(len(ndnf_act_levels))
    pb_response = np.zeros(len(ndnf_act_levels))

    for j, ndnf_act in enumerate(ndnf_act_levels):
        print(f"Running simulation with NDNF activation level {ndnf_act:1.1f}...")

        xFF['N'] = xFF_NDNF_bl + ndnf_act
        t, res_fp, res_op, res_up, _ = run_pc_phases(dur, model, xFF, rN0=rN0, p0=p0, dt=dt, calc_bg_input=False, scale_w_by_p=False)
        fb_response[j] = np.mean(res_fp['rE'][buffer:buffer+dur_stim])-np.mean(res_fp['rE'][:buffer])
        mm_response[j] = np.mean(res_op['rE'][buffer:buffer+dur_stim])-np.mean(res_op['rE'][:buffer])
        pb_response[j]  = np.mean(res_up['rE'][buffer:buffer+dur_stim])-np.mean(res_up['rE'][:buffer])

    fig2, ax2 = plt.subplots(1, 1, dpi=DPI, figsize=(2.5, 1.), gridspec_kw={'right': 0.95, 'left': 0.2, 'bottom': 0.35, 'top': 0.94})
    ax2.plot(ndnf_act_levels, mm_response, c=cpred, lw=1, ls='-', marker='.', label='mismatch')
    ax2.plot(ndnf_act_levels, pb_response, c=csens, lw=1, ls='-', marker='.', label='playback')
    ax2.plot(ndnf_act_levels, fb_response, c='k', ls='-', marker='.', lw=1, label='feedback')
    ax2.set(xlabel='NDNF activation', ylabel=r'$\Delta$ PC act.', ylim=[-0.05, 0.7], yticks=[0, 0.5], xticks=[0, 1])
    # ax.pcolormesh(np.array([fb_response, mm_response, pb_response]))


    if save:
        fig.savefig(f"../results/figs/Naumann23_draft1/exp5-3_prediction-error_inh_exc.pdf", dpi=300)
        fig2.savefig(f"../results/figs/Naumann23_draft1/exp5-4_prediction-error_varyNDNF.pdf", dpi=300)
        [plt.close([ff]) for ff in [fig, fig2]]



    # plot change in pathway activity to PCs in fp phase
    som_inh_d, ndnf_inh_d, pred_exc_d, pv_inh_s, sens_exc_s = get_exc_and_inh_inputs(model, res_fp, xFF, buffer, buffer+dur_stim)
    som_inh_d_act, ndnf_inh_d_act, pred_exc_d_act, pv_inh_s_act, sens_exc_s_act = get_exc_and_inh_inputs(model, res_fp_act, xFF, buffer, buffer+dur_stim)

    fig, ax = plt.subplots(1, 1, dpi=DPI, figsize=(2, 2))

    # plot stacked bar plot
    # ax.bar(0, pred_exc_d, color=cpred, width=0.5)
    ax.bar(0, ndnf_inh_d_act-ndnf_inh_d, bottom=0, color=cNDNF, width=0.5)
    ax.bar(0.5, som_inh_d_act-som_inh_d, color=cSOM, width=0.5)
    ax.bar(1, (ndnf_inh_d_act+som_inh_d_act)-(ndnf_inh_d+som_inh_d), bottom=0, color='silver', width=0.5)

    # som_inh = (model.Ws['DS'] @ np.mean(res_fp['rS'][buffer:buffer+dur_stim], axis=1)).mean()



    # # if save:
    #     # fig5.savefig('../results/figs/Naumann23_draft1/exp5-1_prediction-error.pdf', dpi=300)


def run_pc_phases(dur, model, xFF, rE0=1, rD0=0, rS0=4, rP0=4, rV0=4, rN0=4, p0=0.5, dt=1, calc_bg_input=True, scale_w_by_p=True, p_scale=None):

    # split FF input into three phases
    xFFfp = slice_dict(xFF, 0, dur)
    xFFop = slice_dict(xFF, dur, 2*dur)
    xFFup = slice_dict(xFF, 2*dur, 3*dur)

        # run simulation
    t, rEfp, rDfp, rSfp, rNfp, rPfp, rVfp, pfp, cGABAfp, otherfp = model.run(dur, xFFfp, dt=dt, rE0=rE0, rP0=rP0, rS0=rS0, rV0=rV0, rN0=rN0, rD0=rD0, p0=p0,
                                                                            calc_bg_input=calc_bg_input, scale_w_by_p=scale_w_by_p, init_noise=0,
                                                                            monitor_dend_inh=True, p_scale=p_scale)
    t, rEop, rDop, rSop, rNop, rPop, rVop, pop, cGABAop, otherop = model.run(dur, xFFop, dt=dt, rE0=rE0, rP0=rP0, rS0=rS0, rV0=rV0, rN0=rN0, rD0=rD0, p0=p0,
                                                                            calc_bg_input=calc_bg_input, scale_w_by_p=scale_w_by_p, init_noise=0,
                                                                            monitor_dend_inh=True, p_scale=p_scale)
    t, rEup, rDup, rSup, rNup, rPup, rVup, pup, cGABAup, otherup = model.run(dur, xFFup, dt=dt, rE0=rE0, rP0=rP0, rS0=rS0, rV0=rV0, rN0=rN0, rD0=rD0, p0=p0,
                                                                            calc_bg_input=calc_bg_input, scale_w_by_p=scale_w_by_p, init_noise=0,
                                                                            monitor_dend_inh=True, p_scale=p_scale)
    
    res_fp = dict(rE=rEfp, rD=rDfp, rS=rSfp, rN=rNfp, rP=rPfp, rV=rVfp, p=pfp, cGABA=cGABAfp, other=otherfp)
    res_op = dict(rE=rEop, rD=rDop, rS=rSop, rN=rNop, rP=rPop, rV=rVop, p=pop, cGABA=cGABAop, other=otherop)
    res_up = dict(rE=rEup, rD=rDup, rS=rSup, rN=rNup, rP=rPup, rV=rVup, p=pup, cGABA=cGABAup, other=otherup)

    bg_inputs_calc = model.Xbg

    return t/1000, res_fp, res_op, res_up, bg_inputs_calc


def get_s_and_p_inputs(amp, dur_stim, buffer, nt):

    fp_s = buffer
    op_s = fp_s+buffer*2+dur_stim
    up_s = op_s+buffer*2+dur_stim
    sensory = np.zeros(nt)
    sensory[fp_s:fp_s+dur_stim] = amp
    sensory[up_s:up_s+dur_stim] = amp
    prediction = np.zeros(nt)
    prediction[fp_s:fp_s+dur_stim] = amp
    prediction[op_s:op_s+dur_stim] = amp

    return sensory, prediction
    

def get_exc_and_inh_inputs(model, res, xFF, ts, te):

    som_inh_d = np.array(res['other']['dend_inh_SOM'][ts:te]).mean()
    ndnf_inh_d = np.array(res['other']['dend_inh_NDNF'][ts:te]).mean()
    pred_exc_d = xFF['D'][ts:te].mean()
    pv_inh_s = (model.Ws['EP'] @ np.mean(res['rP'][ts:te], axis=0)).mean()
    sens_exc_s = xFF['E'][ts:te].mean()

    return som_inh_d, ndnf_inh_d, pred_exc_d, pv_inh_s, sens_exc_s


def plot_all_variables_all_phases(t, res_fp, res_op, res_up, prediction, sensory, buffer):

    # plot overview of all cell activities in all conditions
    fig, ax = plt.subplots(6, 3, dpi=DPI, figsize=(4, 3), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1, 1, 0.6, 0.6]}, sharey='row')
    titles = ['P=S', 'P>S', 'P<S']

    dur = len(t)

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


def plot_mismatch_responses(t, res_fp, res_op, res_up, prediction, sensory, buffer, save_name='default', save=False):

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

    # scale bar:
    # ax5[2, 2].plot([2, 4], [-0.6, -0.6], c='k', lw=2)
    # from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    # add rectangle to axis
    ax[2, 2].add_patch(plt.Rectangle((2, -0.6), 2, 0.2, facecolor='k', edgecolor='k', lw=0))
    ax[2, 2].text(3, -0.7, '2s', ha='center', va='top', color='k', fontsize=8)  
    # scalebar = AnchoredSizeBar(ax5[2, 2].transData, 2, '2s', (), frameon=False)
    # ax5[2, 2].add_artist(scalebar)

    if save:
        fig.savefig(f"../results/figs/Naumann23_draft1/exp5-1_prediction-error_{save_name}.pdf", dpi=300)
        plt.close(fig)



def plot_changes_bars(t, res_fp, res_op, res_up, prediction, sensory, buffer, dur_stim):

    fig, ax = plt.subplots(3, 1, dpi=DPI, figsize=(3.8, 2.5), gridspec_kw={'height_ratios': [1, 1, 1], 'wspace': 0.2, 'hspace': 0.2, 'right':0.8, 'top':0.95})

    for i, cond in enumerate(['fp', 'op', 'up']):

        res = eval('res_'+cond)

        # plot changes in neural responses
        ax[0].bar(i, np.mean(res['rE'][buffer:buffer+dur_stim])-np.mean(res['rE'][:buffer]), color=cPC, width=0.3)
        ax[1].bar(i-0.3, np.mean(res['rN'][buffer:buffer+dur_stim])-np.mean(res['rN'][:buffer]), color=cNDNF, width=0.2, label='NDNF' if i==0 else None)
        ax[1].bar(i-0.1, np.mean(res['rS'][buffer:buffer+dur_stim])-np.mean(res['rS'][:buffer]), color=cSOM, width=0.2, label='SOM' if i==0 else None)
        ax[1].bar(i+0.1, np.mean(res['rP'][buffer:buffer+dur_stim])-np.mean(res['rP'][:buffer]), color=cPV, width=0.2, label='PV' if i==0 else None)
        ax[1].bar(i+0.3, np.mean(res['rV'][buffer:buffer+dur_stim])-np.mean(res['rV'][:buffer]), color=cVIP, width=0.2, label='VIP' if i==0 else None)

        # change in dendritic inhibition
        dend_inh_SOM = np.array(res['other']['dend_inh_SOM']).mean(axis=1)
        dend_inh_NDNF = np.array(res['other']['dend_inh_NDNF']).mean(axis=1)
        ddi_SOM = np.mean(dend_inh_SOM[buffer:buffer+dur_stim])-np.mean(dend_inh_SOM[:buffer])
        ddi_NDNF = np.mean(dend_inh_NDNF[buffer:buffer+dur_stim])-np.mean(dend_inh_NDNF[:buffer])
        ax[2].bar(i-0.2, ddi_NDNF, facecolor='none', edgecolor=cNDNF, hatch='/////', width=0.2, label='NDNF' if i==0 else None)
        ax[2].bar(i, ddi_SOM, facecolor='none', edgecolor=cSOM, hatch='/////', width=0.2, label='SOM' if i==0 else None)
        ax[2].bar(i+0.2, ddi_SOM+ddi_NDNF, facecolor='none', edgecolor='silver', hatch='/////', width=0.2, label='sum' if i==0 else None)
        
        # draw zero lines
        ax[0].hlines(0, -0.5, 2.5, color='k', lw=1)
        ax[1].hlines(0, -0.5, 2.5, color='k', lw=1)
        ax[2].hlines(0, -0.5, 2.5, color='k', lw=1)

        # legend
        ax[1].legend(loc=(1.01, 0.1), frameon=False, handlelength=1)
        ax[2].legend(loc=(1.01, 0.1), frameon=False, handlelength=1, title='dend. inh.')
        [ax[k].spines['bottom'].set_visible(False) for k in range(3)]

        # labels
        ax[0].set(ylabel=r'$\Delta$ PC act.', xticks=[], ylim=[-0.2, 0.5], yticks=[0, 0.5])
        ax[1].set(ylabel=r'$\Delta$ IN act.', xticks=[], ylim=[-2.1, 2.1], yticks=[-2, 0, 2])
        ax[2].set(ylabel=r'$\Delta$ inh.', xticks=[0, 1, 2], ylim=[-4, 4.5], yticks=[-3, 0, 3], xticklabels=['P=S', 'P>S', 'P<S'])

        
if __name__ in "__main__":

    exp501_predictive_coding(mean_pop=False, w_hetero=True, reduced=False, pre_inh=True, noise=0.1, with_NDNF=True, xFF_NDNF=0, rNinit=4, calc_bg_input=True,
                            scale_w_by_p=True, save=True, b=0.15)

    plt.show()


