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

def exp501_predictive_coding(mean_pop=True, w_hetero=False, reduced=False, pre_inh=False, noise=0.0, save=False, with_NDNF=True, xFF_NDNF=0, rNinit=0, calc_bg_input=False):

    # define parameter dictionaries
    N_cells, w_mean, conn_prob, bg_inputs, taus = mb.get_default_params(flag_mean_pop=True)

    # set parameters to get negative prediction errors
    w_mean_update = dict(EP=2, DE=0., DS=1, PE=1.2, PP=0.4, PS=0.3, PV=0.15, SE=1, SV=0.5, VE=1, VS=1)
    w_mean_update.update(dict(NS=0.7, DN=0.8, PN=0.1, VN=0.1))

    # w_mean_update.update(dict(PV=0.25, PN=0.02))

    # update mean weights
    w_mean.update(w_mean_update)

    # set bg input
    bg_inputs = {'E': 9, 'D': 4., 'N': 2.8, 'S': 5.0, 'P': 6.2, 'V': 7.0}
        
    # initial activities
    rE0, rD0 = 1, 0
    rP0, rS0, rV0 = 4, 4, 4
    # rN0 = 0

    # simulation parameters
    dur_tot = 14000  # ms
    dur_stim = 2000
    dt = 1  # ms
    t = np.arange(0, dur_tot, dt)
    nt = len(t)

    # construct input: fully prediced, overpredicted, underpredicted
    # TODO: make this a function?
    amp = 1
    fp_s, fp_e = 2000, 2000+dur_stim
    op_s, op_e = 6000, 6000+dur_stim
    up_s, up_e = 10000, 10000+dur_stim
    sensory = np.zeros(nt)
    sensory[fp_s:fp_e] = amp
    sensory[up_s:up_e] = amp
    prediction = np.zeros(nt)
    prediction[fp_s:fp_e] = amp
    prediction[op_s:op_e] = amp

    # set input of cells
    xFF = get_null_ff_input_arrays(nt, N_cells)
    xFF['E'] = np.tile(sensory, [N_cells['E'], 1]).T
    xFF['D'] = np.tile(prediction, [N_cells['D'], 1]).T
    xFF['P'] = np.tile(sensory, [N_cells['P'], 1]).T
    xFF['S'] = np.tile(sensory, [N_cells['S'], 1]).T
    xFF['V'] = np.tile(prediction, [N_cells['V'], 1]).T
    xFF['N'] = np.tile(prediction, [N_cells['N'], 1]).T

    xFF_NDNF_bl = xFF['N']

    # give extra input to NDNF interneurons to get "prediction neurons" ( if input to NDNF is > 0)
    # xFF['N'] += xFF_NDNF

    # run network, simulate 3 conditions separately to avoid long-lasting effects
    dur = 6000 # duration per condition
    buffer = 2000


    calc_bg_input = calc_bg_input
    scale_w_by_p = False

    # fig3, ax3 = plt.subplots(2, 3, figsize=(4, 2), dpi=DPI, sharex=True, gridspec_kw={'height_ratios': [1, 1], 'wspace': 0.15, 'hspace': 0.4}, sharey='row')
    ls_list = ['-', '-']
    alpha_list = [1, 1]

    for j, with_NDNF_i in enumerate([False, True]):

        if not with_NDNF_i:
            bg_inputs['N'] = 0
            rN0 = 0
            pre_inh = False
            calc_bg_input = True
            xFF['N'] = xFF_NDNF_bl
        else:
            rN0 = rNinit
            bg_inputs['N'] = 2.5
            # bg_inputs['V'] = 7.2
            pre_inh = True
            calc_bg_input = False
            xFF['N'] = xFF_NDNF_bl + xFF_NDNF

        xFFfp = slice_dict(xFF, fp_s-buffer, fp_e+buffer)
        xFFop = slice_dict(xFF, op_s-buffer, op_e+buffer)
        xFFup = slice_dict(xFF, up_s-buffer, up_e+buffer)
       
        # instantiate model
        model = mb.NetworkModel(N_cells, w_mean.copy(), conn_prob, taus, bg_inputs.copy(), wED=1, flag_w_hetero=w_hetero, flag_pre_inh=pre_inh, flag_with_NDNF=with_NDNF_i,
                                flag_with_VIP=True, flag_with_PV=True)
        if pre_inh:
            model.b = 0.2
        p0 = model.g_func(rN0) #0.3

        # run simulation
        t, rEfp, rDfp, rSfp, rNfp, rPfp, rVfp, pfp, cGABAfp, otherfp = model.run(dur, xFFfp, dt=dt, rE0=rE0, rP0=rP0, rS0=rS0, rV0=rV0, rN0=rN0, rD0=rD0, p0=p0,
                                                                               calc_bg_input=calc_bg_input, scale_w_by_p=scale_w_by_p, init_noise=0,
                                                                               monitor_dend_inh=True)
        t, rEop, rDop, rSop, rNop, rPop, rVop, pop, cGABAop, otherop = model.run(dur, xFFop, dt=dt, rE0=rE0, rP0=rP0, rS0=rS0, rV0=rV0, rN0=rN0, rD0=rD0, p0=p0,
                                                                               calc_bg_input=calc_bg_input, scale_w_by_p=scale_w_by_p, init_noise=0,
                                                                               monitor_dend_inh=True)
        t, rEup, rDup, rSup, rNup, rPup, rVup, pup, cGABAup, otherup = model.run(dur, xFFup, dt=dt, rE0=rE0, rP0=rP0, rS0=rS0, rV0=rV0, rN0=rN0, rD0=rD0, p0=p0,
                                                                               calc_bg_input=calc_bg_input, scale_w_by_p=scale_w_by_p, init_noise=0,
                                                                               monitor_dend_inh=True)

        print(model.Xbg)
        # print(model.Ws)
        # print(w_mean)
        t = t/1000

        # fig for bar plots
        fig4, ax4 = plt.subplots(3, 1, dpi=DPI, figsize=(3.8, 2.5), gridspec_kw={'height_ratios': [1, 1, 1], 'wspace': 0.2, 'hspace': 0.2, 'right':0.8, 'top':0.95})

        # plot
        fig2, ax2 = plt.subplots(4, 3, figsize=(3, 2.5), dpi=DPI, sharex=True, gridspec_kw={'height_ratios': [0.4, 0.4, 1, 1], 'wspace': 0.15, 'hspace': 0.4, 'bottom': 0.2, 'right': 0.95})
        for i, cond in enumerate(['fp', 'op', 'up']):
            ax2[2, i].plot(t, eval('rE'+cond), c=cPC, ls=ls_list[j], alpha=alpha_list[j])
            ax2[3, i].plot(t, eval('rD'+cond), c='k', ls=ls_list[j], alpha=alpha_list[j])
            ax2[0, i].plot(t, prediction[eval(cond+'_s')-buffer:eval(cond+'_e')+buffer], c=cpred)
            ax2[1, i].plot(t, sensory[eval(cond+'_s')-buffer:eval(cond+'_e')+buffer], c=csens)

            ax2[2, i].set(ylim=[-0.1, 2])
            ax2[3, i].set(ylim=[-0.1, 2], xticks=[0, 2, 4, 6], xlabel='time (s)')
            ax2[0, i].set(ylim=[-0.1, 1.3])
            ax2[1, i].set(ylim=[-0.1, 1.3])
            ax2[0, i].set(yticks=[])
            ax2[1, i].set(yticks=[])
            ax2[0, i].axis('off')
            ax2[1, i].axis('off') #spines['left'].set_visible(False)

            # plot inhibition to dendrite by SOM and NDNF
            dend_inh_SOM = np.array(eval('other'+cond+"['dend_inh_SOM']")).mean(axis=1)
            dend_inh_NDNF = np.array(eval('other'+cond+"['dend_inh_NDNF']")).mean(axis=1)
            # ax3[0,i].plot(t, dend_inh_SOM, c=cSOM, ls=ls_list[j], alpha=alpha_list[j])
            # ax3[0,i].plot(t, dend_inh_NDNF, c=cNDNF, ls=ls_list[j], alpha=alpha_list[j])
            # ax3[1,i].plot(t, dend_inh_SOM+dend_inh_NDNF, c='gray', ls=ls_list[j], alpha=alpha_list[j])

            # plot bars for change in activities and dendritic inhibition
            re_i = eval('rE'+cond)
            rs_i = eval('rS'+cond)
            rn_i = eval('rN'+cond)
            rp_i = eval('rP'+cond)
            rv_i = eval('rV'+cond)
            ax4[0].bar(i, np.mean(re_i[buffer:buffer+dur_stim])-np.mean(re_i[:buffer]), color=cPC, width=0.3)
            ax4[1].bar(i-0.3, np.mean(rn_i[buffer:buffer+dur_stim])-np.mean(rn_i[:buffer]), color=cNDNF, width=0.2, label='NDNF' if i==0 else None)
            ax4[1].bar(i-0.1, np.mean(rs_i[buffer:buffer+dur_stim])-np.mean(rs_i[:buffer]), color=cSOM, width=0.2, label='SOM' if i==0 else None)
            ax4[1].bar(i+0.1, np.mean(rp_i[buffer:buffer+dur_stim])-np.mean(rp_i[:buffer]), color=cPV, width=0.2, label='PV' if i==0 else None)
            ax4[1].bar(i+0.3, np.mean(rv_i[buffer:buffer+dur_stim])-np.mean(rv_i[:buffer]), color=cVIP, width=0.2, label='VIP' if i==0 else None)
            ddi_SOM = np.mean(dend_inh_SOM[buffer:buffer+dur_stim])-np.mean(dend_inh_SOM[:buffer])
            ddi_NDNF = np.mean(dend_inh_NDNF[buffer:buffer+dur_stim])-np.mean(dend_inh_NDNF[:buffer])
            ax4[2].bar(i-0.2, ddi_NDNF, facecolor='none', edgecolor=cNDNF, hatch='/////', width=0.2, label='NDNF' if i==0 else None)
            ax4[2].bar(i, ddi_SOM, facecolor='none', edgecolor=cSOM, hatch='/////', width=0.2, label='SOM' if i==0 else None)
            ax4[2].bar(i+0.2, ddi_SOM+ddi_NDNF, facecolor='none', edgecolor='silver', hatch='/////', width=0.2, label='sum' if i==0 else None)
            ax4[0].hlines(0, -0.5, 2.5, color='k', lw=1)
            ax4[1].hlines(0, -0.5, 2.5, color='k', lw=1)
            ax4[2].hlines(0, -0.5, 2.5, color='k', lw=1)

            if i != 0:
                ax2[2, i].spines['left'].set_visible(False)
                ax2[2, i].set(yticks=[])
                ax2[3, i].spines['left'].set_visible(False)
                ax2[3, i].set(yticks=[])

        ax2[2, 0].set(ylabel='PC', yticks=[0, 1, 2])
        ax2[3, 0].set(ylabel='dend.', yticks=[0, 1, 2])
            
        ax4[1].legend(loc=(1.01, 0.1), frameon=False, handlelength=1)
        ax4[2].legend(loc=(1.01, 0.1), frameon=False, handlelength=1, title='dend. inh.')
        [ax4[k].spines['bottom'].set_visible(False) for k in range(3)]
        # labels
        ax4[0].set(ylabel=r'$\Delta$ PC act.', xticks=[], ylim=[-0.2, 0.5], yticks=[0, 0.5])
        ax4[1].set(ylabel=r'$\Delta$ IN act.', xticks=[], ylim=[-2.1, 2.1], yticks=[-2, 0, 2])
        ax4[2].set(ylabel=r'$\Delta$ inh.', xticks=[0, 1, 2], ylim=[-2.1, 2.1], yticks=[-2, 0, 2], xticklabels=['P=S', 'P>S', 'P<S'])
        


    # # print stuff for debugging
    # ndnf_to_vip_inh = w_mean['VN']/(w_mean['VS']*w_mean['SV']-1)
    # ndnf_to_som_inh = -w_mean['SV']*w_mean['VN']/(w_mean['VS']*w_mean['SV']-1)
    # print(ndnf_to_vip_inh*w_mean['PV'])
    # print(ndnf_to_som_inh*w_mean['PS'])
    # print(w_mean['PN'])
    # print('delta NDNF', np.mean(rNfp[1500:2000]))
    # print('delta VIP', np.mean(rVfp[1500:2000])-4)
    # print('delta PV', np.mean(rPfp[1500:2000])-4)
    # print('rD', np.mean(rDfp[1500:2000]))
    # print('predicted delta VIP', ndnf_to_vip_inh*np.mean(rNfp[1500:2000]))
    # print('predicted delta PV', -w_mean['PV']*(np.mean(rVfp[1500:2000])-4)-w_mean['PN']*np.mean(rNfp[1500:2000]))

        # plot overview of all cell activities in all conditions
        fig, ax = plt.subplots(6, 3, dpi=DPI, figsize=(5, 4), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1, 1, 0.6, 0.6]}, sharey='row')
        titles = ['P=S', 'P>S', 'P<S']

        for i, cond in enumerate(['fp', 'op', 'up']):
            ax[0, i].plot(t, eval('rE'+cond), c=cPC)
            ax[1, i].plot(t, eval('rD'+cond), c='k')
            ax[2, i].plot(t, eval('rP'+cond), c=cPV)
            ax[2, i].plot(t, eval('rV'+cond), c='silver')
            ax[2, i].plot(t, eval('rS'+cond), c=cSOM)
            ax[2, i].plot(t, eval('rN'+cond), c=cNDNF)
            ax[3, i].plot(t, eval('p'+cond), c=cpi)
            ax[4, i].plot(t, prediction[eval(cond+'_s')-buffer:eval(cond+'_e')+buffer], c=cpred)
            ax[5, i].plot(t, sensory[eval(cond+'_s')-buffer:eval(cond+'_e')+buffer], c=csens)
            ax[0, i].set(title=titles[i])

        # labels
        ax[0, 0].set(ylabel='PC', ylim=[0, 2])
        ax[1, 0].set(ylabel='dend.', ylim=[0, 3])
        ax[2, 0].set(ylabel='INs', ylim=[0, 7])
        ax[3, 0].set(ylabel='p', ylim=[-0.05, 1.05])
        ax[4, 0].set(ylabel='sens.', ylim=[-0.05, 1.05])
        ax[5, 0].set(ylabel='pred.', xlabel='time', ylim=[-0.05, 1.05])



    # ax3[0, 0].set(ylabel='dend inh')
    # ax3[1, 0].set(ylabel='sum', xlabel='time (s)')


# exp501_predictive_coding(mean_pop=True, w_hetero=False, reduced=False, pre_inh=True, noise=0.0, save=False, with_NDNF=True, xFF_NDNF=0, rNinit=0, calc_bg_input=True)
exp501_predictive_coding(mean_pop=True, w_hetero=False, reduced=False, pre_inh=True, noise=0.0, save=False, with_NDNF=True, xFF_NDNF=2.8, rNinit=3.72, calc_bg_input=True)

plt.show()


