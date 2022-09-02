import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets

plt.style.use('pretty')
import matplotlib

# matplotlib.use('MACOSX')

colSOM = '#4084BF'
colNDNF = '#EF8961'


class NDNF_SOM_only:
    def __init__(self, wNS=1, tauN=15, tauS=20, taup=100, b=0.3, p0=0.1, r0=0, ba=0, tau_a=100, **kwargs):

        # SOM->NDNF inhibition
        self.wNS = wNS

        # timescales
        self.tauN = tauN
        self.tauS = tauS

        # presynaptic inhibition
        self.taup = taup
        self.b = b
        self.p0 = p0
        self.r0 = r0

        # adaptation
        self.ba = ba
        self.tau_a = tau_a

    def g_func(self, r):
        return np.clip(1 - self.b * (r - self.r0), self.p0, 1)

    def run(self, dur, xN, xS, rN_init=0, rS_init=0, p_init=1, dt=1, store_currents=False):

        t = np.arange(0, dur, dt)

        # make sure inputs to NDNF and SOM are arrays
        if not hasattr(xN, '__len__'):
            xN = np.ones(len(t)) * xN
        if not hasattr(xS, '__len__'):
            xS = np.ones(len(t)) * xS

        # arrays for state variables and initial states
        rN = np.zeros(len(t))
        rS = np.zeros(len(t))
        p = np.zeros(len(t))
        rN[0] = rN_init
        rS[0] = rS_init
        p[0] = p_init
        a = rS_init

        if store_currents:
            store_inp_curr_N = [0]
            store_inp_curr_S = [0]

        # time integration
        for i, ti in enumerate(t[:-1]):

            # compute input currents
            inp_curr_N = xN[i] - p[i] * self.wNS * rS[i]
            inp_curr_S = xS[i]

            # compute state change
            drN = (-rN[i] + inp_curr_N) / self.tauN
            drS = (-rS[i] + inp_curr_S - a) / self.tauS
            dp = (-p[i] + self.g_func(rN[i])) / self.taup

            # perform Euler integration step
            rN[i + 1] = np.maximum(rN[i] + drN * dt, 0)
            rS[i + 1] = np.maximum(rS[i] + drS * dt, 0)
            p[i + 1] = p[i] + dp * dt
            a += (-a + self.ba*rS[i])/self.tau_a*dt

            if store_currents:
                store_inp_curr_N.append(inp_curr_N)
                store_inp_curr_S.append(inp_curr_S)

        if store_currents:
            return t, None, rN, rS, p, store_inp_curr_N, store_inp_curr_S
        else:
            return t, None, rN, rS, p


class NDNF_SOM_PC:

    def __init__(self, wNS=1, wSE=0.8, tauN=15, tauS=20, tauE=20, taup=100, b=0.3, p0=0.1, r0=0.0,  ba=1, tau_a=100,
                 U_S=0.1, tauf=200, tauD=20, wED=0.7, wDN=0.4, wDS=0.5, expS=1.2, expN=0.3,
                 alphaS=0, alphaN=0.1, alphaE=1, wNN=0.5, U_N=0.9, taud=200, dt=1):

        # weights
        self.wNS = wNS  # SOM->NDNF inhibition
        self.wSE = wSE  # PC->SOM excitation
        self.wED = wED  # dendrite-soma coupling
        self.wDN = wDN  # NDNF->dendrite inhibition
        self.wDS = wDS  # SOM->dendrite inhibition
        self.wNN = wNN  # NDNF self-inhibition

        # timescales
        self.tauN = tauN                                
        self.tauS = tauS
        self.tauE = tauE
        self.tauD = tauD

        # presynaptic inhibition
        self.taup = taup
        self.b = b
        self.p0 = p0
        self.r0 = r0

        # transfer function parameters
        self.tfS = {'k': 1, 'rheo': 0, 'n': 1}
        self.tfN = {'k': 1, 'rheo': 0, 'n': 1}
        self.tfE = {'k': 1, 'rheo': 0, 'n': 1}
        self.tfD = {'k': 1, 'rheo': 0, 'n': 1}

        # FF input strength
        self.alphaS = alphaS
        self.alphaN = alphaN
        self.alphaE = alphaE

        # adaptation (of SOMs)
        self.ba = ba
        self.tau_a = tau_a

        # facilitation (of SOM inputs)
        self.U_S = U_S
        self.tauf = tauf
        if self.U_S > 0:
            self.wSE /= self.U_S

        # depression (of NDNF inputs)
        self.U_N = U_N
        self.taud = taud

        # nonlinearities of FF input
        self.expS = expS  # SOM  # ToDo: SOMs do not get direct FF input
        self.expN = expN  # NDNF

        # integration
        self.dt = dt  # integration timestep

    def g_func(self, r):
        """
        Presynaptic inhibition transfer function.
        :param r: input rate
        :return: release probability p
        """
        return np.clip(1 - self.b * (r - self.r0), self.p0, 1)

    def f_func(self, v, k=1, rheo=0, n=1):
        """
        Neuron transfer function.
        :param v: 'activity' variable
        :param k: slope of transfer function
        :param rheo: rheobase of transfer function
        :param n: exponent of transfer function
        :return:
        """

        return k*(np.maximum(v - rheo, 0))**n

    def run(self, dur, xN, xS, xFF=0, xE=0, xD=0.5, rN_init=0, rS_init=0, rE_init=0, p_init=1,
            store_currents=False):

        dt = self.dt

        t = np.arange(0, dur, dt)

        # make sure inputs are arrays
        if not hasattr(xFF, '__len__'):
            xFF = np.ones(len(t)) * xFF
        if not hasattr(xN, '__len__'):
            xN = np.ones(len(t)) * xN
        if not hasattr(xS, '__len__'):
            xS = np.ones(len(t)) * xS
        if not hasattr(xE, '__len__'):
            xE = np.ones(len(t)) * xE

        # arrays for state variables and initial states
        rN = np.zeros(len(t))
        rS = np.zeros(len(t))
        rE = np.zeros(len(t))
        rD = np.zeros(len(t))
        p = np.zeros(len(t))
        rN[0] = rN_init
        rS[0] = rS_init
        rE[0] = rE_init
        rD[0] = 0
        p[0] = p_init

        uS = 1
        hN = 1
        a = 0

        if self.U_S > 0:
            uS = self.U_S
        # else:
        #     uS = 0.1
            # self.wSE /= self.U_S

        uS_store = [uS]

        if store_currents:
            store_inp_curr_N = [0]
            store_inp_curr_S = [0]

        # inputs pre rectification
        vS, vN, vE, vD = rS[0], rN[0], rE[0], rD[0]

        # time integration
        for i, ti in enumerate(t[:-1]):

            # compute input currents
            input_curr_N = - p[i] * self.wNS * rS[i] - self.wNN * rN[i] + xN[i] \
                           + (hN * self.U_N * self.alphaN * xFF[i]) ** self.expN
            # + self.alphaN*xFF[i] + np.heaviside(xFF[i]-0.5, 0)
            input_curr_S = uS * self.wSE * rE[i] + xS[i] + self.alphaS * xFF[i]

            # compute state change of "subthreshold" activity
            vS += (-vS + input_curr_S - a) / self.tauS * dt
            vN += (-vN + input_curr_N) / self.tauN * dt
            vE += (-vE + self.wED * rD[i] + xE[i] + self.alphaE * xFF[i]) / self.tauE * dt
            vD += (-vD - p[i] * self.wDS*rS[i] - self.wDN * rN[i] + xD) / self.tauD * dt

            # integrate presynaptic inhibition, adaptation etc
            p[i+1] = p[i] + (-p[i] + self.g_func(rN[i])) / self.taup * dt
            duS = (self.U_S-uS)/self.tauf + self.U_S*(1-uS)*rE[i]/1000
            uS = (uS + duS*dt) if self.U_S > 0 else 1
            a += (-a + self.ba*rS[i]) / self.tau_a * dt
            dhN = -self.U_N*xFF[i]*hN/1000 + (1-hN)/self.taud
            hN = (hN + dhN*dt) if self.U_N < 1 else 1

            # rectification / transfer function
            rS[i + 1] = self.f_func(vS, **self.tfS)
            rN[i + 1] = self.f_func(vN, **self.tfN)
            rE[i + 1] = self.f_func(vE, **self.tfE)
            rD[i + 1] = self.f_func(vD, **self.tfD)

            uS_store.append(uS)

            if store_currents:
                store_inp_curr_N.append(input_curr_N)
                store_inp_curr_S.append(input_curr_S)

        if store_currents:
            return t, rE, rD, rN, rS, p, store_inp_curr_N, store_inp_curr_S
        else:
            return t, rE, rD, rN, rS, p


def experiment1_invitro_currents(act_NDNF=2, act_SOM=2, bg_NDNF=0, bg_SOM=0, plot_p=False, save=False):

    """
    Experiment: Stimulate NDNF/SOM and record input currents of NDNF and SOM.
                Function performs experiment and plots results

    :param act_NDNF: strength of activation of NDNFs
    :param act_SOM:  strength of activation SOMs
    :param bg_NDNF:  background input to NDNFs
    :param bg_SOM:   background input to SOMs
    :return:
    """

    dpi = 300 if save else 150

    # model = NDNF_SOM_only()
    model = NDNF_SOM_PC(ba=1, U_S=0.0)

    dur = 300
    dt = 1

    act_on, act_off = 100, 150

    input_act_NDNF = np.ones(int(dur / dt))*bg_NDNF
    input_act_NDNF[act_on:act_off] += act_NDNF

    input_act_SOM = np.ones(int(dur / dt))*bg_SOM
    input_act_SOM[act_on:act_off] += act_SOM

    t, rE0, _, rN0, rS0, p0, curr_N_0, curr_S_0 = model.run(dur, bg_NDNF, input_act_SOM, xD=0, store_currents=True)
    t, rE1, _, rN1, rS1, p1, curr_N_1, curr_S_1 = model.run(dur, input_act_NDNF, bg_SOM, xD=0, store_currents=True)

    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, dpi=dpi, figsize=(4, 3),
                           gridspec_kw={'bottom': 0.15})
    ax[0, 0].plot(t, curr_N_0, c=colNDNF)
    ax[0, 1].plot(t, curr_N_1, c=colNDNF)
    ax[1, 0].plot(t, curr_S_0, c=colSOM)
    ax[1, 1].plot(t, curr_S_1, c=colSOM)

    [axx.hlines(-2.1, act_on, act_off, linewidth=3, color=[colSOM, colNDNF][ii % 2], alpha=0.5)
     for ii, axx in enumerate(ax.flatten())]

    ax[0, 0].set(ylabel='input curr. NDNF (au)', title='activate SOM')
    ax[0, 1].set(title='activate NDNF')
    ax[1, 0].set(ylabel='input curr. SOM (au)', xlabel='time (ms)')
    ax[1, 1].set(xlabel='time (ms)')

    # plt.suptitle("Whole-cell paired recordings 'in-vitro'", fontsize=10)

    if save:
        fig.savefig('../figs/in-silico_unidirectional-inh.png', dpi=dpi)

    if plot_p:
        fig2, ax2 = plt.subplots(1, 2, figsize=(4, 2), dpi=150, sharex=True, sharey=True, gridspec_kw={'bottom': 0.2})
        ax2[0].plot(t, p0, c='C2')
        ax2[1].plot(t, p1, c='C2')
        ax2[0].set(ylabel='p', xlabel='time (ms)', ylim=[-0.05, 1.05])
        ax2[1].set(xlabel='time (ms)')


def experiment2_invivo_sensory_stim(bg_SOM=0, bg_NDNF=0.5, tau_stim=100, save=False):

    dpi = 300 if save else 150

    dur = 1200
    dt = 1
    t = np.arange(0, dur, dt)

    tau1 = 150
    tau2 = 50
    ts = 400
    te = 700

    # act_on, act_off = 300, 700

    fig, ax = plt.subplots(4, 1, dpi=dpi, figsize=(4, 4), sharex=True, gridspec_kw={'bottom': 0.12})
    # plt.suptitle("Sensory stimulation & Ca-imaging 'in-vivo' (Abs18, F5G)", fontsize=10)
    ax[0].set(ylabel='NDNF act. (au)')#, yticks=[0, 1])
    ax[1].set(ylabel='SOM act. (au)')
    ax[2].set(ylabel='stimulus (au)', xlabel='time (ms)')

    xff_list = [1, 2, 3, 4, 5]
    model = NDNF_SOM_PC(wNS=1.3, ba=1, U_S=0, U_N=1, expN=0.3, alphaN=1, b=1, wDS=0.5)
    # without pre inh: wNS=0.9, b=0

    resp_NDNF = []
    resp_SOM = []

    for i, xff in enumerate(xff_list):

        # construct feedforward input
        sensory_stim = 2 * np.maximum(xff * (np.exp(-(t - ts) / tau1) - np.exp(-(t - ts) / tau2)), 0)

        # sensory_stim = np.zeros(int(dur/dt))
        # sensory_stim[400:800] = xff
        # xFF[act_sxon:act_off] = xff*(1-np.exp(-(t[act_on:act_off]-act_on)/tau_stim))
        # xFF[act_off:] = xFF[act_off-1]*np.exp(-(t[act_off:]-act_off)/tau_stim)

        t, rE, rD, rN, rS, p = model.run(dur, bg_NDNF, bg_SOM, xFF=sensory_stim, xD=1)

        mean_ts = int(ts/model.dt)
        mean_te = int(te/model.dt)
        resp_NDNF.append(np.mean(rN[mean_ts:mean_te]))
        resp_SOM.append(np.mean(rS[mean_ts:mean_te]))

        ax[0].plot(t, rN, c=colNDNF, alpha=(i+1)/len(xff_list))
        ax[1].plot(t, rS, c=colSOM, alpha=(i+1)/len(xff_list))
        # ax[2].plot(t, rE, c='C3', alpha=(i+1)/len(xff_list))
        ax[2].plot(t, sensory_stim, c='#9B3146', alpha=(i+1)/len(xff_list))
        ax[3].plot(t, rD, c='k', alpha=(i+1)/len(xff_list))

    fig2, ax2 = plt.subplots(2, 1, figsize=(3, 3), dpi=dpi)
    ax2[0].plot(xff_list, resp_NDNF, '-o', c=colNDNF, ms=4)
    ax2[1].plot(xff_list, resp_SOM, '-o', c=colSOM, ms=4)
    ax2[0].set(ylabel='NDNF response')
    ax2[1].set(xlabel='sound pressure level', ylabel='SOM response')
    plt.tight_layout()

    if save:
        fig.savefig('../figs/in-silico_sensory-stim.png', dpi=dpi)
        plt.close(fig)

    # plt.figure()
    # plt.plot(xFF)


def experiment3_invivo_boutons(bg_SOM=1, bg_NDNF=1, act_NDNF=5, bg_PC=1, save=False):

    dpi = 300 if save else 150

    dur = 1200
    dt = 1

    act_on, act_off = 500, 800

    input_act_NDNF = np.ones(int(dur / dt))*bg_NDNF
    input_act_NDNF[act_on:act_off] += act_NDNF

    model = NDNF_SOM_PC(ba=1, wNS=0.9)
    t, rE, rD, rN, rS, p, curr_N, curr_S = model.run(dur, input_act_NDNF, bg_SOM, rS_init=bg_SOM, rN_init=bg_NDNF,
                                                 xE=bg_PC, store_currents=True)
    t, rE0, rD0, rN0, rS0, p0, curr_N0, curr_S0 = model.run(dur, bg_NDNF, bg_SOM,  rS_init=bg_SOM, rN_init=bg_NDNF,
                                                       xE=bg_PC, store_currents=True)

    fig, ax = plt.subplots(2, 1, dpi=dpi, figsize=(2.5, 4), gridspec_kw={'bottom': 0.1, 'hspace': 0.5, 'left': 0.15})

    ax[0].plot(t[200:], (p*rS)[200:], c=colSOM)
    ax[0].plot(t[200:], (p0*rS0)[200:], ':', c=colSOM)

    ax[0].hlines(0.1, act_on, act_off, lw=3, color=colNDNF, alpha=0.5)

    mean_act = (p*rS)[act_on:act_off].mean()
    mean_ctrl = (p0*rS0)[act_on:act_off].mean()

    ax[1].plot([0, 1], [mean_ctrl, mean_act], c=colSOM)
    ax[1].plot([0, 1], [mean_ctrl, mean_act], '.', c=colSOM)

    ax[0].set(xlabel='time (ms)', ylim=[0, 1.2], ylabel='SOM bouton act (au)', yticks=[0, 1])
    ax[1].set(xlim=[-0.5, 1.5], ylim=[0, 1.2], xticks=[0, 1], xticklabels=['ctrl', 'NDNF stim'],
              ylabel='SOM bouton act (mean)', yticks=[0, 1])
    plt.suptitle("Imaging of SOM boutons during NDNF activation 'in-vivo'", fontsize=10)

    if save:
        fig.savefig('../figs/in-silico_bouton-imag.png', dpi=dpi)
        plt.close()


def bistability(save=False):

    dpi = 300 if save else 150

    ba = 1

    model = NDNF_SOM_only(ba=ba, wNS=1, p0=0)

    dur = 3000
    dt = 1

    bg_SOM = 8
    bg_NDNF = 4
    act_NDNF = 3
    inact_NDNF = -3

    rN_init = 0
    rS_init = bg_SOM / (1+ba)

    act_on, act_off = 500, 1000
    inact_on, inact_off = 2000, 2500
    input_act_NDNF = np.ones(int(dur / dt))*bg_NDNF
    input_act_NDNF[act_on:act_off] += act_NDNF

    input_act_NDNF[inact_on:inact_off] += inact_NDNF

    t, rE, rN, rS, p = model.run(dur, input_act_NDNF, bg_SOM, rS_init=rS_init, rN_init=rN_init,
                                 p_init=model.g_func(rN_init))

    fig, ax = plt.subplots(3, 1, sharex=True, dpi=dpi, figsize=(4, 3), gridspec_kw={'bottom': 0.15, 'top':0.95})
    ax[0].plot(t/1000, rN, c=colNDNF)
    ax[1].plot(t/1000, rS, '--', c=colSOM)
    ax[1].plot(t/1000, rS*p, c=colSOM)
    ax[2].plot(t/1000, p, c='#ACC17D')
    ax[0].set(ylim=[-0.1, 7], ylabel='NDNF act. (au)', yticks=[0, 3, 6])
    ax[1].set(ylim=[-0.1, 7], ylabel='SOM act. (au)', yticks=[0, 3, 6])
    ax[2].set(ylim=[-0.05, 1.05], ylabel='rel. prob.', xlabel='time (s)', yticks=[0, 1])

    if save:
        fig.savefig('../figs/bistability_SOM_NDNF.png', dpi=dpi)
        plt.close()

if __name__ in "__main__":
    # experiment1_invitro_currents(plot_p=True, bg_NDNF=0, bg_SOM=0, save=False)
    experiment2_invivo_sensory_stim(save=False)
    # experiment3_invivo_boutons(bg_SOM=1, act_NDNF=3, bg_NDNF=1, save=True)
    # bistability(save=True)
    plt.show()

