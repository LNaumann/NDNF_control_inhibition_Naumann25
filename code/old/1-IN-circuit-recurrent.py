import numpy as np
import matplotlib.pyplot as plt
plt.style.use('pretty')

dtype = np.float32


class Network:
    def __init__(self, wNS=0.6, wDS=0.6, wDN=0.4):

        # time constants
        self.tau_E = 30  # ms  (LH: 60 ms)
        self.tau_S = 5  # LH: 2 ms
        self.tau_N = 5

        # presynaptic inhibition
        self.shift = 1
        self.b = 0
        self.p = 0.5
        self.tau_p = 100

        # weights
        self.wNS = wNS/self.p
        self.wDS = wDS/self.p
        self.wDN = wDN  # 0.4
        self.wSE = 0.5
        self.wED = 0.7
        self.wDE = 1
        self.wPE = 0.5
        self.wEP = 0.2
        self.wPS = 1   # bad choice
        print('wDS - wDN wNS =', self.wDS-self.wDN*self.wNS)

        # external input (tuned s.t. all rates are at target)
        target = 1
        self.target = target
        self.xE = target + (self.wEP - self.wED)*target  # Hz
        self.xD = target + (self.p*self.wDS + self.wDN - self.wDE)*target  # Hz
        self.xS = target - self.wSE*target  # Hz
        self.xN = target + self.p*self.wNS*target  # Hz
        self.xP = target + (self.wPS - self.wPE)*target
        print(f"I_E={self.xE:1.2f}, I_D={self.xD:1.2f}, I_S={self.xS:1.2f}, I_N={self.xN:1.2f}, I_P={self.xP:1.2f}")

        self.alpha_E = 1.5
        self.alpha_N = 1.1
        self.alpha_S = 1

        net_inh = self.p*(self.wDS-self.wDN*self.wNS)
        div = 1 - self.wED*self.wDE + net_inh*self.wSE + self.wEP*(self.wPE - self.wPS*self.wSE)
        num = self.wED*(self.xD-net_inh*self.xS - self.wDN*self.xN) + self.xE + self.wEP*(self.wPS*self.xS - self.xP)
        rE_ss = num/div
        print('rE_ss = ', rE_ss, '(Note: this is slightly off, due to rounding errors??)')

    def g_func(self, r):
        return 1/(1 + np.exp(self.b*(r-self.shift)))

    def fS(self, r):
        # return np.maximum(1/4*r, 0)**2
        return r

    def run(self, duration, actNDNF=0, actSOM=0, xFF_on=0, tau_stim=10, dt=1):

        # initalise rates
        rE = self.target
        rS = self.target
        rN = self.target
        rD = self.target
        rP = self.target
        p = self.p

        # get inputs:
        I_E = self.xE
        I_D = self.xD
        I_S = self.xS
        I_N = self.xN
        I_P = self.xP

        # time array
        t = np.arange(0, duration, dt)

        # create arrays for storage
        rE_store = []
        rS_store = []
        rN_store = []
        rD_store = []
        rP_store = []
        p_store = []

        xS = np.ones(len(t))*self.xS
        xN = np.ones(len(t))*self.xN

        if actSOM:
            xS[400:600] += actSOM

        if actNDNF:
            xN[400:600] += actNDNF

        xFF = 0
        for i, ti in enumerate(t):

            # if 300 <= ti <= 550:
            #     xFF += (-xFF + xFF_on)/tau_stim*dt
            # else:
            #     xFF += (-xFF + 0)/tau_stim*dt

            # input_soma = self.xE + self.alpha_E*xFF
            # input_dend = self.xD - self.p*self.wDS*self.rS - self.wDN*self.rN + self.wEE*self.rE

            # act_dend = self.lambda_E*input_soma + (1-self.lambda_D)*input_dend
            # ca_event = self.alpha_c*np.heaviside(act_dend-self.thresh_c, 1)

            # drE_dt = (-self.rE + (self.lambda_D*np.maximum(input_dend + ca_event, 0) + (1-self.lambda_E)*input_soma
            #                       - self.thresh))/self.tau_E

            drE_dt = (-rE + self.wED*rD + I_E - self.wEP*rP)/self.tau_E

            drD_dt = (-rD + self.wDE*rE - p*self.wDS*rS - self.wDN*rN + I_D)/self.tau_E

            drS_dt = (-rS + self.fS(self.wSE*rE + xS[i]))/self.tau_S

            drN_dt = (-rN + xN[i] - p*self.wNS*rS)/self.tau_N

            drP_dt = (-rN + self.wPE*rE - self.wPS*rS + I_P)/self.tau_S

            if self.b == 0:
                dp_dt = 0
            else:
                dp_dt = (-p + self.g_func(rN))/self.tau_p

            # integrate
            rE = np.maximum(rE + drE_dt*dt, 0)
            rD = np.maximum(rD + drD_dt*dt, 0)
            rS = np.maximum(rS + drS_dt*dt, 0)
            rN = np.maximum(rN + drN_dt*dt, 0)
            rP = np.maximum(rN + drP_dt*dt, 0)
            p = np.maximum(p + dp_dt*dt, 0)

            # store
            rE_store.append(rE)
            rD_store.append(rD)
            rS_store.append(rS)
            rN_store.append(rN)
            rP_store.append(rP)
            # act_dend_store.append(act_dend)
            p_store.append(p)

        return t, rE_store, rD_store, rS_store, rN_store, rP_store, p_store


def opto_experiment(duration, b=0, actNDNF=1, actSOM=1):

    fig, ax = plt.subplots(4, 2, figsize=(5, 4), dpi=150, sharex=True, sharey='row')
    net = Network()
    net.b = b
    t, rE, aD, rS, rN, rP, p = net.run(duration, actSOM=actSOM)
    ax[0, 0].plot(t, rP, c='darkblue')
    ax[0, 0].plot(t, rS, c='C0')
    ax[0, 0].plot(t, rN, c='C1')
    ax[1, 0].plot(t, p, c='C2')
    ax[2, 0].plot(t, rE, c='k')
    ax[3, 0].plot(t, aD, c='silver')
    t, rE, aD, rS, rN, rP, p = net.run(duration, actNDNF=actNDNF)
    ax[0, 1].plot(t, rP, c='darkblue')
    ax[0, 1].plot(t, rS, c='C0')
    ax[0, 1].plot(t, rN, c='C1')
    ax[1, 1].plot(t, p, c='C2')
    ax[2, 1].plot(t, rE, c='k')
    ax[3, 1].plot(t, aD, c='silver')

    ax[0, 0].set(title='activate SOM', ylabel='SOM/NDNF rate')
    ax[0, 1].set(ylim=[-0.2, 3], title='activate NDNF')
    ax[2, 0].set(ylim=[0, 3], ylabel='PC rate')
    ax[3, 0].set(ylim=[0, 3], ylabel='dend act')
    ax[1, 0].set(ylim=[0, 1], ylabel='p')

    # fig, ax = plt.subplots(ncols=2, figsize=(5, 3), dpi=150)
    # rE_som_act = []
    # rE_ndnf_act = []
    # act_strengths = np.arange(0, 2.1, 0.5)
    # for i, act in enumerate(act_strengths):
    #     _, rE, _, _, _, _, _ = net.run(600, actSOM=act)
    #     rE_som_act.append(np.mean(rE[500:600])/rE[0])
    #     _, rE, _, _, _, _, _ = net.run(600, actNDNF=act)
    #     rE_ndnf_act.append(np.mean(rE[500:600])/rE[0])
    #
    # ax[0].plot(act_strengths, rE_som_act)
    # ax[1].plot(act_strengths, rE_ndnf_act)


if __name__ in "__main__":

    opto_experiment(1000, actNDNF=1, actSOM=1, b=0)
    opto_experiment(1000, actNDNF=1, actSOM=1, b=1)
    # opto_experiment(actNDNF=4, b=0)

    # fig, ax0 = plt.subplots(3, 2, figsize=(5, 4), dpi=150, sharex=True)
    #
    # som_fi = []
    #
    # for j, xff in enumerate([2, 4, 6, 8]):
    #     net = Network()
    #     t, rE, aD, rS, rN, p = net.run(1200, xFF_on=xff)
    #     ax0[0, 0].plot(t, rE, 'k', alpha=(j+1)/5)
    #     ax0[1, 0].plot(t, rN, 'C1', alpha=(j+1)/5)
    #     ax0[2, 0].plot(t, rS, 'C0', alpha=(j+1)/5)
    #     ax0[0, 1].plot(t, aD, 'gray', alpha=(j+1)/5)
    #     ax0[1, 1].plot(t, p, 'C2', alpha=(j+1)/5)
    #     ax0[2, 1].plot(t, np.array(p)*np.array(rS), 'C0', alpha=(j+1)/5)
    #     ax0[2, 0].hlines(0.1, 300, 550, color='k', lw=2)
    #     som_fi.append(rS[600])
    # ax0[0, 0].set(ylabel='PC rate')
    # ax0[1, 0].set(ylabel='NDNF rate')
    # ax0[2, 0].set(ylabel='SOM rate', xlabel='time [ms]')
    # ax0[0, 1].hlines(net.thresh_c, 0, 1200, color='C3', ls='--')
    # ax0[1, 1].set(ylim=[0, 1])

    # plt.figure()
    # plt.plot(som_fi)
    # plt.plot(som_fi, '.')

    plt.show()


    # fig, ax = plt.subplots(1, 2, figsize=(5, 2.5), dpi=150)
    #
    # xD_range = np.arange(-20, 40, 5)
    # xE_range = np.arange(-10, 20, 5)
    #
    # rE_final = np.zeros((len(xD_range), len(xE_range)))
    # aD_final = np.zeros((len(xD_range), len(xE_range)))

    # for j, xD in enumerate(xD_range):
    #     for k, xE in enumerate(xE_range):
            # time, rE, aD = net.run(500, xD=xD, xE=xE)
            # rE_final[j, k] = rE[-1]
            # aD_final[j, k] = aD[-1]

        # rE_final.append(rE[-1])
    # cb = ax[0].imshow(rE_final)
    # plt.colorbar(cb, ax=ax[0])
    # cb = ax[1].imshow(aD_final)
    # plt.colorbar(cb, ax=ax[1])

    # plt.figure()
    # plt.plot(rE_final)
    plt.show()
