"""
Model base: contains model classes
"""

# imports
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('pretty')


class NetworkModel:
    """
    Class for network model with two-compartment PCs, SOMs, NDNFs and optionally PVs.
    """

    def __init__(self, N_cells, w_mean, conn_prob, taus, bg_inputs, wED=0.7, b=0.5, r0=0, p0=0, taup=100,
                 flag_w_hetero=False, flag_SOM_ad=False, flag_pre_inh=True, flag_with_VIP=False,
                 flag_with_NDNF=True):

        # network parameters
        self.N_cells = N_cells
        self.w_mean = w_mean

        # flags
        self.flag_w_heteo = flag_w_hetero
        self.flag_with_VIP = flag_with_VIP
        self.flag_with_NDNF = flag_with_NDNF

        # Create weight matrices
        if flag_w_hetero:
            w_std_rel = 0.1
        else:
            w_std_rel = 0

        if not flag_with_VIP:
            self.w_mean['SV'] = 0
            bg_inputs['V'] = 0

        if not flag_with_NDNF:
            bg_inputs['N'] = 0
            self.w_mean['PN'] = 0
            self.w_mean['DN'] = 0

        self.Ws = dict()  # dictionary of weight matrices
        for conn in self.w_mean.keys():
            post, pre = conn[0], conn[1]
            self.Ws[conn] = self.make_weight_mat(N_cells[pre], N_cells[post], conn_prob[conn], self.w_mean[conn],
                                                 w_std_rel=w_std_rel, no_autapse=(pre == post))
            # print('W_'+conn, self.Ws[conn])

        # time constants
        self.taus = taus

        # background inputs
        self.Xbg = bg_inputs

        # PV model parameters
        self.wED = wED

        # SOM adaptation
        self.flag_SOM_ad = flag_SOM_ad
        self.tau_a = 100
        self.ba = 1 if flag_SOM_ad else 0

        # presynaptic inhibition
        self.flag_pre_inh = flag_pre_inh
        self.b = b if flag_pre_inh else 0
        self.p0 = p0
        self.r0 = r0
        self.taup = taup

    def make_weight_mat(self, Npre, Npost, c_prob, w_mean, w_std_rel=0, no_autapse=False):
        """
        Create weight matrix from connection probability and mean weight.
        :param Npre:        number of presynaptic cells
        :param Npost:       number of postsynaptic cells
        :param c_prob:      connection probability
        :param w_mean:      mean weight for connections
        :param w_std_rel:   standard deviation of weights relative to mean
        :param no_autapse:  whether autapses are allowed (i.e. connections from i to i)
        :return:            weight matrix W
        """

        W = np.zeros((Npost, Npre))  # initialise matrix

        n_in = np.round(c_prob * Npre).astype(int)  # determine number of presynaptic cells (in-degree)

        if Npre == 1 and Npost == 1:  # if pre an post are only one neuron, make suer they're connected
            n_in = 1
            no_autapse = False

        for i_post in range(Npost):  # determine presynaptic cells for each postsynaptic cell
            if no_autapse:
                pre_opt = np.delete(np.arange(Npre), i_post)
            else:
                pre_opt = np.arange(Npre)
            js_pre = np.random.choice(pre_opt, n_in, replace=False)  # choose presynaptic partners for neuron i
            W[i_post, js_pre] = np.maximum(np.random.normal(w_mean, w_mean * w_std_rel, size=n_in) / n_in,
                                           0)  # set weights

        return W

    def plot_weight_mats(self):

        fig, ax = plt.subplots(4, 4, figsize=(4, 4), dpi=150)

        for n, k in enumerate(self.Ws.keys()):
            i, j = np.unravel_index(n, (4, 4))
            ax[i, j].imshow(self.Ws[k])
            ax[i, j].set(title='W_' + k)

        # plt.show()

    def g_func(self, r):
        """
        Presynaptic inhibition transfer function.
        :param r: input rate
        :return: release probability p
        """
        return np.clip(1 - self.b * (r - self.r0), self.p0, 1)

    def run(self, dur, xFF, rE0=1, rS0=1, rN0=1, rP0=1, rD0=1, rV0=1, p_init=0.5, init_noise=0.1, dt=1,
            monitor_boutons=False, monitor_currents=False, calc_bg_input=True):
        """
        Function to run the dynamics of the network. Returns arrays for time and neuron firing rates and a dictionary of
        'other' things, such as currents or 'bouton activities'.

        :param dur:         duration of stimulation (in ms)
        :param xFF:         dictionary of inputs (FF or FB) to the cells ('E', 'D', 'P', 'S', 'N', 'V')
        :param rE0:         initial rate for somatic compartment of PCs, also baseline rate if bg input calculated
        :param rS0:         initial rate/ baseline of SOMs
        :param rN0:         initial rate/ baseline of NDNFs
        :param rP0:         initial rate/ baseline of PVs
        :param rD0:         initial rate/ baseline of dendrites (rate = 'activity')
        :param rV0:         initial rate/ baseline of VIPs
        :param p_init:      initial release probability
        :param init_noise:  noise in initial values of variables
        :param dt:          time step (in ms)
        :param monitor_boutons:     whether to monitor SOM boutons
        :param monitor_currents:    whether to monitor input currents to SOM and NDNF
        :param calc_bg_input:       whether to calculcate the background inputs to achieve target rates
        :return:
        """

        # time arrays
        t = np.arange(0, dur, dt)
        nt = len(t)

        # calculate background input to establish baselines specified by initial rates
        if calc_bg_input:
            if self.flag_with_NDNF:
                self.Xbg['N'] = rN0 + self.w_mean['NS'] * rS0 + self.w_mean['NN'] * rN0
            else:
                self.Xbg['N'] = 0
                rN0 = 0
            if self.flag_with_VIP:
                self.Xbg['V'] = rV0 + self.w_mean['VS'] * rS0
            else:
                self.Xbg['V'] = 0
                rV0 = 0
            self.Xbg['E'] = rE0 + self.w_mean['EP'] * rP0 - self.wED * rD0
            self.Xbg['D'] = rD0 + self.w_mean['DS'] * rS0 + self.w_mean['DN'] * rN0
            self.Xbg['S'] = rS0 - self.w_mean['SE'] * rE0 + self.w_mean['SV'] * rV0
            self.Xbg['P'] = rP0 - self.w_mean['PE'] * rP0 + self.w_mean['PS'] * rS0 + self.w_mean['PN'] * rN0 \
                            + self.w_mean['PP'] * rP0
            print('background input:', self.Xbg)

        # create empty arrays
        rE = np.zeros((nt, self.N_cells['E']))
        rD = np.zeros((nt, self.N_cells['D']))
        rS = np.zeros((nt, self.N_cells['S']))
        rN = np.zeros((nt, self.N_cells['N']))
        rP = np.zeros((nt, self.N_cells['P']))
        rV = np.zeros((nt, self.N_cells['V']))

        # set initial rates
        rE[0] = np.random.normal(rE0, rE0*init_noise, size=self.N_cells['E'])
        rD[0] = np.random.normal(rD0, rD0 * init_noise, size=self.N_cells['D'])
        rS[0] = np.random.normal(rS0, rS0*init_noise, size=self.N_cells['S'])
        rN[0] = np.random.normal(rN0, rN0*init_noise, size=self.N_cells['N'])
        rP[0] = np.random.normal(rP0, rP0*init_noise, size=self.N_cells['P'])
        rV[0] = np.random.normal(rV0, rP0*init_noise, size=self.N_cells['P'])

        # variables for other shenanigans
        aS = np.zeros(self.N_cells['S'])
        p = np.ones(nt)
        p[0] = p_init if self.flag_pre_inh else 1

        # optional recording of stuff
        other = dict()
        if monitor_boutons:
            other['boutons_SOM'] = []
        if monitor_currents:
            other['curr_rS'] = []
            other['curr_rN'] = []

        # activations pre rectification
        vS, vN, vE, vD, vP, vV = rS[0], rN[0], rE[0], rD[0], rP[0], rV[0]

        # time integration
        for ti in range(nt-1):
            # ToDo: add presynaptic inhibition

            # compute input currents
            curr_rE = self.wED * rD[ti] - self.Ws['EP'] @ rP[ti] + self.Xbg['E'] + xFF['E'][ti]
            curr_rD = self.Ws['DE'] @ rE[ti] - p[ti]*self.Ws['DS'] @ rS[ti] - p[ti]*self.Ws['DN'] @ rN[ti]\
                      + self.Xbg['D'] + xFF['D'][ti]
            curr_rS = self.Ws['SE'] @ rE[ti] - self.Ws['SV']@rV[ti] + self.Xbg['S'] + xFF['S'][ti]
            curr_rN = -p[ti]*self.Ws['NS'] @ rS[ti] - self.Ws['NN'] @ rN[ti] + self.Xbg['N'] + xFF['N'][ti]
            curr_rP = self.Ws['PE'] @ rE[ti] - self.Ws['PS'] @ rS[ti] - self.Ws['PN'] @ rN[ti] - self.Ws['PP'] @ rP[ti] \
                      + self.Xbg['P'] + xFF['P'][ti]
            curr_rV = -self.Ws['VS']@rS[ti] + self.Xbg['V'] + xFF['V'][ti]

            # Euler integration (pre rectification)  # ToDo: -rE or -vE+
            vE = vE + (-rE[ti] + curr_rE) / self.taus['E'] * dt
            vD = vD + (-rD[ti] + curr_rD) / self.taus['D'] * dt
            vS = vS + (-rS[ti] + curr_rS - aS) / self.taus['S'] * dt
            vN = vN + (-rN[ti] + curr_rN) / self.taus['N'] * dt
            vP = vP + (-rP[ti] + curr_rP) / self.taus['P'] * dt
            vV = vV +(-rV[ti] + curr_rV) / self.taus['V'] * dt

            # adaptation and other shenanigans
            if self.flag_SOM_ad:
                aS += (-aS + self.ba*rS[ti]) / self.tau_a * dt
            if self.flag_pre_inh:
                p[ti+1] = p[ti] + (-p[ti] + self.g_func(np.mean(rN[ti]))) / self.taup * dt

            # rectification and saving
            rE[ti + 1] = np.maximum(vE, 0)
            rD[ti + 1] = np.maximum(vD, 0)
            rS[ti + 1] = np.maximum(vS, 0)
            rN[ti + 1] = np.maximum(vN, 0)
            rP[ti + 1] = np.maximum(vP, 0)
            rV[ti + 1] = np.maximum(vV, 0)

            if monitor_boutons:
                other['boutons_SOM'].append((p[ti]*self.Ws['DS']*rS[ti]).flatten())

            if monitor_currents:
                other['curr_rS'].append(curr_rS)
                other['curr_rN'].append(curr_rN)

        return t, rE, rD, rS, rN, rP, rV, p, other


def get_default_params(flag_mean_pop=False):
    """
    Create dictionaries with default parameters.

    :param flag_mean_pop:   if true, all neuron numbers are set to 1 (= 'mean field picture')
    :return:
    """

    N_cells   = dict(E=70, D=70, S=10, N=10, P=10, V=10)
    if flag_mean_pop:
        N_cells = dict(E=1, D=1, S=1, N=1, P=1, V=1)
    # w_mean = dict(NS=0.5, DS=0.5, DN=0.4, SE=0.8, NN=0.3, PS=0, PN=0, PP=0, PE=0, EP=0, DE=0)  # no PVs
    w_mean = dict(NS=0.7, DS=0.5, DN=0.4, SE=0.8, NN=0.2, PS=0.8, PN=0.5, PP=0.1, PE=1, EP=0.5, DE=0, VS=0.5, SV=0.5)
    conn_prob = dict(NS=0.9, DS=0.55, DN=0.5, SE=0.35, NN=0.5, PS=0.6, PN=0.3, PP=0.5, PE=0.7, EP=0.6, DE=0.1,
                     VS=0.5, SV=0.5)
    bg_inputs = dict(E=0.5, D=2, N=2, S=0.5, P=1.5, V=1.3)
    taus = dict(S=20, N=40, E=10, P=10, D=20, V=15)

    return N_cells, w_mean, conn_prob, bg_inputs, taus


if __name__ in "__main__":

    # ToDo: clean this bottom part here up

    # define parameter dictionaries
    N_cells, w_mean, conn_prob, bg_inputs, taus = get_default_params()

    # w_mean['NS'] = 0  # block SOM->NDNF inhibition

    # N_cells = dict(E=1, D=1, S=1, N=1, P=1)

    # ToDo: add more properties
    # pre_inh_params = dict(taup=100, b=0.3, p0=0.1, r0=0)
    # PC_param = dict(ED=0.7)

    # instantiate model
    model = NetworkModel(N_cells, w_mean, conn_prob, taus, bg_inputs, wED=0.7, flag_SOM_ad=True, flag_w_hetero=True,
                         flag_pre_inh=True)
    model.plot_weight_mats()

    # simulation paramters
    dur = 1000
    dt = 1
    nt = int(dur/dt)

    # generate inputs
    # A: no input
    xFF_null = dict(E=np.zeros((nt, model.N_cells['E'])),
                    D=np.zeros((nt, model.N_cells['E'])),
                    S=np.zeros((nt, model.N_cells['S'])),
                    N=np.zeros((nt, model.N_cells['N'])),
                    P=np.zeros((nt, model.N_cells['P'])),
                    V=np.zeros((nt, model.N_cells['P'])))
    xFF = xFF_null.copy()

    # # B1: stimulate SOM
    # ts, te = 300, 500
    # stim_SOM = 1
    # xFF = xFF_null.copy()
    # xFF['S'][ts:te] = stim_SOM

    # B2: stimulate NDNF
    ts, te = 300, 400
    stim_NDNF = 2
    xFF = xFF_null.copy()
    xFF['N'][ts:te] = stim_NDNF

    # C: sensory stimulation
    # tau1 = 150
    # tau2 = 50
    # ts = 300
    # t = np.arange(0, dur, dt)
    # stim_strength = 3
    # sensory_stim = 2 * np.maximum(stim_strength * (np.exp(-(t - ts) / tau1) - np.exp(-(t - ts) / tau2)), 0)
    # xFF = xFF_null.copy()
    # xFF['E'] = sensory_stim
    # xFF['N'] = sensory_stim**0.3

    # run model
    t, rE, rD, rS, rN, rP, rV, p, other = model.run(dur, xFF, init_noise=0, dt=dt, monitor_boutons=True,
                                                monitor_currents=True)

    # plotting
    fig, ax = plt.subplots(6, 1, figsize=(4, 5), dpi=150, sharex=True)
    ax[0].plot(t, rE, c='C3', alpha=0.5)
    ax[1].plot(t, rD, c='k', alpha=0.5)
    ax[2].plot(t, rS, c='C0', alpha=0.5)
    ax[3].plot(t, rN, c='C1', alpha=0.5)
    ax[4].plot(t, rP, c='darkblue', alpha=0.5)
    ax[5].plot(t, p, c='C2', alpha=1)

    for i, label in enumerate(['PC', 'dend.', 'SOM', 'NDNF', 'PV']):
        ax[i].set(ylabel=label, ylim=[0, 3])
    ax[5].set(ylabel='p', ylim=[0, 1], xlabel='time (ms)')

    fig2, ax2 = plt.subplots(1, 1, figsize=(2, 2), dpi=150)
    boutons = np.array(other['boutons_SOM'])
    boutons_nonzero = boutons[:, np.mean(boutons, axis=0)>0]
    ax2.pcolormesh(boutons_nonzero.T)
    ax2.set(xlabel='time (ms)', ylabel='SOM boutons')

    plt.tight_layout()

    plt.show()

    # wNS = 1, wSE = 0.8, tauN = 15, tauS = 20, tauE = 20, taup = 100, b = 0.3, p0 = 0.1, r0 = 0.0, ba = 1, tau_a = 100,
    # U_S = 0.1, tauf = 200, tauD = 20, wED = 0.7, wDN = 0.4, wDS = 0.5, expS = 1.2, expN = 0.3,
    # alphaS = 0, alphaN = 0.1, alphaE = 1, wNN = 0.5, U_N = 0.9, taud = 200, dt = 1)
