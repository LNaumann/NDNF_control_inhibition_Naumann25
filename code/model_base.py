"""
Model base: contains model class and function for default parameters
"""

# imports
import numpy as np
import matplotlib.pyplot as plt


class NetworkModel:
    """
    Class for network model with two-compartment PCs, SOMs, NDNFs and optionally PVs.

    Parameters:
    - N_cells:          dictionary with the number of cells for each cell type
    - w_mean:           dictionary of mean weights for each synapse type
    - conn_prob:        dictionary of connection probabilities between all neuron types
    - taus:             dictionary of time constants
    - bg_inputs:        dictionary of background inputs
    - wED:              weight of the dendrite->soma coupling
    - b:                presynaptic inhibition parameter
    - r0:               lower threshold to activate presynaptic inhibition (default 0)
    - p_low:            lower bound for release probability (default 0)
    - taup:             time constant for presynaptic inhibition
    - tauG:             time constant for GABA spillover
    - gamma:            scaling factor for GABA spillover
    - w_std_rel:        relative standard deviation of weights
    - flag_w_hetero:    whether to add heterogeneity to weight matrices
    - flag_pre_inh:     whether to include presynaptic inhibition
    - flag_with_VIP:    whether to include VIPs
    - flag_with_NDNF:   whether to include NDNFs
    - flag_with_PV:     whether to include PVs
    - flag_p_on_DN:     whether to include presynaptic inhibition on NDNF->dendrite synapses
    - flag_p_on_VS:     whether to include presynaptic inhibition on SOM->VIP synapses
    """

    def __init__(self, N_cells, w_mean, conn_prob, taus, bg_inputs, wED=1, b=0.5, r0=0, p_low=0, taup=100,
                 tauG=200, gamma=1, w_std_rel=0.1,
                 flag_w_hetero=False, flag_pre_inh=True, flag_with_VIP=True,
                 flag_with_NDNF=True, flag_with_PV=True, flag_p_on_DN=False, flag_p_on_VS=False):

        # network parameters
        self.N_cells = N_cells
        self.w_mean = w_mean

        # flags
        self.flag_w_hetero = flag_w_hetero
        self.flag_with_VIP = flag_with_VIP
        self.flag_with_NDNF = flag_with_NDNF
        self.flag_with_PV = flag_with_PV

        # adapt weights to model variation
        if not flag_w_hetero:
            w_std_rel = 0

        if not flag_with_VIP:
            self.w_mean['SV'] = 0
            self.w_mean['PV'] = 0
            self.w_mean['VE'] = 0
            bg_inputs['V'] = 0

        if not flag_with_NDNF:
            bg_inputs['N'] = 0
            self.w_mean['PN'] = 0
            self.w_mean['DN'] = 0

        if not flag_with_PV:
            bg_inputs['P'] = 0
            self.w_mean['PE'] = 0
            self.w_mean['EP'] = 0

        # create weight matrices
        self.Ws = dict()  # dictionary of weight matrices
        for conn in self.w_mean.keys():
            post, pre = conn[0], conn[1]
            self.Ws[conn] = self.make_weight_mat(N_cells[pre], N_cells[post], conn_prob[conn], self.w_mean[conn],
                                                 w_std_rel=w_std_rel, no_autapse=(pre == post))

        # time constants
        self.taus = taus

        # background inputs
        self.Xbg = bg_inputs

        # PV model parameters
        self.wED = wED

        # GABA spillover & presynaptic inhibition
        self.tauG = tauG
        self.gamma = gamma
        self.flag_pre_inh = flag_pre_inh
        self.b = b if flag_pre_inh else 0
        self.p_low = p_low
        self.r0 = r0
        self.taup = taup
        self.flag_p_on_DN = flag_p_on_DN
        self.flag_p_on_VS = flag_p_on_VS
        self.weights_scaled_by = 1


    def make_weight_mat(self, Npre, Npost, c_prob, w_mean, w_std_rel=0, no_autapse=False):
        """
        Create weight matrix from connection probability and mean weight.

        Parameters:
        - Npre:        number of presynaptic cells
        - Npost:       number of postsynaptic cells
        - c_prob:      connection probability
        - w_mean:      mean weight for connections
        - w_std_rel:   standard deviation of weights relative to mean
        - no_autapse:  whether autapses are allowed (i.e. connections from i to i)

        Returns:
        - weight matrix W
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
    

    def g_func(self, r):
        """
        Presynaptic inhibition transfer function.
        :param r: input rate
        :return: release probability p
        """
        return np.clip(1 - self.b * (r - self.r0), self.p_low, 1)


    def run(self, dur, xFF, rE0=1, rS0=1, rN0=1, rP0=1, rD0=1, rV0=1, p0=0.5, init_noise=0.1, noise=0.1, dt=1,
            monitor_boutons=False, monitor_dend_inh=False, monitor_currents=False, calc_bg_input=True, scale_w_by_p=True, p_scale=None):
        """
        Function to run the dynamics of the network. Returns arrays for time and neuron firing rates and a dictionary of
        'other' things, such as currents.

        Parameters:
        - dur:              duration of stimulation (in ms)
        - xFF:              dictionary of inputs (FF or FB) to the cells ('E', 'D', 'P', 'S', 'N', 'V')
        - rE0:              initial rate for somatic compartment of PCs, also baseline rate if bg input calculated
        - rS0:              initial rate/ baseline of SOMs
        - rN0:              initial rate/ baseline of NDNFs
        - rP0:              initial rate/ baseline of PVs
        - rD0:              initial rate/ baseline of dendrites (rate = 'activity')
        - rV0:              initial rate/ baseline of VIPs
        - p0:               initial release probability
        - noise:            level of white noise added to neural activity
        - init_noise:       noise in initial values of variables
        - dt:               time step (in ms)
        - monitor_boutons:  whether to monitor SOM boutons
        - monitor_dend_inh: whether to monitor dendritic inhibition
        - monitor_currents: whether to monitor input currents to SOM and NDNF
        - calc_bg_input:    whether to calculcate the background inputs to achieve target rates
        - scale_w_by_p:     whether to scale weights by release probability
        - p_scale:          if not None, scale weights by this value

        Returns:
        - t:                time array
        - rE:               array of rates of somatic compartments of PCs
        - rD:               array of activities of dendrites
        - rS:               array of rates of SOMs
        - rN:               array of rates of NDNFs
        - rP:               array of rates of PVs
        - rV:               array of rates of VIPs
        - p:                array of release probabilities
        - cGABA:            array of GABA spillover
        - other:            dictionary of other stuff (boutons_SOM, dend_inh_SOM, dend_inh_NDNF, soma_inh_PV)
        """

        # time arrays
        t = np.arange(0, dur, dt)
        nt = len(t)

        # presynaptic inhibition adjustments to the model
        p_init = p0
        alph_p_on_DN = 0.5 # scaling factor for strength of pre inh on NDNF-dendrite synapses
        if self.flag_pre_inh:
            p0 = p_scale if p_scale else self.g_func(rN0)
            # scale weights by release probability
            if scale_w_by_p:
                self.Ws['NS'] = self.Ws['NS']/p0*self.weights_scaled_by
                self.Ws['DS'] = self.Ws['DS']/p0*self.weights_scaled_by
                if self.flag_p_on_DN:
                    self.Ws['DN'] = self.Ws['DN']/(alph_p_on_DN*p0+(1-alph_p_on_DN)*1)*self.weights_scaled_by
                if self.flag_p_on_VS:
                    self.Ws['VS'] = self.Ws['VS']/p0*self.weights_scaled_by
                self.weights_scaled_by = p0  # we're saving this so we don't scale weights again upon next run
                                            # if the function is called again with the same p0, weights remain the same
        else:
            p0 = 1

        # calculate background input to establish baselines specified by initial rates
        if calc_bg_input:
            if self.flag_with_NDNF:
                self.Xbg['N'] = rN0 + self.w_mean['NS'] * rS0 + self.w_mean['NN'] * rN0
            else:
                self.Xbg['N'] = 0
                rN0 = 0
            if self.flag_with_VIP:
                self.Xbg['V'] = rV0 - self.w_mean['VE'] * rE0 + self.w_mean['VS'] * rS0 + self.w_mean['VN'] * rN0
            else:
                self.Xbg['V'] = 0
                rV0 = 0
            if self.flag_with_PV:
                self.Xbg['P'] = rP0 - self.w_mean['PE'] * rE0 + self.w_mean['PS'] * rS0 + self.w_mean['PN'] * rN0 + self.w_mean['PV'] * rV0 \
                            + self.w_mean['PP'] * rP0
            else:
                self.Xbg['P'] = 0
                rP0 = 0
            self.Xbg['E'] = rE0 + self.w_mean['EP'] * rP0 - self.wED * rD0
            self.Xbg['D'] = rD0 + self.w_mean['DS'] * rS0 + self.w_mean['DN'] * rN0 - self.w_mean['DE'] * rE0
            self.Xbg['S'] = rS0 - self.w_mean['SE'] * rE0 + self.w_mean['SV'] * rV0
            # note: no need to scale weights by p0 here because the weight matrices are divided by p0 and then again
            #       multiplied by the current p during the simulation

        # create empty arrays
        rE = np.zeros((nt, self.N_cells['E']))
        rD = np.zeros((nt, self.N_cells['D']))
        rS = np.zeros((nt, self.N_cells['S']))
        rN = np.zeros((nt, self.N_cells['N']))
        rP = np.zeros((nt, self.N_cells['P']))
        rV = np.zeros((nt, self.N_cells['V']))

        # set initial rates/values
        rE[0] = np.random.normal(rE0, rE0*init_noise, size=self.N_cells['E'])
        rD[0] = np.random.normal(rD0, rD0*init_noise, size=self.N_cells['D'])
        rS[0] = np.random.normal(rS0, rS0*init_noise, size=self.N_cells['S'])
        rN[0] = np.random.normal(rN0, rN0*init_noise, size=self.N_cells['N'])
        rP[0] = np.random.normal(rP0, rP0*init_noise, size=self.N_cells['P'])
        rV[0] = np.random.normal(rV0, rP0*init_noise, size=self.N_cells['P'])

        # variables for other shenanigans
        p = np.ones(nt)
        p[0] = p_init if p_init else p0
        cGABA = np.zeros((nt, self.N_cells['N']))
        cGABA[0] = rN0

        # optional recording of stuff
        other = dict()
        if monitor_boutons:
            other['boutons_SOM'] = []
        if monitor_dend_inh:
            other['dend_inh_SOM'] = [p[0]*self.Ws['DS']@rS[0]]
            other['dend_inh_NDNF'] = [self.Ws['DN']@rN[0]]  # todo: include flag for pi on wDN
            other['soma_inh_PV'] = [self.Ws['EP']@rP[0]]
        if monitor_currents:
            other['curr_rS'] = []
            other['curr_rN'] = []
            other['curr_rE'] = []

        # initialise activations (rates pre rectification)
        vS, vN, vE, vD, vP, vV = rS[0], rN[0], rE[0], rD[0], rP[0], rV[0]

        # time integration
        for ti in range(nt-1):

            # white noise
            xiE = np.random.normal(0, noise, size=self.N_cells['E'])
            xiD = np.random.normal(0, noise, size=self.N_cells['D'])
            xiS = np.random.normal(0, noise, size=self.N_cells['S'])
            xiN = np.random.normal(0, noise, size=self.N_cells['N'])
            xiP = np.random.normal(0, noise, size=self.N_cells['P'])
            xiV = np.random.normal(0, noise, size=self.N_cells['V'])

            # release factor for NDNF->dendrite and SOM->VIP depends on flag
            pDN = alph_p_on_DN*p[ti] + (1-alph_p_on_DN)*1 if self.flag_p_on_DN else 1
            pVS = p[ti] if self.flag_p_on_VS else 1

            # compute input currents
            curr_rE = self.wED * rD[ti] - self.Ws['EP'] @ rP[ti] + self.Xbg['E'] + xFF['E'][ti] + xiE
            curr_rD = self.Ws['DE'] @ rE[ti] - p[ti]*self.Ws['DS'] @ rS[ti] - pDN*self.Ws['DN'] @ cGABA[ti]\
                      + self.Xbg['D'] + xFF['D'][ti] + xiD
            curr_rS = self.Ws['SE'] @ rE[ti] - self.Ws['SV']@rV[ti] + self.Xbg['S'] + xFF['S'][ti] + xiS
            curr_rN = -p[ti]*self.Ws['NS'] @ rS[ti] - self.Ws['NN'] @ rN[ti] + self.Xbg['N'] + xFF['N'][ti] + xiN
            curr_rP = self.Ws['PE'] @ rE[ti] - self.Ws['PS'] @ rS[ti] - self.Ws['PN'] @ rN[ti] - self.Ws['PP'] @ rP[ti] \
                      - self.Ws['PV'] @ rV[ti] + self.Xbg['P'] + xFF['P'][ti] + xiP
            curr_rV = -pVS*self.Ws['VS']@rS[ti] -self.Ws['VN']@rN[ti] + self.Ws['VE']@rE[ti] + self.Xbg['V'] + xFF['V'][ti] + xiV

            # Euler integration (pre rectification)
            vE = vE + (-vE + curr_rE) / self.taus['E'] * dt
            vD = vD + (-vD + curr_rD) / self.taus['D'] * dt
            vS = vS + (-vS + curr_rS )/ self.taus['S'] * dt
            vN = vN + (-vN + curr_rN) / self.taus['N'] * dt
            vP = vP + (-vP + curr_rP) / self.taus['P'] * dt
            vV = vV + (-vV + curr_rV) / self.taus['V'] * dt

            # presynaptic inhibition
            if self.flag_pre_inh:
                p[ti+1] = p[ti] + (-p[ti] + self.g_func(np.mean(cGABA[ti]))) / self.taup * dt

            # GABA spillover
            cGABA[ti+1] = cGABA[ti] + (-cGABA[ti] + self.gamma*rN[ti]) / self.tauG * dt
            cGABA[ti+1] = np.maximum(cGABA[ti+1], 0)  # probably not necesarry, better safe than sorry

            # rectification and saving
            rE[ti + 1] = np.maximum(vE, 0)
            rD[ti + 1] = np.maximum(vD, 0)
            rS[ti + 1] = np.maximum(vS, 0)
            rN[ti + 1] = np.maximum(vN, 0)
            rP[ti + 1] = np.maximum(vP, 0)
            rV[ti + 1] = np.maximum(vV, 0)

            # storage of additional stuff
            if monitor_boutons:
                other['boutons_SOM'].append((p[ti]*self.Ws['DS']@rS[ti]).flatten())
            if monitor_dend_inh:
                other['dend_inh_SOM'].append(p[ti]*self.Ws['DS']@rS[ti])
                other['dend_inh_NDNF'].append(pDN*self.Ws['DN']@cGABA[ti])
                other['soma_inh_PV'].append(self.Ws['EP']@rP[ti])
            if monitor_currents:
                other['curr_rS'].append(curr_rS)
                other['curr_rN'].append(curr_rN)
                other['curr_rE'].append(curr_rE)

        return t, rE, rD, rS, rN, rP, rV, p, cGABA, other


def get_default_params(flag_mean_pop=False):
    """
    Create dictionaries with default parameters.

    Parameters:
    - flag_mean_pop:   if true, all neuron numbers are set to 1 (= 'mean field picture')

    Returns:
    - N_cells:          dictionary with the number of cells for each cell type
    - w_mean:           dictionary of mean weights for each synapse type
    - conn_prob:        dictionary of connection probabilities between all neuron types
    - bg_inputs:        dictionary of background inputs
    - taus:             dictionary of time constants
    """

    # neuron numbers
    if flag_mean_pop:
        N_cells = dict(E=1, D=1, S=1, N=1, P=1, V=1)
    else:
        N_cells   = dict(E=70, D=70, S=10, N=10, P=10, V=10)

    # mean weights
    w_mean = dict(NS=0.7, DS=0.5, DN=0.4, SE=0.8, NN=0.2, PS=0.8, PN=0.3, PP=0.1, PE=1, EP=0.5, DE=0.2, VS=0.5, SV=0.4, PV=0.2, VE=0.3, VN=0.2)

    # connection probabilities
    conn_prob = dict(NS=0.9, DS=0.55, DN=0.5, SE=0.35, NN=0.5, PS=0.6, PN=0.3, PP=0.5, PE=0.7, EP=0.6, DE=0.1,
                     VS=0.5, SV=0.5, PV=0.5, VE=0.1, VN=0.3)
    
    # background inputs
    bg_inputs = dict(E=0.5, D=1.7, N=1.9, S=0.6, P=1.4, V=1.4)

    # time constants
    taus = dict(S=20, N=40, E=10, P=10, D=20, V=15)

    return N_cells, w_mean, conn_prob, bg_inputs, taus


if __name__ in "__main__":

    # optional custom style sheet
    if 'pretty' in plt.style.available:
        plt.style.use('pretty')

    from helpers import get_null_ff_input_arrays, get_model_colours
    cPC, cPV, cSOM, cNDNF, cVIP, cpi = get_model_colours()

    # set some paremeters
    mean_pop = False
    w_hetero = True
    pre_inh = True
    wnoise = 0.1
    noise = 0.1 

    # define parameter dictionaries
    N_cells, w_mean, conn_prob, bg_inputs, taus = get_default_params(flag_mean_pop=mean_pop)

    # instantiate model
    model = NetworkModel(N_cells, w_mean, conn_prob, taus, bg_inputs, wED=1, flag_w_hetero=w_hetero,
                         flag_pre_inh=pre_inh, flag_p_on_VS=False)

    # simulation paramters
    dur = 3000
    dt = 1
    nt = int(dur/dt)

    # generate null inputs
    xFF_null = get_null_ff_input_arrays(nt, N_cells)

    # stimulate NDNF
    ts, te = 1000, 1500
    stim_strength = 1.5
    xFF = xFF_null.copy()
    xFF['N'][ts:te] = stim_strength

    # run model
    t, rE, rD, rS, rN, rP, rV, p, cGABA, other = model.run(dur, xFF, init_noise=wnoise, noise=noise, dt=dt, monitor_boutons=True,
                                                           monitor_currents=True, calc_bg_input=True)

    # plotting
    fig, ax = plt.subplots(8, 1, figsize=(4, 5), dpi=150, sharex=True, gridspec_kw={'top': 0.95})
    ax[0].plot(t, rE, c=cPC, alpha=0.5)
    ax[1].plot(t, rD, c='k', alpha=0.5)
    ax[2].plot(t, rS, c=cSOM, alpha=0.5)
    ax[3].plot(t, rN, c=cNDNF, alpha=0.5)
    ax[4].plot(t, rV, c=cVIP, alpha=0.5)
    ax[5].plot(t, rP, c=cPV, alpha=0.5)
    ax[6].plot(t, cGABA, c=cpi, alpha=1)
    ax[7].plot(t, p, c=cpi, alpha=1)

    for i, label in enumerate(['PC', 'dend.', 'SOM', 'NDNF', 'VIP', 'PV', 'GABA']):
        ax[i].set(ylabel=label, ylim=[0, 3])
    ax[-1].set(ylabel='p', ylim=[0, 1], xlabel='time (ms)')

    plt.show()