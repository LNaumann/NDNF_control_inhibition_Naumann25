import numpy as np
import matplotlib.pyplot as plt
plt.style.use('pretty')


def simulate_circuit(tmax, I_s, I_n, w=1, tau_s=10, tau_n=10, tau_p=50, b=1, bias=0,
                     act_SOM=0, act_NDNF=0):
    """
    Simulate a circuit that only consists of an NDNF and a SOM population. SOM inhibits NDNF directly, but NDNF
    only inhibits the 'superficial outputs of SOM.
    :param tmax:     duration of time intgration (dt = 1ms)
    :param I_s:      input to the SOM population
    :param I_n:      input to the NDNF population
    :param w:        weight of SOM->NDNF projection
    :param tau_s:    timescale of SOM population
    :param tau_n:    timescale of NDNF population
    :param tau_p:    timescale of presynaptic inhibition of SOM outputs by NDNF
    :param b:        strength of presynaptic inhibition (slope of transfer function p)
    :param bias:     strength of bias to NDNF population
    :param act_SOM:  strength of additional activating input to SOM
    :param act_NDNF: strength of additional activating input to NDNF
    :return:         time-vector t, rate of SOM r_s, rate of NDNF r_n and release prob of SOM outputs
    """
    dt = 1
    t = np.arange(0, tmax, dt)
    nt = len(t)

    r_s = np.zeros(nt)
    r_n = np.zeros(nt)
    p = np.zeros(nt)
    p[0] = 0.73
    r_n[0] = 2 + bias
    r_s[0] = 2

    stim_s = np.zeros(nt)
    stim_n = np.zeros(nt)

    act_t_start, act_t_stop = 200, 250
    if act_SOM:
        stim_s[act_t_start:act_t_stop] = act_SOM
    elif act_NDNF:
        stim_n[act_t_start:act_t_stop] = act_NDNF
    stim_s += I_s
    stim_n += I_n

    for i in range(1, nt):
        r_s[i] = r_s[i - 1] + (-r_s[i - 1] + stim_s[i]) * dt / tau_s
        r_n[i] = r_n[i - 1] + (-r_n[i - 1] + stim_n[i] - w * p[i - 1] * r_s[i - 1]) * dt / tau_n
        p[i] = p[i - 1] + (-p[i - 1] + p_func(r_n[i - 1], b=b)) * dt / tau_p

        r_s[i] = np.maximum(r_s[i], 0)
        r_n[i] = np.maximum(r_n[i], 0)
        p[i] = np.maximum(p[i], 0)

    return t, r_s, r_n, p


def p_func(x, b=1, sh=3):
    """
    Transfer function (decreasing sigmoid) of presynaptic inhibition (release probability)
    :param x:   input rate
    :param b:   slope of transfer function
    :param sh:  shift of transfer function
    :return:    'release probability'
    """
    #     return 1-0.1*x  #(linear presynaptic inhibition transfer function)
    return 1 - 1 / (1 + np.exp(-b * (x - sh)))


if __name__ in "__main__":

    dpi = 250

    #######################################
    # Measure bi-directional connectivity #
    #######################################

    # activate SOM and NDNF
    input_SOM = 2
    input_NDNF = 3.5
    t1, r_s_SOMact, r_n_SOMact, _ = simulate_circuit(400, input_SOM, input_NDNF, act_SOM=1)
    t1, r_s_NDNFact, r_n_NDNFact, _ = simulate_circuit(400, input_SOM, input_NDNF, act_NDNF=1)

    # plotting
    # csom, cndnf = 'cornflowerblue', 'coral'
    csom, cndnf = '#4084BF', '#EF8961'

    fig1, ax1 = plt.subplots(1, 2, dpi=dpi, figsize=(5, 2), sharey=True, gridspec_kw={'bottom': 0.2})
    t1 -= 100  # remove first 100 ms from plot
    ax1[0].plot(t1, r_n_SOMact, c=cndnf)
    ax1[0].plot(t1, r_s_SOMact, c=csom)
    ax1[1].plot(t1, r_n_NDNFact, c=cndnf)
    ax1[1].plot(t1, r_s_NDNFact, c=csom)
    ax1[0].hlines(4, 100, 150, linewidth=5, color=csom)
    ax1[1].hlines(4, 100, 150, linewidth=5, color=cndnf)
    ax1[0].set(ylim=[0, 4], xlabel='time (ms)', ylabel='activity (au)', title='activate SOM', xlim=[0, 300])
    ax1[1].set(xlabel='time (ms)', title='activate NDNF', xlim=[0, 300])

    #######################################
    # Measure bi-directional connectivity #
    #######################################

    inputs_SOM = np.arange(1, 5.1, 1.5)
    inputs_NDNF = np.arange(0, 6.1, 0.2)

    w = 1
    m, n = len(inputs_SOM), len(inputs_NDNF)
    som1 = np.zeros((n, m))
    som2 = np.zeros((n, m))
    ndnf = np.zeros((n, m))

    for ii, In in enumerate(inputs_NDNF):
        for jj, Is in enumerate(inputs_SOM):
            time, rate_SOM, rate_NDNF, p = simulate_circuit(500, Is, In, w=w, tau_p=20)

            som1[ii, jj] = rate_SOM[-1] * p[-1]
            som2[ii, jj] = rate_SOM[-1]
            ndnf[ii, jj] = rate_NDNF[-1]

    # plotting
    fig2, ax2 = plt.subplots(1, m, figsize=(5, 2), dpi=dpi, sharey=True, gridspec_kw={'bottom': 0.2})
    str_ndnf = ['weak', 'medium', 'strong']
    for ii in range(m):
        ax2[ii].plot(inputs_NDNF, som1[:, ii], c=csom, label='SOM upper')
        ax2[ii].plot(inputs_NDNF, som2[:, ii], ':', c=csom, label='SOM lower')
        ax2[ii].plot(inputs_NDNF, ndnf[:, ii], c=cndnf, label='NDNF upper')
        #     ax2[ii].plot(inputs_NDNF, ndnf[:,ii]+som1[:,ii], c='gray', alpha=0.8, label='total L1')
        ax2[ii].set(xlabel='NDNF input (au)', title=f"{str_ndnf[ii]} SOM input", xticks=[0, 5])
    ax2[0].set(ylabel='output strength (au)')
    ax2[0].legend(loc=(0.05, 0.7), handlelength=1, fontsize=6, frameon=False)

    plt.show()