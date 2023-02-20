"""
Experiments 3: Slow inhibition by NDNF interneurons preferentially transmits certain signals.
- 
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('pretty')
import model_base as mb
from experiments import get_null_ff_input_arrays, get_model_colours, plot_violin

cPC, cPV, cSOM, cNDNF, cVIP, cpi = get_model_colours()

DPI = 150


def exp301_frequency_preference(noise=0.0, wNS=1.4, w_hetero=False, mean_pop=True, pre_inh=True, reduced=False, save=False):
    """
    TODO
    """

    freqs = np.array([0.5, 1, 2, 4, 8, 16])  # np.arange(1, 20)

    # define parameter dictionaries
    N_cells, w_mean, conn_prob, bg_inputs, taus = mb.get_default_params(flag_mean_pop=mean_pop)

    if reduced:  # remove 
        w_mean['EP'], w_mean['PE'], w_mean['SE'] = 0, 0, 0

    # instantiate model
    model = mb.NetworkModel(N_cells, w_mean, conn_prob, taus, bg_inputs, wED=1, flag_w_hetero=w_hetero,
                           flag_pre_inh=pre_inh)

    # simulation paramters
    dur = 5000
    dt = 1
    nt = int(dur/dt)


    amplitudes_n = np.zeros(len(freqs))
    amplitudes_s = np.zeros(len(freqs))

    # fig, ax = plt.subplots(2, 1, figsize=(3, 3), dpi=150)

    for i, ff in enumerate(freqs):

        # make input
        xff_null = np.zeros(nt)  
        xff_sine = make_sine(nt, ff)-0.5

        # simulate with sine input to SOM
        xFF = get_null_ff_input_arrays(nt, N_cells)
        xFF['S'] = np.tile(xff_sine, [N_cells['S'], 1]).T
        xFF['N'] = np.tile(xff_null, [N_cells['N'], 1]).T
        t, rE1, rD1, rS1, rN1, rP1, rV1, p1, cGABA1, other1 = model.run(dur, xFF, dt=dt, p0=0.5, monitor_dend_inh=True)

        # simulate with sine input to NDNF
        xFF = get_null_ff_input_arrays(nt, N_cells)
        xFF['S'] = np.tile(xff_null, [N_cells['S'], 1]).T
        xFF['N'] = np.tile(xff_sine, [N_cells['N'], 1]).T
        t, rE2, rD2, rS2, rN2, rP2, rV2, p2, cGABA2, other2 = model.run(dur, xFF, dt=dt, p0=0.5, monitor_dend_inh=True)

        # quantify   


        


def make_sine(nt, freq):
    t = np.arange(nt)/1000
    return (np.sin(2*np.pi*freq*t)+1)/2


if __name__ in "__main__":


    plt.show()
