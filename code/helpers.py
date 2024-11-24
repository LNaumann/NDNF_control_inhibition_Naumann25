import numpy as np
import matplotlib.pyplot as plt


def get_null_ff_input_arrays(nt, N_cells):
    """
    Generate empty arrays for feedforward input.

    Parameters:
    ----------
    - nt: number of timesteps
    - N_cells: dictionary of cell numbers (soma E, dendrite E, SOMs S, NDNFs N, PVs P)

    Returns:
    -------
    - dictionary with an empty array of size nt x #cell for each neuron type
    """

    xFF_null = dict()
    for key in N_cells.keys():
        xFF_null[key] = np.zeros((nt, N_cells[key]))

    return xFF_null


def get_model_colours():
    """
    Get the colours for the different cell types.
    """

    # colours
    cPC = '#B83D49'
    cPV = '#345377'
    cSOM = '#5282BA'
    cNDNF = '#E18E69'
    cVIP = '#D1BECF'
    cpi = '#A7C274'
    return cPC, cPV, cSOM, cNDNF, cVIP, cpi


def make_sine(nt, freq, plot=False):
    """
    Make a sine wave.

    Parameters:
    ----------
    - nt: number of time points
    - freq: frequency of the sine wave
    - plot: whether to plot the sine wave

    Returns:
    -------
    - sine wave as numpy array
    """
    t = np.arange(nt)/1000
    sine = (np.sin(2*np.pi*freq*t)+1)/2
    if plot:
        plt.figure(figsize=(3, 2), dpi=300)
        plt.plot(t, sine, lw=1, c='k')
    return sine


def slice_dict(dic, ts, te):
    """
    Slice a dictionary of arrays.

    Parameters:
    ----------
    - dic: dict, dictionary of arrays
    - ts:  int, start index
    - te:  int, end index

    Returns:
    -------
    - dic_new: dict, sliced dictionary
    """

    dic_new = dict()
    for k in dic.keys():
        dic_new[k] = dic[k][ts:te]

    return dic_new