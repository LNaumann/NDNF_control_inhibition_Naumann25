# Layer-specific control of inhibition by NDNF interneurons

Publication code for Naumann et al., 2024

## Overview

This repository contains the code for reproducing the computational experiments for the research article "Layer-specific control of inhibition by NDNF interneurons" (Naumann, Hertäg, Müller, Letzkus & Sprekeler, 2024).

## Network model

The network model is implemented as a class `NetworkModel` in `code/model_base.py`. The script also contains an example of how to run the network model and visualise the activity traces of each neuron type.

## Experiments for the publication figures

The scripts for running the experiments shown in the publication are `exp_fig...py`. They contain individual methods running different experiments and you can run the whole script to obtain all simulation results for a figure:

- Figure 3: `exp_fig3_competition.py`
- Figure 4: `exp_fig4_switching.py`
- Figure 5: `exp_fig5_timescale.py`
- Figure 6: `exp_fig6_predictive_coding.py`

At the bottom of each script the different methods are called and you can decide whether to save figures (set `SAVE=True`) and whether to plot the supplementary figures (set `plot_supps=True`). 
