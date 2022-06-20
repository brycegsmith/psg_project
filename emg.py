# This class should provide an easy way of managing EMG data.
# It will also be where the EMG feature extraction lives.

import numpy as np
from scipy import signal
import pandas as pd

# Helper function to get energy of a signal:
def energy(signal):
    # Remove DC offset (find and subtract the mean)
    dcOffset = signal.mean()
    signal = signal - dcOffset

    # Square the signal
    signal = signal ** 2

    # Sum the signal
    return signal.sum()

class EMG:

    def __init__(self, emgData):
        # Where emgData is a dataframe with 2 columns: [Timestamp, ECG column of PSG]
        self.emgData = emgData
        self.rawEmg = np.array(emgData.loc[:, "EMG1-EMG2"])

    def emgEnergy(self, filterOrder = 10, hpCutoff = 30, lpCutoff = 200):
        # Filter
        filterOrder = 10
        hpCutoff = 30
        lpCutoff = 200
        samplingRate = 500
        sosBandpass = signal.butter(filterOrder, (hpCutoff, lpCutoff), btype = "bandpass", fs = samplingRate, output = "sos")
        filteredEmg = signal.sosfilt(sosBandpass, self.rawEmg)
        self.emgData['filtered'] = filteredEmg

        filteredWithEpochs = self.emgData[['epoch', 'filtered']]

        filteredSignalGroupedByEpoch = (filteredWithEpochs.groupby(['epoch']))['filtered']
        energyByEpoch = filteredSignalGroupedByEpoch.apply(energy)
        return energyByEpoch