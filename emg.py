# This class should provide an easy way of managing EMG data.
# It will also be where the EMG feature extraction lives.

from tracemalloc import start
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

    def __init__(self, emgData, filterOrder = 10, hpCutoff = 30, lpCutoff = 200):
        # Where emgData is a dataframe with 2 columns: [Epoch, ECG column of PSG]
        self.emgData = emgData
        self.rawEmg = np.array(emgData.loc[:, "EMG1-EMG2"])
        self.length = len(self.rawEmg)
        self.samplingRate = 500

        # Filter the signal:
        filterOrder = 10
        hpCutoff = 30
        lpCutoff = 200
        samplingRate = 500
        sosBandpass = signal.butter(filterOrder, (hpCutoff, lpCutoff), btype = "bandpass", fs = samplingRate, output = "sos")
        filteredEmg = signal.sosfilt(sosBandpass, self.rawEmg)
        self.emgData['filtered'] = filteredEmg


    def getRawDataByEpoch(self, epoch, filtered = True):
        # Return raw heart rate data [time, ECG] for the specified epoch (for visualization)
        
        # Get EMG dataframe at specified epoch:
        emgDf = self.emgData[self.emgData["epoch"] == epoch]
        if filtered:
            return emgDf["filtered"]
        else:
            return emgDf["EMG1-EMG2"]

    def getRawDataByTime(self, startTime, range, filtered = True):
        startTime = startTime * 500 # startTime * samplingRate to convert from desired second to desired sample
        range = range * 500 # Multiply by sampling rate to get number of datapoints 
        endTime = startTime + range

        # Get EMG dataframe at specified time:
        emgDf = self.emgData.loc[startTime:endTime, :]
        if filtered:
            return emgDf["filtered"]
        else:
            return emgDf["EMG1-EMG2"]
    
    def emgEnergy(self, window = 30, windowStep = 1):
        if window == 30: # Get energy for each epoch
            filteredWithEpochs = self.emgData[['epoch', 'filtered']]

            filteredSignalGroupedByEpoch = (filteredWithEpochs.groupby(['epoch']))['filtered']
            energyByEpoch = filteredSignalGroupedByEpoch.apply(energy)
            return energyByEpoch

        else: # Do sliding window calculation
            # Create a column to group by
            windowSizeIndices = window * self.samplingRate
            windowStepSize = windowStep * self.samplingRate # How many indices to move window by
            windowCount = (self.length - windowSizeIndices) + 1
            #windowCount = (self.length / windowStepSize) - 2 * int(window/2)
                # Subtract 1 window because the first half of the first window will not get a window
                # and the second half of the last window will not get a window (because beyond ends of array)
            print(self.length)
            indexer = np.arange(windowSizeIndices)[None, :] + np.arange(0, windowCount, windowStepSize)[:, None]
            indexer = indexer.astype(int)

            filteredEmg = (self.emgData["filtered"]).to_numpy()
            windowApplied = filteredEmg[indexer]
            energyByWindow = np.apply_along_axis(energy, 1, windowApplied)
                # This will have the first and last floor(window/2) elements removed.
            
            leadValues = np.ones(int(window/2)) * energyByWindow[0] # Set leading values to value of first complete window
            trailingValues = np.ones(int(window/2)) * energyByWindow[-1] # Set trailing values to value of last complete window
            energyByWindow = np.concatenate([leadValues, energyByWindow, trailingValues])
            print(len(energyByWindow))
            return energyByWindow

    def getNHighestSeconds(self, n, window):
        # Gets the average value of the n highest w-second windows within an epoch. Where w is window size in seconds.
        
        energyByWindow = self.emgEnergy(window = window)

        # Need to get list of epochs corresponding to correct energyByWindow values
        # energyByWindow is an w-second window for every 1-second in the emg data
        reindexedEmg = self.emgData.reset_index()
        startingEpoch = reindexedEmg.loc[0, 'epoch']
        endEpoch = reindexedEmg['epoch'].iloc[-1]
        pointsInStartingEpoch = self.emgData[self.emgData['epoch'] == startingEpoch]
        numPointsInStartingEpoch = pointsInStartingEpoch['epoch'].count()
        pointsInEndEpoch = self.emgData[self.emgData['epoch'] == endEpoch]
        numPointsInEndEpoch = pointsInEndEpoch['epoch'].count()

        numSecondsStartEpoch = int(numPointsInStartingEpoch / self.samplingRate)
        numSecondsEndEpoch = int(numPointsInEndEpoch / self.samplingRate)

        startEpochArray = np.ones(numSecondsStartEpoch) * startingEpoch
        endEpochArray = np.ones(numSecondsEndEpoch) * endEpoch

        #numberIntermediateEpochs = endEpoch - startingEpoch - 1
        stepSize = 1/30
        intermediateEpochs = np.arange(startingEpoch + 1, endEpoch, stepSize).astype(int)
        
        # Epochs:
        epochs = np.concatenate([startEpochArray, intermediateEpochs, endEpochArray])
        print(len(epochs))
        print(len(energyByWindow))

        # Create dataframe: [epoch, energyByWindow]
        windowEnergyByEpoch = pd.DataFrame(epochs, columns = ['epoch'])
        windowEnergyByEpoch['energy'] = energyByWindow

        # Group by epochs
        groupedByEpoch = windowEnergyByEpoch.groupby(['epoch'])
        
        # Pick out n largest from each epoch
        nLargestByEpoch = groupedByEpoch['energy'].nlargest(n)
        #display(nLargestByEpoch)

        # Get average of n-largest:
        averagesByEpoch = nLargestByEpoch.groupby(['epoch']).mean()
        return averagesByEpoch

    def getMetrics(self, metrics = ["epochEnergy", "5H5S", "epochEnergy_norm", "5H5S_norm"]):
        # Metrics is a list of metrics to access
        # Returns a dataframe with format [epoch, metric1, metric2, etc]

        # Metric options:
        # epochEnergy - Energy calculated over entire epoch
        # 5H5S - Average of the 5 highest-energy 5 second periods within epoch
        # epochEnergy_norm - Energy calculated over entire epoch divided by average energy of all epochs
        # 5H5S - 5H5S divided by average energy of all epochs

        reindexedEmg = self.emgData.reset_index()
        startingEpoch = int(reindexedEmg.loc[0, 'epoch'])
        endEpoch = int(reindexedEmg['epoch'].iloc[-1])

        metricsDf = pd.DataFrame(range(startingEpoch, endEpoch), columns = ["epoch"])
        print(metricsDf)

        if "epochEnergy" in metrics:
            epochEnergy = self.emgEnergy()
            epochEnergy = epochEnergy.to_frame() # Convert series to dataframe
            epochEnergy.index = epochEnergy.index.astype(int) # Set index to type int for merge to work correctly, otherwise get NaN
            metricsDf = metricsDf.merge(epochEnergy, on = 'epoch')
            metricsDf = metricsDf.rename(columns = {'filtered': 'EMG_epochEnergy'})

        if "epochEnergy_norm" in metrics:
            metricsDf["EMG_epochEnergy_norm"] = metricsDf["EMG_epochEnergy"] / metricsDf["EMG_epochEnergy"].mean()

        if "5H5S" in metrics:
            fiveH5S = self.getNHighestSeconds(5, window = 5)
            fiveH5S = fiveH5S.to_frame() # Convert series to dataframe
            fiveH5S.index = fiveH5S.index.astype(int)
            metricsDf = metricsDf.merge(fiveH5S, on = 'epoch')
            metricsDf = metricsDf.rename(columns = {'energy': "EMG_5Highest5Sec"})

        if "5H5S_norm" in metrics:
            metricsDf["EMG_5Highest5Sec_norm"] = metricsDf["EMG_5Highest5Sec"] / metricsDf["EMG_epochEnergy"].mean()

        return metricsDf