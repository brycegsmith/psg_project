# This class should provide an easy way of managing ECG data.
# It will also be where the ECG feature extraction lives.

import heartpy as hp
import pandas as pd
import numpy as np

class ECG:

    def __init__(self, ecgData, windowSize = 240):
        # Where ecgData is a dataframe with 2 columns: [Epoch, ECG column of PSG]
        # windowSize is window duration in seconds

        self.ecgData = ecgData
        self.lastEpoch = ecgData['epoch'].max()

        self.windowSizeSec = windowSize

        # Constants:
        samplingRate = 500
        windowStepSec = 30 # Length of epoch

        
        rawEcg = np.array(ecgData["ECG1-ECG2"])
        windowOverlap = (windowSize - windowStepSec)/windowSize
        print(windowOverlap)
        workingData, measures = hp.process_segmentwise( # Find heart beats and calculate heart rate metrics
            rawEcg, 
            samplingRate, 
            segment_width = self.windowSizeSec, 
            segment_overlap = windowOverlap,
            calc_freq = True,
            segment_min_size = self.windowSizeSec
        )

        self.workingData = workingData
        self.measures = measures # Dict containing HR metrics (such as BPM, RMSSD, LF, HF, etc.)
        

    def getRawDataByEpoch(self, epoch):
        # Return raw heart rate data [time, ECG] for the specified epoch (for visualization)
        return self.ecgData[self.ecgData["epoch"] == epoch]

    def getRawDataByTime(self, startTime, range):
        startTime = startTime * 500 # startTime * samplingRate to convert from desired second to desired sample
        range = range * 500 # Multiply by sampling rate to get number of datapoints 
        endTime = startTime + range
        return self.ecgData.loc[startTime:endTime, :]

    def getMetrics(self, metrics = ["bpm", "rmssd", "lf", "hf", "lf/hf"], normalized = True):
        # Metrics is a list of metrics to access
        # Returns a dataframe with format [epoch, metric1, metric2, etc]

        # Metric options:
        # bpm, ibi, sdnn, sdsd, rmssd, pnn50, pnn20, mad, lf, hf, lf/hf

        startEpoch = int(self.ecgData['epoch'].min())
        initializeEpochColumn = True
        metricsDf = pd.DataFrame()

        for metric in metrics:
            if initializeEpochColumn:
                thisMetricDf = pd.DataFrame(self.measures[metric], columns = [metric])
                epochCount = thisMetricDf[metric].size
                epochs = range(startEpoch, startEpoch + epochCount, 1)
                metricsDf = pd.DataFrame(epochs, columns = ["epoch"])
                metricsDf = metricsDf.join(thisMetricDf)

                if normalized:
                    thisMetricNorm = self.measures[metric] / np.mean(self.measures[metric])
                    thisMetricDfNorm = pd.DataFrame(thisMetricNorm, columns = [metric + "_norm"])
                    metricsDf = metricsDf.join(thisMetricDfNorm)

                initializeEpochColumn = False
            else:
                thisMetricDf = pd.DataFrame(self.measures[metric], columns = [metric])
                metricsDf = metricsDf.join(thisMetricDf)

                if normalized:
                    thisMetricNorm = self.measures[metric] / np.mean(self.measures[metric])
                    thisMetricDfNorm = pd.DataFrame(thisMetricNorm, columns = [metric + "_norm"])
                    metricsDf = metricsDf.join(thisMetricDfNorm)

        return metricsDf



    #def getBeats(self): # Find R waves of ECG signal
        # Return a dataframe: [timeOfBeat, amplitudeOfBeat/averageAmplitudeOfBeat]
        # pass

    #def heartRate(self, startTime=0, duration=0):
        #if startTime == 0 and duration == 0: 
            # Go through entire dataset and calculate Average HR for every epoch
        # Call getBeats
        # Return a dataframe: [epoch number, instantaneoustimestamp heart rate]


    #def rri(self, startTime = 0, duration = 0):
        # R-R intervals
        # Should probably normalize to average over entire dataset
    
    #def lf(self, startTime = 0, duration = 0):
        #if startTime == 0 and duration == 0: 
            # Go through entire dataset and calculate Average LF power for every epoch
        # Call getBeats
        # Return a dataframe: [timestamp, lf power]
        # Should probably normalize to average over entire dataset

    #def hf(self, startTime = 0, duration = 0):
        #pass

    #def rmssd(self, startTime = 0, duration = 0):
        #pass