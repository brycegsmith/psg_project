# This class should provide an easy way of managing ECG data.
# It will also be where the ECG feature extraction lives.

import heartpy as hp
import pandas as pd
import numpy as np

def removeNaN(dataframe):
    NaN_df = dataframe[dataframe.iloc[:,0].isna()]
    repaired = dataframe.copy()
    length = dataframe.shape[0]

    for i in list(NaN_df.index):
        pointsAbove = []
        pointsBelow = []
        pointsAboveCount = 0
        pointsBelowCount = 0
        minHalfWindow = 2
        includePointsAbove = True
        includePointsBelow = True
        searchDist = 1 # Start looking 1 above and 1 below
        while pointsAboveCount < minHalfWindow or pointsBelowCount < minHalfWindow:
            # Search up:
            if i - searchDist > 0:
                thisPointAbove = dataframe.iloc[i - searchDist, 0]
                if not np.isnan(thisPointAbove):
                    pointsAbove.append(thisPointAbove)
                    pointsAboveCount += 1
                    
            else:
                pointsAboveCount = minHalfWindow # So loop continues
                includePointsAbove = False

            # Search down:
            if i + searchDist < length:
                thisPointBelow = dataframe.iloc[i + searchDist, 0]
                if not np.isnan(thisPointBelow):
                    pointsBelow.append(thisPointBelow)
                    pointsBelowCount += 1
                    
            else:
                pointsBelowCount = minHalfWindow # So loop continues
                includePointsBelow = False

            searchDist += 1
        
        # Calculate average:
        if includePointsBelow and includePointsAbove:
            average = (sum(pointsAbove) + sum(pointsBelow)) / (sum([len(pointsBelow), len(pointsAbove)]))
        elif includePointsBelow:
            average = sum(pointsBelow) / len(pointsBelow)
        elif includePointsAbove:
            average = sum(pointsAbove) / len(pointsAbove)
        else:
            average = 0
        
        repaired.iloc[i, 0] = average
    
    return repaired


class ECG:

    def __init__(self, ecgData, windowSize = 240, signalType = 'ECG1-ECG2'):
        # Where ecgData is a dataframe with 2 columns: [Epoch, ECG column of PSG]
        # windowSize is window duration in seconds

        self.ecgData = ecgData
        self.lastEpoch = ecgData['epoch'].max()
        self.signalType = signalType
        self.windowSizeSec = windowSize

        # Constants:
        self.samplingRate = 512
        windowStepSec = 30 # Length of epoch

        
        if self.signalType == 'ECG1-ECG2':
            rawEcg = np.array(ecgData["ECG1-ECG2"])
        elif self.signalType == 'PLETH':
            rawEcg = np.array(ecgData["PLETH"])
        windowOverlap = (windowSize - windowStepSec)/windowSize
        workingData, measures = hp.process_segmentwise( # Find heart beats and calculate heart rate metrics
            rawEcg, 
            self.samplingRate, 
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
        startTime = startTime * self.samplingRate # startTime * samplingRate to convert from desired second to desired sample
        range = range * self.samplingRate # Multiply by sampling rate to get number of datapoints 
        endTime = startTime + range
        return self.ecgData.loc[startTime:endTime, :]

    def getMetrics(self, metrics = ["bpm", "rmssd", "lf", "hf", "lf/hf", "breathingrate"], normalized = True):
        # Metrics is a list of metrics to access
        # Returns a dataframe with format [epoch, metric1, metric2, etc]

        # Metric options:
        # bpm, ibi, sdnn, sdsd, rmssd, pnn50, pnn20, mad, lf, hf, lf/hf, breathingrate

        startEpoch = int(self.ecgData['epoch'].min())
        initializeEpochColumn = True
        metricsDf = pd.DataFrame()

        for metric in metrics:
            if initializeEpochColumn:
                thisMetricDf = pd.DataFrame(self.measures[metric], columns = [metric + "_" + self.signalType])
                # Repair NaN's:
                thisMetricDf = removeNaN(thisMetricDf)

                epochCount = thisMetricDf[metric + "_" + self.signalType].size
                epochs = range(startEpoch, startEpoch + epochCount, 1)
                metricsDf = pd.DataFrame(epochs, columns = ["epoch"])
                metricsDf = metricsDf.join(thisMetricDf)

                if normalized:
                    thisMetricNorm = thisMetricDf.iloc[:,0].to_numpy() / np.mean(thisMetricDf.iloc[:,0].to_numpy())
                    thisMetricDfNorm = pd.DataFrame(thisMetricNorm, columns = [metric + "_norm" + "_" + self.signalType])
                    metricsDf = metricsDf.join(thisMetricDfNorm)

                initializeEpochColumn = False
            else:
                thisMetricDf = pd.DataFrame(self.measures[metric], columns = [metric + "_" + self.signalType])
                # Repair NaN's:
                thisMetricDf = removeNaN(thisMetricDf)

                metricsDf = metricsDf.join(thisMetricDf)

                if normalized:
                    thisMetricNorm = thisMetricDf.iloc[:,0].to_numpy() / np.mean(thisMetricDf.iloc[:,0].to_numpy())
                    thisMetricDfNorm = pd.DataFrame(thisMetricNorm, columns = [metric + "_norm" + "_" + self.signalType])
                    metricsDf = metricsDf.join(thisMetricDfNorm)

        return metricsDf