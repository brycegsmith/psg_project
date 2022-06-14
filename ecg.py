# This class should provide an easy way of managing ECG data.
# It will also be where the ECG feature extraction lives.

class ECG:

    def __init__(self, ecgData):
        # Where ecgData is a dataframe with 2 columns: [Timestamp, ECG column of PSG]
        self.ecgData = ecgData

    def getBeats(self): # Find R waves of ECG signal
        # Return a dataframe: [timeOfBeat, amplitudeOfBeat/averageAmplitudeOfBeat]
        pass

    def heartRate(self, startTime=0, duration=0):
        if startTime == 0 and duration == 0: 
            # Go through entire dataset and calculate Average HR for every epoch
        # Call getBeats
        # Return a dataframe: [epoch number, instantaneoustimestamp heart rate]


    def rri(self, startTime = 0, duration = 0):
        # R-R intervals
        # Should probably normalize to average over entire dataset
    
    def lf(self, startTime = 0, duration = 0):
        if startTime == 0 and duration == 0: 
            # Go through entire dataset and calculate Average LF power for every epoch
        # Call getBeats
        # Return a dataframe: [timestamp, lf power]
        # Should probably normalize to average over entire dataset

    def hf(self, startTime = 0, duration = 0):
        pass

    def rmssd(self, startTime = 0, duration = 0):
        pass