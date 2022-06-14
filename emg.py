# This class should provide an easy way of managing EMG data.
# It will also be where the EMG feature extraction lives.

class EMG:

    def __init__(self, emgData):
        # Where emgData is a dataframe with 2 columns: [Timestamp, ECG column of PSG]
        self.emgData = emgData