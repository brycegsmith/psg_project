# This class should give an easy way of managing the querying and data of the raw PSG
# There will be 1 PSG object created per person

class PSG:

    def __init__(self, filename, start):
        self.filename = filename

        self.metadata = {"age" = }

        edf = mne.io.read_raw_edf(filename) ### Replace with code to query Cloud database
        header = ','.join(edf.ch_names)

        ### Ashok's code to create dataframe:
        #
        #
        #
        # self.data = edfDataframe # Create the dataframe using Ashok's code. Then assign to object property


        # Create an ECG Object:
        self.ecg = ecg(data[["Timestamp", "ECG"]])

        # Create an EMG Object:
        self.emg = emg(data[["Timestamp", "EMG"]])