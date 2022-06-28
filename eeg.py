"""
Class to handle feature extraction of EEG data from PSG object.

Created on 06/20/2022

@author: Bryce Smith (brycegsmith@hotmail.com)
"""

import constants
import pandas as pd
import numpy as np
from scipy import signal
from scipy.integrate import simps
import seaborn as sns
from matplotlib import pyplot as plt


class EEG:
    def __init__(self, psg_df):
        """
        Initialize EEG object.

        Parameters:
            psg_df (DataFrame): Dataframe of individual's PSG data.

        Returns:
            None
        """

        # Build dataframe of EEG data
        eeg_columns_time = ["epoch", "elapsed_seconds"]
        eeg_columns = eeg_columns_time
        for column in list(psg_df.columns):
            if column in constants.EEG_COLUMNS:
                eeg_columns.append(column)
        self.eeg_data = psg_df.loc[:, eeg_columns]

        # Create dataframe to store features
        self.eeg_features = pd.DataFrame({"epoch": self.eeg_data["epoch"].unique()})

        # Populate features
        for column in eeg_columns:
            if column not in eeg_columns_time:
                print(column)

    def get_eeg_dataframe_by_epoch(self, epoch):
        """
        Get raw EEG data for a specified epoch. Useful for plotting.

        Parameters:
            epoch (int): epoch of interest

        Returns:
            eeg_dataframe_epoch (DataFrame): dataframe of EEG data for specified epoch
        """

        eeg_dataframe_epoch = self.eeg_data[self.eeg_data["epoch"] == epoch]
        return eeg_dataframe_epoch

    def get_eeg_vector_epoch(self, epoch, channel):
        """
        Get vector of EEG data as numpy vector. Use for efficient feature extraction.
        Single epoch.

        Parameters:
            epoch (int): epoch of interest
            channel (str): EEG channel of interest

        Returns:
            eeg_vector_epoch (array): 1D array of EEG data for specified epoch
        """

        mask = self.eeg_data["epoch"] == epoch
        eeg_vector_epoch = self.eeg_data.loc[mask, [channel]].values.flatten()
        return eeg_vector_epoch

    def get_eeg_vector_epoch_range(self, epoch_start, epoch_end, channel):
        """
        Get vector of EEG data as numpy vector. Use for efficient feature extraction.
        Epoch range.

        Parameters:
            epoch_start (int): starting epoch (inclusive)
            epoch_end (int): ending epoch (exclusive)
            channel (str): EEG channel of interest

        Returns:
            eeg_vector_epochs (array): 1D array of EEG data for specified epochs
        """

        mask = (self.eeg_data["epoch"] >= epoch_start) & (
            self.eeg_data["epoch"] < epoch_end
        )
        eeg_vector_epochs = self.eeg_data.loc[mask, [channel]].values.flatten()
        return eeg_vector_epochs

    def preprocess(self, eeg_vector):
        """
        Preprocess data to remove artifacts & focus on only relevant signals

        Parameters:
            eeg_vector (array): 1D array of EEG data

        Returns:
            eeg_vector_preprocessed (array): preprocessed 1D array of EEG data
        """

        # Convert voltage (V to uV)
        eeg_vector = eeg_vector * 10 ** 6

        # 5-th order bandpass filter with cutoffs of 0.5 - 45 Hz (Forward & Backward)
        b, a = signal.butter(5, [0.5, 45], "bp", fs=500)
        eeg_vector = signal.filtfilt(b, a, eeg_vector, axis=0)

        # Return vector
        eeg_vector_preprocessed = eeg_vector
        return eeg_vector_preprocessed

    def plot_eeg(self, eeg_vector):
        """
        Plot processed EEG data. Vector must have sampling frequency of 500.
        15-30 second intervals are recommended for data clarity.

        Parameters:
            eeg_vector (array): 1D array of EEG processed data

        Returns:
            None
        """

        # Extract time from vector
        time = np.linspace(0, len(eeg_vector) / constants.FREQUENCY, len(eeg_vector))

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(25, 4))
        plt.plot(time, eeg_vector, lw=1.5, color="k")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Voltage (uV)")
        plt.xlim([0, time[len(time) - 1]])
        plt.title("EEG Data")

    def plot_welch_psd(self, eeg_vector):
        """
        Plot non-log transformed Welch power spectral density (PSD) of EEG vector.

        See here for more info: https://raphaelvallat.com/bandpower.html

        Parameters:
            eeg_vector (array): 1D array of EEG processed data

        Returns:
            None
        """

        # Window size to capture minium frequency (4 seconds)
        fq = constants.FREQUENCY
        win = int(4 * fq)

        # Calculate Welch PSD (Median Averaging)
        freqs, psd = signal.welch(eeg_vector, fq, nperseg=win, average="median")

        # Plot
        plt.plot(freqs, psd, "k", lw=2)
        plt.fill_between(freqs, psd, cmap="Spectral")
        plt.xlim(0, 30)
        sns.despine()
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("PSD ($uV^2$/Hz)")

    def relative_psd_power(self, eeg_vector):
        """
        Calculate relative power for each brain wave using non-log transformed Welch
        power spectral density (PSD) of EEG vector.

        See here for more info: https://raphaelvallat.com/bandpower.html

        Brain Waves
            Beta: 12-30 Hz
            Alpha: 8-12 Hz
            Theta: 4-8 Hz
            Delta: 0.5-4 Hz

        Parameters:
            eeg_vector (array): 1D array of EEG processed data

        Returns:
            None
        """

        # Window size to capture minium frequency (4 seconds)
        fq = constants.FREQUENCY
        win = int(4 * fq)

        # Calculate Welch PSD (Median Averaging)
        freqs, psd = signal.welch(eeg_vector, fq, nperseg=win, average="median")

        # Create list of waves
        wave_names = ["Beta", "Alpha", "Theta", "Delta"]
        wave_freqs = [[12, 30], [8, 12], [4, 8], [0.5, 4]]

        # Frequency resolution
        freq_res = freqs[1] - freqs[0]  # = 1 / 4 = 0.25

        # Total Power
        total_power = simps(psd, dx=freq_res)

        for i in range(len(wave_names)):
            # Define low and high wave frequency cutoffs
            low = wave_freqs[i][0]
            high = wave_freqs[i][1]

            # Find intersecting values in frequency vector
            idx = np.logical_and(freqs >= low, freqs <= high)

            # Compute the absolute power by approximating the area under the curve
            power = simps(psd[idx], dx=freq_res)

            # Relative delta power (expressed as a percentage of total power)
            rel_power = power / total_power
            print(wave_names[i] + " Relative power: %.3f" % rel_power)
