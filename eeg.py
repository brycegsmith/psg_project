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
import antropy as ant


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
        eeg_columns = ["epoch", "elapsed_seconds"]
        for column in list(psg_df.columns):
            if column in constants.EEG_COLUMNS:
                eeg_columns.append(column)
        self.eeg_data = psg_df.loc[:, eeg_columns]

        # Preprocess data
        for i in range(2, self.eeg_data.shape[1]):
            column = self.eeg_data.columns.values.tolist()[i]
            channel_vector = self.eeg_data.loc[:, column].values
            self.eeg_data[column] = self.preprocess(channel_vector)

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

    def extract_features(self):
        """
        Perform feature extraction on EEG object.

        Parameters:
            None

        Returns:
            None
        """
        # Get channel columns
        eeg_channels = self.eeg_data.columns.values.tolist()
        eeg_channels.remove("epoch")
        eeg_channels.remove("elapsed_seconds")

        # Create dataframe to store features
        self.eeg_features = pd.DataFrame({"epoch": self.eeg_data["epoch"].unique()})
        self.eeg_features["beta_relative_power"] = np.nan
        self.eeg_features["alpha_relative_power"] = np.nan
        self.eeg_features["theta_relative_power"] = np.nan
        self.eeg_features["delta_relative_power"] = np.nan
        self.eeg_features["perm_entropy"] = np.nan
        self.eeg_features["spectral_entropy"] = np.nan
        self.eeg_features["svd_entropy"] = np.nan
        self.eeg_features["approx_entropy"] = np.nan
        self.eeg_features["sample_entropy"] = np.nan
        self.eeg_features["petrosian"] = np.nan
        self.eeg_features["katz"] = np.nan
        self.eeg_features["higuchi"] = np.nan
        self.eeg_features["dfa"] = np.nan

        # Populate features
        for epoch in self.eeg_data["epoch"].unique():
            # Relative Powers
            rps, _ = self.relative_psd_power_avg(epoch, eeg_channels)
            self.eeg_features.at[epoch, "beta_relative_power"] = rps[0]
            self.eeg_features.at[epoch, "alpha_relative_power"] = rps[1]
            self.eeg_features.at[epoch, "theta_relative_power"] = rps[2]
            self.eeg_features.at[epoch, "delta_relative_power"] = rps[3]

            # Antropy Metrics
            antropy, _ = self.antropy_metrics_avg(epoch, eeg_channels)
            self.eeg_features.at[epoch, "perm_entropy"] = antropy[0]
            self.eeg_features.at[epoch, "spectral_entropy"] = antropy[1]
            self.eeg_features.at[epoch, "svd_entropy"] = antropy[2]
            self.eeg_features.at[epoch, "approx_entropy"] = antropy[3]
            self.eeg_features.at[epoch, "sample_entropy"] = antropy[4]
            self.eeg_features.at[epoch, "petrosian"] = antropy[5]
            self.eeg_features.at[epoch, "katz"] = antropy[6]
            self.eeg_features.at[epoch, "higuchi"] = antropy[7]
            self.eeg_features.at[epoch, "dfa"] = antropy[8]

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
            relative_powers (list): [beta_rp, alpha_rp, theta_rp, delta_rp]
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

        # Relative Power List
        relative_powers = []

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
            relative_powers.append(rel_power)

        return relative_powers

    def relative_psd_power_avg(self, epoch, eeg_channels):
        """
        Calculate relative power for each brain wave using non-log transformed Welch
        power spectral density (PSD) for a single epoch and AVERAGE ACROSS CHANNELS.

        See here for more info: https://raphaelvallat.com/bandpower.html

        Brain Waves
            Beta: 12-30 Hz
            Alpha: 8-12 Hz
            Theta: 4-8 Hz
            Delta: 0.5-4 Hz

        Parameters:
            epoch (int): epoch to average data
            eeg_channels (list of strs): list of channels to average data across

        Returns:
            average_relative_powers (np.array): 1D vector of averaged relative powers
            channel_relative_powers (np.array): matrix of channel's relative powers

            Both ordered by beta_rp, alpha_rp, theta_rp, delta_rp
        """

        channel_relative_powers = np.zeros((len(eeg_channels), 4))

        # Populate features
        for i in range(len(eeg_channels)):
            # Get channel
            channel = eeg_channels[i]

            # Get vector
            eeg_vector = self.get_eeg_vector_epoch(epoch, channel)

            # Calculate relative powers
            rps = self.relative_psd_power(eeg_vector)
            channel_relative_powers[i, :] = rps

            # Average across channels
            average_relative_powers = np.mean(channel_relative_powers, axis=0)

        return average_relative_powers, channel_relative_powers

    def antropy_metrics(self, eeg_vector):
        """
        Calculate all metrics associated eith Raphael Vallat's antropy library.

        See here for more info:  https://raphaelvallat.com/antropy/build/html/index.html

        Parameters:
            eeg_vector (array): 1D array of EEG processed data

        Returns:
            antropy_features (list): list of antropy features in order below
                * Permutation Entropy
                * Spectral Entropy
                * Singular Value Decomposition Entropy
                * Approximate Entropy
                * Sample Entropy
                * Petrosian Fractal Dimension
                * Katz Fractal Dimension
                * Higuchi Fractal Dimension
                * Derended Fluctuation Analysis
        """

        # Initialize list
        antropy_features = []

        # Permutation Entropy
        antropy_features.append(ant.perm_entropy(eeg_vector, normalize=True))

        # Spectral Entropy
        sf = constants.FREQUENCY
        antropy_features.append(
            ant.spectral_entropy(eeg_vector, sf=sf, method="welch", normalize=True)
        )

        # Singular Value Decomposition Entropy
        antropy_features.append(ant.svd_entropy(eeg_vector, normalize=True))

        # Approximate Entropy
        antropy_features.append(ant.app_entropy(eeg_vector))

        # Sample Entropy
        antropy_features.append(ant.sample_entropy(eeg_vector))

        # Petrosian Fractal Dimension
        antropy_features.append(ant.petrosian_fd(eeg_vector))

        # Katz Fractal Dimension
        antropy_features.append(ant.katz_fd(eeg_vector))

        # Higuchi Fractal Dimension
        antropy_features.append(ant.higuchi_fd(eeg_vector))

        # Derended Fluctuation Analysis
        antropy_features.append(ant.detrended_fluctuation(eeg_vector))

        return antropy_features

    def antropy_metrics_avg(self, epoch, eeg_channels):
        """
        Calculate all metrics associated eith Raphael Vallat's antropy library for a
        single epoch and AVERAGE ACROSS CHANNELS.

        See here for more info:  https://raphaelvallat.com/antropy/build/html/index.html

        Parameters:
            epoch (int): epoch to average data
            eeg_channels (list of strs): list of channels to average data across

        Returns:
            average_antropy_metrics (np.array): 1D vector of averaged antropy metrics
            channel_antropy_metrics (np.array): matrix of antropy metrics

            Antropy Features Ordered By:
                * Permutation Entropy
                * Spectral Entropy
                * Singular Value Decomposition Entropy
                * Approximate Entropy
                * Sample Entropy
                * Petrosian Fractal Dimension
                * Katz Fractal Dimension
                * Higuchi Fractal Dimension
                * Derended Fluctuation Analysis
        """

        channel_antropy_metrics = np.zeros((len(eeg_channels), 9))

        # Populate features
        for i in range(len(eeg_channels)):
            # Get channel
            channel = eeg_channels[i]

            # Get vector
            eeg_vector = self.get_eeg_vector_epoch(epoch, channel)

            # Calculate relative powers
            antropy_metrics = self.antropy_metrics(eeg_vector)
            channel_antropy_metrics[i, :] = antropy_metrics

            # Average across channels
            average_antropy_metrics = np.mean(channel_antropy_metrics, axis=0)

        return average_antropy_metrics, channel_antropy_metrics
