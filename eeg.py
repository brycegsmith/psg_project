"""
Class to handle feature extraction of EEG data from PSG object.

Created on 06/20/2022

@author: Bryce Smith (brycegsmith@hotmail.com)
"""

import constants
import pandas as pd


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

        # Create dataframe to store features
        self.eeg_features = pd.DataFrame({"epoch": self.eeg_data["epoch"].unique()})

        # Populate features

    def get_eeg_data_by_epoch(self, epoch):
        """
        Initialize EEG object.

        Parameters:
            epoch (int): epoch of interest

        Returns:
            eeg_data_epoch (DataFrame): dataframe of raw EEG data for specified epoch
        """

        eeg_data_epoch = self.eeg_data[self.eeg_data["epoch"] == epoch]
        return eeg_data_epoch
