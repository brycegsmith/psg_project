# This class should give an easy way of managing the querying and data of the raw PSG
# There will be 1 PSG object created per person

from constants import startTimes
import numpy as np
import pandas as pd
import re
import datetime
import mne


class PSG:
    def __init__(self, individual):
        self.individual = individual

        # Set up Metadata:
        # Defines gender, age, and condition for the individual:
        gender_age = pd.read_excel("gender-age.xlsx", header=None)
        # ^^^ Should probably move this so it doesn't have to run for every new PSG file
        self.gender = list(gender_age[gender_age[0] == individual.upper()][1])[0]
        self.age = list(gender_age[gender_age[0] == individual.upper()][2])[0]
        condition_code = re.search(r"\D+", individual).group()
        conditions = {
            "n": "Normal",
            "nfle": "Nocturnal Frontal Lobe Epilepsy",
            "ins": "Insomnia",
            "plm": "Periodic Leg Movement",
            "rbd": "REM Behavior Disorder",
        }
        self.condition = conditions[condition_code]

        # Extract Data from Text File (Sleep stage labels):
        # Reads txt file and filters to just sleep stages:
        df2 = pd.read_csv(individual + ".txt")[17:]
        df2 = df2.iloc[:, 0].str.split("\t", expand=True)
        df2.columns = [x for x in df2.iloc[0]]
        df2 = df2.iloc[1:]
        primary_loc = (
            df2.groupby("Location")
            .count()["Event"]
            .sort_values(ascending=False)
            .index[0]
        )
        df2 = df2[df2["Location"] == primary_loc]

        # Get start time information from constants.py
        study_start_year = startTimes[individual]["study_start_year"]
        study_start_month = startTimes[individual]["study_start_month"]
        study_start_day = startTimes[individual]["study_start_day"]
        study_start_hour = startTimes[individual]["study_start_hour"]
        study_start_min = startTimes[individual]["study_start_min"]
        study_start_second = startTimes[individual]["study_start_second"]

        # Calculates number of waveform measurements between study start time and first txt timestamp:
        timestamp_comps = df2["Time [hh:mm:ss]"].iloc[0].split(":")
        first_recording_day = (
            study_start_day + 1
            if int(timestamp_comps[0]) < study_start_hour
            else study_start_day
        )
        first_timestamp = datetime.datetime(
            study_start_year,
            study_start_month,
            first_recording_day,
            int(timestamp_comps[0]),
            int(timestamp_comps[1]),
            int(timestamp_comps[2]),
        )
        study_start = datetime.datetime(
            study_start_year,
            study_start_month,
            study_start_day,
            study_start_hour,
            study_start_min,
            study_start_second,
        )
        lag = 512 * (first_timestamp - study_start).seconds

        # Adds epochs and features:
        df2["epoch"] = [i for i in range(0, len(df2.iloc[:, 0]))]
        df2["condition"] = [self.condition for i in range(0, len(df2.iloc[:, 0]))]
        df2["gender"] = [self.gender for i in range(0, len(df2.iloc[:, 0]))]
        df2["age"] = [self.age for i in range(0, len(df2.iloc[:, 0]))]
        df2 = df2[["epoch", "Sleep Stage", "condition", "gender", "age"]]

        # Read in EDF:
        edf = mne.io.read_raw_edf(individual + ".edf")
        F = edf.get_data(start=lag)

        # Defines list of signals in edf
        signal = edf.ch_names
        signal = ["DX1-DX2" if x == "Dx1-DX2" else x for x in signal]
        signal = ["SAO2" if x == "SpO2" else x for x in signal]
        signal = ["Fp2-F4" if x == "F2-F4" else x for x in signal]

        # Adds epochs and defines column names based on signals
        D = F.transpose()
        df = pd.DataFrame(D)
        df.columns = signal
        df["elapsed_seconds"] = [(1 / 512) * i for i in range(0, len(df.iloc[:, 0]))]
        df["epoch"] = df["elapsed_seconds"] // 30
        df = df[
            [
                "epoch",
                "elapsed_seconds",
                "Fp2-F4",
                "F4-C4",
                "C4-P4",
                "P4-O2",
                "C4-A1",
                "ROC-LOC",
                "EMG1-EMG2",
                "ECG1-ECG2",
                "DX1-DX2",
                "SX1-SX2",
                "PLETH",
                "SAO2",
            ]
        ]

        self.txtData = df2
        self.data = df

        # Create an ECG Object:
        # self.ecg = ecg(data[["Timestamp", "ECG"]])

        # Create an EMG Object:
        # self.emg = emg(data[["Timestamp", "EMG"]])
