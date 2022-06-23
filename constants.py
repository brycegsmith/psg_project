# Start Times dict with the needed start time params associated with each EDF
# Data Previews: https://archive.physionet.org/cgi-bin/atm/ATM
# Set Input:
#    Set Database to CAP Sleep Database (capslpdb)
#    Set Record to the Correct Individual
# Set Toolbox to "Export Signals as CSV"
# You should be able to see the start time of the study.

# Copy the start time parameters below:

startTimes = {
    "n3": {
        "study_start_year": 2009,
        "study_start_month": 1,
        "study_start_day": 1,
        "study_start_hour": 22,
        "study_start_min": 15,
        "study_start_second": 42,
    },
    "n10": {
        "study_start_year": 2008,
        "study_start_month": 1,
        "study_start_day": 1,
        "study_start_hour": 22,
        "study_start_min": 24,
        "study_start_second": 52,
    },
}

# FREQUENCY
FREQUENCY = 512
