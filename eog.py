import pandas as pd

class EOG:

    def __init__(self, data):

        self.rawEOG = data[['epoch', 'ROC-LOC']]
        self.absrawEOG = (self.rawEOG).abs()

    def get_EOG_metrics(self):

        # Maximum Peak Amplitude Value
        EOG_PAV = (self.rawEOG).groupby('epoch').max()

        # Minimum Valley Amplitude Value
        EOG_VAV = (self.rawEOG).groupby('epoch').min()

        # EOG Standard Deviation Value
        EOG_STD = (self.rawEOG).groupby('epoch').std()

        # Area Under Curve
        EOG_AUC = (self.absrawEOG).groupby('epoch').sum()

        final_EOG = pd.concat([EOG_PAV, EOG_VAV, EOG_STD, EOG_AUC], axis=1)
        final_EOG.columns = ['EOG_PAV', 'EOG_VAV', 'EOG_STD', 'EOG_AUC']
        return final_EOG




