import numpy as np
import pandas as pd
from scipy import signal as sig, integrate

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

        # Energy Content Band using Power Spectrum
        ECB = []
        for i in range(0, self.rawEOG['epoch'].nunique() - 1):
            focus = self.rawEOG[self.rawEOG['epoch'] == i]
            f, ps = sig.welch(focus['ROC-LOC'], nperseg = 1048576, scaling='spectrum', fs=512)
            bandenergy = integrate.simps(ps[(f >= 0.35) & (f <= 0.5)])
            ECB.append(bandenergy)
        ECB.append(0)

        # Data Assembly
        final_EOG = pd.concat([pd.Series([i for i in range(0, len(EOG_PAV))]), EOG_PAV, EOG_VAV, EOG_STD, EOG_AUC], axis=1)
        final_EOG.columns = ['epoch', 'EOG_PAV', 'EOG_VAV', 'EOG_STD', 'EOG_AUC']
        final_EOG['EOG_ECB'] = ECB
        return final_EOG




