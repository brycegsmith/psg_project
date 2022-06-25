import pandas as pd

class SAO2:

    def __init__(self, data):
        #data should be in form [epoch, SAO2]
        self.sao2Data = data

    def percentAbove(self, percent, data):
        #helper function for percentage calculation of time spent above certain threshold
        #should be noted that this is usually done in intervals of 10min and not 30secs
        #Further reading: https://arxiv.org/pdf/2008.03382.pdf
        #                 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5128354/#:~:text=That%20is%2C%20during%20REM%20sleep,arterial%20oxygen%20saturation%20decreases%20further.
        paData = []

        for i in range(0, self.sao2Data['epoch'].nunique() - 1):
            focus = self.sao2Data[self.sao2Data['epoch'] == i]

            paData.append((focus.loc[focus['SAO2'] >= percent].size)/(focus.size))
        paData.append(0) #temporary fix for mismatch in columns that I could not find the cause for

        return paData

    def get_SAO2_metrics(self):

        SAO2_min = (self.sao2Data).groupby('epoch').min()

        SAO2_avg = (self.sao2Data).groupby('epoch').mean()

        above90 = self.percentAbove(90, (self.sao2Data).groupby('epoch'))

        above80 = self.percentAbove(80, (self.sao2Data).groupby('epoch'))

        above70 = self.percentAbove(70, (self.sao2Data).groupby('epoch'))

        # Data Assembly
        final_SAO2 = pd.concat([pd.Series([i for i in range(0, len(SAO2_min))]), SAO2_min, SAO2_avg], axis=1)
        final_SAO2.columns = ['epoch', 'SAO2_min', 'SAO2_avg']
        final_SAO2['above90'] = above90
        final_SAO2['above80'] = above80
        final_SAO2['above70'] = above70

        return final_SAO2
