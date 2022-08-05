from tkinter import Y
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("SVM_tuning/rbf_underAndOverSampling_balancedAccuracy.csv")
#data = data[not data["param_gamma"] == "scale" and not data["param_gamma"] == "auto"]
#data = data[data["param_gamma"] != "scale"]
#data = data[data["param_gamma"] != "auto"]

rowCount = np.size(data["param_C"].unique())
colCount = np.size(data["param_gamma"].unique())
values = data["mean_test_score"].to_numpy()
values = np.reshape(values, [rowCount, colCount])

ax = sns.heatmap(values, xticklabels = data["param_gamma"].unique(), yticklabels = data["param_C"].unique())
ax.set_title("Parameter Selection for RBF Kernel")
ax.set_xlabel("Gamma")
ax.set_ylabel("Misclassification")
plt.show()