## CS4641 Summer 2022 Project - Sleep Stage Classification

### Infographic
<img src="https://brycegsmith.github.io/psg_project/images/infographic.png">

### Introduction
Sleep is an important physiological process directly correlated with physical health, mental well-being, and chronic disease risk. Unfortunately, nearly 70 million Americans suffer from sleep disorders.<sup>1</sup> The most effective measurement of sleep quality to date is collecting polysomnography (PSG) data in a sleep laboratory and measuring the duration of sleep stages. However, sleep studies are expensive, time-consuming, and inaccessible to the majority of the population. Wearables have attempted to use heart rate data and machine learning algorithms to predict sleep stage, but suffer from low accuracy.<sup>2</sup> We intend to create a machine learning model for the automatic classification of sleep stages using a minimum viable subset of biosignals from PSG data.

### Methodology
#### Dataset
Our data source is the CAP Sleep Database on PhysioNet.<sup>3</sup> It contains PSG recordings for 108 individuals; each waveform has over 10 channels including EEGs (brain), EMGs (muscle), ECGs (heart), EOGs (eyes), and SAO2 (respiration) signals.<sup>4</sup> From each voltage waveform we extracted numerical measurements taken every 1.9 milliseconds. Additionally, for each individual, a text file provides labeled sleep stages every epoch (30 second interval) along with age, gender, and sleep disease information. We selected 23 individuals with various sleep diseases for our initial dataset, which will become our training dataset when we move to supervised learning. After data preparation and feature extraction for these individuals, there were ~20,000 data points and ~47 features, most of which were engineered. The target values are discrete sleep stages (Wake, REM, NREM 1-4). An overview of the distribution of sleep stages for all individuals in the dataset is shown below.

<img src="https://brycegsmith.github.io/psg_project/images/sleep_stage_distribution.png" width="400" height="400">

#### Data Preparation
Our initial data cleaning involved converting each individual’s waveform data to numerical measurements and capturing a common set of ~10 signals. The original data for each individual had a sampling frequency of 512 Hz, meaning measurements were taken every 1.9 milliseconds. However, the sleep stage labels provided for each individual were taken every 30 seconds. Therefore, we needed a way to encapsulate our original data into 30 second chunks in order to align PSG data with sleep stages. Simply averaging measurements across each epoch is not the best way to encapsulate the data for each epoch; we implemented feature extraction methods based on previous research on effective features for each type of signal to get multiple features from each signal. Feature Extraction methods are detailed below.

Due to physiological variability between PSG subjects and many of the recordings being taken years apart, there is high variability between the subjects' baseline values. For example, the baseline ECG level for some subjects is significantly higher than others, due to factors like physiological differences and electrode connection. These differences in baselines are unique to each subject but persist through the entire recording. This was remedied by centering each individual’s data by subtracting the mean of each of their features before combining their data with the rest of the subjects’ data.

Outliers were detected in the dataset using the Local Outlier Factor (LOF) method. This algorithm considers if a point is an outlier among its nearest neighbors, as opposed to considering the point in relation to the entire dataset. Thus, extreme outliers due to recording errors are removed, but expected outliers, such as spikes in muscle activity are not removed. For example, when heart beats are overrun by noise due to recording abnormalities, certain heart rate metrics, such as the low frequency change in heart rate can spike (often multiple orders of magnitude higher than expected). Such outliers were detected and removed based on the LOF method. Other statistical outliers, such as those due to spikes in EMG (muscle) activity are not removed using this method, which is advantageous because this type of outlier is a valid measurement that can be used to detect motion during sleep, often associated with REM and Wake sleep stages.

Finally, we applied robust scaling to our dataset using the interquartile range. We opted for robust scaling over standard scaling due to concerns regarding the effect of outliers on our dataset. The Box Cox Transformation was used to standardize all features to a normal distribution. The results of Box-Cox transformation applied to EOG Energy Content Band are shown below.

Before Box-Cox:

<img src="https://brycegsmith.github.io/psg_project/images/before_bc.png" width="450" height="300">

After Box-Cox:

<img src="https://brycegsmith.github.io/psg_project/images/after_bc.png" width="450" height="300">

Additionally, we used encoding methods to express sleep disease and sleep stage (target) in numerical form. For sleep disease, we employed “dummy” encoding; with five disease classes including “Normal”,  we created four new binary variables that took value  “1” if an individual had a certain disease and “0” if not, with a “Normal” individual having all four of these binary variables equal to 0. For sleep stage, we used “ordinal” encoding in which we simply assigned each sleep stage a numerical value; the “awake” stage was assigned value “0”, NREM stages 1-4 were assigned their respective stage numbers, and REM was assigned value “5”.

#### Feature Engineering
Feature extraction methods for each type of signal from the PSG data are described:
* __EEG__: EEG (electroencephalogram) is a technique used to detect electrical activity in the brain. Manual sleep stage classification is largely dependent on the fraction of brain waves with specific frequencies (e.g., delta waves with a frequency of 1 - 4 Hz) and secondary time-domain features. In our dataset, available EEG signals differ slightly between individuals, but broadly follow the International 10-20 System. Extensive literature exists on useful EEG features, so a subset of suggested features were selected. First, the time-domain EEG signal was decomposed into the frequency-domain using Welch’s method (see image below), and the power of each frequency band of each brain wave was computed. Second, multiple entropy-based metrics (i.e., metrics conveying the amount of information given by a signal) were computed. Finally, miscellaneous more sophisticated time-domain metrics (e.g., Petrosian fractal dimension) were calculated. In total, thirteen unique features were computed using the provided EEG signals. All EEG features were averaged across each individual’s EEG channels.

<img src="https://brycegsmith.github.io/psg_project/images/eeg_report_image.png" width="450" height="300">

* __ECG & PPG__: ECG (electrocardiogram) and PPG (photoplethysmogram) are two methods used to record heart beats during the sleep studies. First, Python's heartpy library was used to detect heart beats (see image below).<sup>5</sup> Once heart beats were located, heart rate could be calculated. Beyond heart rate, an informative set of metrics consist of those that quantify variation in heart rate. The root mean square of the differences in time between adjacent heart beats (RMSSD) is one measure of heart rate variability, which is useful in our application because it can be meaningfully calculated over short time periods, such as 30 second epochs.<sup>6</sup> Heart rate changes in the frequency domain, specifically “low frequency” changes (0.04-0.15Hz) and “high frequency” changes (0.15-0.5 Hz) have been observed to vary with sleep stage, so these were also applied using the implementation in the heartpy library.<sup>7,8</sup>

<img src="https://brycegsmith.github.io/psg_project/images/ecg_report_image.png" width="450" height="300">

* __EMG__ - EMG (electromyography) is a method for measuring electrical activity of muscles. The main metric used to quantify the EMG activity was energy, calculated as the sum of squared differences between each point and the sample mean, divided by the number of samples.<sup>9</sup> Progression into deeper stages of sleep is typically correlated with a decrease in muscle tone, which corresponds to a decrease in baseline EMG energy, but REM sleep is also associated with brief spikes in muscle activity (see image below).<sup>10</sup> To capture these transient spikes in EMG energy that were “averaged out” over an entire 30 second epoch, a moving average with a five second window was applied over each second, and the average of the five highest windows was recorded within each 30 second epoch.<sup>11</sup>

<img src="https://brycegsmith.github.io/psg_project/images/emg_report_image.png" width="450" height="300">

* __EOG__: EOG (electrooculography) is used to detect activity within the human eye. One study aimed at Human-Computer Interaction applications mentioned a few useful features that were extracted from EOG signals, including: Maximum Peak Amplitude, which measures the maximum positive amplitude, Maximum Valley Amplitude, which measures the maximum negative amplitude, Area Under Curve, which is a summation of the absolute values of amplitude under positive and negative curves, and Signal Variance.<sup>12</sup> All of these metrics were calculated within each epoch. Another study that focused specifically on sleep staging estimated the Power Spectrum for the EOG signal and calculated the Energy Content Band by integrating this function over the frequency range 0.35-0.5 Hz, where REM activity is concentrated.<sup>9</sup> Using a Welch method to estimate the power spectrum, we calculated the Energy Content Band for each epoch.

* __SAO2__: SAO2 (or SPO2) refers to a blood-oxygen saturation reading which indicates the percentage of hemoglobin molecules that are saturated with oxygen. Readings can vary from 0 to 100% . Normal reading will range from 94% to 100%. Literature suggests readings below 50% are artifacts. Related literature to sleep staging using oximetry data engineered features by taking the peaks of each time period and the percentage of time spent above a certain threshold.<sup>13</sup> We followed suit with our data by taking the maximum of each epoch and the percentage of time spent above 70%, 80%, and 90% oxygen saturation by epoch. In addition, we included the average oxygen saturation of each epoch.

#### Feature Selection
After feature extraction, the correlation and mutual information methods were used to eliminate unnecessary features.
* __Correlation Method__: Correlated features were detected and removed using the method proposed by Kuhn and Johnson.<sup>14</sup> This method involves first calculating a correlation matrix for the data. Then, correlations are assessed pairwise. For any pair of features with a correlation above a set threshold (0.8 was used here), the feature in this pair with the larger average correlation between itself and every other feature was removed. This method eliminated 13 features, as shown in the results section.

* __Mutual Information Method__: After using the correlation method, we calculated the normalized mutual information between each feature and the sleep stages (target values) and defined four feature sets: the top 5, 10, 20, and 30 features with the greatest normalized mutual information values. As of this point, only the top 5 and top 10 sets have been evaluated by unsupervised learning, but we may incorporate use of the other two sets as we move into supervised learning.

#### Dimensionality Reduction
After feature selection, two methods were employed to reduce the dimensionality of data - Principal Component Analysis (PCA) and T-Distributed Stochastic Neighbor Embedding (TSNE). Broadly, PCA linearly transforms combinations of features such that variance is maximized along each principal component (i.e., axis). TSNE is a more sophisticated dimensionality reduction  technique that is able to account for nonlinear features in data. Both techniques were employed on the four feature groups (i.e., top 5, 10, 20, & 30 features) and were used to reduce to 1, 2, and 3 components. Isomap, a non-linear, manifold-based dimensionality reduction algorithm was also applied to the dataset. The results of this dimensionality reduction can be observed in the Results section, but because the method did not yield noticeably improved differentiation of sleep stages in the reduced feature space, these results were not used with the unsupervised learning algorithms.

#### Unsupervised Learning
Following dimensionality reduction, we applied several unsupervised learning methods to our data, including K-Means, GMM, and DBSCAN. To determine the quality of our clustering, we used the external measures of homogeneity, F1 score, normalized mutual information, Rand Statistic, and Fowlkes-Mallows measure. The sleep stages were taken as the “ground-truth” assignments, and each cluster was assigned a sleep stage based on the sleep stage of the majority of points in that cluster. We defined the “predicted” label of a point as the sleep stage of the cluster that it was assigned to.
* __K-Means__: K-Means was applied to our dataset via the sklearn implementation. As K-Means is notoriously sensitive to outliers, we expected suboptimal results. Thus, we explored similar methods to K-Means such as K-Medians and K-Medoids which are both more resistant to outliers.<sup>15</sup> K-Means gave the baseline behavior while K-Medoids was chosen as it was the most outlier resistant due to the nature of cluster center selection. We utilized the elbow method and found that 3 clusters was optimal for K-Means & K-Medoids (see image below). It should be noted that 6 clusters is expected for our dataset as it would capture each stage of sleep. Thus, we ran the K-Means & K-Medoids on both 3 and 6 clusters. The 6 cluster models consistently gave better performances by all metrics.

<img src="https://brycegsmith.github.io/psg_project/images/kmeans_plot.png" width="450" height="300">

* __GMM__: GMM was applied to our dataset via the sklearn implementation. Like K-Means, the most important parameter for GMM is the specified number of clusters. For our data, six clusters are used to yield the best results across all metrics.

* __DBSCAN__: DBSCAN was applied using the implementation in the sklearn package. The critical parameters to set for the algorithm are epsilon, or the maximum radius of a neighborhood around a point, and MinPts, the minimum number of points required to be in a point’s epsilon neighborhood for that point to be considered a core point. The starting value of MinPts was determined based on the dimensionality of the data being clustered, using the rule of thumb that in noisy datasets, a MinPts of 2xD is often appropriate. Epsilon was calculated using the distance to the 4 nearest neighbors of each point (see image below). These distances were sorted and plotted, yielding a graph that shows a flat region followed by a sharp increase in distance to outliers. A starting value of epsilon was selected as a value in the flat region of this graph, and it was adjusted further by steps of 0.1 to increase the clustering metrics.

<img src="https://brycegsmith.github.io/psg_project/images/dbscan_plot.png" width="450" height="300">

### Results
#### Feature Engineering & Selection
The feature engineering and selection process discussed in the Methodology section was followed. As discussed, we selected the top 5, 10, 20, and 30 features and consider these sets separate for dimensionality reduction & unsupervised learning tasks. As an example, the image of the confusion matrix below shows the original 37 features (left) being reduced to a set of the 30 features with the lowest correlation and highest mutual information (right).

* _Figure 1: Correlation Heat Map Before and After Eliminating Highly Correlated Features_

<img src="https://brycegsmith.github.io/psg_project/images/before_after_reduction.png" width="700" height="325">

* _Figure 2: Remaining Features Sorted by Normalized Mutual Information Values_

<img src="https://brycegsmith.github.io/psg_project/images/nmi.png" width="325" height="575">

#### Dimensionality Reduction
Again, the dimensionality reduction process using PCA and TSNE described in the Methodology section was followed. The results of dimensionality reduction when reducing the top 5 and top 10 features to 2-3 dimensions is shown below.

* _Figure 3: PCA Results_

<img src="https://brycegsmith.github.io/psg_project/images/flat_pca.png" width="575" height="600">

* _Figure 4: TSNE Results_

<img src="https://brycegsmith.github.io/psg_project/images/tsne_results.png" width="575" height="600">

#### Unsupervised Learning
After dimensionality reduction, K-Means, GMM, and DBSCAN were implemented on data according to the process outlined in the Methodology section. All of the algorithms performed best on the Top 10 feature sets, so only these results are provided. Although each algorithm was applied to each number of reduced components, only the best results are shared here: K-Means (3rd & 4th PCA components), GMM (3rd & 4th PCA components), and DBSCAN (3 TSNE Components).

* _Figure 5: K-Means Best Outcome_

<img src="https://brycegsmith.github.io/psg_project/images/best_kmeans.png" width="550" height="250">

* _Figure 6: GMM Best Outcome_

<img src="https://brycegsmith.github.io/psg_project/images/best_gmm.png" width="550" height="250">

* _Figure 7: DBSCAN Best Outcome_

<img src="https://brycegsmith.github.io/psg_project/images/best_dbscan.png" width="575" height="600">

The external quality measures for the best result of each algorithm are provided in the bar plot below.

* _Figure 8: Comparison of Performance for Each Clustering Algorithm_

<img src="https://brycegsmith.github.io/psg_project/images/metrics_barplot.png" width="550" height="400">

### Ancillary Results:
#### Isomap Dimensionality Reduction
Isomap dimensionality reduction was also applied to the data. Because the results did not yield greater differentiation of the sleep stages than PCA and showed the added difficulties of a mixture of low density and high density regions and abnormally shaped clusters, these results were not put through the unsupervised learning algorithms.

* _Figure 9: Isomap Dimensionality Reduction of 5 Top Features_

<img src="https://brycegsmith.github.io/psg_project/images/isomap_5features.png" width="550" height="250">

* _Figure 10: Isomap Dimensionality Reduction of 5 Top Features_

<img src="https://brycegsmith.github.io/psg_project/images/isomap_10features.png" width="550" height="250">

#### Simplification of K-Means and GMM Clustering
As mentioned, applying the elbow method for K-Means suggested an ideal cluster size of 3. Because the six sleep stage targets present in this dataset (Wake, NREM1, NREM2, NREM3, NREM4, and REM) can be more broadly grouped into only three more general targets (Wake, NREM, and REM), K-Means and GMM with three clusters were applied and compared to these broader sleep stage classifications.

* _Figure 11: K-Means - 3 Target Values_

<img src="https://brycegsmith.github.io/psg_project/images/best_kmeans_3targets.png" width="550" height="250">

* _Figure 12: GMM - 3 Target Values_

<img src="https://brycegsmith.github.io/psg_project/images/best_gmm_3targets.png" width="550" height="250">

* _Table 1: Clustering Metrics - 3 Target Values_

|   |NMI|F1 |Homogeneity   |Rand-Stat   | Fowlkes-Mallows|
|---|---|---|---|---|---|
|K-Means   |0.2303   |0.8074   |0.2515   |0.6750   |0.7297|
|GMM   |0.2940   |0.8282   |0.3227   |0.7013   |0.7509|

### Discussion

#### Note: All figures referenced are defined in the Results section.

#### Distribution of Targets & Important Features
Based on the pie chart in the Methodology section showing distribution of target values, the most frequent sleep stage in our data is NREM stage 2, but other stages like W (awake) and NREM stage 4 also comprise significant proportions. The class imbalance here may have contributed to poor results with our unsupervised learning methods. Since NREM stage 2 is the most prevalent sleep stage, most clusters will likely be assigned NREM stage 2 as well based on our definition. This means the predicted label of points in those clusters will also be NREM stage 2, but there is still a significant likelihood that a good portion of these points correspond to other sleep stages.

The histograms in the Methodology section compare overall EOG energy content band data before and after Box-Cox transformation. As evidenced by Figure 2, EOG energy content band has one of the highest normalized mutual information scores with sleep stage labels, making it an important feature, and it reflects the extreme right skewness that most other features in our dataset exhibit. However, we see that the Box-Cox transformation makes the distribution significantly more normal-shaped. Standardizing the shape of all features through Box-Cox transformation allows the data to be more balanced and reduces the impact of outliers.

#### Feature Engineering & Selection
As evidenced by the left image in Figure 1, there was high correlation among features within the same domain, especially EEG and EOG. For example, the EOG metrics AUC (area under curve) and STD (standard deviation) were closely correlated (correlation = 0.99) because AUC is the sum of the absolute values of individual voltage values in an epoch, so when AUC is higher, this indicates that there were more spikes in the EOG signal, causing the measurements to be further spread out overall. Similarly, two EMG metrics were used to measure EMG energy, which corresponds to muscle activity. One metric was an average EMG energy across an entire 30 second epoch, while the other was generated using a moving average with a 5 second window and then averaging the five highest 5 second periods. Although this second metric is more sensitive to spikes in EMG activity, across most epochs, it was very closely correlated with the average across the entire epoch (correlation = 0.97). As a whole, co-linearity between features in this dataset can be explained by similarity in the feature extraction calculations used to generate metrics for the same signal type. These closely correlated features do not add significant insights to the model, so they can be removed to simplify the model. Beyond simplification, removing these correlated features within signal domains can improve the performance of later algorithms, such as PCA, by reducing imbalanced impacts on variance resulting from the uneven number of features used for each type of signal. Following removal of the thirteen features identified during this step, the dataset had much lower correlation overall, as seen in the right image in Figure 1, in which there are no large regions of highly correlated data that would skew PCA and other algorithms.

Due to the high correlation between features within these domains, most of the features removed via our average correlation threshold were EEG and EOG features. We selected different sets of features with the top 5-30 normalized mutual information values with sleep stages for comparison purposes, since one of our project goals is determining a minimum viable subset of signals that can be used for accurate predictions. The features that had the lowest mutual information with sleep stages were Oxygen Saturation metrics, which is surprising given that breathing rate is known to decrease during deep sleep. We applied two stages of feature extraction in order to avoid issues with high-dimensional data like overfitting and distorted distance calculations in clustering. For example, the EMG metrics for the 5 second moving average and the 30 second epoch average were closely correlated (correlation = 0.97), so the moving 5 second average was removed because of the two metrics, it had a higher average correlation across all other features (average correlation = 0.3418). From there the EMG epoch average moved on to the mutual information feature selection step where it was removed from the set of “Top 5” features because it had the seventh highest mutual information with sleep stage.

#### Dimensionality Reduction
We investigated two dimensionality reduction algorithms, PCA and TSNE. In our results from PCA, shown in Figure 3, outliers cause variance to be highly concentrated in the first principal component, leading to a tight distribution of data points in others. This can be observed in Figure 3, in which the majority of data points appear “flattened” onto one axis or plane, corresponding to the lower-variance components. This issue becomes more apparent when we increase the dimensionality of the dataset, as evidenced by comparing the PCA results from the Top 5 versus Top 10 feature sets. When we examine the target values of points across the entire first principal component in these plots, we see that most are in the “awake” stage since most of the spread-out points correspond to spikes in signals, such as muscle activity, which are most common when someone is awake.

We also implemented TSNE, as the algorithm better accounts for nonlinear relationships in our data. For optimal performance, sklearn recommends that TSNE be applied to a dataset with less than 50 features; given that we have 5 to 30 features, TSNE appears to be a viable method for our data. Figure 4 shows the output obtained from TSNE. The spread of data points is higher than PCA results, but the data points still tend to be tightly distributed with significant overlap between points corresponding to different sleep stages. The clear presence of outliers and irregularly shaped clusters indicate that after applying TSNE, DBSCAN will have much better performance than K-Means or GMM.

#### Unsupervised Learning
For the purposes of this project, since we are looking at the accuracy of predicting sleep stage, supervised metrics using target values as “ground truth” is the most logical method of evaluating the quality of our clustering. The use of internal measures is not useful unless each cluster clearly corresponds to a distinct sleep stage, which is not the case.

The bar chart comparing external metrics between unsupervised learning approaches, shown in Figure 8, provides insight into algorithm performance. All of the metrics shown in the chart range from 0 to 1, with higher values indicating better clustering performance. The results from K-Means and GMM are extremely close, with K-Means performing slightly better in all metrics with the exception of the Rand-Stat. Since the metrics for DBSCAN are all greater than for K-Means and GMM, we can conclude that our best DBSCAN outcome seems to be the best clustering for our data. However, there seems to be a positive correlation between each metric value for the different algorithms; the metrics that have higher values for one algorithm also tend to have higher values for the other. For all three algorithms, the metric with the highest value is Rand-Stat and the metric with the lowest value is Homogeneity.

All of our clustering algorithms run into issues due to the tight distribution of points in reduced dimensionality. The reduced dimensionality from both PCA and TNSE looks very compact, with a lot of overlap between points corresponding to different sleep stages. As mentioned earlier, periodic outliers are common in our data. This may explain why K-means did not perform very well since it is especially sensitive to outliers. The sleep stage clusters appear to have highly irregular shapes, which is difficult for both K-means and GMM to categorize. K-Means and GMM had very similar performance when applied on the 3rd and 4th components of a 4D PCA with our top 10 dataset. Our PCA results indicate that the vast majority of data variance is concentrated within the first principal component, leaving much less in the others. Therefore, by isolating these components, we can focus on data with very low variance. As variance approaches zero, GMM converges to K-Means, which can explain the similarity in results.

Due to the irregularly shaped clusters resulting from the TSNE dimensionality reduction, DBSCAN was the most suitable clustering algorithm to apply to the TSNE dimensions. The best results were achieved by applying DBSCAN to a three-dimensional TSNE dimensionality reduction. Although relatively high values were achieved for the clustering metrics, especially the Rand Stat (0.74), Fowlkes-Mallows (0.53), and F1 measures (0.66), these results were only achieved by setting a low enough epsilon that all of the small clusters in the TSNE feature space could be captured as unique clusters by the DBSCAN algorithm. This helps with the purity and accuracy of each cluster, but the downside is an uninterpretable cluster assignment of ten clusters to describe only five sleep stages. This clustering results will therefore have poor generalizability to other datasets, and would require knowing the target values in order to interpret the large number of clusters as capturing one sleep stage or another, ultimately defeating the purpose of this being an unsupervised method that could be meaningfully applied without have access to the data labels.

Sleep stage prediction seems to be more tailored towards supervised learning methods. Most machine learning projects on sleep staging have gone directly to supervised learning algorithms with great accuracy. Part of our objective in applying unsupervised learning was to take a novel approach and evaluate the viability of unsupervised learning methods for this problem. Assigning each cluster the “majority” sleep stage is not an accurate way to summarize clusters since, as evidenced by the dimensionality reduction plots, there is significant overlap between data points with different sleep stage labels. Another potentially significant issue is that there is not enough distinction between different NREM stages. Machine learning projects that deal with sleep staging often focus on classifying awake, REM, and NREM sleep since there is a lot of overlap in the metrics corresponding to the different stages of NREM sleep, which are often difficult to distinguish based on metrics extracted from PSG data. Additionally, the proportion of individual NREM sleep stages in our data is significantly uneven. The significant overlap between data points from different sleep stages may be attributable to our pooling of individuals with different diseases. We hope to investigate this further as we proceed into the supervised learning portion, but our initial analysis indicates that when we examine the spread of data for each sleep stage for individuals without any sleep diseases, there seems to be less overlap compared to when all the individuals are combined.


### References
1. Malekzadeh M, Hajibabaee P, Heidari M, Berlin B. Review of Deep Learning Methods for Automated Sleep Staging. 2022:0080-0086.
2. de Zambotti M, Goldstone A, Claudatos S, Colrain IM, Baker FC. A validation study of Fitbit Charge 2™ compared with polysomnography in adults. Chronobiology International. 2018/04/03 2018;35(4):465-476. doi:10.1080/07420528.2017.1413578
3. Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation 101(23):e215-e220 [Circulation Electronic Pages; http://circ.ahajournals.org/content/101/23/e215.full]; 2000 (June 13).
4. Terzano MG, Parrino L, Sherieri A, et al. Atlas, rules, and recording techniques for the scoring of cyclic alternating pattern (CAP) in human sleep. Sleep Med. Nov 2001;2(6):537-53. doi:10.1016/s1389-9457(01)00149-6
5. van Gent P, Farah H, van Nes N, van Arem B. HeartPy: A novel heart rate algorithm for the analysis of noisy signals. Transportation Research Part F: Traffic Psychology and Behaviour. 2019/10/01/ 2019;66:368-378. doi:https://doi.org/10.1016/j.trf.2019.09.015
6. Shaffer, F., & Ginsberg, J. P. (2017). An Overview of Heart Rate Variability Metrics and Norms. Frontiers in public health, 5, 258. https://doi.org/10.3389/fpubh.2017.00258
7. Vanoli E, Adamson PB, Ba-Lin n, Pinna GD, Lazzara R, Orr WC. Heart Rate Variability During Specific Sleep Stages. Circulation. 1995/04/01 1995;91(7):1918-1922. doi:10.1161/01.CIR.91.7.1918
8. Boudreau P, Yeh WH, Dumont GA, Boivin DB. Circadian variation of heart rate variability across sleep stages. Sleep. 2013;36(12):1919-1928. Published 2013 Dec 1. doi:10.5665/sleep.3230
9. E. Estrada, H. Nazeran, J. Barragan, J. R. Burk, E. A. Lucas and K. Behbehani, "EOG and EMG: Two Important Switches in Automatic Sleep Stage Classification," 2006 International Conference of the IEEE Engineering in Medicine and Biology Society, 2006, pp. 2458-2461, doi: 10.1109/IEMBS.2006.260075.
10. Gonzalez AA, Rezvan, Kaveh, Valladares, Edwin, Hammond, Terese. Sleep Stage Scoring. Medscape. 2019;doi:https://emedicine.medscape.com/article/1188142-overview#a3
11. Levendowski DJ, St Louis EK, Strambi LF, Galbiati A, Westbrook P, Berka C. Comparison of EMG power during sleep from the submental and frontalis muscles. Nat Sci Sleep. 2018;10:431-437. Published 2018 Dec 6. doi:10.2147/NSS.S189167
12. S. Aungsakul, A. Phinyomark, P. Phukpattaranont, C. Limsakul, Evaluating Feature Extraction Methods of Electrooculography (EOG) Signal for Human-Computer Interface, Procedia Engineering, Volume 32, 2012, Pages 246-252, ISSN 1877-7058, https://doi.org/10.1016/j.proeng.2012.01.1264.
13. Choi, E., Park, D.-H., Yu, J.-H., Ryu, S.-H., & Ha, J.-H. (2016, November). The severity of sleep disordered breathing induces different decrease in the oxygen saturation during rapid eye movement and non-rapid eye movement sleep. Psychiatry investigation.
14. Kuhn, M., Johnson, K. (2013). Data Pre-processing. In: Applied Predictive Modeling. Springer, New York, NY. https://doi.org/10.1007/978-1-4614-6849-3_3
15. Clustering with KMEDOIDS and common-nearest-neighbors¶. scikit. (n.d.).
