#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq
import gc
import pywt
from statsmodels.robust import mad
import scipy
from scipy import signal
from scipy.signal import butter

plt.style.use('ggplot')

data_dir = './Input'

#%%
metadata_train = pd.read_csv(data_dir + '/metadata_train_subset.csv', sep = ";")
metadata_train.head()

#%%
subset_train = pq.read_pandas(data_dir + '/subset_train.parquet', columns=[str(i) for i in range(0,4)]).to_pandas()
subset_train.head()

#%% # Signal characteristics
# From @randxie https://github.com/randxie/Kaggle-VSB-Baseline/blob/master/src/utils/util_signal.py
SAMPLING_FREQ = 800000 / 0.02  # 800,000 data points taken over 20 ms
print(SAMPLING_FREQ)

def add_high_pass_filter(x, low_freq=1000, sample_fs=SAMPLING_FREQ):
    """
    From @randxie https://github.com/randxie/Kaggle-VSB-Baseline/blob/master/src/utils/util_signal.py
    Modified to work with scipy version 1.1.0 which does not have the fs parameter
    """
    
    cutoff = 1000
    nyq = 0.5 * sample_fs
    normal_cutoff = cutoff / nyq
    
    # Fault pattern usually exists in high frequency band. According to literature, the pattern is visible above 10^4 Hz.
    # scipy version 1.2.0
    #sos = butter(10, low_freq, btype='hp', fs=sample_fs, output='sos')
    
    # scipy version 1.1.0
    sos = butter(10, normal_cutoff, btype='hp', output='sos')
    filtered_sig = signal.sosfilt(sos, x)

    return filtered_sig

def denoise_signal( x, wavelet='db4', level=1):
    """
    1. Adapted from waveletSmooth function found here:
    http://connor-johnson.com/2016/01/24/using-pywavelets-to-remove-high-frequency-noise/
    2. Threshold equation and using hard mode in threshold as mentioned
    in section '3.2 denoising based on optimized singular values' from paper:
    http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    """
    
    # Decompose to get the wavelet coefficients
    coeff = pywt.wavedec( x, wavelet, mode="per" )
    
    # Calculate sigma for threshold as defined in 
    # http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    sigma = (1/0.6745) * mad( coeff[-level] )
    #sigma = mad( coeff[-level] )
    
    # Calculte the univeral threshold
    uthresh = sigma * np.sqrt( 2*np.log( len( x ) ) )
    coeff[1:] = ( pywt.threshold( i, value=uthresh, mode='hard' ) for i in coeff[1:] )
    
    # Reconstruct the signal using the thresholded coefficients
    return pywt.waverec( coeff, wavelet, mode='per' )

#%% 
# Signal Denoising
# Before starting the feature extraction, it is very important to remove the noise from the signal
# in order to correctly eliminate false PD signals and background noise that could compromise
# the accuracy of the classifier

# First, a high pass signal will be applied to the signal. Since PD signals are characterized by very short pulses, its spectral content locate in higher frequencies. 
# 
# Second, a wavelet based denoising technique is applied to the signal. Wavelet denoising is the most effective
# and used approach available in the literature. 

train_length = subset_train.shape[1]

for i in range(0, train_length):
    signal_id = str(i)
    meta_row = metadata_train[metadata_train['signal_id'] == i]
    measurement = str(meta_row['id_measurement'].values[0])
    signal_id = str(meta_row['signal_id'].values[0])
    phase = str(meta_row['phase'].values[0])
    
    subset_train_row = subset_train[signal_id]
    
    # Apply high pass filter
    x_hp= add_high_pass_filter(subset_train_row)
    
    # Apply denoising
    x_dn = denoise_signal(x_hp, wavelet='haar', level=1)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))
    
    ax[0].plot(subset_train_row, alpha=0.5)
    ax[0].set_title(f"m: {measurement}, signal id: {signal_id}, phase: {phase}", fontsize=24)
    ax[0].legend(['Original'], fontsize=24)
    
    ax[1].plot(x_hp, 'r', alpha=0.5)
    ax[1].set_title(f"m: {measurement}, signal id: {signal_id}, phase: {phase}", fontsize=24)
    ax[1].legend(['HP filter'], fontsize=24)
    
    ax[2].plot(x_dn, 'g', alpha=0.5)
    ax[2].set_title(f"m: {measurement}, signal id: {signal_id}, phase: {phase}", fontsize=24)
    ax[2].legend(['HP filter and denoising'], fontsize=24)
    
    plt.show()
    
    subset_train.loc[:, signal_id] = x_dn




#%% 
# Feature extraction
# The most relevant features will be extracted following the results from:     
# http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf

# The signal is divided into 4 regions equally spaced (200000), and the features extracted for each one
# of them.

SIGNAL_SPACE_STEP = 200000
features =  metadata_train.copy()
features['number_of_peaks_p2'] = np.zeros(metadata_train.shape[0])
features['number_of_peaks_p3'] = np.zeros(metadata_train.shape[0])
features['number_of_peaks_p4'] = np.zeros(metadata_train.shape[0])

features['mean_width_of_peaks_p2'] = np.zeros(metadata_train.shape[0])
features['mean_width_of_peaks_p4'] = np.zeros(metadata_train.shape[0])
features['mean_width_of_peaks_p3'] = np.zeros(metadata_train.shape[0])
features['mean_width_of_peaks_overall'] = np.zeros(metadata_train.shape[0])

for i in range(0, train_length):

    signal_id = str(i)
    meta_row = metadata_train[metadata_train['signal_id'] == i]
    measurement = str(meta_row['id_measurement'].values[0])
    signal_id = str(meta_row['signal_id'].values[0])
    phase = str(meta_row['phase'].values[0])
    
    subset_train_row = subset_train[signal_id]

    # 1. Number of peaks
    peaks, properties = signal.find_peaks(subset_train_row, prominence = 10, width=1, distance = 50)
    features.loc[meta_row.index, 'number_of_peaks_p2'] = len(peaks[(peaks >= SIGNAL_SPACE_STEP) & (peaks < SIGNAL_SPACE_STEP * 2)])
    features.loc[meta_row.index, 'number_of_peaks_p3'] = len(peaks[(peaks >= SIGNAL_SPACE_STEP * 2) & (peaks < SIGNAL_SPACE_STEP * 3)])
    features.loc[meta_row.index, 'number_of_peaks_p4'] = len(peaks[(peaks >= SIGNAL_SPACE_STEP * 3) & (peaks < SIGNAL_SPACE_STEP * 4)])
    #prominences = signal.peak_prominences(subset_train_row, peaks)[0]
    #display(prominences.min())
    #display(prominences.max())
    #display(prominences.mean())

    #display(properties)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(30, 10))

    ax.plot(subset_train_row)
    ax.plot(peaks, subset_train_row[peaks], "x")
    ax.set_title(f"m: {measurement}, signal id: {signal_id}, phase: {phase}", fontsize=24)
    plt.show()

    # 2. Mean width of peaks
    widths = properties["widths"]
    features.loc[meta_row.index, 'mean_width_of_peaks_p2'] = widths[(peaks >= SIGNAL_SPACE_STEP) & (peaks < SIGNAL_SPACE_STEP * 2)].mean() / SAMPLING_FREQ
    features.loc[meta_row.index, 'mean_width_of_peaks_p3'] = widths[(peaks >= SIGNAL_SPACE_STEP * 2) & (peaks < SIGNAL_SPACE_STEP * 3)].mean() / SAMPLING_FREQ
    features.loc[meta_row.index, 'mean_width_of_peaks_p4'] = widths[(peaks >= SIGNAL_SPACE_STEP * 3) & (peaks < SIGNAL_SPACE_STEP * 4)].mean() / SAMPLING_FREQ
    features.loc[meta_row.index, 'mean_width_of_peaks_overall'] = widths[peaks >= SIGNAL_SPACE_STEP].mean()  / SAMPLING_FREQ

    # 3. Max width of peaks
    features.loc[meta_row.index, 'max_width_of_peaks_p2'] = widths[(peaks >= SIGNAL_SPACE_STEP) & (peaks < SIGNAL_SPACE_STEP * 2)].max() / SAMPLING_FREQ
    features.loc[meta_row.index, 'max_width_of_peaks_p3'] = widths[(peaks >= SIGNAL_SPACE_STEP * 2) & (peaks < SIGNAL_SPACE_STEP * 3)].max() / SAMPLING_FREQ
    features.loc[meta_row.index, 'max_width_of_peaks_p4'] = widths[(peaks >= SIGNAL_SPACE_STEP * 3) & (peaks < SIGNAL_SPACE_STEP * 4)].max() / SAMPLING_FREQ
    features.loc[meta_row.index, 'max_width_of_peaks_overall'] = widths[peaks >= SIGNAL_SPACE_STEP].max()  / SAMPLING_FREQ

    # 4. Min width of peaks
    features.loc[meta_row.index, 'min_width_of_peaks_p2'] = widths[(peaks >= SIGNAL_SPACE_STEP) & (peaks < SIGNAL_SPACE_STEP * 2)].min() / SAMPLING_FREQ
    features.loc[meta_row.index, 'min_width_of_peaks_p3'] = widths[(peaks >= SIGNAL_SPACE_STEP * 2) & (peaks < SIGNAL_SPACE_STEP * 3)].min() / SAMPLING_FREQ
    features.loc[meta_row.index, 'min_width_of_peaks_p4'] = widths[(peaks >= SIGNAL_SPACE_STEP * 3) & (peaks < SIGNAL_SPACE_STEP * 4)].min() / SAMPLING_FREQ
    features.loc[meta_row.index, 'min_width_of_peaks_overall'] = widths[peaks >= SIGNAL_SPACE_STEP].min()  / SAMPLING_FREQ

    # 5. Min width of peaks

    # 6. Mean height of peaks

    # 7. Max height of peaks

    # 8. Min height of peaks


features.head()

#%%
