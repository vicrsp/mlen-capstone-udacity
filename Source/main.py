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
#subset_train = pq.read_pandas(data_dir + '/subset_train.parquet', columns=[str(i) for i in range(0,4)]).to_pandas()
#subset_train.head()

#%% # Signal characteristics
SAMPLING_FREQ = 800000 / 0.02  # 800,000 data points taken over 20 ms
#%% 
# Signal Denoising functions
# Before starting the feature extraction, it is very important to remove the noise from the signal
# in order to correctly eliminate false PD signals and background noise that could compromise
# the accuracy of the classifier

# First, a high pass signal will be applied to the signal. Since PD signals are characterized by very short pulses, its spectral content locate in higher frequencies. 
# 
# Second, a wavelet based denoising technique is applied to the signal. Wavelet denoising is the most effective
# and used approach available in the literature. 

def add_high_pass_filter(x, low_freq=1000, sample_fs=SAMPLING_FREQ):
    """
    From @randxie https://github.com/randxie/Kaggle-VSB-Baseline/blob/master/src/utils/util_signal.py
    Modified to work with scipy version 1.1.0 which does not have the fs parameter
    """
    
    cutoff = low_freq
    nyq = 0.5 * sample_fs
    normal_cutoff = cutoff / nyq
    
    # Fault pattern usually exists in high frequency band. According to literature, the pattern is visible above 10^4 Hz.
    # scipy version 1.2.0
    #sos = butter(10, low_freq, btype='hp', fs=sample_fs, output='sos')
    
    # scipy version 1.1.0
    sos = butter(10, normal_cutoff, btype='hp', output='sos')
    filtered_sig = signal.sosfilt(sos, x)

    return filtered_sig

def wavelet_denoising( x, wavelet='db4', level=1, mode = 'per'):
    """
    1. Adapted from waveletSmooth function found here:
    http://connor-johnson.com/2016/01/24/using-pywavelets-to-remove-high-frequency-noise/
    2. Threshold equation and using hard mode in threshold as mentioned
    in section '3.2 denoising based on optimized singular values' from paper:
    http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    """
    
    # Decompose to get the wavelet coefficients
    coeff = pywt.wavedec( x, wavelet, mode)
    
    # Calculate sigma for threshold as defined in 
    # http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    sigma = (1/0.6745) * mad( coeff[-level] )
    #sigma = (1/0.6745) * mad(coeff)

    #sigma = mad( coeff[-level] )
    
    # Calculte the univeral threshold
    uthresh = sigma * np.sqrt( 2*np.log( len( x ) ) )
    coeff[1:] = ( pywt.threshold( i, value=uthresh, mode='hard' ) for i in coeff[1:] )
    
    # Reconstruct the signal using the thresholded coefficients
    return pywt.waverec( coeff, wavelet, mode )


def denoise_signal(x, wavelet='db4', level=1, low_freq=1000, sample_fs=SAMPLING_FREQ, mode = 'per', plot = False):
    """
    Apply a high pass filter followed by a wavelete denoising algorithm
    """
    x_hp = add_high_pass_filter(x, low_freq, sample_fs)
    x_dn = wavelet_denoising(x_hp, wavelet, level, mode)

    if plot:
        
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))
    
        ax[0].plot(x, alpha=0.5)
        ax[0].set_title(f"m: {measurement}, signal id: {signal_id}, phase: {phase}", fontsize=24)
        ax[0].legend(['Original'], fontsize=24)
        
        ax[1].plot(x_hp, 'r', alpha=0.5)
        ax[1].set_title(f"m: {measurement}, signal id: {signal_id}, phase: {phase}", fontsize=24)
        ax[1].legend(['HP filter'], fontsize=24)
        
        ax[2].plot(x_dn, 'g', alpha=0.5)
        ax[2].set_title(f"m: {measurement}, signal id: {signal_id}, phase: {phase}", fontsize=24)
        ax[2].legend(['HP filter and denoising'], fontsize=24)
        
        plt.show()

    return x_dn

#%% 
# Feature extraction functions
# The most relevant features will be extracted following the results from:     
# http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf

# The signal is divided into 4 regions equally spaced (200000), and the features extracted for each one
# of them.

SIGNAL_SPACE_STEP = 200000

def extract_time_based_features(x, step = SIGNAL_SPACE_STEP, fs = SAMPLING_FREQ, plot = False):

    features = []

    # 1. Number of peaks
    peaks, _ = signal.find_peaks(x, prominence = 10, distance = 50)
    
    features.append(len(peaks[(peaks >= SIGNAL_SPACE_STEP) & (peaks < SIGNAL_SPACE_STEP * 2)]))
    features.append(len(peaks[(peaks >= SIGNAL_SPACE_STEP * 2) & (peaks < SIGNAL_SPACE_STEP * 3)]))
    features.append(len(peaks[(peaks >= SIGNAL_SPACE_STEP * 3) & (peaks < SIGNAL_SPACE_STEP * 4)]))

    if plot:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(30, 10))

        ax.plot(x)
        ax.plot(peaks, x[peaks], "x")
        plt.show()

    # 2. Mean width of peaks
    widths = signal.peak_widths(x, peaks, rel_height=0.9)[0]
    features.append(widths[(peaks >= SIGNAL_SPACE_STEP) & (peaks < SIGNAL_SPACE_STEP * 2)].mean() / fs)
    features.append(widths[(peaks >= SIGNAL_SPACE_STEP * 2) & (peaks < SIGNAL_SPACE_STEP * 3)].mean() / fs)
    features.append(widths[(peaks >= SIGNAL_SPACE_STEP * 3) & (peaks < SIGNAL_SPACE_STEP * 4)].mean() / fs)
    features.append(widths[peaks >= SIGNAL_SPACE_STEP].mean()  / fs)

    # 3. Max width of peaks
    features.append(max(widths[(peaks >= SIGNAL_SPACE_STEP) & (peaks < SIGNAL_SPACE_STEP * 2)], default=0) / fs)
    features.append(max(widths[(peaks >= SIGNAL_SPACE_STEP * 2) & (peaks < SIGNAL_SPACE_STEP * 3)], default=0) / fs)
    features.append(max(widths[(peaks >= SIGNAL_SPACE_STEP * 3) & (peaks < SIGNAL_SPACE_STEP * 4)], default=0) / fs)
    features.append(max(widths[peaks >= SIGNAL_SPACE_STEP], default=0) / fs)

    # 4. Min width of peaks
    features.append(min(widths[(peaks >= SIGNAL_SPACE_STEP) & (peaks < SIGNAL_SPACE_STEP * 2)], default=0) / fs)
    features.append(min(widths[(peaks >= SIGNAL_SPACE_STEP * 2) & (peaks < SIGNAL_SPACE_STEP * 3)], default=0) / fs)
    features.append(min(widths[(peaks >= SIGNAL_SPACE_STEP * 3) & (peaks < SIGNAL_SPACE_STEP * 4)], default=0) / fs)
    features.append(min(widths[peaks >= SIGNAL_SPACE_STEP], default=0) / fs)
 
    # 5. Min width of peaks

    # 6. Mean height of peaks

    # 7. Max height of peaks

    # 8. Min height of peaks

    features = [0 if np.isnan(x) else x for x in features]

    return features

#%% Execute feature extraction

features =  metadata_train.copy().drop('Unnamed: 0', axis = 1)

features['number_of_peaks_p2'] = np.zeros(metadata_train.shape[0])
features['number_of_peaks_p3'] = np.zeros(metadata_train.shape[0])
features['number_of_peaks_p4'] = np.zeros(metadata_train.shape[0])

features['mean_width_of_peaks_p2'] = np.zeros(metadata_train.shape[0])
features['mean_width_of_peaks_p4'] = np.zeros(metadata_train.shape[0])
features['mean_width_of_peaks_p3'] = np.zeros(metadata_train.shape[0])
features['mean_width_of_peaks_overall'] = np.zeros(metadata_train.shape[0])

features['max_width_of_peaks_p2'] = np.zeros(metadata_train.shape[0])
features['max_width_of_peaks_p3'] = np.zeros(metadata_train.shape[0])
features['max_width_of_peaks_p4'] = np.zeros(metadata_train.shape[0])
features['max_width_of_peaks_overall'] = np.zeros(metadata_train.shape[0])

features['min_width_of_peaks_p2'] = np.zeros(metadata_train.shape[0])
features['min_width_of_peaks_p4'] = np.zeros(metadata_train.shape[0])
features['min_width_of_peaks_p3'] = np.zeros(metadata_train.shape[0])
features['min_width_of_peaks_overall'] = np.zeros(metadata_train.shape[0])

signals = metadata_train.signal_id

for signal_id in signals:

    signal_raw = pq.read_pandas(data_dir + '/subset_train.parquet', columns = [str(signal_id)]).to_pandas()
    
    meta_row = metadata_train[metadata_train['signal_id'] == signal_id]
    measurement = str(meta_row['id_measurement'].values[0])
    signal_id = str(meta_row['signal_id'].values[0])
    phase = str(meta_row['phase'].values[0])
    
    signal_dn = denoise_signal(signal_raw.squeeze(), wavelet='db4', level=1, low_freq=1000, mode='zero')
    feat = extract_time_based_features(signal_dn)

    #feat = extract_time_based_features(np.asarray(signal_dn).squeeze())
    
    features.iloc[meta_row.index, 4:] = extract_time_based_features(signal_dn)


#%%
