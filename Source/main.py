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

data_dir = '../Data'

#%%
metadata_train = pd.read_csv(data_dir + '/metadata_train.csv')
metadata_train.head()

#%%
subset_train = pq.read_pandas(data_dir + '/train.parquet', columns=[str(i) for i in range(3)]).to_pandas()


#%% # Signal characteristics
# From @randxie https://github.com/randxie/Kaggle-VSB-Baseline/blob/master/src/utils/util_signal.py
SAMPLING_FREQ = 1 / 0.02 # 80,000 data points taken over 20 ms

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

train_length = 3
for i in range(train_length):
    signal_id = str(i)
    meta_row = metadata_train[metadata_train['signal_id'] == i]
    measurement = str(meta_row['id_measurement'].values[0])
    signal_id = str(meta_row['signal_id'].values[0])
    phase = str(meta_row['phase'].values[0])
    
    subset_train_row = subset_train[signal_id]
    
    # Apply high pass filter
    x_hp = add_high_pass_filter(subset_train_row)
    
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
