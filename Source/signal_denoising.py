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

SAMPLING_FREQ = 800000 / 0.02  # 800,000 data points taken over 20 ms

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
        
        _, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))
    
        ax[0].plot(x, alpha=0.5)
        ax[0].legend(['Original'], fontsize=24)
        
        ax[1].plot(x_hp, 'r', alpha=0.5)
        ax[1].legend(['HP filter'], fontsize=24)
        
        ax[2].plot(x_dn, 'g', alpha=0.5)
        ax[2].legend(['HP filter and denoising'], fontsize=24)
        
        plt.show()

    return x_dn
