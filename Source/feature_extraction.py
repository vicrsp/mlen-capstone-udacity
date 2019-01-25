import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import sys
import time 

SAMPLING_FREQ = 800000 / 0.02  # 800,000 data points taken over 20 ms
SIGNAL_SPACE_STEP = 200000

def extract_time_based_features(x, step = SIGNAL_SPACE_STEP, fs = SAMPLING_FREQ):

    features = []

    # 1. Number of peaks
    peaks, properties = signal.find_peaks(x, prominence = 10, distance = 50)
    
    features.append(len(peaks[(peaks >= SIGNAL_SPACE_STEP) & (peaks < SIGNAL_SPACE_STEP * 2)]))
    features.append(len(peaks[(peaks >= SIGNAL_SPACE_STEP * 2) & (peaks < SIGNAL_SPACE_STEP * 3)]))
    features.append(len(peaks[(peaks >= SIGNAL_SPACE_STEP * 3) & (peaks < SIGNAL_SPACE_STEP * 4)]))

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
 
    # 5. Mean height of peaks
    x_height = x
    x_height[peaks] = properties["prominences"]
    features.append(x_height[peaks[(peaks >= SIGNAL_SPACE_STEP) & (peaks < SIGNAL_SPACE_STEP * 2)]].mean())
    features.append(x_height[peaks[(peaks >= SIGNAL_SPACE_STEP * 2) & (peaks < SIGNAL_SPACE_STEP * 3)]].mean())
    features.append(x_height[peaks[(peaks >= SIGNAL_SPACE_STEP * 3) & (peaks < SIGNAL_SPACE_STEP * 4)]].mean())

    # 6. Max height of peaks
    features.append(max(x_height[peaks[(peaks >= SIGNAL_SPACE_STEP) & (peaks < SIGNAL_SPACE_STEP * 2)]], default=0))
    features.append(max(x_height[peaks[(peaks >= SIGNAL_SPACE_STEP * 2) & (peaks < SIGNAL_SPACE_STEP * 3)]], default=0))
    features.append(max(x_height[peaks[(peaks >= SIGNAL_SPACE_STEP * 3) & (peaks < SIGNAL_SPACE_STEP * 4)]], default=0))

    # 7. Min height of peaks
    features.append(min(x_height[peaks[(peaks >= SIGNAL_SPACE_STEP) & (peaks < SIGNAL_SPACE_STEP * 2)]], default=0))
    features.append(min(x_height[peaks[(peaks >= SIGNAL_SPACE_STEP * 2) & (peaks < SIGNAL_SPACE_STEP * 3)]], default=0))
    features.append(min(x_height[peaks[(peaks >= SIGNAL_SPACE_STEP * 3) & (peaks < SIGNAL_SPACE_STEP * 4)]], default=0))

    # Set NaN values as 0
    features = [0 if np.isnan(x) else x for x in features]

    return features


def extract_fourier_features(x, split_size = 0.25, step = SIGNAL_SPACE_STEP, fs = SAMPLING_FREQ, plot = False):

    features = []

    nyq = fs / 2

    # Calculate Welch periodogram
    f, Pxx = signal.welch(x[step:], fs)

    # split the power spectrum into equal length sizes and calculate its statistics
    ratio = split_size
    while ratio <= 1.0:
        subset_spectrum = Pxx[(f >  (ratio - split_size) * nyq) & (f <= (ratio) * nyq)]
        features.append(subset_spectrum.sum()) # Sum of power spectra
        features.append(subset_spectrum.max()) # Max of power spectra
        features.append(subset_spectrum.mean()) # Mean of power spectra
        
        ix = np.argmax(Pxx[(f >  (ratio - split_size) * nyq) & (f <= (ratio) * nyq)])
        start =  int((ratio - split_size) * len(f))
        features.append(f[start + ix ]) # Frequency of max value

        ratio += split_size


    if plot == True:
        plt.figure()
        plt.semilogy(f , Pxx)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
    
    return features



