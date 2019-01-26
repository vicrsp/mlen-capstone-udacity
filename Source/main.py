#%%
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq
import time 

from Source.feature_extraction import *
from Source.signal_denoising import denoise_signal
from Source.models import *

plt.style.use('ggplot')
data_dir = './Input'

#%%
metadata_train = pd.read_csv(data_dir + '/metadata_train.csv', sep = ",")
metadata_train.head()

#%%
#subset_train = pq.read_pandas(data_dir + '/subset_train.parquet', columns=[str(i) for i in range(0,4)]).to_pandas()
#subset_train.head()

#%% # Signal characteristics
SAMPLING_FREQ = 800000 / 0.02  # 800,000 data points taken over 20 ms
SIGNAL_SPACE_STEP = 200000

#%% Execute feature extraction

features =  metadata_train.copy()

features['number_of_peaks_p1'] = np.zeros(metadata_train.shape[0])
features['number_of_peaks_p2'] = np.zeros(metadata_train.shape[0])
features['number_of_peaks_p3'] = np.zeros(metadata_train.shape[0])
features['number_of_peaks_p4'] = np.zeros(metadata_train.shape[0])

features['mean_width_of_peaks_p1'] = np.zeros(metadata_train.shape[0])
features['mean_width_of_peaks_p2'] = np.zeros(metadata_train.shape[0])
features['mean_width_of_peaks_p4'] = np.zeros(metadata_train.shape[0])
features['mean_width_of_peaks_p3'] = np.zeros(metadata_train.shape[0])

features['max_width_of_peaks_p1'] = np.zeros(metadata_train.shape[0])
features['max_width_of_peaks_p2'] = np.zeros(metadata_train.shape[0])
features['max_width_of_peaks_p3'] = np.zeros(metadata_train.shape[0])
features['max_width_of_peaks_p4'] = np.zeros(metadata_train.shape[0])

features['min_width_of_peaks_p1'] = np.zeros(metadata_train.shape[0])
features['min_width_of_peaks_p2'] = np.zeros(metadata_train.shape[0])
features['min_width_of_peaks_p4'] = np.zeros(metadata_train.shape[0])
features['min_width_of_peaks_p3'] = np.zeros(metadata_train.shape[0])

features['mean_height_of_peaks_p1'] = np.zeros(metadata_train.shape[0])
features['mean_height_of_peaks_p2'] = np.zeros(metadata_train.shape[0])
features['mean_height_of_peaks_p4'] = np.zeros(metadata_train.shape[0])
features['mean_height_of_peaks_p3'] = np.zeros(metadata_train.shape[0])

features['max_height_of_peaks_p1'] = np.zeros(metadata_train.shape[0])
features['max_height_of_peaks_p2'] = np.zeros(metadata_train.shape[0])
features['max_height_of_peaks_p4'] = np.zeros(metadata_train.shape[0])
features['max_height_of_peaks_p3'] = np.zeros(metadata_train.shape[0])

features['min_height_of_peaks_p1'] = np.zeros(metadata_train.shape[0])
features['min_height_of_peaks_p2'] = np.zeros(metadata_train.shape[0])
features['min_height_of_peaks_p4'] = np.zeros(metadata_train.shape[0])
features['min_height_of_peaks_p3'] = np.zeros(metadata_train.shape[0])

features['sum_spectrum_p1'] = np.zeros(metadata_train.shape[0])
features['max_spectrum_p1'] = np.zeros(metadata_train.shape[0])
features['mean_spectrum_p1'] = np.zeros(metadata_train.shape[0])
features['max_freq_spectrum_p1'] = np.zeros(metadata_train.shape[0])

features['sum_spectrum_p2'] = np.zeros(metadata_train.shape[0])
features['max_spectrum_p2'] = np.zeros(metadata_train.shape[0])
features['mean_spectrum_p2'] = np.zeros(metadata_train.shape[0])
features['max_freq_spectrum_p2'] = np.zeros(metadata_train.shape[0])

features['sum_spectrum_p3'] = np.zeros(metadata_train.shape[0])
features['max_spectrum_p3'] = np.zeros(metadata_train.shape[0])
features['mean_spectrum_p3'] = np.zeros(metadata_train.shape[0])
features['max_freq_spectrum_p3'] = np.zeros(metadata_train.shape[0])

features['sum_spectrum_p4'] = np.zeros(metadata_train.shape[0])
features['max_spectrum_p4'] = np.zeros(metadata_train.shape[0])
features['mean_spectrum_p4'] = np.zeros(metadata_train.shape[0])
features['max_freq_spectrum_p4'] = np.zeros(metadata_train.shape[0])

signals = metadata_train.signal_id

for signal_id in signals:

    start = time.time()

    signal_raw = pq.read_pandas(data_dir + '/train.parquet', columns = [str(signal_id)]).to_pandas()
    
    meta_row = metadata_train[metadata_train['signal_id'] == signal_id]
    measurement = str(meta_row['id_measurement'].values[0])
    signal_id = str(meta_row['signal_id'].values[0])
    phase = str(meta_row['phase'].values[0])
    
    signal_dn = denoise_signal(signal_raw.squeeze(), wavelet='db4', level=1, low_freq=10000, mode='zero')
    feat_fft = extract_fourier_features(signal_dn, step = SIGNAL_SPACE_STEP, fs = SAMPLING_FREQ)
    feat_ts = extract_time_based_features(signal_dn, SIGNAL_SPACE_STEP, SAMPLING_FREQ)
    features.iloc[meta_row.index, 4:] = feat_ts + feat_fft

    end = time.time()

    sys.stdout.flush()
    print('Processed signal #{}/{} - Elapsed time: {}'.format(signal_id, len(signals), end-start))

#%%
features.to_csv('./Output/all_features.csv', sep=";")
features.head()

#%% Features statistics - Save to file to allow better visualization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

features_norm = 
#%%
