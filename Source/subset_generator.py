#%% [markdown]
# This code extracts a subset of the total dataset to be used in the project

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq

plt.style.use('ggplot')

data_dir = './Input'

#%%
# Load the train data metadata
metadata_train = pd.read_csv(data_dir + '/metadata_train.csv')
metadata_train.head()

#print('Number of signals: {}'.format(metadata_train.shape[0]) )

#%%
# Load the test data metadata
metadata_test = pd.read_csv(data_dir + '/metadata_test.csv')
metadata_test.head()

print('Number of signals: {}'.format(metadata_test.shape[0]) )

#%% Visualize the distribution of failures
plt.figure(figsize = (8,6))
sns.countplot(data=metadata_train, x = 'target', hue='phase')
plt.title('Target values distribution per phase')

#%% Calculate statistics for the training dataset
# % of each class
metadata_train.target.value_counts() / metadata_train.shape[0]
#print('% of failures: {}'.format(metadata_train.target.value_counts() / metadata_test.shape[0]))

#%% Extract a subset of the data using the train_test_split function
from sklearn.model_selection import train_test_split
X = metadata_train[['signal_id','id_measurement','phase']]
y = metadata_train.target

X_subset, _, y_subset, _ = train_test_split(X, y, test_size = 0.75, random_state = 77, shuffle=False)

metadata_subset = X_subset.join(y_subset)
display(metadata_subset.head())

plt.figure(figsize = (8,6))
sns.countplot(data=metadata_subset, x = 'target', hue='phase')
plt.title('Target values distribution per phase for subset')

# % of each class
display(metadata_subset.target.value_counts() / metadata_subset.shape[0])

#%%
subset_train = pq.read_pandas(data_dir + '/train.parquet', columns=[str(i) for i in metadata_subset.signal_id]).to_pandas()


#%% save all the data
subset_train.to_parquet(data_dir +  '/subset_train.parquet')
metadata_subset.to_csv(data_dir +  '/metadata_train_subset.csv', sep = ";")

#%%
