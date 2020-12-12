## Data Loader

import numpy as np
import pandas as pd
import pickle as pk

from sklearn import preprocessing

import utils

def load_synthetic_one_hot():
	dataPath = "../robustnessExperiments/data/synthetic_exp4/synthetic_one_hot"
	sample_ids = np.random.choice(np.arange(0, len(data)), 2000)
	data_samples = data.iloc[sample_ids]
	data = pk.load(open(dataPath, 'rb')).data_frame_kurz
	FEATURE_COLUMNS = ['x1', 'x2', 'x3']
	TARGET_COLUMNS = ['y']
	return data_samples, FEATURE_COLUMNS, TARGET_COLUMNS

def load_german_dataset():
	dataPath = "./data/german_credit.pk"
	datasetName = 'german_credit'

	data = pk.load(open(dataPath, 'rb'))
	FEATURE_COLUMNS = data.columns[1:]
	TARGET_COLUMNS = data.columns[0]

	### Scale the dataset
	min_max_scaler = preprocessing.MinMaxScaler()
	data_scaled = min_max_scaler.fit_transform(data)
	data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

	return data_scaled, FEATURE_COLUMNS, TARGET_COLUMNS

def load_synthetic_face():
	dataPath = "./data/synthetic_face_dataset.pk"
	datasetName = 'synthetic_face'

	data = pk.load(open(dataPath, 'rb'))
	FEATURE_COLUMNS = ['x1', 'x2']
	TARGET_COLUMNS = ['y']

	### Scale the dataset
	min_max_scaler = preprocessing.MinMaxScaler()
	data_scaled = min_max_scaler.fit_transform(data)
	data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

	return data_scaled, FEATURE_COLUMNS, TARGET_COLUMNS

def load_german_synthetic_dataset():
	dataPath = "./data/synthetic_german_one_hot.pk"
	datasetName = 'synthetic_german_one_hot'

	data = pk.load(open(dataPath, 'rb')).data_frame_kurz

	data = data.sample(frac=0.10, random_state=utils.random_seed)
	pk.dump(data, open("./data/synthetic_german_one_hot_sampled0.1.pk", 'wb'))
	data.reset_index()

	FEATURE_COLUMNS = ['x'+str(i) for i in range(1, 8)] #data.columns[1:]
	TARGET_COLUMNS = data.columns[0]

	### Scale the dataset
	min_max_scaler = preprocessing.MinMaxScaler()
	data_scaled = min_max_scaler.fit_transform(data)
	data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

	return data_scaled, FEATURE_COLUMNS, TARGET_COLUMNS

def load_german_synthetic_sampled_dataset():
	dataTrainPath = "./data/synthetic_german_one_hot_sampled_train.pk"
	dataTestPath = "./data/synthetic_german_one_hot_sampled_test.pk"

	data_train = pk.load(open(dataTrainPath, 'rb'))
	data_test = pk.load(open(dataTestPath, 'rb'))

	datasetName = 'synthetic_german_oneHot_sampled'

	FEATURE_COLUMNS = ['x'+str(i) for i in range(1, 8)]
	TARGET_COLUMNS= data_train.columns[0]

	### Scale the dataset
	data_train_std = (data_train - data_train.min(axis=0)) / (data_train.max(axis=0) - data_train.min(axis=0))
	data_test_std = (data_test - data_train.min(axis=0)) / (data_train.max(axis=0) - data_train.min(axis=0))

	data_train_std = data_train_std.reset_index()
	data_test_std = data_test_std.reset_index()
	print(data_test_std)

	return data_train_std, data_test_std, FEATURE_COLUMNS, TARGET_COLUMNS


def load_dataset(datasetName='german_credit'):
	if (datasetName == 'synthetic_3lin'):
		return load_synthetic_one_hot()
	if (datasetName == 'german_credit'):
		return load_german_dataset()
	if (datasetName == 'synthetic_german_one_hot'):
		return load_german_synthetic_dataset()
	if (datasetName == 'synthetic_face'):
		return load_synthetic_face()

	return pd.DataFrame([]), [], []