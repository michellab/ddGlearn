
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"
import tensorflow as tf 

import numpy as np

import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib

import warnings

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error

from scipy import stats
##################################################

from tensorflow import keras
from sklearn.decomposition import PCA


# variables for future use:
dataset_path = None
num_epochs = None
num_steps = None
scaler = None
activation_fn = None
remove_outliers = True


hidden_units_params = None		#?????

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# load dataset:
dataset_file = "../datasets/ddG_offset_compiled/dataset_13.csv"
dataset_perts = pd.read_csv("../datasets/ddG_offset_compiled/dataset_13.csv", usecols=["Perturbation"])


dataset = pd.read_csv("../datasets/ddG_offset_compiled/dataset_13.csv").drop("Perturbation", axis=1)
dataset = dataset.apply(pd.to_numeric).astype(float)

#dataset = dataset.loc[:, (dataset != 0).any(axis=0)]	# drops all null-columns. not recommended


###################################################################

train_dataset = dataset.sample(frac=0.8)
test_dataset = dataset.drop(train_dataset.index)

print("Z-scoring and cleaning data..")
train_stats = train_dataset.describe()
train_stats.pop("ddG_offset")
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('ddG_offset')
test_labels = test_dataset.pop('ddG_offset')

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset).fillna(0).replace([np.inf, -np.inf], 0.0)
normed_test_data = norm(test_dataset).fillna(0).replace([np.inf, -np.inf], 0.0)

###################################################################

def reduce_features(training_data, testing_data):
	# Initialise PCA object, keep components up to 99% variance explained:
	pca = PCA(n_components=0.99)
	
	train_postPCA = pd.DataFrame(pca.fit_transform(training_data))

	# Apply fit to testing data
	test_postPCA = pd.DataFrame(pca.transform(testing_data))

	if train_postPCA.shape[1] != test_postPCA.shape[1]:
		print("Something went wrong with PCA generation, training and testing features are not equal.")
		return
	return train_postPCA, test_postPCA
print("Computing PCA, reducing features up to 99% VE..")

train_postPCA, test_postPCA = reduce_features(normed_train_data, normed_test_data)

train_postPCA = train_postPCA.set_index(normed_train_data.index)



###################################################################


def build_model():
  model = keras.Sequential([
    keras.layers.Dense(round(len(train_postPCA.columns)/2), activation=tf.nn.leaky_relu, input_shape=[len(train_postPCA.keys())]),

    keras.layers.Dense(round(len(train_postPCA.columns)/4), activation=tf.nn.leaky_relu),
    keras.layers.Dense(1, activation=tf.nn.leaky_relu)
  ])
  model.add(keras.layers.Dropout(0.2))				#??????
  #optimizer = tf.keras.optimizers.RMSprop(0.01)
  #optimizer = tf.keras.optimizers.SGD(
  #	lr=0.05,
  #	momentum=0.01,
  #	decay=0,
  #	nesterov=True
  #	)
  optimizer = tf.keras.optimizers.SGD(lr=0.1, clipnorm=1)
  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse']
                )
  return model

model = build_model()

print(str(model.summary()))


# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')


EPOCHS = 300


history = model.fit(
  train_postPCA, train_labels,
  epochs=EPOCHS, 
  validation_split = 0.2, 
  verbose=0,
  #callbacks=[PrintDot()],
  batch_size=128)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch


import matplotlib.pyplot as plt

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [ddG offset]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Training Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Validation Error')
  plt.legend()
  plt.ylim([0,5])
  
  #plt.figure()
  #plt.xlabel('Epoch')
  #plt.ylabel('Mean Square Error [$MPG^2$]')
  #plt.plot(hist['epoch'], hist['mean_squared_error'],
  #         label='Train Error')
  #plt.plot(hist['epoch'], hist['val_mean_squared_error'],
  #         label = 'Val Error')
  #plt.legend()
  #plt.ylim([0,20])
  plt.show()
  return

plot_history(history)
