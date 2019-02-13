
# TF-related imports & some settings to reduce TF verbosity:
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf 
from tensorflow import keras
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# SciKit-Optimize:
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from tensorflow.python.keras import backend as K
from skopt.utils import use_named_args

# General imports:
import glob
import numpy as np
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Misc. imports:
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from scipy import stats


# variables for future use:
dataset_path = '../datasets/tmp_dataset1/dataset_1.csv'
train_test_split = 0.8
PCA_threshold = 0.99999
startpoint_mae = 5.0


momentum = 0.01
decay = 0


def LoadDatasets(datasets_directory):
	# Make list of datasets:
	contents = glob.glob(datasets_directory)
		
	loaded_dataframes = {}
	# Loop through contents:
	print("Reading input files..")
	for dataset in contents:
		dataframe_name = os.path.basename(dataset).replace(".csv", "")

		dataframe = pd.read_csv(dataset, index_col="Perturbation")
		dataframe = dataframe.apply(pd.to_numeric).astype(float)
		dataframe

	# Return dictionary with dataset names and respective dataframes:
	return loaded_dataframes

collection = pd.read_csv(dataset_path, index_col="Perturbation")



def SplitAndNormalise(dataset, train_test_split):
	split_and_normalised_dataframes = {}

	
	print("Z-scoring and cleaning data..")
	
	# Split dataframes according to indicated variable:
	train_dataset = dataset.sample(frac=train_test_split)
	test_dataset = dataset.drop(train_dataset.index)
	

	# Calculate statistics, compute Z-scores, clean:
	train_stats = train_dataset.describe()
	train_stats.pop("ddG_offset")
	train_stats = train_stats.transpose()

	train_labels = train_dataset.pop('ddG_offset')
	test_labels = test_dataset.pop('ddG_offset')
	def norm(x):
		return (x - train_stats['mean']) / train_stats['std']

	normed_train_data = norm(train_dataset).fillna(0).replace([np.inf, -np.inf], 0.0)
	normed_test_data = norm(test_dataset).fillna(0).replace([np.inf, -np.inf], 0.0)	

	return [[normed_train_data, normed_test_data], [train_labels, test_labels]]

normalised_collection, labels = SplitAndNormalise(collection, train_test_split)

def ReduceFeatures(normalised_collection, PCA_threshold):
	print("Computing PCA, reducing features up to "+str(round(PCA_threshold*100, 5))+"% VE..")
	training_data = normalised_collection[0]
	testing_data = normalised_collection[1]

	# Initialise PCA object, keep components up to x% variance explained:
	PCA.__init__
	pca = PCA(n_components=PCA_threshold)

	# Fit to training set, apply to both training and testing set:			
	train_postPCA = pd.DataFrame(pca.fit_transform(training_data))
	test_postPCA = pd.DataFrame(pca.transform(testing_data))

	if train_postPCA.shape[1] != test_postPCA.shape[1]:
		print("Something went wrong during PCA generation, training and testing features are not equal.")
		return


	return [ train_postPCA, test_postPCA ]


preprocessed_data = ReduceFeatures(normalised_collection, PCA_threshold)






def FF_DNN_KERAS(dataframe, labels):
	# Display training progress by printing a single dot per epoch:
	class PrintDot(keras.callbacks.Callback):
	  def on_epoch_end(self, epoch, logs):
	    if epoch % 100 == 0: print('')
	    print('.', end='')

	# Set early stopping variable:
	early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=0)



	# Retrieve datasets and set some variables:
	train_postPCA, test_postPCA, train_labels, test_labels = dataframe[0], dataframe[1], labels[0], labels[1]
	
	# Build keras DNN using global params:
	def create_model(learning_rate, num_dense_layers, num_dense_nodes, activation):
		model = keras.Sequential()

		# Add input layer of length of the dataset columns:
		model.add(keras.layers.Dense(len(train_postPCA.columns), input_shape=[len(train_postPCA.keys())]))

		# Loop over number of layers to optimise on:
		for i in range(num_dense_layers):
			model.add(keras.layers.Dense(num_dense_nodes,
			activation=activation
			))

		# Add output layer:
		model.add(keras.layers.Dense(1, activation=activation))

		optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

		model.compile(
			loss='mae',
			optimizer=optimizer,
			metrics=['mae']
			)
		return model

	# Set hyperparameter ranges, append to list:
	dim_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform',
		name='learning_rate')
	dim_num_dense_layers = Integer(low=1, high=5, name='num_dense_layers')
	dim_num_dense_nodes = Integer(low=5, high=512, name='num_dense_nodes')
	dim_activation = Categorical(categories=[tf.nn.leaky_relu, tf.nn.relu, tf.nn.sigmoid],
		name='activation')
	dimensions = [dim_learning_rate,
				dim_num_dense_layers,
				dim_num_dense_nodes,
				dim_activation]

	
	



	@use_named_args(dimensions=dimensions)
	def fitness(learning_rate, num_dense_layers, num_dense_nodes, activation):
		print('learning rate: {0:.1e}'.format(learning_rate))
		print('num_dense_layers:', num_dense_layers)
		print('num_dense_nodes:', num_dense_nodes)
		print('activation_fn:', activation)

    # Create the neural network with these hyper-parameters.
		model = create_model(learning_rate=learning_rate,
							num_dense_layers=num_dense_layers,
							num_dense_nodes=num_dense_nodes,
							activation=activation)

		print(str(model.summary()))
		print("Fitting model to data..")
		history = model.fit(
			train_postPCA, train_labels,
		epochs= 300, 
		validation_split = 1 - train_test_split, 
		verbose=0,
		callbacks=[PrintDot(), early_stopping],
		batch_size=128)

		hist = pd.DataFrame(history.history)
		hist['epoch'] = history.epoch
		pd.options.display.max_rows = 8
	# calculate MAE on external set:
		MAE = hist["val_mean_absolute_error"].tail(10).mean()
		print("MAE:", MAE)

		global startpoint_mae

		if MAE < startpoint_mae:
			if not os.path.exists("./opt_output"):
				os.makedirs("./opt_output")

			model.save_weights("opt_output/ALFRESCO_TopPerform_weights.h5")
			with open("opt_output/ALFRESCO_TopPerform_architecture.json", "w") as file:

				file.write(model.to_json())
			startpoint_mae = MAE

		del model
		K.clear_session()

		return MAE

	# Bayesian Optimisation to search through hyperparameter space. Default parameters were found by manual search.
	default_parameters = [0.01, 2, 10, tf.nn.leaky_relu]
	print("###########################################")
	print("Created model, optimising hyperparameters..")
	search_result = gp_minimize(func=fitness,
								dimensions=dimensions,
								acq_func='EI', # Expected Improvement.
								n_calls=100,
								x0=default_parameters)


	print(search_result.x)
	plot_convergence(search_result)
	plt.ylabel("ddG offset prediction MAE for n calls")
	plt.show()


	


FF_DNN_KERAS(preprocessed_data, labels)


"""

from keras.model import model_from_json

with open("opt_output/ALFRESCO_TopPerform_architecture.json", "r") as file:
	model = model_from_json(file.read())
model.load_weights("opt_output/ALFRESCO_TopPerform_weights.h5")

"""


