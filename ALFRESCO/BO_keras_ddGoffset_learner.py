
# TF-related imports & some settings to reduce TF verbosity:
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1, 2, 3"
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
import csv
import seaborn as sns
import matplotlib.cbook
import warnings
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

# Misc. imports:
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from scipy import stats


# some global variables:
dataset_paths = {
#'../datasets/traingsets_compiled/dataset_1.csv': "1",
#'../datasets/traingsets_compiled/dataset_2.csv': "2",
#'../datasets/traingsets_compiled/dataset_3.csv': "3",
#'../datasets/traingsets_compiled/dataset_12.csv': "12",
'../datasets/traingsets_compiled/dataset_13.csv': "13",
#'../datasets/traingsets_compiled/dataset_23.csv': "23",
#'../datasets/traingsets_compiled/dataset_123.csv': "123"
}

def LoadDatasets(datasets_directory):
	# Make list of datasets:
	contents = glob.glob(datasets_directory)
		
	loaded_dataframes = {}
	# Loop through contents:
	print("Reading input files..")
	for dataset in contents:
		dataframe_name = os.path.basename(dataset).replace(".csv", "")

		dataframe = pd.read_csv(dataset, index_col="Perturbation")
		# ensure every value is a float, shuffle all rows
		dataframe = dataframe.apply(pd.to_numeric).astype(float).sample(frac=1)
		

	# Return dictionary with dataset names and respective dataframes:
	return loaded_dataframes





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
	print("# of PCA features after reduction: "+str(len(test_postPCA.columns)))

	if train_postPCA.shape[1] != test_postPCA.shape[1]:
		print("Something went wrong during PCA generation, training and testing features are not equal.")
		return


	return [ train_postPCA, test_postPCA ]




def FF_DNN_KERAS(dataframe, labels, n_calls):
	# Display training progress by printing a single dot per epoch:
	class PrintDot(keras.callbacks.Callback):
	  def on_epoch_end(self, epoch, logs):
	    if epoch % 100 == 0: print('')
	    #print('.', end='')

	# Set early stopping variable:
	early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=0)



	# Retrieve datasets and set some variables:
	train_postPCA, test_postPCA, train_labels, test_labels = dataframe[0], dataframe[1], labels[0], labels[1]
	
	# Build keras DNN using global params:
	def create_model(
		learning_rate, 
		num_dense_layers_base, 
		num_dense_nodes_base, 
		num_dense_layers_end, 
		num_dense_nodes_end, 
		activation,
		adam_b1,
		adam_b2,
		num_batch_size):
		model = keras.Sequential()

		# Add input layer of length of the dataset columns:
		model.add(keras.layers.Dense(len(train_postPCA.columns), input_shape=[len(train_postPCA.keys())]))

		# Generate n number of hidden layers (base, i.e. first half):
		for i in range(num_dense_layers_base):
			model.add(keras.layers.Dense(num_dense_nodes_base,
			activation=activation
			))
		# Generate n number of hidden layers (end, i.e. last half):
		for i in range(num_dense_layers_end):
			model.add(keras.layers.Dense(num_dense_nodes_end,
			activation=activation
			))

		# Add output layer:
		model.add(keras.layers.Dense(1, activation=activation))

		optimizer = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=adam_b1, beta_2=adam_b2)

		model.compile(
			loss='mae',
			optimizer=optimizer,
			metrics=['mae']
			)
		return model

	# Set hyperparameter ranges, append to list:
	dim_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform',
		name='learning_rate')

	dim_num_dense_layers_base = Integer(low=1, high=4, name='num_dense_layers_base')
	dim_num_dense_nodes_base = Integer(low=5, high=512, name='num_dense_nodes_base')
	dim_num_dense_layers_end = Integer(low=1, high=4, name='num_dense_layers_end')
	dim_num_dense_nodes_end = Integer(low=5, high=512, name='num_dense_nodes_end')


	dim_activation = Categorical(categories=[tf.nn.leaky_relu, tf.nn.relu, tf.nn.sigmoid],
		name='activation')
	dim_adam_b1 = Real(low=8e-1, high=9.9e-1, prior="log-uniform", name="adam_b1")
	dim_adam_b2 = Real(low=8e-1, high=9.9e-1, prior="log-uniform", name="adam_b2")
	dim_num_batch_size = Integer(low=32, high=128, name='num_batch_size')
	



	dimensions = [dim_learning_rate,
				dim_num_dense_layers_base,
				dim_num_dense_nodes_base,
				dim_num_dense_layers_end,
				dim_num_dense_nodes_end,
				dim_adam_b1,
				dim_adam_b2,
				dim_num_batch_size]

	
	



	@use_named_args(dimensions=dimensions)
	def fitness(
		learning_rate, 
		num_dense_layers_base, 
		num_dense_nodes_base, 
		num_dense_layers_end, 
		num_dense_nodes_end,
		adam_b1,
		adam_b2,
		num_batch_size):
		# print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
		# print('learning rate: {0:.1e}'.format(learning_rate))
		# print('num_dense_layers_base:', num_dense_layers_base)
		# print('num_dense_nodes_base:', num_dense_nodes_base)
		# print('num_dense_layers_end', num_dense_layers_end)
		# print('num_dense_nodes_end:', num_dense_nodes_end)
		# print('adam_b1: {0:.1e}'.format(adam_b1))
		# print('adam_b2: {0:.1e}'.format(adam_b2))
		# print('num_batch_size', num_batch_size)
		

    # Create the neural network with these hyper-parameters.
		model = create_model(learning_rate=learning_rate,
							num_dense_layers_base=num_dense_layers_base,
							num_dense_nodes_base=num_dense_nodes_base,
							num_dense_layers_end=num_dense_layers_end,
							num_dense_nodes_end=num_dense_nodes_end,
							activation=tf.nn.leaky_relu,
							adam_b1=adam_b1,
							adam_b2=adam_b2,
							num_batch_size=num_batch_size)

		#print(str(model.summary()))
		#print("Fitting model to data..")
		history = model.fit(
			train_postPCA, train_labels,
		epochs= 300, 
		validation_split = 1 - train_test_split, 
		verbose=0,
		callbacks=[early_stopping],		#insert PrintDot() if you want verbosity on epochs
		batch_size=num_batch_size)

		hist = pd.DataFrame(history.history)
		hist['epoch'] = history.epoch
		#pd.options.display.max_rows = 8
	# calculate MAE on external set:
		MAE = hist["val_mean_absolute_error"].tail(10).mean()
		print("MAE:", MAE)

		prediction = model.predict(test_postPCA)
		prediction_list = []

		for item in prediction:
			prediction_list.append(item[0])


		perts_list = test_labels.index.tolist()

		exp_list = test_labels.values.tolist()

		tuples_result = list(zip(perts_list, exp_list, prediction_list))
		nested_list_result = [ list(elem) for elem in tuples_result ]
		

		################################################
		# Plot training validation:
		plt.cla()
		plt.subplot()
		sns.set()
		plt.xlabel('Epoch')
		plt.ylabel('Mean Abs Error [ddG offset (kcal/mol)]')
		plt.plot(hist['epoch'], hist['mean_absolute_error'],
		       label='Training Error')
		plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
		       label = 'Validation Error')
		plt.legend()
		plt.ylim([0,2.5])
		plt.xlim([0,100])
		#plt.draw()
		#plt.pause(0.01)
		#plt.title(name, fontsize=16)
		
		################################################
		# Do stuff with best performing model:
		global startpoint_mae

		if MAE < startpoint_mae:
			if not os.path.exists("./opt_output"):
				os.makedirs("./opt_output")
			# write model files:
			model.save_weights("opt_output/ALFRESCO_TopPerform_weights.h5")
			with open("opt_output/ALFRESCO_TopPerform_architecture.json", "w") as file:

				file.write(model.to_json())
			startpoint_mae = MAE

			# write internal validation figure: 
			#plt.savefig("opt_output/TopPerformer_ValPlot.png")

			# write external validation DF:
			with open("opt_output/TopPerformer_externalVal_df.csv", "w") as file:
				writer = csv.writer(file)
				writer.writerow(["Perturbation", "Experimental ddGoffset (kcal/mol)", "Predicted ddGoffset (kcal/mol)"])
				for row in nested_list_result:
					writer.writerow(row)
			print("poep")
		del model
		K.clear_session()

		return MAE

	# Bayesian Optimisation to search through hyperparameter space. Prior parameters were found by manual search.
	default_parameters = [0.01, 1, 10, 1, 10, 0.9, 0.99, 32]
	print("###########################################")
	print("Created model, optimising hyperparameters..")
	search_result = gp_minimize(func=fitness,
								dimensions=dimensions,
								acq_func='EI', # Expected Improvement.
								n_calls=n_calls,
								x0=default_parameters)


	print("###########################################")
	print("Concluded optimal hyperparameters:")
	print(search_result.x)
	print("###########################################")

	#return plot_convergence(search_result)
	return search_result

	# plot_objective(search_result, dimensions=[
	# 	"Learning Rate", 
	# 	"# Hidden Layers First Half", 
	# 	"# Neurons First Half",
	# 	"# Hidden Layers Second Half", 
	# 	"# Neurons Second Half",
	# 	"Adam Beta1",
	# 	"Adam Beta2",
	# 	"Batch Size"
	# 	])


	#plt.savefig("test.png")

train_test_split = 0.8
PCA_threshold = 0.99999
startpoint_mae = 5.0	
n_calls=200

df = pd.DataFrame()

for dataset, split in dataset_paths.items():
	print("Working on dataset: "+split)
	collection = pd.read_csv(dataset, index_col="Perturbation")

	normalised_collection, labels = SplitAndNormalise(collection, train_test_split)

	preprocessed_data = ReduceFeatures(normalised_collection, PCA_threshold)

	OptimizeResult = FF_DNN_KERAS(preprocessed_data, labels, n_calls)
	
	#plot_objective(OptimizeResult)
	#plt.show()


	split_column = { split : OptimizeResult.func_vals}
	
	df = pd.concat([df, pd.DataFrame(split_column)],axis=1)


#df.to_csv("results.csv")



"""
df = pd.read_csv("results.csv", index_col=[0])

df = df.cummin()


#sns.set()
#sns.lineplot(data=df)
plt.style.use("seaborn")
plt.plot(df)
plt.legend(["123","2","23", "1", "3", "13", "12"], loc=(1,0.6))




plt.ylabel("MAE after n calls (ddG)")
plt.xlabel("n calls")

plt.savefig("convergence_plot.png")
plt.show()
"""



#plot_convergence(descent_objects)

"""
# for later:
from keras.model import model_from_json

with open("opt_output/ALFRESCO_TopPerform_architecture.json", "r") as file:
	model = model_from_json(file.read())
model.load_weights("opt_output/ALFRESCO_TopPerform_weights.h5")

"""


