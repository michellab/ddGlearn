
# TF-related imports & some settings to reduce TF verbosity:
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"
import tensorflow as tf 
from tensorflow import keras
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
dataset_path = '../datasets/ddG_offset_compiled/data*'
train_test_split = 0.8
PCA_threshold = 0.99999

loss_function = 'mse'
activation_function = 'leaky_relu'							#TD find way to loop over different modules
dropout = 0.2


n_epochs = 300
batch_size = 128

optimizer = 'SGD'											#TD find way to loop over different modules
learning_rate = 0.05
clipnorm = 1
momentum = 0.01
decay = 0
nesterov = True



hidden_units_params = None		#?????


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
		loaded_dataframes[dataframe_name] = dataframe

	# Return dictionary with dataset names and respective dataframes:
	return loaded_dataframes





def SplitAndNormalise(dataframe_dict, train_test_split):
	split_and_normalised_dataframes = {}

	# Loop through dataframes dict:
	print("Z-scoring and cleaning data..")
	for name, dataset in dataframe_dict.items():
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
	# Append split data to new dictionary:
		split_and_normalised_dataframes[name] = [normed_train_data, normed_test_data, train_labels, test_labels]
	return split_and_normalised_dataframes


def ReduceFeatures(normalised_data_dict, PCA_threshold):
	print("Computing PCA, reducing features up to "+str(round(PCA_threshold*100, 5))+"% VE..")
	reduced_dataframes = {}

	# Loop through dict:
	for name, datasets in normalised_data_dict.items():
		training_data = datasets[0]
		testing_data = datasets[1]

		train_labels = datasets[2]
		test_labels = datasets[3]

	# Initialise PCA object, keep components up to x% variance explained:
		PCA.__init__
		pca = PCA(n_components=PCA_threshold)
	
	# Fit to training set, apply to both training and testing set:			
		train_postPCA = pd.DataFrame(pca.fit_transform(training_data))
		test_postPCA = pd.DataFrame(pca.transform(testing_data))
		if train_postPCA.shape[1] != test_postPCA.shape[1]:
			print("Something went wrong during PCA generation, training and testing features are not equal.")
			return

	# Append PCA columns to new dict together with label columns (i.e. untouched ddG values per pert):	
		reduced_dataframes[name] = [ train_postPCA, test_postPCA, train_labels, test_labels ]
	
	return reduced_dataframes


def FF_DNN_KERAS(preprocessed_data_dict):
	# Display training progress by printing a single dot per epoch:
	class PrintDot(keras.callbacks.Callback):
	  def on_epoch_end(self, epoch, logs):
	    if epoch % 100 == 0: print('')
	    print('.', end='')

	# Set early stopping variable:
	early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=0)

	resulting_MAEs = {}
	# Loop through dict:
	for name, dataframes in preprocessed_data_dict.items():

	# Retrieve datasets and set some variables:
		train_postPCA, test_postPCA, train_labels, test_labels = dataframes[0], dataframes[1], dataframes[2], dataframes[3],
		
	# Build keras DNN using global params:
		def build_model():
			  model = keras.Sequential([
			    keras.layers.Dense(round(len(train_postPCA.columns)/2), activation=tf.nn.leaky_relu, input_shape=[len(train_postPCA.keys())]),

			    keras.layers.Dense(round(len(train_postPCA.columns)/4), activation=tf.nn.leaky_relu),
			    keras.layers.Dense(1, activation=tf.nn.leaky_relu)
			  ])
			  model.add(keras.layers.Dropout(dropout))				#??????

			  optimizer = tf.keras.optimizers.Adam(lr=learning_rate, clipnorm=clipnorm)
			  model.compile(loss=loss_function,
			                optimizer=optimizer,
			                metrics=['mae', 'mse']
			                )
			  return model

		model = build_model()
		print("#########################  "+str(name)+":  ##########################")
		print(str(model.summary()))
		print("Fitting model to data..")
		history = model.fit(
		  train_postPCA, train_labels,
		  epochs=n_epochs, 
		  validation_split = 1 - train_test_split, 
		  verbose=0,
		  callbacks=[PrintDot(), early_stopping],
		  batch_size=batch_size)

		
		hist = pd.DataFrame(history.history)
		hist['epoch'] = history.epoch
		pd.options.display.max_rows = 8
	# calculate MAE on external set; append to dict:
		MAE = hist["val_mean_absolute_error"].tail(10).mean()
		print("Model MAE on external testing set (n="+str(len(test_labels))+"): ", MAE)
		resulting_MAEs[name] = MAE
		"""
		#Not used in optimisation:
		plt.subplot()
		plt.xlabel('Epoch')
		plt.ylabel('Mean Abs Error [ddG offset]')
		plt.plot(hist['epoch'], hist['mean_absolute_error'],
		       label='Training Error')
		plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
		       label = 'Validation Error')
		plt.legend()
		plt.ylim([0,5])
		plt.draw()
		plt.pause(0.01)
		plt.title(name, fontsize=16)


		plt.show()"""

	return resulting_MAEs

MAE_distribution_per_model = pd.DataFrame()

for i in range(30):
	print("########## iteration: ", i)
	collection = LoadDatasets(dataset_path)
	normalised_collection = SplitAndNormalise(collection, train_test_split)
	preprocessed_data_dict = ReduceFeatures(normalised_collection, PCA_threshold)
	results = FF_DNN_KERAS(preprocessed_data_dict)

	MAE_df = pd.DataFrame.from_records([results])
	MAE_distribution_per_model = MAE_distribution_per_model.append(MAE_df)


print(MAE_distribution_per_model)
MAE_distribution_per_model.to_csv("output.csv", index=False)
import seaborn as sns 

sns.distplot(MAE_distribution_per_model["dataset_1"], label="pFP", hist=False)
sns.distplot(MAE_distribution_per_model["dataset_2"], label="dFEAT", hist=False)
sns.distplot(MAE_distribution_per_model["dataset_3"], label="dPLEC", hist=False)

sns.distplot(MAE_distribution_per_model["dataset_12"], label="pFP + dFEAT", hist=False)
sns.distplot(MAE_distribution_per_model["dataset_23"], label="dFEAT + dPLEC", hist=False)
sns.distplot(MAE_distribution_per_model["dataset_13"], label="pFP + dPLEC", hist=False)

sns.distplot(MAE_distribution_per_model["dataset_123"], label="pFP + dFEAT + dPLEC", hist=False)


plt.xlabel("MAE (ddG)")
plt.ylabel("Frequency")
plt.title('Histogram of ddG offset prediction MAEs')
plt.legend()
plt.show()