
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



# variables for future use:
dataset_path = None
num_epochs = None
num_steps = None
scaler = None
activation_fn = None
remove_outliers = True


hidden_units_params = None		#?????



# load dataset:
dataset_file = "../datasets/datasets/dataset_2.csv"
dataset_perts = pd.read_csv("../datasets/datasets/dataset_2.csv", usecols=["Perturbation"])


dataset = pd.read_csv("../datasets/datasets/dataset_2.csv").drop("Perturbation", axis=1)
dataset = dataset.apply(pd.to_numeric).astype(float)


# Define columns:
FEATURES = dataset.drop("ddG", axis=1).columns.tolist()
COLUMNS = dataset.columns.tolist()
LABEL = "ddG"


y = dataset["ddG"]

for i in range(20):

	# Split into Train and Test (80/20)
	train, test = train_test_split(dataset, test_size=0.2)

	# Outlier exclusion step:

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		outlier_forest = IsolationForest(max_samples = "auto")
		outlier_forest.fit(train)

		y_no_outliers = outlier_forest.predict(train)
		y_no_outliers = pd.DataFrame(y_no_outliers, columns = ['Top'])
		y_no_outliers[y_no_outliers['Top'] == 1].index.values

		train = train.iloc[y_no_outliers[y_no_outliers['Top'] == 1].index.values]
		train.reset_index(drop = True, inplace = True)

		print("Number of outliers in training data:", y_no_outliers[y_no_outliers['Top'] == -1].shape[0])

	# Normalisation step:

	scaler_ddG = StandardScaler()
	mat_ddG = np.array(train.ddG).reshape((len(train)), 1)
	scaler_ddG.fit(mat_ddG)


	scaler_train = StandardScaler()
	mat_train = np.matrix(train)
	scaler_train.fit(mat_train)


	train_norm = scaler_train.transform(train)
	train = pd.DataFrame(train_norm, columns=COLUMNS)


	scaler_test = StandardScaler()

	scaler_test.fit(test.drop("ddG", axis=1))

	test_norm = scaler_test.transform(test.drop("ddG", axis=1))
	test = pd.DataFrame(test_norm, columns=FEATURES)

	# Set up splits:
	dataset_no_ddG = list(dataset.columns)
	dataset_no_ddG.remove("ddG")

	COLUMNS = list(dataset.columns)
	FEATURES = dataset_no_ddG
	LABEL = "ddG"

	feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]

	training_set = train[COLUMNS]
	prediction_set = train.ddG

	x_train, x_test, y_train, y_test = train_test_split(training_set[FEATURES], 
		prediction_set, 
		test_size=0.2, 
		)

	y_train = pd.DataFrame(y_train, columns = [LABEL])
	training_set = pd.DataFrame(x_train, columns = FEATURES).merge(y_train, 
		left_index = True, 
		right_index = True)


	y_test = pd.DataFrame(y_test, columns = [LABEL])
	test_set = pd.DataFrame(x_test, columns = FEATURES).merge(y_test, 
		left_index = True, 
		right_index = True)


	# Shuffle the training set:
	training_set.reset_index(drop = True, inplace =True)

	# Set up TF regressor:
	tf.logging.set_verbosity(tf.logging.ERROR)
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


	feature_columns = [tf.feature_column.numeric_column(k) for k in FEATURES]

	def get_input_fn(data_set, num_epochs=None):
		return tf.estimator.inputs.pandas_input_fn(
			x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
			y=pd.Series(data_set[LABEL].values),
			num_epochs=num_epochs,
			shuffle=False
			)

	regressor = tf.estimator.DNNRegressor(feature_columns=feature_columns, 
	                                          activation_fn = tf.nn.tanh, 
	                                          hidden_units=[50, 10])

	# Execute training:
	print("Training model..")
	regressor.train(input_fn=get_input_fn(training_set), steps=30)

	# Evaluate:
	ev = regressor.evaluate(input_fn=get_input_fn(test_set, num_epochs=50))
	loss_score = ev["loss"]
	print("Loss: {0:f}".format(loss_score))

	# Predict on test set:
	y = regressor.predict(input_fn=get_input_fn(test_set, num_epochs=50))


	# Reshape prediction results:
	predictions = [ p["predictions"] for p in itertools.islice(y, test_set.shape[0])]


	# Call scaler_ddG to retrieve original scale of ddG data:
	predictions = pd.DataFrame(scaler_ddG.inverse_transform(np.array(predictions).reshape(len(predictions),1)), columns = ['Prediction'])



	test_index = test_set.index.tolist()

	perts_column = pd.DataFrame([dataset_perts.iloc[i].values.tolist() for i in test_index], columns=["Perturbation"])
	reality = pd.read_csv("../datasets/datasets/dataset_2.csv", usecols=["Perturbation", "ddG"])

	# Combine predicted with experimental; plot correlations:
	predictions.reset_index(drop=True, inplace=True)
	reality.reset_index(drop=True, inplace=True)

	result = pd.merge(perts_column, reality, on="Perturbation", how="left")

	result = pd.concat([result, predictions], axis=1, join="outer").rename(index=str, columns={"ddG":"Experimental_ddG", "Prediction":"Predicted_ddG"})
	result.index = test_index
	print(result)




	slope, intercept, r_value, p_value, std_err = stats.linregress(result.iloc[:, 1].values, result.iloc[:, 2].values)
	mae = mean_absolute_error(result.iloc[:, 1].values, result.iloc[:, 2].values)

	print("R2-Score: "+str(r_value**2))
	print("MAE: "+str(mae))


	#print("R2-Score: "+str(r2_score(result["Experimental_ddG"], result["Predicted_ddG"])))
	




	print("################################################")



	import matplotlib.pyplot as plt
	ts = result.plot.scatter(x="Experimental_ddG", y="Predicted_ddG")

	mn = np.min(result["Experimental_ddG"].values)
	mx = np.max(result["Experimental_ddG"].values)
	x1=np.linspace(mn,mx, 500)
	y1=slope*x1+intercept


	result[['Experimental_ddG','Predicted_ddG','Perturbation']].apply(lambda x: ts.text(*x),axis=1)
	ts.plot()
	plt.plot(x1, y1)
	plt.show()


	
