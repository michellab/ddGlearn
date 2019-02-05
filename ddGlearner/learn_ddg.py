import pandas as pd 
#import deepchem as dc 
import sklearn
import numpy as np
import statistics

"""
# SKLearn standard regressor:

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

from sklearn.neural_network import MLPRegressor

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

for i in range(len(np.arange(1, 10))):
	r2_scores = []
	for i in range(len(np.arange(1, 50))):
		mlp = MLPRegressor(max_iter=400, hidden_layer_sizes=(100, 100, 100))
		mlp.fit(X_train,y_train)
		predictions = mlp.predict(X_test)
		#print(r2_score(y_test,predictions))
		#print(mean_absolute_error(y_test,predictions))
		#print("#########################")
		r2_scores.append(mean_absolute_error(y_test,predictions))
	print(statistics.mean(r2_scores))
	print(statistics.stdev(r2_scores))
	print("###################")



"""



"""
# Deepchem:

import deepchem as dc
featurizer = dc.feat.UserDefinedFeaturizer(all_columns)


loader = dc.data.UserCSVLoader(["ddG"], 
	smiles_field=None, 
	id_field=index, 
	mol_field=None, 
	featurizer=featurizer, 
	verbose=True, 
	log_every_n=1000)

data = loader.featurize(dataset_file)

"""