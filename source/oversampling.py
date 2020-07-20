import os
from collections import Counter
import glob
import numpy as np
import pandas as pd
from dtosmote import DTO
from imblearn.base import BaseSampler
from imblearn.metrics import classification_report_imbalanced
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE
from matplotlib import pyplot as plt
from scipy.io.arff import loadarff
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.metrics import roc_auc_score, r2_score, mean_absolute_error, mean_squared_error, max_error, \
	mean_poisson_deviance, mean_gamma_deviance
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import normalize

from dto_smoter import dtosmoter
from parameters import projectors, order, alphas, train_smote_ext, datasets
from gsmote import GeometricSMOTE
import warnings
from regression_algorithms import REGRESSION

warnings.filterwarnings('ignore')


class FakeSampler(BaseSampler):
	_sampling_type = 'bypass'
	
	def _fit_resample(self, X, y):
		return X, y


class Oversampling:
	
	def __init__(self):
		pass
	
	def runSMOTEvariationsGen(self, folder):
		"""
		Create files with SMOTE preprocessing and without preprocessing.
		:param datasets: datasets.
		:param folder:   cross-validation folders.
		:return:
		"""
		smote = SMOTE()
		borderline1 = BorderlineSMOTE(kind='borderline-1')
		borderline2 = BorderlineSMOTE(kind='borderline-2')
		smoteSVM = SVMSMOTE()
		geometric_smote = GeometricSMOTE(n_jobs=-1)
		
		for dataset in datasets:
			for fold in range(5):
				path = os.path.join(folder, dataset, str(fold), ''.join([dataset, "_train.csv"]))
				train = np.genfromtxt(path, delimiter=',')
				X = train[:, 0:train.shape[1] - 1]
				Y = train[:, train.shape[1] - 1]
				Y = Y.reshape(len(Y), 1)
				
				# SMOTE
				print("SMOTE..." + dataset)
				data_r = np.hstack([X, Y])
				data_r = pd.DataFrame(data_r)
				data_r.columns = data_r.columns.astype(str)
				colunas = list(data_r.columns)
				y_name = colunas[-1]
				
				dtoregression = dtosmoter(
						
						data=data_r,
						y=y_name,
						oversampler=smote
				)
				
				dtoregression.to_csv(os.path.join(folder, dataset, str(fold), ''.join([dataset, "_SMOTE.csv"])),
				                     header=False, index=False)
				# SMOTE BORDERLINE1
				print("Borderline1..." + dataset)
				data_r = np.hstack([X, Y])
				data_r = pd.DataFrame(data_r)
				data_r.columns = data_r.columns.astype(str)
				colunas = list(data_r.columns)
				y_name = colunas[-1]
				dtoregression = dtosmoter(
						
						data=data_r,
						y=y_name,
						oversampler=borderline1
				)
				
				dtoregression.to_csv(os.path.join(folder, dataset, str(fold), ''.join([dataset, "_Borderline1.csv"])),
				                     header=False, index=False)
				# SMOTE BORDERLINE2
				print("Borderline2..." + dataset)
				data_r = np.hstack([X, Y])
				data_r = pd.DataFrame(data_r)
				data_r.columns = data_r.columns.astype(str)
				colunas = list(data_r.columns)
				y_name = colunas[-1]
				dtoregression = dtosmoter(
						
						data=data_r,
						y=y_name,
						oversampler=borderline2
				)
				
				dtoregression.to_csv(os.path.join(folder, dataset, str(fold), ''.join([dataset, "_Borderline2.csv"])),
				                     header=False, index=False)
				# SMOTE SVM
				print("SMOTE SVM..." + dataset)
				data_r = np.hstack([X, Y])
				data_r = pd.DataFrame(data_r)
				data_r.columns = data_r.columns.astype(str)
				colunas = list(data_r.columns)
				y_name = colunas[-1]
				dtoregression = dtosmoter(
						
						data=data_r,
						y=y_name,
						oversampler=smoteSVM
				)
				
				dtoregression.to_csv(os.path.join(folder, dataset, str(fold), ''.join([dataset, "_smoteSVM.csv"])),
				                     header=False, index=False)
				
				# GEOMETRIC SMOTE
				print("GEOMETRIC SMOTE..." + dataset)
				data_r = np.hstack([X, Y])
				data_r = pd.DataFrame(data_r)
				data_r.columns = data_r.columns.astype(str)
				colunas = list(data_r.columns)
				y_name = colunas[-1]
				dtoregression = dtosmoter(
						
						data=data_r,
						y=y_name,
						oversampler=geometric_smote
				)
				
				dtoregression.to_csv(
						os.path.join(folder, dataset, str(fold), ''.join([dataset, "_Geometric_SMOTE.csv"])),
						header=False, index=False)
	
	def runDelaunayVariationsGen(self, folder):
		
		for dataset in datasets:
			for fold in range(5):
				for p in projectors:
					for o in order:
						for a in alphas:
							path = os.path.join(folder, dataset, str(fold), ''.join([dataset, "_train.csv"]))
							train = np.genfromtxt(path, delimiter=',')
							X = train[:, 0:train.shape[1] - 1]
							Y = train[:, train.shape[1] - 1]
							Y = Y.reshape(len(Y), 1)
							print("dtosmote..." + dataset)
							data_r = np.hstack([X, Y])
							data_r = pd.DataFrame(data_r)
							data_r.columns = data_r.columns.astype(str)
							colunas = list(data_r.columns)
							y_name = colunas[-1]
							delaunay = DTO(dataset, geometry=o, dirichlet=a)
							name = "delaunay_" + p.__class__.__name__ + "_" + o + "_" + str(a)
							dtoregression = dtosmoter(data=data_r, y=y_name, oversampler=delaunay)
							dtoregression.to_csv(
									os.path.join(folder, dataset, str(fold), ''.join([dataset, "_" + name + ".csv"])),
									header=False, index=False)
	
	def runRegression(self, folder):
		print("INIT REGRESSION DTO")
		dfcol = ['ID', 'DATASET', 'FOLD', 'PREPROC', 'ALGORITHM', 'MODE', 'ORDER', 'ALPHA',
		         'R2score',  # sklearn.metrics.r2_score
		         'MAE',  # sklearn.metrics.mean_absolute_error
		         'MSE',  # sklearn.metrics.mean_squared_error
		         'MAX',  # sklearn.metrics.max_error
		         ]
		df = pd.DataFrame(columns=dfcol)
		i = 0
		for dataset in datasets:
			print(dataset)
			for fold in range(5):
				print('Fold: ', fold)
				test_path = os.path.join(folder, dataset, str(fold), ''.join([dataset, "_test.csv"]))
				test = np.genfromtxt(test_path, delimiter=',')
				X_test = test[:, 0:test.shape[1] - 1]
				Y_test = test[:, test.shape[1] - 1]
				# SMOTER REGRESSION
				print("SMOTER REGRESSION")
				for ext in train_smote_ext:
					train_path = os.path.join(folder, dataset, str(fold), ''.join([dataset, ext, ".csv"]))
					train = np.genfromtxt(train_path, delimiter=',')
					X_train = train[:, 0:train.shape[1] - 1]
					Y_train = train[:, train.shape[1] - 1]
					for name, rgs in REGRESSION.items():
						identificador = dataset + '_' + ext + '_' + name
						print(identificador)
						rgs.fit(X_train, Y_train)
						Y_pred = rgs.predict(X_test)
						df.at[i, 'ID'] = identificador
						df.at[i, 'DATASET'] = dataset
						df.at[i, 'FOLD'] = fold
						df.at[i, 'PREPROC'] = ext
						df.at[i, 'ALGORITHM'] = name
						df.at[i, 'MODE'] = 'PCA'
						df.at[i, 'ORDER'] = 'NONE'
						df.at[i, 'ALPHA'] = 'NONE'
						df.at[i, 'R2score'] = r2_score(Y_test, Y_pred)
						df.at[i, 'MAE'] = mean_absolute_error(Y_test, Y_pred)
						df.at[i, 'MSE'] = mean_squared_error(Y_test, Y_pred)
						df.at[i, 'MAX'] = max_error(Y_test, Y_pred)
						i = i + 1
				
				# DTO-SMOTER REGRESSION
				print("DTO-SMOTER REGRESSION")
				for p in projectors:
					for o in order:
						for a in alphas:
							id = "_delaunay_" + p.__class__.__name__ + "_" + o + "_" + str(a)
							train_path = os.path.join(folder, dataset, str(fold),
							                          ''.join([dataset, id, ".csv"]))
							train = np.genfromtxt(train_path, delimiter=',')
							X_train = train[:, 0:train.shape[1] - 1]
							Y_train = train[:, train.shape[1] - 1]
							for name, rgs in REGRESSION.items():
								identificador = dataset + '_' + ext + '_' + name
								print(identificador)
								rgs.fit(X_train, Y_train)
								Y_pred = rgs.predict(X_test)
								df.at[i, 'ID'] = identificador
								df.at[i, 'DATASET'] = dataset
								df.at[i, 'FOLD'] = fold
								df.at[i, 'PREPROC'] = '_delaunay' + "_" + o + "_" + str(a)
								df.at[i, 'ALGORITHM'] = name
								df.at[i, 'MODE'] = p.__class__.__name__
								df.at[i, 'ORDER'] = o
								df.at[i, 'ALPHA'] = a
								df.at[i, 'R2score'] = r2_score(Y_test, Y_pred)
								df.at[i, 'MAE'] = mean_absolute_error(Y_test, Y_pred)
								df.at[i, 'MSE'] = mean_squared_error(Y_test, Y_pred)
								df.at[i, 'MAX'] = max_error(Y_test, Y_pred)
								i = i + 1
			
			df.to_csv('./../output/results_regression.csv', index=False)
			print('DATA SAVED')
	
	def createValidationData(self, folder):
		"""
		Create sub datasets for cross validation purpose
		:param datasets: List of datasets
		:param folder: Where datasets was stored
		:return:
		"""
		for filename in glob.glob(os.path.join(folder, '*.arff')):
			print(filename)
			raw_data = loadarff(filename)
			df_data = pd.DataFrame(raw_data[0])
			drop_na_col = True,  ## auto drop columns with nan's (bool)
			drop_na_row = True,  ## auto drop rows with nan's (bool)
			## pre-process missing values
			if bool(drop_na_col) == True:
				df_data = df_data.dropna(axis=1)  ## drop columns with nan's
			
			if bool(drop_na_row) == True:
				df_data = df_data.dropna(axis=0)  ## drop rows with nan's
			
			## quality check for missing values in dataframe
			if df_data.isnull().values.any():
				raise ValueError("cannot proceed: data cannot contain NaN values")
			
			df_data = df_data.select_dtypes(exclude=['object'])
			Y = np.array(df_data.iloc[:, -1])
			X = normalize(np.array(df_data.iloc[:, 0:-1]))
			skf = KFold(n_splits=5, shuffle=True)
			dataset = filename.replace(folder, "")
			dataset = dataset.replace('.arff', '')
			for fold, (train_index, test_index) in enumerate(skf.split(X, Y)):
				X_train, X_test = X[train_index], X[test_index]
				y_train, y_test = Y[train_index], Y[test_index]
				y_train = y_train.reshape(len(y_train), 1)
				y_test = y_test.reshape(len(y_test), 1)
				train = pd.DataFrame(np.hstack((X_train, y_train)))
				test = pd.DataFrame(np.hstack((X_test, y_test)))
				os.makedirs(os.path.join(folder, dataset, str(fold)))
				train.to_csv(os.path.join(folder, dataset, str(fold), ''.join([dataset, "_train.csv"])), header=False,
				             index=False)
				test.to_csv(os.path.join(folder, dataset, str(fold), ''.join([dataset, "_test.csv"])), header=False,
				            index=False)
