# for imbalanced data 
# I'm using tutorial from this link
# https://elitedatascience.com/imbalanced-classes
# using the technique "Up-sample Minority Class"
# https://github.com/nickkunz/smogn	
import os, shutil
import pandas as pd 
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
from scipy import stats
from scipy import linalg
import pickle

from collections import Counter
from numpy import mean

from sklearn.utils import resample
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.covariance import ShrunkCovariance, LedoitWolf
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 	 sudo pip install imbalanced-learn
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
import smogn

# models
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


debug = 1

START_DRUG = 1

MAX_DRUGS = 0
CLEAN_DATA_DIR='data_clean/'


# model = 'NN'
run_model = 0
run_model_NN = 1
run_model_RF = 1
do_plot = 0

def svm_trivial(X_train, X_test, y_train, y_test):
	svm_kernel = 'rbf' #'poly'#'linear'

	regr = make_pipeline(StandardScaler(), SVR(kernel=svm_kernel,C=1000.0, epsilon=0.01, gamma='auto'))
	regr.fit(X_train, y_train)
	score = regr.score(X_test, y_test)
	y_pred = regr.predict(X_test)

	
	# clf = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1)
	# clf = SVR(kernel='linear', C=1, max_iter=1000).fit(X_train, y_train)
	# score = clf.score(X_test, y_test)


	print('SVM 1 fold score: ', score)
	# print('y_pred: ', list(y_pred))
	# print('y_test: ', list(y_test))
	
	print('Mean squared error: %.2f'
      % mean_squared_error(y_true=y_test, y_pred=y_pred))

def svm (X_train, X_test, y_train, y_test):

	svm_kernel = 'poly'
	param_grid = {'svr__C': [10],
              'svr__gamma': [0.01], }
	regr = make_pipeline(StandardScaler(), SVR(kernel=svm_kernel))
	grid = GridSearchCV(regr, param_grid,
					    n_jobs=-1,
					    return_train_score=True)
	grid = grid.fit(X_train, y_train)
	y_pred_test = grid.predict(X_test)
	y_pred_train = grid.predict(X_train)
	MSE_test = mean_squared_error(y_true=y_test, y_pred=y_pred_test)
	MSE_train = mean_squared_error(y_true=y_train, y_pred=y_pred_train)

	print(grid)
	# print('cv_results_: ', grid.cv_results_)
	# print('Best score: ', grid.best_score_)
	# print('Best parameter: ', grid.best_estimator_)
	# print('Best parameters: ', grid.best_params_)
	print('MSE train: %.2f , MSE test: %.2f'
      % (MSE_train, MSE_test))

	test_scores = grid.cv_results_['mean_test_score']
	train_scores = grid.cv_results_['mean_train_score'] 

	print('test_scores:', test_scores)
	print('train_scores:', train_scores)

	return test_scores, train_scores, MSE_train, MSE_test

def RF(X_train, X_test, y_train, y_test,param=None):

	param_grid = param #{'max_depth': [10]}

	regr = RandomForestRegressor(
		 	criterion='mse', min_samples_split=50, min_samples_leaf=50, 
		 	max_features='log2',  oob_score=False, n_jobs=-1, 
		 	random_state=47,  max_samples=len(X_train)//2)

	grid = GridSearchCV(regr, param_grid, 
					    n_jobs=-1,
					    return_train_score=True)
	grid = grid.fit(X_train, y_train)
	y_pred_test = grid.predict(X_test)
	y_pred_train = grid.predict(X_train)
	MSE_test = mean_squared_error(y_true=y_test, y_pred=y_pred_test)
	MSE_train = mean_squared_error(y_true=y_train, y_pred=y_pred_train)

	print(grid)
	# print(' cv_results_: ', grid.cv_results_)
	# print('Best score: ', grid.best_score_)
	# print('Best parameter: ', grid.best_estimator_)
	# print('Best parameters: ', grid.best_params_)
	print('MSE train: %.2f , MSE test: %.2f'
      % (MSE_train, MSE_test))

	test_scores = grid.cv_results_['mean_test_score']
	train_scores = grid.cv_results_['mean_train_score'] 

	print('test_scores:', test_scores)
	print('train_scores:', train_scores)
	print()
	res = {'y_pred_test': list(y_pred_test), 'y_test': list(y_test)
		, 'y_pred_train': list(y_pred_train) , 'y_train': list(y_train) }
	print('y_Pred: ')
	print(res)

	return test_scores, train_scores, MSE_train, MSE_test

def NN(X_train, X_test, y_train, y_test,param=None):
	param_grid = param 
	regr = MLPRegressor(activation='relu', solver='adam' , shuffle=True, 
		random_state=47, tol=0.0001)

	grid = GridSearchCV(regr, param_grid, 
					    n_jobs=-1,
					    return_train_score=True)
	grid = grid.fit(X_train, y_train)
	y_pred_test = grid.predict(X_test)
	y_pred_train = grid.predict(X_train)
	MSE_test = mean_squared_error(y_true=y_test, y_pred=y_pred_test)
	MSE_train = mean_squared_error(y_true=y_train, y_pred=y_pred_train)

	print(grid)
	# print(' cv_results_: ', grid.cv_results_)
	# print('Best score: ', grid.best_score_)
	# print('Best parameter: ', grid.best_estimator_)
	# print('Best parameters: ', grid.best_params_)
	print('MSE train: %.2f , MSE test: %.2f'
      % (MSE_train, MSE_test))

	test_scores = grid.cv_results_['mean_test_score']
	train_scores = grid.cv_results_['mean_train_score'] 

	print('test_scores:', test_scores)
	print('train_scores:', train_scores)

	return test_scores, train_scores, MSE_train, MSE_test


def main():
	
	# load drug ids
	drug_ids = []
	with open(CLEAN_DATA_DIR+'drug_ids.pkl', 'rb') as f:
		drug_ids = pickle.load(f)

	cnt = 0
	for i in range(START_DRUG,len(drug_ids)):
	# for drug_id in drug_ids:
		drug_id = drug_ids[i]
		if cnt > MAX_DRUGS:
			break
		cnt += 1

		# read PCA for drug_id
		with open(CLEAN_DATA_DIR+'data_pca_'+str(drug_id)+'.pkl', 'rb') as f:
				PCAS = pickle.load( f)	

		
		score = {}
		for n in PCAS: # for each component value

			X_train = PCAS[n]['X_train']
			y_train = PCAS[n]['y_train']
			X_test = PCAS[n]['X_test'] 
			y_test = PCAS[n]['y_test'] 

			print('\n=========================================')
			print('Run: n_components:', n,', drug_id:', drug_id)

			models = ['SVM', 'RF', 'NN']
			score[n] = {}
			score[n]['train'] = []
			score[n]['test'] = []
			score[n]['MSE_train'] = []
			score[n]['MSE_test'] = []
			# ===================
			# SVM
			if run_model == 1:
				
				test_scores, train_scores, MSE_train, MSE_test = svm(X_train, X_test, y_train, y_test)
				score[n]['train'].append(train_scores[0])
				score[n]['test'].append(test_scores[0])
				score[n]['MSE_train'].append(MSE_train)
				score[n]['MSE_test'].append(MSE_test)
				
			# ===================
			# RF
			if run_model_RF == 1:
				param = {'max_depth': {}, 'n_estimators':{}, 'bootstrap':{}}
				
				param['max_depth']['n_estimators'] = [100]
				param['max_depth']['max_depth'] = [15]
				param['max_depth']['bootstrap'] = [False]

				
				whichlist = ['max_depth']
				for which in whichlist:
				
					print('\n')
					print(which)
					print()
					
					test_scores, train_scores, MSE_train, MSE_test = RF(X_train, X_test, y_train, y_test, param[which])
					
					score[n]['train'].append(train_scores[0])
					score[n]['test'].append(test_scores[0])
					score[n]['MSE_train'].append(MSE_train)
					score[n]['MSE_test'].append(MSE_test)

			# ===================
			# NN
			if run_model_NN == 1:
				param = {'hidden_layer_sizes': {}, 'alpha':{}, 'max_iter':{}}
				
				param['hidden_layer_sizes']['hidden_layer_sizes'] = [15]
				param['hidden_layer_sizes']['alpha'] = [0.0001, 0.001, 0.01, 0.1]
				param['hidden_layer_sizes']['max_iter'] = [5000]

				whichlist = ['hidden_layer_sizes']
				for which in whichlist:
				
					print('\n')
					print(which)
					print()
					
					test_scores, train_scores, MSE_train, MSE_test = NN(X_train, X_test, y_train, y_test, param[which])
					
					score[n]['train'].append(train_scores[0])
					score[n]['test'].append(test_scores[0])
					score[n]['MSE_train'].append(MSE_train)
					score[n]['MSE_test'].append(MSE_test)

			print()
			print(score)
			print()

			if do_plot == 1:
				fig, ax = plt.subplots()
				ax.plot(models, score[n]['test'], marker="o", label='test')
				ax.plot(models, score[n]['train'], marker="o", label='train')
				ax.plot(models, score[n]['MSE_train'], marker="o", label='MSE_train')
				ax.plot(models, score[n]['MSE_test'], marker="o", label='MSE_test')
				plt.title('R2 Scores with PCA ='+str(n))
				plt.legend(loc='best')
				plt.tight_layout()
				# plt.show()
				plt.savefig(str(n)+'_'+str(drug_id)+'_scores.png')



if __name__ == '__main__':
	


	main()
	

