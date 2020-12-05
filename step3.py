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
from sklearn.linear_model import Ridge

debug = 0

MAX_DRUGS = 266
CLEAN_DATA_DIR='data_clean/'
CLEAN_DATA_DIR2 = 'data_clean2/'

model = 'ALL'

allmodel = 1

READ_PCA = 0

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

def svm (X_train, X_test, y_train, y_test, param_grid):

	
	
	regr = make_pipeline(StandardScaler(), SVR())
	grid = GridSearchCV(regr, param_grid, 
					    n_jobs=-1,
					    return_train_score=True)
	grid = grid.fit(X_train, y_train)

	# re-predict using best parameter:
	
	params = {}
	params['svr__kernel'] = [grid.best_params_['svr__kernel']]
	
	grid = GridSearchCV(regr, params, 
					    n_jobs=-1,
					    return_train_score=True)
	grid = grid.fit(X_train, y_train)


	y_pred_test = grid.predict(X_test)
	y_pred_train = grid.predict(X_train)
	MSE_test = mean_squared_error(y_true=y_test, y_pred=y_pred_test)
	MSE_train = mean_squared_error(y_true=y_train, y_pred=y_pred_train)

	# print(grid)
	# print('cv_results_: ', grid.cv_results_)
	# print('Best score: ', grid.best_score_)
	# print('Best parameter: ', grid.best_estimator_)
	# print('Best parameters: ', grid.best_params_)
	# print('MSE train: %.2f , MSE test: %.2f'
 #      % (MSE_train, MSE_test))

	test_scores = grid.cv_results_['mean_test_score']
	train_scores = grid.cv_results_['mean_train_score'] 

	slope, intercept, r_value, p_value, std_err = stats.linregress(y_test ,y_pred_test)
	return test_scores, train_scores, MSE_train, MSE_test, r_value, p_value, std_err

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
	# re-predict using best parameter:

	params = {}
	params['bootstrap'] = [grid.best_params_['bootstrap']]
	params['max_depth'] = [grid.best_params_['max_depth']]
	params['n_estimators'] = [grid.best_params_['n_estimators']]
	
	grid = GridSearchCV(regr, params, 
					    n_jobs=-1,
					    return_train_score=True)
	grid = grid.fit(X_train, y_train)


	y_pred_test = grid.predict(X_test)
	y_pred_train = grid.predict(X_train)
	MSE_test = mean_squared_error(y_true=y_test, y_pred=y_pred_test)
	MSE_train = mean_squared_error(y_true=y_train, y_pred=y_pred_train)

	# print(grid)
	# print('cv_results_: ', grid.cv_results_)
	# print('Best score: ', grid.best_score_)
	# print('Best parameter: ', grid.best_estimator_)
	# print('Best parameters: ', grid.best_params_)
	# print('MSE train: %.2f , MSE test: %.2f'
 #      % (MSE_train, MSE_test))

	test_scores = grid.cv_results_['mean_test_score']
	train_scores = grid.cv_results_['mean_train_score'] 

	slope, intercept, r_value, p_value, std_err = stats.linregress(y_test ,y_pred_test)
	return test_scores, train_scores, MSE_train, MSE_test, r_value, p_value, std_err

	

def NN(X_train, X_test, y_train, y_test,param=None):
	param_grid = param 
	regr = MLPRegressor(activation='relu', solver='adam' , shuffle=True, 
		random_state=47, tol=0.0001)

	grid = GridSearchCV(regr, param_grid, 
					    n_jobs=-1,
					    return_train_score=True)
	grid = grid.fit(X_train, y_train)
	
	# re-predict using best parameter:
	
	params = {}
	params['alpha'] = [grid.best_params_['alpha']]
	params['hidden_layer_sizes'] = [grid.best_params_['hidden_layer_sizes']]
	params['max_iter'] = [grid.best_params_['max_iter']]
	grid = GridSearchCV(regr, params ,
					    n_jobs=-1,
					    return_train_score=True)
	grid = grid.fit(X_train, y_train)


	y_pred_test = grid.predict(X_test)
	y_pred_train = grid.predict(X_train)
	MSE_test = mean_squared_error(y_true=y_test, y_pred=y_pred_test)
	MSE_train = mean_squared_error(y_true=y_train, y_pred=y_pred_train)

	# print(grid)
	# print('cv_results_: ', grid.cv_results_)
	# print('Best score: ', grid.best_score_)
	# print('Best parameter: ', grid.best_estimator_)
	# print('Best parameters: ', grid.best_params_)
	# print('MSE train: %.2f , MSE test: %.2f'
 #      % (MSE_train, MSE_test))

	test_scores = grid.cv_results_['mean_test_score']
	train_scores = grid.cv_results_['mean_train_score'] 

	slope, intercept, r_value, p_value, std_err = stats.linregress(y_test ,y_pred_test)
	return test_scores, train_scores, MSE_train, MSE_test, r_value, p_value, std_err



def RidgeR(X_train, X_test, y_train, y_test,param=None):
	param_grid = param 
	regr = Ridge()
	grid = GridSearchCV(regr, param_grid, 
					    n_jobs=-1,
					    return_train_score=True)
	grid = grid.fit(X_train, y_train)
	
	# re-predict using best parameter:
	params = {}
	params['alpha'] = [grid.best_params_['alpha']]

	grid = GridSearchCV(regr, params, 
					    n_jobs=-1,
					    return_train_score=True)
	grid = grid.fit(X_train, y_train)


	y_pred_test = grid.predict(X_test)
	y_pred_train = grid.predict(X_train)
	MSE_test = mean_squared_error(y_true=y_test, y_pred=y_pred_test)
	MSE_train = mean_squared_error(y_true=y_train, y_pred=y_pred_train)

	# print(grid)
	# print('cv_results_: ', grid.cv_results_)
	# print('Best score: ', grid.best_score_)
	# print('Best parameter: ', grid.best_estimator_)
	# print('Best parameters: ', grid.best_params_)
	# print('MSE train: %.2f , MSE test: %.2f'
 #      % (MSE_train, MSE_test))

	test_scores = grid.cv_results_['mean_test_score']
	train_scores = grid.cv_results_['mean_train_score'] 

	slope, intercept, r_value, p_value, std_err = stats.linregress(y_test ,y_pred_test)
	return test_scores, train_scores, MSE_train, MSE_test, r_value, p_value, std_err


def main():
	
	# load drug ids
	drug_ids = []
	with open(CLEAN_DATA_DIR+'drug_ids.pkl', 'rb') as f:
		drug_ids = pickle.load(f)

	cnt = 0
	for drug_id in drug_ids:
		if cnt > MAX_DRUGS:
			break
		cnt += 1

		if READ_PCA == 1:
			# read PCA for drug_id
			with open(CLEAN_DATA_DIR+'data_pca_'+str(drug_id)+'.pkl', 'rb') as f:
					PCAS = pickle.load( f)	
			n_pcas = {}
			n_pcas[1026] = 100
			n_pcas[1028] = 10
			n_pcas[1029] = 50
			n_pcas[1030] = 50
			n_pcas[1031] = 50
			n_pcas[1032] = 100
			n_pcas[1033] = 200
			n_pcas[1036] = 50
			n_pcas[1037] = 200
			n_pcas[1038] = 200
			n_pcas[1039] = 50
			
		else:
			# read CorrF for drug_id
			with open(CLEAN_DATA_DIR2+'data_corr_'+str(drug_id)+'.pkl', 'rb') as f:
					PCAS = pickle.load( f)	
		
		if debug == 1:					
			print()
			print('START')
			print()
		for n in PCAS: # for each component value

			if READ_PCA == 1:
				n = n_pcas[drug_id]

			X_train = PCAS[n]['X_train']
			y_train = PCAS[n]['y_train']
			X_test = PCAS[n]['X_test'] 
			y_test = PCAS[n]['y_test'] 

			if X_train.shape[1] == 0 or X_test.shape[1] == 0:
				if debug == 1:
					print('\ndrug_id: ', drug_id, ', No Features\n\n')
				continue

			if debug == 1:
				print('\n=========================================')
				print('Run:', model,',n_components:', n,', drug_id:', drug_id, ',shape: ', X_train.shape)

			res = {}

			if model == 'SVM_TRIVIAL':
				svm_trivial(X_train, X_test, y_train, y_test)
			if model == 'SVM' or allmodel == 1:

				# param_grid = { 'svr_kernel': ['poly'],
				#   'svr__C': [1, 10, 1e3, 5e3],
	   #            'svr__gamma': [ 0.0005, 0.01, 0.1], }

				param = {'svr_kernel': {}}
				
				param['svr_kernel']['svr__kernel'] = ['linear', 'poly', 'rbf']
				# param['svr_kernel']['svr__C'] = [10]
				# param['svr_kernel']['svr__gamma'] = [0.01]
				

				# param['svr__C']['svr__C'] = [10,100,1000]
				# param['svr__C']['svr_kernel'] = ['poly']
				# param['svr__C']['svr__gamma'] = [0.01]
				
				whichlist = ['svr_kernel']
				for which in whichlist:
				
					# print('\n')
					# print(which)
					# print()
					
					test_scores, train_scores, MSE_train, MSE_test, r_value, p_value, std_err = svm(X_train, X_test, y_train, y_test, param[which])
					
					res['SVM'] = [test_scores, train_scores, MSE_train, MSE_test, r_value, p_value, std_err]

					# print()
					# print('SVM Results: ', drug_id)
					# print(res['SVM'])
					# fig, ax = plt.subplots()
					# ax.plot(param[which][which], test_scores, marker="o", label='test')
					# ax.plot(param[which][which], train_scores, marker="o", label='train')
					# plt.title('Tuning '+which)
					# plt.legend(loc='best')
					# plt.tight_layout()
					# # plt.show()
					# plt.savefig(model+'_'+str(drug_id)+'_'+str(n)+'_'+which+'_scores.png')

			if model == 'RANDOM_FOREST' or allmodel == 1:
				param = {'max_depth': {}, 'n_estimators':{}, 'bootstrap':{}}
				
				param['max_depth']['n_estimators'] = [100]
				param['max_depth']['max_depth'] = [3,8,15]
				param['max_depth']['bootstrap'] = [False]

				param['n_estimators']['n_estimators'] = [10,100,500]
				param['n_estimators']['max_depth'] = [8]
				param['n_estimators']['bootstrap'] = [False]

				param['bootstrap']['n_estimators'] = [100]
				param['bootstrap']['max_depth'] = [8]
				param['bootstrap']['bootstrap'] = [False, True]
				
				# whichlist = ['max_depth', 'n_estimators','bootstrap']
				whichlist = ['max_depth']
				for which in whichlist:
				
					# print('\n')
					# print(which)
					# print()
					
					test_scores, train_scores, MSE_train, MSE_test, r_value, p_value, std_err = RF(X_train, X_test, y_train, y_test, param[which])
					

					res['RF'] = [test_scores, train_scores, MSE_train, MSE_test, r_value, p_value, std_err]

					# print()
					# print('RF Results: ', drug_id)
					# print(res['RF'])

					# fig, ax = plt.subplots()
					# ax.plot(param[which][which], test_scores, marker="o", label='test')
					# ax.plot(param[which][which], train_scores, marker="o", label='train')
					# plt.title('Tuning '+which)
					# plt.legend(loc='best')
					# plt.tight_layout()
					# # plt.show()
					# plt.savefig(model+'_'+str(drug_id)+'_'+str(n)+'_'+which+'_scores.png')

			if model == 'NN' or allmodel == 1:
				param = {'hidden_layer_sizes': {}, 'alpha':{}, 'max_iter':{}}
				
				param['hidden_layer_sizes']['hidden_layer_sizes'] = [2,5,7,10]
				param['hidden_layer_sizes']['alpha'] = [0.0001]
				param['hidden_layer_sizes']['max_iter'] = [1000]

				param['alpha']['hidden_layer_sizes'] = [50]
				param['alpha']['alpha'] = [0.0001, 0.01, 0.1]
				param['alpha']['max_iter'] = [200]

				param['max_iter']['hidden_layer_sizes'] = [50]
				param['max_iter']['alpha'] = [0.0001]
				param['max_iter']['max_iter'] = [200,5000, 10000]
				
				# whichlist = ['hidden_layer_sizes', 'alpha','max_iter']
				whichlist = ['hidden_layer_sizes']
				for which in whichlist:
				
					# print('\n')
					# print(which)
					# print()
					
					test_scores, train_scores, MSE_train, MSE_test, r_value, p_value, std_err = NN(X_train, X_test, y_train, y_test, param[which])
					
					res['NN'] = [test_scores, train_scores, MSE_train, MSE_test, r_value, p_value, std_err]

					# print()
					# print('NN Results: ', drug_id)
					# print(res['NN'])

					# fig, ax = plt.subplots()
					# ax.plot(param[which][which], test_scores, marker="o", label='test')
					# ax.plot(param[which][which], train_scores, marker="o", label='train')
					# plt.title('Tuning '+which)
					# plt.legend(loc='best')
					# plt.tight_layout()
					# # plt.show()
					# plt.savefig(model+'_'+str(drug_id)+'_'+str(n)+'_'+which+'_scores.png')

			if model == 'Ridge' or allmodel == 1:
				param = {'alpha': {}}
				
				
				
				param['alpha']['alpha'] = [0.1, 1.0, 5.0]
				

				
				whichlist = ['alpha']
				for which in whichlist:
				
					# print('\n')
					# print(which)
					# print()
					
					test_scores, train_scores, MSE_train, MSE_test, r_value, p_value, std_err = RidgeR(X_train, X_test, y_train, y_test, param[which])
					
					res['Ridge'] = [test_scores, train_scores, MSE_train, MSE_test, r_value, p_value, std_err]

					# print()
					# print('Ridge Results: ', drug_id)
					# print(res['Ridge'])

					# fig, ax = plt.subplots()
					# ax.plot(param[which][which], test_scores, marker="o", label='test')
					# ax.plot(param[which][which], train_scores, marker="o", label='train')
					# plt.title('Tuning '+which)
					# plt.legend(loc='best')
					# plt.tight_layout()
					# # plt.show()
					# plt.savefig(model+'_'+str(drug_id)+'_'+str(n)+'_'+which+'_scores.png')

			print(drug_id,',', res['SVM'][4], ',',res['RF'][4], ',',res['NN'][4],',', res['Ridge'][4] ,',', res['SVM'][0][0], ',',res['RF'][0][0], ',',res['NN'][0][0],',', res['Ridge'][0][0] ,',', res['SVM'][1][0], ',',res['RF'][1][0], ',',res['NN'][1][0],',', res['Ridge'][1][0] ,',', res['SVM'][2], ',',res['RF'][2], ',',res['NN'][2],',', res['Ridge'][2] ,',', res['SVM'][3], ',',res['RF'][3], ',',res['NN'][3],',', res['Ridge'][3])
			if READ_PCA == 1:
				break



if __name__ == '__main__':
	


	main()
	

