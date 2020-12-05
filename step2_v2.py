# for imbalanced data 
# I'm using tutorial from this link
# https://elitedatascience.com/imbalanced-classes
# using the technique "Up-sample Minority Class"
# https://github.com/nickkunz/smogn	
import pandas as pd 
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
from scipy import stats
import os, shutil
import pickle
from sklearn.utils import resample
from collections import Counter
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
import smogn
# models
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn.svm import SVR
# from sklearn import svm
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from scipy import linalg

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.covariance import ShrunkCovariance, LedoitWolf
from sklearn.model_selection import GridSearchCV

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 	 sudo pip install imbalanced-learn

debug = 1

if debug == 1:
	folds = 2
else:
	folds = 5

svm_kernel = 'rbf' #'poly'#'linear'
do_PCA = 0
do_overSampling = 1
n_components = [10]

def test():



	# example of random oversampling to balance the class distribution
	# define dataset
	X, y = make_classification(n_samples=10000, weights=[0.99], flip_y=0)
	# summarize class distribution
	print(Counter(y))
	# define oversampling strategy
	oversample = RandomOverSampler(sampling_strategy='minority')
	# fit and apply the transform
	X_over, y_over = oversample.fit_resample(X, y)
	# summarize class distribution
	print(Counter(y_over))


def test2(data):
	# define dataset
	# X, y = make_classification(n_samples=10000, weights=[0.99], flip_y=0)
	y = data['AUC'].to_numpy(dtype ='float32')
	X = data.drop(columns=['AUC']).to_numpy(dtype ='float32')

	# define pipeline
	# steps = [('over', RandomOverSampler()), ('model', DecisionTreeClassifier())]
	steps = [('over', RandomOverSampler()), ('model', linear_model.LinearRegression())]

	lr = linear_model.LinearRegression()
	
	pipeline = Pipeline(steps=steps)
	# evaluate pipeline
	# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(lr, X, y, scoring='f1_micro', cv=10)
	score = mean(scores)
	print('F1 Score: %.3f' % score)

def svm (X_train, X_test, y_train, y_test):
	# regr = make_pipeline(StandardScaler(), SVR(kernel='linear',C=1.0, epsilon=0.2, gamma='auto'))
	# regr.fit(X_train, y_train)
	# 

	# clf = SVR(kernel='linear', C=1)

	# scores = cross_val_score(regr, X_train, y_train, cv=5)
	# score_valid = mean(scores)
	# score_test = regr.score(X_test, y_test)
	# print('Validation Score: %.3f' %(score_valid))
	# print('Testing Score: %.3f'%(score_test))

	param_grid = {'svr__C': [1, 10, 1e3, 5e3, 1e4],
              'svr__gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
	regr = make_pipeline(StandardScaler(), SVR(kernel=svm_kernel))
	grid = GridSearchCV(regr, param_grid)
	grid = grid.fit(X_train, y_train)
	print(grid)
	
	print('Best score: ', grid.best_score_)
	print('Best parameter: ', grid.best_estimator_)
	print('Best parameters: ', grid.best_params_)



def PCA_compute_scores(X, n_components):
    pca = PCA(svd_solver='full')
    fa = FactorAnalysis()

    pca_scores, fa_scores = [], []
    for n in n_components:
        pca.n_components = n
        # fa.n_components = n
        sc = np.mean(cross_val_score(pca, X))
        pca_scores.append(sc)
        print('score: ',sc)
        # fa_scores.append(np.mean(cross_val_score(fa, X)))

    return pca_scores, fa_scores

def PCA_fn(data):
	y = data['AUC'].to_numpy(dtype ='float32')
	X = data.drop(columns=['AUC']).to_numpy(dtype ='float32')

	n_features = 17421
	n_components = np.arange(0, n_features//10, 100)  # options for n_components

	pca_scores, fa_scores = PCA_compute_scores(X, n_components)
	n_components_pca = n_components[np.argmax(pca_scores)]
	# n_components_fa = n_components[np.argmax(fa_scores)]

	# pca = PCA(svd_solver='full', n_components='mle')
	# pca.fit(X)
	# n_components_pca_mle = pca.n_components_

	print("best n_components by PCA CV = %d" % n_components_pca)
	# print("best n_components by FactorAnalysis CV = %d" % n_components_fa)
	# print("best n_components by PCA MLE = %d" % n_components_pca_mle)

	rank = 10
	plt.figure()
	plt.plot(n_components, pca_scores, 'b', label='PCA scores')
	# plt.plot(n_components, fa_scores, 'r', label='FA scores')
	# plt.axvline(rank, color='g', label='TRUTH: %d' % rank, linestyle='-')
	# plt.axvline(n_components_pca, color='b',
	        # label='PCA CV: %d' % n_components_pca, linestyle='--')
	# plt.axvline(n_components_fa, color='r',
	            # label='FactorAnalysis CV: %d' % n_components_fa,
	            # linestyle='--')
	# plt.axvline(n_components_pca_mle, color='k',
	            # label='PCA MLE: %d' % n_components_pca_mle, linestyle='--')

	#  # compare with other covariance estimators
	# plt.axhline(shrunk_cov_score(X), color='violet',
	#              label='Shrunk Covariance MLE', linestyle='-.')
	#  plt.axhline(lw_score(X), color='orange',
	#              label='LedoitWolf MLE' % n_components_pca_mle, linestyle='-.')

	plt.xlabel('nb of components')
	plt.ylabel('CV scores')
	plt.legend(loc='lower right')
	plt.title('Tuning')

	plt.show()

def apply_PCA(X,y, n_components):
	data = []
	for n in n_components:	
		pca = PCA(n_components=n)
		principalComponents = pca.fit_transform(X)
		data.append(principalComponents)
	return data

def apply_PCA2(X,y, n_components=[10]):
	pcas = []
	for n in n_components:	
		pca = PCA(n_components=n)
		principalComponents = pca.fit(X)
		pcas.append(principalComponents)
	return pcas
	
# def handle_imbalanced_Data_try1():
	# ========================
		# handle im-balanced data 
		# ========================

		# # Separate majority and minority classes
		# minval = df['AUC'].min()
		# maxval = df['AUC'].max()

		# if debug == 1:
		# 	print('min,max: ', minval, maxval)

		# df_majority = df[df['AUC'] > minval+0.5]
		# df_minority = df[df['AUC'] <= minval+0.5]
		# majority_N = len(df_majority)

		# if debug ==1 :
		# 	print('Before Up Sampling: ', len(df_majority), len(df_minority))
		# 	print('majority_N: ', majority_N)

		
		# # Upsample minority class
		# df_minority_upsampled = resample(df_minority, 
  #                                replace=True,     # sample with replacement
  #                                n_samples=majority_N,    # to match majority class
  #                                random_state=123) # reproducible results
 
		# # Combine majority class with upsampled minority class
		# df_upsampled = pd.concat([df_majority, df_minority_upsampled])
		
		# if debug == 1:
		# 	# Display new class counts
		# 	df_majority = df_upsampled[df_upsampled['AUC'] > minval+0.5]
		# 	df_minority = df_upsampled[df_upsampled['AUC'] <= minval+0.5]
		# 	print('After Up Sampling: ', len(df_majority), len(df_minority))


def lr (X_train, X_test, y_train, y_test):

	regr = linear_model.LinearRegression()
	regr.fit(X_train, y_train)
	y_pred = regr.predict(X_test)

	print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

	print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))
	print('Score: %.2f'% regr.score(X_test, y_test) )

	y_test = np.reshape(y_test, (len(y_test), 1))
	y_pred = np.reshape(y_pred, (len(y_pred), 1))

	print('LR: ', y_test.shape, X_test.shape , y_pred.shape)
	# Plot outputs
	plt.scatter(X_test[:,0], y_test,  color='black')
	# plt.plot(X_test[:,0], y_pred, color='blue', linewidth=3)
	plt.scatter(X_test[:,0], y_pred, color='blue')

	plt.xticks(())
	plt.yticks(())

	plt.show()



def svm_trivial(X_train, X_test, y_train, y_test):
	
	regr = make_pipeline(StandardScaler(), SVR(kernel=svm_kernel,C=1000.0, epsilon=0.01, gamma='auto'))
	regr.fit(X_train, y_train)
	score = regr.score(X_test, y_test)
	y_pred = regr.predict(X_test)

	
	# clf = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1)
	# clf = SVR(kernel='linear', C=1, max_iter=1000).fit(X_train, y_train)
	# score = clf.score(X_test, y_test)


	print('SVM 1 fold score: ', score)
	print('y_pred: ', list(y_pred))
	print('y_test: ', list(y_test))

	

def main(data , drug_ids):
	# if debug == 1:
	# 	print(drug_ids)

	for drug_id in data:
		print('drug_id: ', drug_id)
		if debug == 1:
			print(data[drug_id].shape)

		df = data[drug_id]
		y = df['AUC'].to_numpy(dtype ='float32')
		X = df.drop(columns=['AUC']).to_numpy(dtype ='float32')
		X = StandardScaler().fit_transform(X)
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
		# test2(df)
		# svm(df)
		#PCA_fn(df)
		

		# ====================================
		# perform PCA
		if do_PCA == 1:
			
			if debug == 1:
				print('X_train shape: ', X_train.shape)

			
			pca = apply_PCA2(X_train,y_train,n_components)

			# save drug ids
			with open('data_pca_'+str(drug_id)+'_'+str(n_components[0])+'.pkl', 'wb') as f:
				pickle.dump(pca[0], f)

			# X = pca[0]
			X_train = pca[0].transform(X_train)
			X_test = pca[0].transform(X_test)

			if debug == 1:
				print('n_components: ', n_components[0], ', new shape: ', X_train.shape)

		else:
			# read saved data

			with open('data_pca_'+str(drug_id)+'_'+str(n_components[0])+'.pkl', 'rb') as f:
				d = pickle.load( f)

			X_train = d.transform(X_train)
			X_test = d.transform(X_test)

		# ====================================
		# Over Sampling
		if do_overSampling == 1:
			dfr = pd.DataFrame(X_train)
			dfr['AUC'] = pd.Series(y_train)

			# oversample = RandomOverSampler(sampling_strategy='minority')
			# oversample = SMOTE(sampling_strategy='minority')
			print(len(dfr[dfr['AUC'] > 0.5]))
			print(len(dfr[dfr['AUC'] <= 0.5]))

			upsampled_data = smogn.smoter(data=dfr, y='AUC')
			# fit and apply the transform
			# X, y = oversample.fit_resample(X, y)
			print(len(upsampled_data[upsampled_data['AUC'] > 0.5]))
			print(len(upsampled_data[upsampled_data['AUC'] <= 0.5]))
			# break
			y_train = upsampled_data['AUC'].to_numpy(dtype='float32')
			X_train = upsampled_data.drop(columns=['AUC']).to_numpy(dtype='float32')
		

		# ==================
		# split for training and testing

		
		if debug == 1:
			print("Split: ",X_train.shape, X_test.shape)

		# lr(X_train, X_test, y_train, y_test)
		svm_trivial(X_train, X_test, y_train, y_test)
		# svm(X_train, X_test, y_train, y_test)
		

		break # To do : should comment this line for delivery 

if __name__ == '__main__':
	
	
	
	# load drug ids
	drug_ids = []
	with open('data/drug_ids.pkl', 'rb') as f:
		drug_ids = pickle.load(f)

	# load data
	data = None
	with open('data/data.pkl', 'rb') as f:
		data = pickle.load(f)

	main(data = data, drug_ids=drug_ids)
	

