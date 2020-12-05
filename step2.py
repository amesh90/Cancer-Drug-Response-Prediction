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

MAX_DRUGS = 10
CLEAN_DATA_DIR='data_clean/'

# for PCA
N_COMPONENTS = [10, 50, 100, 200]

do_PCA = 1
do_overSampling = 2
test_model = 1

apply_CORR = 1
apply_PCA = 0



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

def apply_PCA2(X,y, n):
	# pcas = []
	# for n in n_components:	
		pca = PCA(n_components=n)
		principalComponents = pca.fit(X)
		# pcas.append(principalComponents)
		return pca
	
def handle_imbalanced_Data_try1(df):
	# ========================
	# 	handle im-balanced data 
	# 	========================

		# Separate majority and minority classes
		minval = df['AUC'].min()
		maxval = df['AUC'].max()
		cutVal = (maxval+minval)/2
		if debug == 1:
			print('min,max: ', minval, maxval)

		df_majority = df[df['AUC'] > cutVal]
		df_minority = df[df['AUC'] <= cutVal]
		majority_N = len(df_majority)

		if debug ==1 :
			print('Before Up Sampling: ', len(df_majority), len(df_minority))
			print('majority_N: ', majority_N)

		
		# Upsample minority class
		df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=majority_N,    # to match majority class
                                 random_state=123) # reproducible results
 
		# Combine majority class with upsampled minority class
		df_upsampled = pd.concat([df_majority, df_minority_upsampled])
		
		if debug == 1:
			# Display new class counts
			df_majority = df_upsampled[df_upsampled['AUC'] > cutVal]
			df_minority = df_upsampled[df_upsampled['AUC'] <= cutVal]
			print('After Up Sampling: ', len(df_majority), len(df_minority))

		return df_upsampled


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



def main(data , drug_ids):
	

	
	for drug_id in data:
		print('drug_id: ', drug_id)
		if debug == 1:
			print(data[drug_id].shape)

		
		df = data[drug_id]
		df = df.drop(columns=['Cell_id'])


		# ===============
		# CAtegorize the labels
		# binInterval = np.arange(0,1.1,0.1)
		# df['AUC'] = pd.cut(df['AUC'], bins = binInterval, labels=binInterval[1:])
		# y = df['AUC'].values

		# ==================================
		# Split the data

		y = df['AUC'].to_numpy(dtype ='float32')
		X = df.drop(columns=['AUC']).to_numpy(dtype ='float32')
		X = StandardScaler().fit_transform(X)
		X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X, y, test_size=0.2, random_state=0)
		
		if debug == 1:
			print("Step1: Split: ",X_train_orig.shape, X_test_orig.shape)
		

		# ====================================
		if apply_PCA == 1:
			# ===========
			# perform PCA
			PCAS = {}
			for n in N_COMPONENTS:
				
				if n not in PCAS:
					PCAS[n] = {}

				if debug == 1:
					print('PCA_',n)

				if do_PCA == 1:
					
					pca_v = PCA(n_components=n)
					principalComponents = pca_v.fit(X_train_orig)
					
				else:
					
					with open(CLEAN_DATA_DIR+'data_pca_'+str(drug_id)+'_'+str(n)+'.pkl', 'rb') as f:
						principalComponents = pickle.load( f)

					
				X_train = principalComponents.transform(X_train_orig)
				X_test = principalComponents.transform(X_test_orig)
				
				if debug == 1:
						print('new shape: ', X_train.shape)

				# ====================================
				# Over Sampling
				if do_overSampling == 1:
					dfr = pd.DataFrame(X_train)
					dfr['AUC'] = pd.Series(y_train_orig)

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
				
				elif do_overSampling == 2: #try2
					dfr = pd.DataFrame(X_train)
					dfr['AUC'] = pd.Series(y_train_orig)
					upsampled_data = handle_imbalanced_Data_try1(dfr)
					y_train = upsampled_data['AUC'].to_numpy(dtype='float32')
					X_train = upsampled_data.drop(columns=['AUC']).to_numpy(dtype='float32')

				# ==================
				
				# 
				y_test = y_test_orig
				PCAS[n]['X_train'] = X_train
				PCAS[n]['y_train'] = y_train
				PCAS[n]['X_test'] = X_test
				PCAS[n]['y_test'] = y_test

				if test_model == 1:
					svm_trivial(X_train, X_test, y_train, y_test)

			# save PCA
			with open(CLEAN_DATA_DIR+'data_pca_'+str(drug_id)+'.pkl', 'wb') as f:
				pickle.dump(PCAS, f)

		elif apply_CORR == 1:
			s = X_train_orig.shape[1]
			corr = []
			for i in range(s):
				slope, intercept, r_value, p_value, std_err = stats.linregress(X_train_orig[:,i], y_train_orig)
				corr.append(r_value)

			print('drug_id:'+str(drug_id)+', minVal: '+str(min(corr)) +', maxVal: '+str(max(corr)))
			
			corr = np.array(corr)
			l1 = corr[corr > 0.4]
			l2 = corr[corr < -0.4]
			combined = np.concatenate((l1,l2), axis=0)

			inx = np.argwhere( (corr > 0.4) | (corr < -0.4))
			X_train = X_train_orig[:, inx.T.tolist()[0]]
			y_train = y_train_orig

			X_test = X_test_orig[:, inx.T.tolist()[0]]
			y_test = y_test_orig

			print('> 0.4: ', len(l1), ', < -0.4', len(l2), ', total: ', len(inx))

			# save PCA
			with open(CLEAN_DATA_DIR+'data_corr_val_'+str(drug_id)+'.pkl', 'wb') as f:
				pickle.dump(corr, f)

			corrF = {0:{}}
			corrF[0]['X_train'] = X_train
			corrF[0]['y_train'] = y_train
			corrF[0]['X_test'] = X_test
			corrF[0]['y_test'] = y_test

			with open(CLEAN_DATA_DIR+'data_corr_'+str(drug_id)+'.pkl', 'wb') as f:
				pickle.dump(corrF, f)

		# break # To do : should comment this line for delivery 

if __name__ == '__main__':
	
	
	

	# load drug ids
	drug_ids = []
	with open(CLEAN_DATA_DIR+'drug_ids.pkl', 'rb') as f:
		drug_ids = pickle.load(f)

	# load data
	data = None
	with open(CLEAN_DATA_DIR+'data.pkl', 'rb') as f:
		data = pickle.load(f)

	main(data = data, drug_ids=drug_ids)
	

