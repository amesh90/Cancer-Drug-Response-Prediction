import pandas as pd 
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
from scipy import stats
import os, shutil
import pickle

debug = 2
execute_loop = 1

MAX_DRUGS = 10
CLEAN_DATA_DIR='data_clean/'

def test(trainingfilename , labels):
	
	data = pd.read_csv(trainingfilename, sep='\t')
	labels_d = pd.read_csv(labels, sep='\t')


	ids = labels_d['COSMIC_ID'].unique()

	it = 0
	for id in ids:
		print(it, id)
		cols = list(data.columns)
		if 'DATA.'+str(id) in cols:
			x = data['DATA.'+str(id)]
			it += 1


def main(trainingfilename , labels):
	
	data = pd.read_csv(trainingfilename, sep='\t')
	labels_d = pd.read_csv(labels, sep='\t')

	# drop duplicate columns but with different column like
	# DATA.1503362 and DATA.1503362.1
	# DATA.1330983 and DATA.1330983.1
	# DATA.909976 and DATA.909976.1
	# DATA.905954 and DATA.909976.1
	# Drop Gene title columns no need


	data = data.drop(columns=['GENE_title', 'DATA.1503362.1', 'DATA.1330983.1', 'DATA.909976.1', 'DATA.909976.1']) # drop titles 
	data = data.dropna(axis=0) # drop NAN values

	# flip rwos with cloumns (cell lines in rows , gene types in columns)
	data_trans = data.set_index('GENE_SYMBOLS').T.rename_axis('Cell_id').rename_axis(None, axis=1).reset_index()
	
	# remove 'DATA.' from cell_id columns
	data_trans['Cell_id'] = data_trans['Cell_id'].map(lambda x: x.strip().lstrip('DATA.'))
	
	# change cell_id column to numeric data type
	data_trans['Cell_id'] = pd.to_numeric(data_trans['Cell_id'])

	data_trans = data_trans.sort_values('Cell_id')


	cell_ids = labels_d['COSMIC_ID'].unique()
	drug_ids = labels_d['DRUG_ID'].unique()

	dataset_grouped_drug = {}
	dataset = {}
	cnt = 0
	if execute_loop == 1:
		for drug_id in drug_ids:
			if debug == 1:
				print(cnt,': drug_id: ', drug_id)

			# group by drug_id
			d = labels_d[labels_d['DRUG_ID'] == drug_id][['COSMIC_ID','AUC']]
			d = d.sort_values('COSMIC_ID').reset_index(drop=True)
			dataset_grouped_drug[drug_id] = d

			if debug == 1:
				print('d: ', d.shape)
				print(d)

			# sub-select data correspond to the cell_ids belong to drug_id
			filterd = data_trans[data_trans['Cell_id'].isin(list(d['COSMIC_ID']))].reset_index(drop=True)
			
			# append AUC (label) column
			filterd['AUC'] = filterd.apply(lambda x: float(d[d['COSMIC_ID'] == x['Cell_id']]['AUC']), axis=1)

			if debug == 1:
				print('filterd: ', filterd.shape)
				print(filterd)

			if debug == 1:
				f = filterd[['Cell_id', 'AUC']]
				d2 = d[d['COSMIC_ID'].isin(list(filterd['Cell_id']))]
				df = pd.merge(d2,f, how='left',left_on=['COSMIC_ID'],right_on=['Cell_id'])
				print(cnt, ': df: ', df.shape)
				print(df)
				print('\n\n')
			
			if debug == 2:
				if cnt > MAX_DRUGS:
					break

			cnt += 1
			
			dataset[drug_id] = filterd
			# save clean dataset
			# outputfilename = directory+'/'+str(drug_id)+'.csv'
			# data.to_csv(outputfilename, index=False)
	

		with open(CLEAN_DATA_DIR+'data.pkl', 'wb') as f:
			pickle.dump(dataset, f)


	# save drug ids
	with open(CLEAN_DATA_DIR+'drug_ids.pkl', 'wb') as f:
		pickle.dump(drug_ids, f)

	if debug == 1:
		print(len(drug_ids), len(cell_ids))
		print('total drugs: ', len(dataset))
		for drug_id in dataset:
			print(len(dataset[drug_id]))



def empty_directory(folder):
	
	for filename in os.listdir(folder):
	    file_path = os.path.join(folder, filename)
	    try:
	        if os.path.isfile(file_path) or os.path.islink(file_path):
	            os.unlink(file_path)
	        elif os.path.isdir(file_path):
	            shutil.rmtree(file_path)
	    except Exception as e:
	        print('Failed to delete %s. Reason: %s' % (file_path, e))		
		


if __name__ == '__main__':
	
	training_dataset = 'data/Cell_line_RMA_proc_basalExp.txt'
	labels = 'data/v17_fitted_dose_response.csv'

	# handle output directory 
	
	if not os.path.exists(CLEAN_DATA_DIR):
		os.makedirs(CLEAN_DATA_DIR)

	# empty_directory(CLEAN_DATA_DIR)

	main(trainingfilename = training_dataset, labels=labels)
	

	
	
