# 
# 
# Just started this file (didn't finish nor tested)
# 
# 
# ------------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division, print_function

import os
import sys
import time
import numpy as np
import pandas as pd

# from utils_rnaseq import update_metadata_comb_may2018, extract_specific_datasets
# from utils_ml import balance_df
# import utils_ml as utils_ml

# utils_path = os.path.abspath(os.path.join('..', 'utils_py'))
# sys.path.append(utils_path)
import utils_all as utils_all
import utils_rnaseq as utils_rnaseq

DATADIR = '/vol/ml/apartin/Benchmarks/Data/Pilot1'  # '/Users/apartin/work/jdacs/Benchmarks/Data/Pilot1'
PDM_METADATA_FILENAME = 'combined_metadata_2018May.txt'


def extract_target_from_sample(df_rna):
    y = df_rna['Sample'].map(lambda x: x.split('.')[0].lower())
    return y


def update_target_names(y):
    """ `ncipdm` --> `pdm`
        `nci60` & `ccle` --> cell_lines """
    y = y.map(lambda x: 'pdm' if x == 'ncipdm' else x)
    y = y.map(lambda x: 'cell_line' if (x=='ccle') | (x=='nci60') else x)
    return y


# ======================================================================================================================
# TODO : finish creating the class
# class CombinedRNASeq():


# ======================================================================================================================


def load_lincs1000(dataset='combat', datadir=DATADIR, sources=[], meta=True, verbose=True):
	""" Load lincs1000.
	TODO: convert this to class.
	"""
	if dataset == 'raw':
		DATASET = 'combined_rnaseq_data_lincs1000'
	elif dataset == 'source_scale':
		DATASET = 'combined_rnaseq_data_lincs1000_source_scale'
	elif dataset == 'combat':
		DATASET = 'combined_rnaseq_data_lincs1000_combat'
	else:
		raise ValueError(f'The passed dataset ({DATASET}) is not supprted.')

	df_rna = pd.read_csv(os.path.join(datadir, DATASET), sep='\t')

	if sources:
		df_rna = utils_rnaseq.extract_specific_datasets(df_rna, datasets_to_keep=sources)

	if meta:
		meta = pd.read_csv(os.path.join(datadir, PDM_METADATA_FILENAME), sep='\t')
		meta = utils_rnaseq.update_metadata_comb_may2018(meta)
		df_rna, meta = utils_rnaseq.update_df_and_meta(df_rna, meta, on='Sample')
	else:
		meta = None

	if verbose:
		print(f'\nDataset: {DATASET}')
		print(f'df_rna {df_rna.shape}')
		if meta is not None:
			print(f'meta   {meta.shape}')

	return df_rna, meta


def create_dataset_balanced_by_souce(dataset='combat'):
	""" (rick asked) """
	df_rna = extract_specific_datasets(df_rna, datasets_to_keep=['gdc', 'ncipdm', 'ccle', 'nci60'])
	y = extract_target_from_sample(df_rna)
	
	# y = update_target_names(y)
	y = y.map(lambda x: 'pdm' if x == 'ncipdm' else x)  # `ncipdm` --> `pdm`
	y = y.map(lambda x: 'cell_line' if (x=='ccle') | (x=='nci60') else x)  # `nci60` & `ccle` --> `cell_lines`

	df_rna, y, _ = utils.balance_df(df_rna, y, seed=SEED)
	print(y.value_counts())

	df = pd.concat([y, df_rna.iloc[:, 1:]], axis=1)
	df.rename(columns={'Sample': 'y'}, inplace=True)
	df = df.sample(frac=1.0).reset_index(drop=True)

	xdata = df.iloc[:, 1:]
	ydata = df['y'].map(lambda x: x.split('.')[0].lower()).to_frame()

	return df_rna, meta



