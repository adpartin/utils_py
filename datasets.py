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

# DATADIR = '/vol/ml/apartin/Benchmarks/Data/Pilot1'
DATADIR = '/Users/apartin/work/jdacs/Benchmarks/Data/Pilot1'
CELLMETA_FILENAME = 'combined_metadata_2018May.txt'

na_values = ['na', '-', '']


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
class CombinedRNASeqLINCS():
    """ Combined LINCS dataset. """
    def __init__(self, datadir=DATADIR, dataset='combat', cellmeta_filename=CELLMETA_FILENAME, sources=[],
                 na_values=['na', '-', ''], verbose=True):
        """ Note that df_rna file must have the following structure:
        df_rna.columns[0] --> 'Sample'
        df_rna.columns[1:] --> gene names
        df_rna.iloc[:, 0] --> strings of sample names
        df_rna.iloc[:, 1:] --> gene expression values
        
        Example:
            DATADIR = '/Users/apartin/work/jdacs/Benchmarks/Data/Pilot1'
            METADATA_FILENAME = 'combined_metadata_2018May.txt'
            lincs = CombinedLINCS(dataset='combat', datadir=DATADIR, cellmeta_filename=CELLMETA_FILENAME)
        """
        if dataset == 'raw':
            DATASET = 'combined_rnaseq_data_lincs1000'
        elif dataset == 'source_scale':
            DATASET = 'combined_rnaseq_data_lincs1000_source_scale'
        elif dataset == 'combat':
            DATASET = 'combined_rnaseq_data_lincs1000_combat'
        else:
            raise ValueError(f'The passed dataset ({DATASET}) is not supprted.')
            
        # Load RNA-Seq
        path = os.path.join(datadir, DATASET)
        cols = pd.read_table(path, nrows=0, sep='\t')
        dtype_dict = {c: np.float32 for c in cols.columns[1:]}
        df_rna = pd.read_table(path, dtype=dtype_dict, sep='\t', na_values=na_values, warn_bad_lines=True)
        # df_rna = pd.read_csv(path, sep='\t')
        df_rna = self._keep_sources(df_rna, sources=sources) 

        # Load metadata
        meta = pd.read_table(os.path.join(datadir, cellmeta_filename), sep='\t')
        meta = self._update_metadata_comb_may2018(meta)
        
        # Merge df_rna and meta
        df_rna, meta = self._update_df_and_meta(df_rna, meta, on='Sample')

        if verbose:
            print(f'\nDataset: {DATASET}')
            print(f'df_rna {df_rna.shape}')
            if meta is not None:
                print(f'meta   {meta.shape}')
            print(df_rna['Sample'].map(lambda s: s.split('.')[0]).value_counts())
            
        self._df_rna, self._meta = df_rna, meta


    def _keep_sources(self, df_rna, sources=[]):
        """ Keep specific data sources.
		Args:
			sources (list) : list of strings indicating the sources/studies to extract.
                (e.g., source=['ccle', 'ctrp'])
		"""
        if len(sources) == 0:
            return df_rna

        if isinstance(sources, str):
            sources = [sources]
            
        if len(sources) > 0:
            sources = [s.lower() for s in sources]
            df_rna = df_rna.loc[df_rna['Sample'].map(lambda s: s.split('.')[0].lower() in sources), :].reset_index(drop=True)
        else:
            print('Empty list was passed to the arg `sources`. Returns the same dataframe.')

        return df_rna  


    def _update_metadata_comb_may2018(self, meta):
        """ Update the metadata of the combined RNA-Seq (Judith metadata):
        /nfs/nciftp/private/tmp/jcohn/metadataForRNASeq2018Apr/combined_metadata_2018May.txt
        Remove "unnecessary" columns.
        Use Argonne naming conventions (e.g. GDC rather than TCGA).
        """
        # Rename columns
        meta = meta.rename(columns={'sample_name': 'Sample',
                                    'dataset': 'source',
                                    'sample_category': 'category',
                                    'sample_descr': 'descr',
                                    'tumor_site_from_data_src': 'csite',
                                    'tumor_type_from_data_src': 'ctype',
                                    'simplified_tumor_site': 'simplified_csite',
                                    'simplified_tumor_type': 'simplified_ctype'
                                    })

        meta['source'] = meta['source'].map(lambda x: x.lower())
        meta['csite'] = meta['csite'].map(lambda x: x.strip())
        meta['ctype'] = meta['ctype'].map(lambda x: x.strip())
        meta['source'] = meta['source'].map(lambda x: 'gdc' if x=='tcga' else x)
        return meta

    
    def _update_df_and_meta(self, df_rna, meta, on='Sample'):
        """ Merge df_rna and meta on a column specified by `on`.
        Args:
            df_rna (df) : df rna
            meta (df) : df meta
        Returns:
            df_rna (df) : df rna updated
            meta (df) : df meta updated
        """
        df_rna = df_rna.copy()
        meta = meta.copy()
        df = pd.merge(meta, df_rna, how='inner', on=on).reset_index(drop=True)

        df_rna = df[['Sample'] + df_rna.columns[1:].tolist()]
        meta = df.drop(columns=df_rna.columns[1:].tolist())
        return df_rna, meta
    

    def df_rna(self):
        """ df_rna getter. """
        df_rna = self._df_rna.copy()
        return df_rna
    
    
    def meta(self):
        """ meta getter. """
        meta = self._meta.copy()
        return meta
    
    
    # def extract_specific_datasets(self, sources=[]):
    def get_subset(self, sources=[]):
        """ Get samples of the specified data sources (this is a getter method).
        Args:
            sources (list) : list of strings indicating the sources/studies to extract
        Returns:
            df_rna (df) : df rna for the data sources specified by `sources`
            meta (df) : df meta for the data sources specified by `sources`
        Example:
            cells_rna, cells_meta = lincs.get_subset(sources=['ccle','nci60'])
        """
        df_rna = self._df_rna.copy()
        meta = self._meta.copy()

        if len(sources) > 0:
            sources = [s.lower() for s in sources]
            df_rna = df_rna.loc[df_rna['Sample'].map(lambda s: s.split('.')[0].lower() in sources), :].reset_index(drop=True)
            df_rna, meta = self._update_df_and_meta(df_rna, meta, on='Sample')
        else:
            print('Empty list was passed to the arg `sources`. Returns the same dataframe.')

        return df_rna, meta
# ======================================================================================================================


def load_lincs1000(dataset='combat', datadir=DATADIR, sources=[], meta=True, verbose=True):
	""" Load lincs1000.
	Args:
		dataset : dataset type specified by batch-effect removal approach
		datadir : path to the data
		sources : list of strings indicating the data sources to keep
		meta (bool) : whether to include the metadata
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
		meta = pd.read_csv(os.path.join(datadir, CELLMETA_FILENAME), sep='\t')
		meta = utils_rnaseq.update_metadata_comb_may2018(meta)
		df_rna, meta = utils_rnaseq.update_df_and_meta(df_rna, meta, on='Sample')
	else:
		meta = None

	if verbose:
		print(f'\nDataset: {DATASET}')
		print(f'df_rna {df_rna.shape}')
		if meta is not None:
			print(f'meta   {meta.shape}')
		print(df_rna['Sample'].map(lambda s: s.split('.')[0]).value_counts())

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



