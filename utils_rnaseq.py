from __future__ import absolute_import
from __future__ import division, print_function

import os
import time
import random
import numpy as np
import pandas as pd

import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
matplotlib.rcParams['font.size'] = 14

import seaborn as sns

from pandas.api.types import is_string_dtype, is_numeric_dtype
from itertools import cycle, islice

from collections import OrderedDict

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ComBat imports
import patsy  # for combat
from combat import *

from utils_generic import read_file_by_chunks


P1B3_URL = 'http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/P1B3/'
DATA_URL = 'http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/'


# PDM_METADATA_FILENAME = 'pdm_file_011817_metadata.txt'
# PDM_METADATA_FILENAME = 'comb_sample_metadata.txt'

# DATASET = 'combined_rnaseq_data_lincs1000'
# DATASET = 'combined_rnaseq_data'
# DATASET = 'combined_rnaseq_data_filtered'
# DATASET = 'combined_rnaseq_data_filtered_norm'
# DATASET = ncipdm_rnaseq_tpm_gene


# ======================================================================================================================
class kNNrnaseq():
    """ Compute the closest neighbors across sources/studies.
    Args:
        df_train : training set
        df_test : for each sample in test set (df_query) a total of n_neighbors closest nbrs
                  from df_train (df_nbrs) are found
        meta_train : 
        meta_test : 
        label (str) : `label` is a col name that both meta_train and meta_test have (e.g. phenotype: 'ctype').
                      The returned table knn_labels shows the label for each query sample
                      and the corresling neighbors.
        n_neighbors (int) : number of neighbors to find
        ref_col_name (str) : ref col name based on which query samples (default: 'Sample')
        algorithm (str) : 
        metric (str) : distance metric

    Returns:
        dd_label (df) : df where each row corresponds to a sample in the test set (df_query);
                        each row contains the value of eval_col_name of each nbr
    """
    # TODO : write test code with some of the sklearn datasets (!!)
    def __init__(self, df_train, meta_train,
                 df_test=None, meta_test=None,
                 ref_col_name='Sample', label=None, 
                 n_neighbors=5, algorithm='auto',
                 metric='minkowski', p=2,
                 metric_params=None):
        # http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
        from sklearn.neighbors import NearestNeighbors

        self.df_train = df_train.copy().reset_index(drop=True)
        self.meta_train = meta_train.copy().reset_index(drop=True)

        if df_test is not None:
            self.df_test = df_test.copy().reset_index(drop=True)
        else:
            self.df_test = self.df_train.copy().reset_index(drop=True)

        if meta_test is not None:
            self.meta_test = meta_test.copy().reset_index(drop=True)
        else:
            self.meta_test = self.meta_train.copy().reset_index(drop=True)
            
        if ref_col_name is not None:
            self.ref_col_name = ref_col_name
        
        if label is not None:
            self.label = label

        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.metric = metric
        self.p = p
        self.metric_params = metric_params

        self.pred_dist = []     # distances from the predicted nbrs
        self.pred_indices = []  # indices of the predicted nbrs

        # kNN model
        self.knn_model = NearestNeighbors(n_neighbors=self.n_neighbors,
                                          algorithm=self.algorithm,
                                          metric=self.metric,
                                          p=self.p,
                                          metric_params=self.metric_params,
                                          n_jobs=-1)


    def fit(self):
        # Training data (e.g. cell lines)
        tr_values = self.df_train.iloc[:, 1:].values

        # Train the model (knn just stores the samples)
        self.knn_model.fit(tr_values)


    def neighbors(self):
        """ Generate 3 tables that summarize the results.
        knn_samples :
            df that stores the sample names of the closest nbrs
            columns = [ref_col_name, nbr1, ..., nbrk]
        knn_distances :
            df that stores the distances of the closest nbrs from the sample
            columns = [ref_col_name, nbr1, ..., nbrk]
        knn_labels :
            df that stores the label (e.g. tissue type) of the closest nbrs
            columns = [ref_col_name, 'label', 'match_total', nbr1, ..., nbrk]
        """
        # Compute k nearest nbrs for each sample in the test (query) dataset (e.g. PDM)
        te_values = self.df_test.iloc[:, 1:].values
        self.pred_dist, self.pred_indices = self.knn_model.kneighbors(te_values)

        n_test_samples = self.df_test.shape[0]

        # --- df with Sample names ---
        knn_samples = pd.DataFrame(index=range(n_test_samples),
                                     columns=[self.ref_col_name] + [f'nbr{n}' for n in range(self.n_neighbors)])

        # --- df with distances ---
        knn_distances = knn_samples.copy()
        # knn_distances = pd.DataFrame(index=range(n_test_samples),
        #                              columns=[self.ref_col_name] + [f'nbr{n}' for n in range(self.n_neighbors)])

        # --- df with label names ---
        if self.label:
            # columns=[self.ref_col_name, 'label', 'match_total', 'match_total_wgt', 'match_first'] +
            #                                     ['nbr {}'.format(n) for n in range(self.n_neighbors)]
            knn_labels = pd.DataFrame(index=range(n_test_samples),
                                      columns=[self.ref_col_name, 'label', 'total_matches'] +
                                              [f'nbr{n}' for n in range(self.n_neighbors)])
            # knn_labels = knn_samples.copy()
            # knn_labels.insert(loc=1, column='label', value=None)
            # knn_labels.insert(loc=2, column='match_total', value=None)

        for i in range(n_test_samples):  # iter over test/query samples
            query_sample_name = self.meta_test.loc[i, self.ref_col_name]  # get test (query) sample name
            k_nbrs = self.meta_train.iloc[self.pred_indices[i], :]  # get the nbrs (train) for the current test (query) sample

            # Assign to samples table
            knn_samples.loc[i, self.ref_col_name] = query_sample_name  # assign test (query) sample name
            ##knn_samples.iloc[i, -self.n_neighbors:] = k_nbrs.loc[:, self.ref_col_name].values  # assign the neighbors sample name
            knn_samples.iloc[i, -self.n_neighbors:] = k_nbrs[self.ref_col_name].values  # assign the neighbors sample name

            # Assign to distances table
            knn_distances.loc[i, self.ref_col_name] = query_sample_name  # assign test (query) sample name
            knn_distances.iloc[i, -self.n_neighbors:] = self.pred_dist[i, :]  # distances across all k nbrs
            # knn_distances.iloc[i, -self.n_neighbors:] = self.pred_dist[i, :] / self.pred_dist[i, :].sum()  # normalize distances across all k nbrs

            # Assign to labels table
            if self.label:
                knn_labels.loc[i, self.ref_col_name] = query_sample_name  # assign test (query) sample name
                query_label = self.meta_test.loc[i, self.label]
                
                knn_labels.loc[i, 'label'] = query_label  # label of the test (query) sample
                ##knn_labels.iloc[i, -self.n_neighbors:] = k_nbrs.loc[:, self.label].values
                knn_labels.iloc[i, -self.n_neighbors:] = k_nbrs[self.label].values
                n_matches = k_nbrs[self.label].tolist().count(query_label)
                knn_labels.loc[i, 'total_matches'] = n_matches
                # knn_labels.loc[i, 'match_first'] = 1 if k_nbrs.loc[:, self.label].values[0] == query_label else 0
                # knn_labels.loc[i, 'bingo'] = 1 if num_matches == self.n_neighbors else 0

                # TODO: come up with a new weighting method
                val = 0
                D = self.pred_dist[i, :].sum()  # sum across all distances of the k nbrs
                for ii, lbl in enumerate(k_nbrs[self.label].values):
                    if lbl == query_label:
                        val += self.pred_dist[i, ii] / D
                # knn_labels.loc[i, 'match_total_wgt'] = val

        self.knn_samples = knn_samples
        self.knn_distances = knn_distances
        if self.label:
            self.knn_labels = knn_labels


    def print_results(self):
        if hasattr(self, 'knn_labels'):
            print(self.knn_labels.sort_values(['label']))


    def summary(self):
        if hasattr(self, 'knn_labels'):
            n_matches = self.knn_labels['total_matches'].sum()
            possible_matches = self.n_neighbors * self.df_test.shape[0]
            print(f'Total {n_matches} matches out of {possible_matches} ({n_matches/possible_matches:.3f}).')


    def save_to_file(self, filename='knn.csv'):
        # Merge tables horizontally and save
        tmp1 = self.knn_labels.sort_values(['label'])
        tmp2 = self.knn_samples.reindex(tmp1.index)
        tmp1[' '] = None  # add one blank column for readability of csv
        tmp = pd.concat([tmp1, tmp2], axis=1)

        foldername = 'save'
        verify_folder(foldername)
        if len(filename.split('.')) < 2:
            filename += '.csv'
        filepath = os.path.join('.', foldername, filename)
        tmp.to_csv(filepath)
# ======================================================================================================================


# def read_file_by_chunks(path, chunksize=1000, sep='\t', non_numeric_cols=None):
#     """ Reads a large file in chunks (e.g., RNA-Seq).
#     Args:
#         path : path to the data file
#         non_numeric_cols : a list of column names which contain non numeric values
#     """
#     t0 = time.time()
#     columns = pd.read_table(path, header=None, nrows=1, sep=sep, engine='c').iloc[0, :].tolist()  # read col names only
#     types = OrderedDict((c, str if c in non_numeric_cols else np.float32) for c in columns)  # organize types in dict
#     chunks = pd.read_table(path, chunksize=chunksize, dtype=types, sep=sep, engine='c')

#     print('Loading dataframe by chunks...')
#     chunk_list = []
#     for i, chunk in enumerate(chunks):
#         # print('Chunk {}'.format(i+1))
#         chunk_list.append(chunk)

#     # print('Loading time:  {:.2f} mins'.format((time.time() - t0)/60))

#     df = pd.concat(chunk_list, ignore_index=True)
#     return df


def load_combined_rnaseq(dataset, chunksize=1000, verbose=False):
    """ Load combined RNA-Seq dataframe.
    TODO: remove; not using
    """
    # non_numeric_cols = ['Sample']
    # df = read_file_by_chunks(dataset, chunksize=chunksize, sep='\t', non_numeric_cols=non_numeric_cols)
    df = pd.read_csv(dataset, sep='\t')

    print('{}: {}'.format(dataset, df.shape))
    if verbose:
        tmp_source = pd.Series(df['Sample'].map(lambda x: x.split('.')[0].lower()))
        print(tmp_source.value_counts())
    return df


def update_metadata_comb(meta):
    """ Update metadata for the combined RNA-Seq.
    Remove "irrelevant" columns.
    Use ANL naming conventions (e.g. GDC rather than TCGA).
    TODO: move to the class dataset (but not using)
    """
    # Drop columns (Judith's clustering and LANL names)
    meta = meta.drop(columns=['lanl_sample_name',
                              'k200_clust_num', 'k200_clust_tissue', 'k200_clust_pct_normal', 'k200_clust_descr',
                              'k50_clust_grp_num', 'k50_clust_grp_tissue', 'k50_clust_grp_descr',
                              'k25_clust_grp_num', 'k25_clust_grp_tissue', 'k25_clust_grp_descr'])

    meta = meta.drop(columns=['stage'])  # relevant to tcga only

    # Rename columns
    meta = meta.rename(columns={'anl_2018jan_sample_name': 'Sample',
                                'dataset': 'source',
                                'sample_field1': 'field1',
                                'sample_field2': 'field2',
                                'sample_field3': 'field3',
                                'sample_category': 'category',
                                'sample_descr': 'descr',
                                })

    # Rename 'tcga' to 'gdc'
    meta['source'] = meta['source'].map(lambda x: 'gdc' if x == 'tcga' else x)
    return meta


def update_metadata_comb_may2018(meta):
    """ Update metadata for the combined RNA-Seq (Judith metadata):
    /nfs/nciftp/private/tmp/jcohn/metadataForRNASeq2018Apr/combined_metadata_2018May.txt
    Remove "irrelevant" columns.
    Use ANL naming conventions (e.g. GDC rather than TCGA).
    TODO: move to the class dataset
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
    
    # Rename 'tcga' to 'gdc'
    meta['source'] = meta['source'].map(lambda x: 'gdc' if x == 'tcga' else x)
    return meta


def update_df_and_meta(df_rna, meta, on='Sample'):
    """ Update df_rna and metadata to match.
    TODO: move to the class dataset
    """
    df_rna = df_rna.copy()
    meta = meta.copy()
    df = pd.merge(meta, df_rna, how='inner', on=on).reset_index(drop=True)

    df_rna = df[['Sample'] + df_rna.columns[1:].tolist()]
    meta = df.drop(columns=df_rna.columns[1:].tolist())
    return df_rna, meta


def extract_specific_datasets(df, datasets_to_keep=[]):
    """ Extract samples of the specified data sources.
    Args:
        datasets_to_keep : list of strings indicating the datasets/sources/studies to keep
    TODO: move to the class dataset
    """
    df = df.copy()
    # datasets_to_keep = ['gdc', 'ccle', 'ncipdm', 'nci60']

    if len(datasets_to_keep) > 0:
        datasets_to_keep = [d.lower() for d in datasets_to_keep]
        df = df.loc[df['Sample'].map(lambda d: d.split('.')[0].lower() in datasets_to_keep), :].reset_index(drop=True)
    else:
        print('Empty list was passed to the arg `datasets_to_keep`. Returns the same dataframe.')
        
    return df


def scale_rnaseq(df, per_source=False):
    """ Scale df values and return updated df. """
    df = df.copy()

    if per_source:
        sources = df['Sample'].map(lambda x: x.split('.')[0].lower()).unique().tolist()
        for i, source in enumerate(sources):
            print('Scaling {}'.format(source))
            source_vec = df['Sample'].map(lambda x: x.split('.')[0].lower())
            source_idx_bool = source_vec.str.startswith(source)

            data_values = df.loc[source_idx_bool, df.columns[1:].values].values
            values_scaled = StandardScaler().fit_transform(data_values)
            df.loc[source_idx_bool, 1:] = values_scaled
    else:
        if is_numeric_dtype(df.iloc[:, 1:]):
            df.iloc[:, 1:] = StandardScaler().fit_transform(df.iloc[:, 1:])
        # data_values = df.iloc[:, 1:].values
        # values_scaled = StandardScaler().fit_transform(data_values)
        # df.iloc[:, 1:] = values_scaled

    return df


def copy_rna_profiles_to_cell_lines(df_rna, cl_mapping):
    """
    Use mapping to copy ccle and nci60 gene expression to gdsc, gcsi, and ctrp.
    Args:
        df_rna : df which contains only datasets that have their original rna expression data
        cl_mapping : contains cell line mappings that are used to generate new samples for which
                     we have dose response but don't have rna expression
    Returns:
        df : contains the merged df which contains the original and the mapped samples
    """
    df = df_rna.copy()
    
    # Check that only the allowed datasets are passed
    datasets_to_keep = ['ccle', 'gdc', 'nci60', 'ncipdm']
    bl = df['Sample'].map(lambda x: True if x.split('.')[0].lower() in datasets_to_keep else False)
    assert all(bl), 'only these datasets are allowed: {}'.format(datasets_to_keep)
    
    # Merge in order to copy rnaseq profiles
    cells = cl_mapping.merge(df, left_on='from_cell', right_on='Sample', how='inner')
    
    # Drop and rename columns
    cells = cells.drop(columns=['from_cell', 'Sample']).rename(columns={'to_cell': 'Sample'})

    # Concat 'df' (df that contains original rna profiles) and 'cells' replicated profiles
    df = pd.concat([df, cells], axis=0).sort_values('Sample').reset_index(drop=True)
    
    return df


def get_union(df, meta, base_col, pivot_col, to_plot=True):
    """ Returns a dataframe in which a union is taken on pivot_col with respect to base_col.
    This can be done with groupby (look at fastai).
    Args:
        df :
        meta :
        base_col :
        pivot_col :
    """
    assert set([base_col, pivot_col]).issubset(set(meta.columns)), '`base` and `pivot` are not in the metadata file.'
    # cols_pivot = ['source', 'tissue']
    df = df.copy()

    tissue_table = gen_contingency_table(df=meta, cols=[base_col, pivot_col], to_plot=False)
    tissue_union = [c for c in tissue_table.columns if (tissue_table[c]!=0).all()]

    # tissue_table = tissue_table.loc[:, tissue_union]
    meta = meta.loc[meta['tissue'].map(lambda x: x in tissue_union), :].reset_index(drop=True)
    df, meta = update_df_and_meta(df, meta, on='Sample')
    tissue_table = gen_contingency_table(df=meta, cols=[base_col, pivot_col], to_plot=to_plot)

    return tissue_table, df, meta


# TODO: Are these still used ???
def ap_combat(df_rna, meta):
    """ ... """
    dat, pheno, _, _ = py_df_to_R_df(data = df_rna,
                                     meta = meta,
                                     filename=None, to_save=False, to_scale=False, var_thres=None)
    # dat.columns.name = None
    # pheno.index.name = pheno.columns.name
    # pheno.columns.name = None

    mod = patsy.dmatrix("~1", data=pheno, return_type="dataframe")
    ebat = combat(data = dat,
                  batch = pheno['source'],  # pheno['batch']
                  model = mod)

    df_rna_be = R_df_to_py_df(ebat)
    return df_rna_be


def R_df_to_py_df(data):
    """ This is applied to the output of combat.py """
    return data.T.reset_index().rename(columns={'index': 'Sample'})


def py_df_to_R_df(data, meta, filename=None, to_save=False, to_scale=False, var_thres=None):
    """ Convert python dataframe to R dataframe (transpose). """
    data, meta = update_df_and_meta(data.copy(), meta.copy(), on='Sample')

    # Remove low var columns (this is required for SVA)
    if var_thres is not None:
        data, low_var_col_names = rmv_low_var_genes(data, var_thres=var_thres, per_source=True, verbose=True)

    # Scale dataset
    # todo: Scaling provides worse results in terms of kNN(??)
    # if to_scale:
    #     dat = scale_rnaseq(dat, per_source=False)

    # Transpose df for processing in R
    data_r = data.set_index('Sample', drop=True)
    data_r = data_r.T

    # This is required for R
    meta_r = meta.set_index('Sample', drop=True)
    del meta_r.index.name
    meta_r.columns.name = 'Sample'

    if to_save:
        print('Data shape to save:', data_r.shape)
        file_format = '.txt'
        foldername = 'save'
        verify_folder(foldername)

        if filename is not None:
            data_r.to_csv(os.path.join(foldername, filename + '_dat_R' + file_format), sep='\t')
            meta_r.to_csv(os.path.join(foldername, filename + '_pheno_R' + file_format), sep='\t')
        else:
            data_r.to_csv(os.path.join(foldername, 'dat_R' + file_format), sep='\t')
            meta_r.to_csv(os.path.join(foldername, 'pheno_R' + file_format), sep='\t')

    return data_r, meta_r, data, meta


def rmv_low_var_genes(df, var_thres=0, per_source=False, verbose=True):
    """
    Return:
        df_out : updated df
        gene_column_names : the updated list of gene names
        low_var_col_names : genes that have been removed due to low variance
    """
    df = df.copy()
    ddof = 0  # for var calculation
    low_var_col_names = []

    if per_source:
        sources = df['Sample'].map(lambda x: x.split('.')[0].lower()).unique().tolist()
        for i, source in enumerate(sources):
            source_vec = df['Sample'].map(lambda x: x.split('.')[0].lower())
            source_idx_bool = source_vec.str.startswith(source)
            data = df.loc[source_idx_bool, df.columns[1:].values]
            cols = data.columns[data.var(axis=0, ddof=ddof) <= var_thres].tolist()
            low_var_col_names.extend(cols)
            if verbose:
                print('Columns removed from {:10} {}'.format(source, len(cols)))
    else:
        data = df.loc[:, df.columns[1:].values]
        cols = data.columns[data.var(axis=0, ddof=ddof) <= var_thres].tolist()
        low_var_col_names.extend(cols)
        if verbose:
            print('Columns removed: {:5d}'.format(len(cols)))

    df_out = df.drop(columns=low_var_col_names)
    return df_out, low_var_col_names
# Are these still used ???



# def balance_df(df, y, label_size=None, seed=None):
#     """ Balace input dataframe based on vector y.
#     Args:
#         df : input dataframe
#         y : vector based on which to balance the dataframe
#         label_size : drop labels/classes that have less than label_size samples in df and sample
#             label_size samples from each label/class.  If label_size is not passed then use the
#             size of the least represented label.
#     """
#     assert len(df) == len(y), "df_in and y must contain same number of samples."
#     df = df.copy()
#     df_out = pd.DataFrame(columns=df.columns)
    
#     if label_size is None:
#         label_size = y.value_counts().min()
    
#     # List of dataframes (df per class)
#     df_list = []
    
#     # Returns a list of dropped labels
#     dropped_labels = {}
    
#     for label in y.unique():
#         idx = y == label
#         # print(label, np.sum(idx))
#         if np.sum(idx) >= label_size:
#             tmp_df = df.loc[idx, :].sample(n=label_size, replace=False, axis=0, random_state=seed)
#             df_out = pd.concat([df_out, tmp_df], axis=0).reset_index(drop=True)
#         else:
#             dropped_labels[label] = np.sum(idx)
            
#     # Shuffle
#     df_out = df_out.sample(frac=1.0).reset_index(drop=True)
    
#     return df_out, dropped_labels


# def drop_samples_on_label_count(df, y, min_label_size=100):
#     """ Keep classes which contain at least `min_label_size` samples (apply on dataframe)
#     Returns:
#         df : df with labels that have at least `min_label_size` samples
#         df_small : df with labels that have less than `min_label_size` samples
#     """
#     assert len(df) == len(y), "df_in and y must contain same number of samples."
#     df = df.copy()
#     bincount = y.value_counts(sort=True)
#     df_list = []
#     small_list = []
    
#     for idx in bincount.index:
#         # print('class label:', idx, 'count:', bincount[idx])
#         if bincount[idx] >= min_label_size:
#             tmp = df[y == idx]
#             df_list.append(tmp)
#         else:
#             tmp = df[y == idx]
#             small_list.append(tmp)

#     assert len(df_list), '`min_label_size` is too high (all samples were thrown away).'
#     df_out = pd.concat(df_list, axis=0)

#     if len(small_list):
#         df_out_small = pd.concat(small_list, axis=0)
#     else:
#         df_out_small = None
    
#     return df_out, df_out_small


# def plot_pca(df, components=[1, 2], figsize=(8, 5),
#              color_vector=None, marker_vector=None,
#              to_scale=False, title=None, verbose=True):
#     """
#     Apply PCA to input dataframe.
#     Args:
#         color_vector : each element corresponds to a row in df. The unique elements will be colored
#             with a different color.
#         marker_vector : each element corresponds to a row in df. The unique elements will be marked
#             with a different marker.
#     Returns:

#     https://stackoverflow.com/questions/12236566/setting-different-color-for-each-series-in-scatter-plot-on-matplotlib
#     """
#     if color_vector is not None:
#         assert len(df) == len(color_vector), 'len(df) and len(color_vector) shuold be the same size.'
#         n_colors = len(color_vector.unique())
#         colors = iter(cm.rainbow(np.linspace(0, 1, n_colors)))

#     if marker_vector is not None:
#         assert len(df) == len(marker_vector), 'len(df) and len(marker_vector) shuold be the same size.'
#         all_markers = ('o', 'v', 's', 'p', '^', '<', '>', '8', '*', 'h', 'H', 'D', 'd', 'P', 'X')
#         markers = all_markers[:len(marker_vector.unique())]

#     df = df.copy()

#     # PCA
#     if to_scale:
#         arr = StandardScaler().fit_transform(df.values)
#     else:
#         arr = df.values

#     n_components = max(components)
#     pca_obj = PCA(n_components=n_components)
#     pca = pca_obj.fit_transform(arr)
#     pc0 = components[0] - 1
#     pc1 = components[1] - 1

#     # Start plotting
#     fig, ax = plt.subplots(figsize=figsize)

#     if (color_vector is not None) and (marker_vector is not None):
#         for i, marker in enumerate(marker_vector.unique()):
#             colors = iter(cm.rainbow(np.linspace(0, 1, n_colors)))

#             for color in color_vector.unique():
#                 # print(i, 'marker:', marker, 'color:', color)
#                 idx = (marker_vector == marker) & (color_vector == color)
#                 ax.scatter(pca[idx, pc0], pca[idx, pc1], alpha=0.7,
#                             marker=markers[i],
#                             edgecolors='black',
#                             color=next(colors),
#                             label='{}, {}'.format(marker, color))

#     elif (color_vector is not None):
#         for color in color_vector.unique():
#             idx = (color_vector == color)
#             ax.scatter(pca[idx, pc0], pca[idx, pc1], alpha=0.7,
#                         marker='o',
#                         edgecolors='black',
#                         color=next(colors),
#                         label='{}'.format(color))

#     elif (marker_vector is not None):
#         for i, marker in enumerate(marker_vector.unique()):
#             idx = (marker_vector == marker)
#             ax.scatter(pca[idx, pc0], pca[idx, pc1], alpha=0.7,
#                         marker=markers[i],
#                         edgecolors='black',
#                         color='blue',
#                         label='{}'.format(marker))

#     else:
#         ax.scatter(pca[:, pc0], pca[:, pc1], alpha=0.7,
#                    marker='s', edgecolors='black', color='blue')


#     if title: ax.set_title(title)
#     ax.set_xlabel('PC'+str(components[0]))
#     ax.set_ylabel('PC'+str(components[1]))
#     ax.legend(loc='lower left', bbox_to_anchor= (1.01, 0.0), ncol=1,
#               borderaxespad=0, frameon=True)

#     if verbose:
#         print('Explained variance by PCA components [{}, {}]:  [{:.5f}, {:.5f}]'.format(
#             components[0], components[1],
#             pca_obj.explained_variance_ratio_[pc0],
#             pca_obj.explained_variance_ratio_[pc1]))

#     return pca_obj, pca, fig


# Variance filtering ===================================================================================================
# def variance_filtering(df):
#     """ (ap) Variance filtering for co-clustering (Judith & Ben).
#     They got 9994 genes. I got 9955. --> why?? """
#     df = df.copy()
#
#     # For clustering, duplicated data labels such as GDSC have been removed
#     sources_to_keep = ['CCLE', 'GDC', 'NCI60', 'NCIPDM']
#     sources_to_keep_bool = [True if sc in sources_to_keep else False for sc in df['source']]
#     df = df.loc[sources_to_keep_bool, :]
#
#     gene_column_names = df.columns[2:]
#     df1 = df.drop(gene_column_names, axis=1)  # df1 contains all the cols except the genes cols
#
#     # For each dataset for each gene, center the TPM expression values by subtracting the mean value for that gene
#     # across all samples
#     print('\nSubtract a mean from individual data sources...')
#     sources_to_norm = ['GDC', 'CCLE', ['NCI60', 'NCIPDM']]
#     for i, sc in enumerate(sources_to_norm):  # df['source'].unique()
#         print('Subtract mean from {}'.format(sc))
#         source_idx_bool = [True if s in sc else False for s in df['source']]
#         print(df.loc[source_idx_bool, 'source'].value_counts())
#         arr = df.loc[source_idx_bool, gene_column_names].values  #
#         tmp = arr - arr.mean(axis=0)
#         df.loc[source_idx_bool, gene_column_names] = tmp  # plt.plot(arr.mean(axis=0))
#
#     # For each gene...
#     # Calculate the variance for the gene across all TCGA/GDC samples
#     # Calculate the variance for the gene across all other samples
#     # Calculate ratio of the variance for TCGA/GDC vs other data (Vtcga/Vother) for the gene
#     df_var = pd.DataFrame(index={'v_gdc', 'v_other', 'v_ratio'}, columns=gene_column_names)
#     gdc = df.loc[df['source'] == 'GDC', gene_column_names].values
#     other = df.loc[df['source'] != 'GDC', gene_column_names].values
#     df_var.loc['v_gdc', :] = gdc.var(axis=0)
#     df_var.loc['v_other', :] = other.var(axis=0)
#     df_var.loc['v_ratio', :] = df_var.loc['v_gdc', :].values / df_var.loc['v_other', :].values
#
#     # Keep genes where the ratio is between 0.5 and 2
#     genes_to_keep = (df_var.loc['v_ratio'].values > 0.5) & (df_var.loc['v_ratio'].values < 2.0)
#     df2 = df.loc[:, df_var.columns[genes_to_keep].tolist()]
#
#     assert df1.shape[0] == df2.shape[0], 'The number of samples (rows) in the dataframes should match.'
#     df_filt = pd.concat([df1, df2], axis=1)
#
#     return df_filt
# Variance filtering ===================================================================================================


# Plot functions =======================================================================================================
# def plot_gene_values(df, gene_id=0):
#     """ (ap) Plot gene values across samples. """
#     gene_name = df.columns[2 + gene_id]  # genes start from index 2
#     markers = ['.', '+', 'x', 'o', 'h']
#     colors = ['b', 'k', 'g', 'r', 'c']
#     legend = []
#     plt.figure()
#     for i, sc in enumerate(df['source'].unique()):
#         legend.append(sc)
#         dff = df[df['source'] == sc]
#         plt.scatter(dff.index, dff.loc[:, gene_name], marker=markers[i], color=colors[i], alpha=0.5)
#     plt.xlabel('Sample index')
#     plt.ylabel('Gene expression values (normalized)')
#     plt.legend(legend)
#     plt.title('Gene name:  {}'.format(gene_name))


# fontsize = 20
# fontsize_leg = 14
# alpha = 0.5
# figsize = (20, 6)
# colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
#                                      '#f781bf', '#a65628', '#984ea3',
#                                      '#999999', '#e41a1c', '#dede00']),
#                               int(10 + 1))))  # len(y_pred) instead of 10


# def plot_means_per_sample(datasets, column_names=None, labels=None, title=None, savefig=False):
#     """ Plot of mean vs sample.
#     Each data point in the plot is the mean across all gene values for a single sample.
#     Args:
#         datasets : list of pandas dataframes or numpy arrays
#     """
#     axis_to_compute = 1  # Dimension across which to compute the statistic !!

#     plt.figure(figsize=figsize)

#     x2 = 0
#     for i, data in enumerate(datasets):
#         if len(column_names):
#             data = data.loc[:, column_names]
#         if len(labels):
#             label = labels[i]

#         x1 = x2
#         x2 = x1 + data.shape[0]  # !!
#         x = np.arange(x1, x2)
#         mu = data.mean(axis=axis_to_compute)  # !!
#         plt.scatter(x, mu, c=colors[i], alpha=alpha, marker='o', label=label)

#     plt.title(title, fontsize=fontsize)
#     plt.xlabel('Sample index', fontsize=fontsize)  # !!
#     plt.ylabel('Mean', fontsize=fontsize)  # !!
#     plt.legend(loc='best', fontsize=fontsize_leg)
#     plt.grid('on')

#     if savefig:
#         if title:
#             plt.savefig(title + '.png')
#         else:
#             plt.savefig('sample_means.png')  # !!


# def plot_vars_per_sample(datasets, column_names=None, labels=None, title=None, savefig=False):
#     """ Plot of variance vs sample.
#     Each data point in the plot is the variance across all gene values for a single sample.
#     Args:
#         datasets : list of pandas dataframes or numpy arrays
#     """
#     axis_to_compute = 1  # Dimension across which to compute the statistic !!

#     plt.figure(figsize=figsize)

#     x2 = 0
#     for i, data in enumerate(datasets):
#         if len(column_names):
#             data = data.loc[:, column_names]
#         if len(labels):
#             label = labels[i]

#         x1 = x2
#         x2 = x1 + data.shape[0]  # !!
#         x = np.arange(x1, x2)
#         var = data.var(axis=axis_to_compute, ddof=0)  # !!
#         plt.scatter(x, var, c=colors[i], alpha=alpha, marker='o', label=label)

#     plt.title(title, fontsize=fontsize)
#     plt.xlabel('Sample index', fontsize=fontsize)  # !!
#     plt.ylabel('Variance', fontsize=fontsize)  # !!
#     plt.legend(loc='best', fontsize=fontsize_leg)
#     plt.grid('on')

#     if savefig:
#         if title:
#             plt.savefig(title + '.png')
#         else:
#             plt.savefig('sample_variances.png')  # !!


# def plot_means_per_gene(datasets, column_names=None, labels=None, title=None, savefig=False):
#     """ Plot of mean vs gene.
#     Each data point in the plot is the mean across all gene values for a single gene.
#     Args:
#         datasets : list of pandas dataframes or numpy arrays
#     """
#     axis_to_compute = 0  # Dimension across which to compute the statistic !!

#     plt.figure(figsize=figsize)

#     # x2 = 0
#     for i, data in enumerate(datasets):
#         if len(column_names):
#             data = data.loc[:, column_names]
#         if len(labels):
#             label = labels[i]

#         # x1 = x2
#         # x2 = x1 + data.shape[1]  # !!
#         # x = np.arange(x1, x2)
#         x = np.arange(0, data.shape[1])
#         mu = data.mean(axis=axis_to_compute)  # !!
#         plt.scatter(x, mu, c=colors[i], alpha=alpha, marker='o', label=label)

#     plt.title(title, fontsize=fontsize)
#     plt.xlabel('Gene index', fontsize=fontsize)  # !!
#     plt.ylabel('Mean', fontsize=fontsize)  # !!
#     plt.legend(loc='best', fontsize=fontsize_leg)
#     plt.grid('on')

#     if savefig:
#         if title:
#             plt.savefig(title + '.png')
#         else:
#             plt.savefig('gene_means.png')  # !!


# def plot_vars_per_gene(datasets, column_names=None, labels=None, title=None, savefig=False):
#     """ Plot of variance vs gene.
#     Each data point in the plot is the variance across all sample values for a single gene.
#     Args:
#         datasets : list of pandas dataframes or numpy arrays
#     """
#     axis_to_compute = 0  # Dimension across which to compute the statistic !!

#     plt.figure(figsize=figsize)

#     # x2 = 0
#     for i, data in enumerate(datasets):
#         if len(column_names):
#             data = data.loc[:, column_names]
#         if len(labels):
#             label = labels[i]

#         # x1 = x2
#         # x2 = x1 + data.shape[1]  # !!
#         # x = np.arange(x1, x2)
#         x = np.arange(0, data.shape[1])
#         var = data.var(axis=axis_to_compute, ddof=0)  # !!
#         plt.scatter(x, var, c=colors[i], alpha=alpha, marker='o', label=label)

#     plt.title(title, fontsize=fontsize)
#     plt.xlabel('Gene index', fontsize=fontsize)  # !!
#     plt.ylabel('Variance', fontsize=fontsize)  # !!
#     plt.legend(loc='best', fontsize=fontsize_leg)
#     plt.grid('on')

#     if savefig:
#         if title:
#             plt.savefig(title + '.png')
#         else:
#             plt.savefig('gene_variances.png')  # !!


# def plot_hist_of_means_and_vars(datasets, axis_to_compute=0, labels=None, column_names=None):
#     """ Plot histogram of samples mean.
#     Each bin is the number of samples in a given source (study) having `this` value for mean.
#     Args:
#         datasets : list of pandas dataframes or numpy arrays
#     """
#     ddof = 0

#     if axis_to_compute == 0:
#         var_name = 'genes'
#     elif axis_to_compute == 1:
#         var_name = 'samples'

#     fig, ax = plt.subplots(2, 1, figsize=(14, 10))

#     for i, data in enumerate(datasets):
#         if len(column_names):
#             data = data.loc[:, column_names]
#         if len(labels):
#             label = labels[i]

#         mu = data.mean(axis=axis_to_compute)
#         var = data.var(axis=axis_to_compute)

#         ax[0].hist(mu, bins=60, density=False, color=colors[i], alpha=0.5, label=label)
#         ax[0].set_ylabel('Count')
#         ax[0].set_title('mean across {}'.format(var_name), fontsize=fontsize)
#         ax[0].legend(loc='best', fontsize=fontsize_leg)
#         ax[0].grid('on')

#         ax[1].hist(var, bins=60, density=False, color=colors[i], alpha=0.5, label=label)
#         ax[1].set_ylabel('Count')
#         ax[1].set_title('variance across {}'.format(var_name), fontsize=fontsize)
#         ax[1].legend(loc='best', fontsize=fontsize_leg)
#         ax[1].grid('on')
# Plot functions =======================================================================================================


# EDA ==================================================================================================================
# def eda_combined_rnaseq_data(df, var_thres=1e-5):
#     df = df.copy()
#     sources = df['Sample'].map(lambda x: x.split('.')[0].lower()).unique().tolist()

#     ddof = 0  # for var calculation

#     cols_means = df.iloc[:, 1:].mean(axis=0)
#     cols_vars = df.iloc[:, 1:].var(axis=0, ddof=0)
#     print('\nDataset {}'.format(df.shape))
#     print('Genes mean range:  {:.7f}, {:.7f}'.format(cols_means.min(),
#                                                      cols_means.max()))
#     print('Genes var range:   {:.7f}, {:.7f}'.format(cols_vars.min(),
#                                                      cols_vars.max()))
#     print('Num of genes where var <= {}:   {:3d}'.format(var_thres, (cols_vars <= var_thres).sum()))
#     print('Num of samples where var <= {}: {:3d}'.format(var_thres, (cols_vars <= var_thres).sum()))

#     for i, source in enumerate(sources):
#         source_vec = df['Sample'].map(lambda x: x.split('.')[0].lower())
#         source_idx_bool = source_vec.str.startswith(source)
#         data = df.loc[source_idx_bool, df.columns[1:].values]

#         cols_means = data.mean(axis=0)
#         print('\n{} {}'.format(source, data.shape))
#         print('Genes mean range:  {:.7f}, {:.7f}'.format(cols_means.min(),
#                                                          cols_means.max()))
#         cols_vars = data.var(axis=0, ddof=ddof)
#         print('Genes var range:   {:.7f}, {:.7f}'.format(cols_vars.min(),
#                                                          cols_vars.max()))
#         print('Num of genes where var <= {}:   {:3d}'.format(var_thres, (cols_vars <= var_thres).sum()))
#         print('Num of samples where var <= {}: {:3d}'.format(var_thres, (cols_vars <= var_thres).sum()))


# def eda_pdm_metadata():
#     """ (ap)
#     There are primarily 3 types of filenames:
#     172845~064-121-B~M770J05J48~RNASeq.RSEM.genes.results
#     172845~064-121-B~M770J05J48~RNASeq.RSEM.isoforms.results
#     172845~064-121-B~M770J05J48~WES.vcf

#     File name interpretation: PATIENTID~SPECIMENID~SAMPLEID~FILE_TYPE

#     RNASeq files:  /nfs/nciftp/private/NCI_PDM/RNASeq/results,  1426 out of 1429 are *RNASeq.RSEM*
#     WES files:     /nfs/nciftp/private/NCI_PDM/WES/vcf,  780 out of 781 are *.vcf

#     """
#     path = get_file(DATA_URL + PDM_METADATA_FILENAME)

#     df = global_cache.get(path)
#     if df is None:
#         df = pd.read_csv(path, sep='\t', engine='c')
#         global_cache[path] = df

#     print('\n==============  EDA for metadata ==============')
#     print('PDM metadata: {}'.format(df.shape))
#     print('\nPDM metadata columns:\n{}'.format(df.columns.values))

#     print('\nNaN values per column:\n{}'.format(df.isnull().sum(axis=0)))

#     print('\nUnique values per column (include NaN as unique):\n{}'.format(df.nunique(dropna=False)))

#     print('\nSAMPLE_TYPE:\n{}'.format(df['SAMPLE_TYPE'].value_counts(dropna=False)))

#     print('\nSAMPLE_PASSAGE:\n{}'.format(df['SAMPLE_PASSAGE'].value_counts(dropna=False)))


# def eda_data_Sep2017_files():
#     """ (ap) EDA for genes.results and isoforms.results
#     There are 590 files of each type (genes.results, isoforms.results).
#     Each pair of files corresponds to a tissue sample and contains various values for each gene (e.g., TPM, FPKM).
#     The shape of genes.results and isoforms.results files is different:
#     genes shape:     (28109, 7), i.e. (gene_id, 7)
#     isomorfs shape:  (78375, 8)
#     Each element in column `transcript_id.s.` in genes file is flattened and placed across the `transcript_id` column in
#     isoforms file.
#     """
#     path = '/Users/apartin/Documents/JDACS/nciftp/private/NCI_PDM/data_Sep2017'
#     genes_filenames = glob(pathname=os.path.join(path, '*genes.results'))
#     iso_filenames = glob(pathname=os.path.join(path, '*isoforms.results'))
#     print('\nTotal .genes.results files:  {}'.format(len(genes_filenames)))
#     print('Total .isoform.results files:  {}'.format(len(iso_filenames)))

#     # EDA a random genes.results filename
#     gene_filename = genes_filenames[np.random.randint(low=0, high=len(genes_filenames), size=1)[0]]
#     df_genes = pd.read_csv(gene_filename, sep='\t', engine='c')
#     print('\n==============  EDA for genes.results  ==============')
#     print('df_genes:  {}'.format(df_genes.shape))
#     print('\ndf_genes columns:  {}'.format(df_genes.columns.tolist()))
#     print('\ndf_genes head:\n{}'.format(df_genes.head()))

#     # EDA a random isoforms.results filename
#     iso_filename = iso_filenames[np.random.randint(low=0, high=len(iso_filenames), size=1)[0]]
#     df_iso = pd.read_csv(iso_filename, sep='\t', engine='c')
#     print('\n==============  EDA for iso.results  ==============')
#     print('df_iso:  {}'.format(df_iso.shape))
#     print('\ndf_iso columns:  {}'.format(df_iso.columns.tolist()))
#     print('\ndf_iso head:\n{}'.format(df_iso.head()))


# def eda_gene_files():
#     """ (ap) EDA for genes.results and isoforms.results
#     There are 713 files of each type (genes.results, isoforms.results).
#     Each pair of files corresponds to a tissue sample and contains various values for each gene (e.g., TPM, FPKM).
#     The shape of genes.results and isoforms.results files is different:
#     genes shape:     (28109, 7), i.e. (gene_id, 7)
#     isomorfs shape:  (78375, 8)
#     Each element in column `transcript_id.s.` in genes file is flattened and placed across the `transcript_id` column in
#     isoforms file.
#     """
#     path = '/Users/apartin/Documents/JDACS/nciftp/private/NCI_PDM/RNASeq/results'
#     genes_filenames = glob(pathname=os.path.join(path, '*genes.results'))
#     iso_filenames = glob(pathname=os.path.join(path, '*isoforms.results'))
#     print('\nTotal .genes.results files:  {}'.format(len(genes_filenames)))
#     print('Total .isoform.results files:  {}'.format(len(iso_filenames)))

#     # EDA a random genes.results filename
#     gene_filename = genes_filenames[np.random.randint(low=0, high=len(genes_filenames), size=1)[0]]
#     df_genes = pd.read_csv(gene_filename, sep='\t', engine='c')
#     print('\n==============  EDA for genes.results  ==============')
#     print('df_genes:  {}'.format(df_genes.shape))
#     print('\ndf_genes columns:  {}'.format(df_genes.columns.values))
#     print('\ndf_genes head:\n{}'.format(df_genes.head()))

#     # EDA a random isoforms.results filename
#     iso_filename = iso_filenames[np.random.randint(low=0, high=len(iso_filenames), size=1)[0]]
#     df_iso = pd.read_csv(iso_filename, sep='\t', engine='c')
#     print('\n==============  EDA for iso.results  ==============')
#     print('df_iso:  {}'.format(df_iso.shape))
#     print('\ndf_iso columns:  {}'.format(df_iso.columns.values))
#     print('\ndf_iso head:\n{}'.format(df_iso.head()))


# def eda_pdm_by_type():
#     """ (ap)
#     This function looks at the data located in /private/NCI_PDM/data_frames/RNASeq/PDM_ByType.1
#     The effective shape (after removing NaN vectors) of X is (697 x 28109), i.e. (sample x gene_id)
#     """
#     pathdir = '/Users/apartin/Documents/JDACS/nciftp/private/NCI_PDM/data_frames/RNASeq/PDM_ByType.1'
#     X = pd.read_csv(os.path.join(pathdir, 'X'), sep='\t',
#                     na_values=['na', '-', '', 'n/a'],  # why it doesn't infer by default that 'n/a' is NaN
#                     header=None)

#     all_nan_cols = X.columns[X.isnull().sum(axis=0) == X.shape[0]].values
#     any_nan_cols = X.columns[X.isnull().sum(axis=0) > 0].values
#     all_nan_rows = X.index[X.isnull().sum(axis=1) == X.shape[1]].values
#     any_nan_rows = X.index[X.isnull().sum(axis=1) > 0].values
#     print('\n==============  EDA for X  ==============')
#     print('X dataframe: {}'.format(X.shape))
#     print('\n{}'.format(X.head()))
#     print('\nColumn names which contain all NaN values:\n{}'.format(all_nan_cols))
#     print('\nColumn names which contain any NaN values:\n{}'.format(any_nan_cols))
#     print('\nRow names which contain all NaN values:\n{}'.format(all_nan_rows))
#     print('\nRow names which contain any NaN values:\n{}'.format(any_nan_rows))

#     col = pd.read_csv(os.path.join(pathdir, 'col.h'), sep='\t', engine='c')
#     print('\n==============  EDA for col.h  ==============')
#     print('col.h dataframe: {}'.format(col.shape))
#     print('\ncol.head:\n{}'.format(col.head()))
#     print('\ncol.tail:\n{}'.format(col.tail()))

#     # row = pd.read_csv(os.path.join(pathdir, 'row.h'), sep='\t', engine='c')
#     row = pd.read_csv(os.path.join(pathdir, 'row.h'), sep='\t', engine='c',
#                       header=None, names=['id', 'Sample'])
#     print('\n==============  EDA for row.h  ==============')
#     print('row.h dataframe: {}'.format(row.shape))
#     print('\nrow.head:\n{}'.format(row.head()))
#     print('\nrow.tail:\n{}'.format(row.tail()))

#     # Drop all NaN cols and rows (why the first column in X is all NaNs ??)
#     X = X.drop(labels=X.columns[all_nan_cols], axis=1)
#     X = X.drop(labels=X.columns[all_nan_rows], axis=0)

#     # Update column names with gene names
#     col_mapper = OrderedDict((X.columns[i], col.iloc[i, 1]) for i in range(col.shape[0]))
#     X = X.rename(columns=col_mapper)

#     # Add a column of (tissue) sample identifier
#     X.insert(loc=0, column='source', value=row['Sample'])

#     # Add column of tumor names (use y and y_map)
#     y = pd.read_csv(os.path.join(pathdir, 'y'), sep='\t', engine='c')
#     y_map = pd.read_csv(os.path.join(pathdir, 'y.map'), sep='\t', engine='c')

#     # Add a column of tumor type
#     # (note that the last element in tissue_type is NaN)
#     X.insert(loc=1, column='tissue_type', value=y.astype(np.int16))
# EDA ==================================================================================================================



