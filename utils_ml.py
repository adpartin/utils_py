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

import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from scipy.stats import spearmanr


def balance_df(df, y, class_size=None, seed=None):
    """ Balace input dataframe based on vector y.
    Args:
        df : input dataframe
        y : vector based on which to balance the dataframe
        class_size : drop labels/classes that have less than label_size samples in df and sample
            class_size samples from each label/class.  If the arg `class_size` is not passed then
            use the size of the least represented class.
    Returns:
        df_out : updated/balanced df
        y_out : updated vector of classes
        dropped_classes : dict of class labels that have been dropped and their count
    """
    assert df.shape[0] == len(y), "df and y must contain same number of samples."
    df = df.copy()
    df_out = pd.DataFrame(columns=df.columns)
    y_out = []

    if class_size is None:
        class_size = y.value_counts().min()
    
    # Returns a list of dropped labels
    dropped_classes = {}
    
    for c in y.unique():
        idx = y == c
        # print(label, np.sum(idx))
        if np.sum(idx) >= class_size:
            c_df = df.loc[idx, :].sample(n=class_size, replace=False, axis=0, random_state=seed)
            df_out = pd.concat([df_out, c_df], axis=0).reset_index(drop=True)
            y_out.extend([c] * c_df.shape[0])
        else:
            dropped_classes[c] = np.sum(idx)

    y_out = pd.Series(y_out, name='y')
            
    # Shuffle
    # df_out = df_out.sample(frac=1.0).reset_index(drop=True)
    
    return df_out, y_out, dropped_classes


def drop_samples_on_class_count(df, y, min_class_size=100):
    """ Keep classes which contain at least `min_class_size` samples (apply on dataframe)
    Returns:
        df : df with classes that have at least `min_class_size` samples
        df_small : df with classes that have less than `min_class_size` samples
    """
    assert len(df) == len(y), "df and y must contain same number of samples."
    df = df.copy()
    bincount = y.value_counts(sort=True)
    df_list = []
    small_list = []
    
    for idx in bincount.index:
        # print('class label:', idx, 'count:', bincount[idx])
        if bincount[idx] >= min_class_size:
            tmp = df[y == idx]
            df_list.append(tmp)
        else:
            tmp = df[y == idx]
            small_list.append(tmp)

    assert len(df_list), '`min_class_size` is too high (all samples were thrown away).'
    df_out = pd.concat(df_list, axis=0)

    if len(small_list):
        df_out_small = pd.concat(small_list, axis=0)
    else:
        df_out_small = None
    
    return df_out, df_out_small


def plot_confusion_matrix(y_true, y_pred, labels=None, figsize=None, title=None,
                          cmap=sns.light_palette(color='navy', n_colors=100),
                          savefig=False, img_name='confusion'):
    """ Create a confusion matrix for a classification results.
    Args:
        y_true : numpy array of true labels
        y_pred : numpy array of predictions
        labels : list of unique label names
    Returns:
        df_conf : df of confusion matrix
    https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html
    """
    # TODO: should be a shorter way to put this assertion, or reshape the input arrays to support multi-dim ...
    if (y_true.ndim > 1) and (y_true.shape[1] > 1):
        raise TypeError('`y_true` must be a 1-D array.')
    else: 
        y_true = y_true.reshape(-1,)
    
    if (y_pred.ndim > 1) and (y_pred.shape[1] > 1):
        raise TypeError('`y_pred` must be a 1-D array.')
    else: 
        y_pred = y_pred.reshape(-1,)

    np_conf = confusion_matrix(y_true, y_pred)
    df_conf = pd.DataFrame(np_conf, index=labels, columns=labels)

    fontsize = 10  # font size of labels (not table numbers)
    if figsize is None:
        sc_x, sc_y = 0.5, 0.5
        figsize = sc_x * df_conf.shape[1], sc_y * df_conf.shape[0]

    fig, ax = plt.subplots(figsize=figsize)
    #sns.set(font_scale=2.0)
    sns.heatmap(df_conf, annot=True, annot_kws={"size": fontsize}, fmt='d',
                linewidths=0.99, cmap=cmap, linecolor='white', cbar_kws=dict(ticks=[]))
    ax.set_ylabel('True', fontsize=fontsize)
    ax.set_xlabel('Predicted', fontsize=fontsize)
    if labels is not None:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=fontsize)
        ax.set_yticklabels(labels, fontsize=fontsize)
    # fig.tight_layout()

    if title:
        ax.set_title(title, fontsize=fontsize)

    if savefig:
        fig.savefig(img_name, bbox_inches='tight')

    return df_conf


def compute_cor_mat(X, zero_diag=False, decimals=None):
    """ Compute Spearman correlation matrix of dataframe X.
    Args:
        X : input dataframe
        zero_diag : whether to zero out the diagonal
        decimals : number of decimal places to round
    Returns:
        cor : the computed corr dataframe
    """
    X = X.copy()
    X, _ = drop_low_var_cols(X, th=10**-16, verbose=False)  # required for Spearman rank correlationn

    if decimals:
        cor = np.around(spearmanr(X).correlation, 4)
    else:
        cor = spearmanr(X).correlation
    
    if zero_diag:
        np.fill_diagonal(cor, val=0)

    cor = pd.DataFrame(cor, columns=X.columns, index=X.columns)
    return cor


def plot_cor_heatmap(cor, value_range=[-1, 1], title=None, cmap='jet', figsize=None, full=True):
    """ TODO : This function runs too long for large arrays.
    Implement with regular matplotlib(??).
    https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html
    """
    if len(value_range)==2:
        vmin, vmax = value_range
    else:
        vmin, vmax = cor.min().min(), cor.max().max()

    fontsize = 8
    if figsize is None:
        sc_x, sc_y = 0.5, 0.5
        figsize = sc_x * cor.shape[1], sc_y * cor.shape[0]

    fig, ax = plt.subplots(figsize=figsize)
    if full == True:
        ax = sns.heatmap(cor, vmin=vmin, vmax=vmax, cmap=cmap, annot=True, annot_kws={"size": fontsize},
                         fmt='.2f', linewidths=0.99, linecolor='white')
    else:
        mask = np.zeros_like(cor)
        # mask[np.triu_indices_from(mask)] = True
        mask[np.tril_indices_from(mask)] = True
        ax = sns.heatmap(cor, vmin=vmin, vmax=vmax, cmap=cmap, annot=True, annot_kws={"size": fontsize},
                         fmt='.2f', linewidths=0.99, linecolor='white', mask=mask)        
        
    # ax.invert_yaxis()
    ax.xaxis.tick_top()
    if isinstance(cor, pd.DataFrame):
        ax.set_xticklabels(cor.columns, rotation=60)

    # plt.xticks(range(len(cor.columns)), cor.columns)
    # plt.yticks(range(len(cor.columns)), cor.columns)
    
    if title:
        plt.title(title)
    
    return fig


def plot_rf_fi(rf_model, figsize=(8, 5), plot_direction='h', columns=None, max_cols_plot=None,
               color='g', title=None, errorbars=True):
    """ Plot feature importance from a random forest.
    Args:
        plot_direction : direction of the bars (`v` for vertical, `h` for hrozontal)
        columns : list of columns names (df.columns)
        max_cols_plot (int) : number of top most important features to plot
    Returns:
        indices : all feature indices ordered by importance
        fig : handle for plt figure
    """
    fontsize=14
    alpha=0.7

    importance = rf_model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)
    indices = np.argsort(importance)[::-1]  # feature indices ordered by importance
    top_indices = indices[:max_cols_plot]    # get indices of top most important features
    if columns is None:
        columns = top_indices
    else:
        columns = np.array(columns)[top_indices]

    # Start plotting
    fig, ax = plt.subplots(figsize=figsize)
    if title:
        ax.set_title(title)
        
    if plot_direction=='v':
        if errorbars:
            ax.bar(range(len(top_indices)), importance[top_indices], color=color, align='center', alpha=alpha,
                   yerr=std[top_indices], ecolor='black')
        else:
            ax.bar(range(len(top_indices)), importance[top_indices], color=color, align='center', alpha=alpha)
        ax.set_xticks(range(len(top_indices)))
        ax.set_xticklabels(columns, rotation='vertical', fontsize=fontsize)
        ax.set_xlim([-1, len(top_indices)])
        ax.set_xlabel('Feature', fontsize=fontsize)
        ax.set_ylabel('Importance', fontsize=fontsize)
        [tick.label.set_fontsize(fontsize-4) for tick in ax.yaxis.get_major_ticks()]
    else:
        if errorbars:
            ax.barh(range(len(top_indices)), importance[top_indices], color=color, align='center', alpha=alpha,
                    xerr=std[top_indices], ecolor='black')
        else:
            ax.barh(range(len(top_indices)), importance[top_indices], color=color, align='center', alpha=alpha)
        ax.set_yticks(range(len(top_indices)))
        ax.set_yticklabels(columns, rotation='horizontal', fontsize=fontsize)
        ax.set_ylim([-1, len(top_indices)])
        # ax.invert_yaxis()
        ax.set_ylabel('Feature', fontsize=fontsize)
        ax.set_xlabel('Importance', fontsize=fontsize)
        [tick.label.set_fontsize(fontsize-4) for tick in ax.xaxis.get_major_ticks()]

    # ax.grid()
    # plt.tight_layout()

    return indices, fig


def get_rf_fi(rf_model, columns=None):
    """ Return df of feature importance from a random forest object. """
    importance = rf_model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)
    if columns is None:
        columns = range(len(importance))

    fi = pd.DataFrame({'cols': columns, 'imp': importance, 'std': std})
    
    fi.sort_values('imp', ascending=False, inplace=True)
    fi.reset_index(drop=False, inplace=True)
    fi.rename(columns={'index': 'col_id'}, inplace=True)
    return fi


def drop_most_imp_cols(rf_model, df, n):
    """ Drop the n most important features from rf model. """
    importance = rf_model.feature_importances_
    indices = np.argsort(importance)[::-1]  # feature indices ordered by importance (descending)
    indices_to_drop = indices[:n]           # get indices of top most important features
    cols_to_drop = np.array(df.columns)[indices_to_drop]
    df = df.drop(columns=cols_to_drop)
    
    return df, cols_to_drop


def drop_least_imp_cols(rf_model, df, n):
    """ Drop the n least important features from rf model. """
    importance = rf_model.feature_importances_
    indices = np.argsort(importance)  # feature indices ordered by importance (ascending)
    indices_to_drop = indices[:n]     # get indices of top most important features
    cols_to_drop = np.array(df.columns)[indices_to_drop]
    df = df.drop(columns=cols_to_drop)
    
    return df, cols_to_drop


def dropna(df, axis=0, th=0.4):
    """ Drop rows or cols based on the ratio of NA values along the axis.
    Args:
        df : input df
        th (float) : if the ratio of NA values along the axis is larger that th, then drop all the values
        axis (int) : 0 to drop rows; 1 to drop cols
    Returns:
        df : updated df
    """
    df = df.copy()
    axis = 0 if axis==1 else 1
    col_idx = df.isna().sum(axis=axis)/df.shape[axis] <= th
    df = df.iloc[:, col_idx.values]
    return df


def drop_low_var_cols(df, th=10**-16, skipna=True, verbose=True):
    """ Drop cols in which the variance is lower than th.
    Args:
        df : input dataframe
        th (float): threshold variance to drop the cols
    Returns:
        df : updated dataframe
        idx : indexes of the dropped columns
    """
    # TODO: modify the function to accept numpy array in addition to dataframe
    df = df.copy()
    idx = df.var(axis=0, skipna=skipna) <= th
    df = df.loc[:, ~idx]
    if verbose:
        print(f'Dropped {np.sum(idx)} cols out of {len(idx)} based on col variance (th={th}).')

    return df, idx    


def drop_cols_on_cor(df, th=0.95, verbose=True):
    """ Remove cols whose correlations with other cols exceed th.
    Note: cols with var=0 (yield corr=nan) and na values are not processed here; should be processed before/after.
    For example: th=1 => drops col if there is a perfect correlation with other col
                 th=0.5 => drops col if there is 0.5 correlation with other col
    Args:
        df : input dataframe (df should not contain any missing values)
        th (float): min correlation value to drop the cols
    Returns:
        df : updated dataframe
        cols_dropped (list): cols/features dropped
    """
    assert (th >= 0) & (th <= 1), "th must be in the range [0, 1]."
    df = df.copy()

    col_len = len(df.columns)

    # Ignore cols which contain missing values, or where var = 0
    idx_tmp = (df.var(axis=0) == 0).values | (df.isnull().sum(axis=0) > 0).values
    ## df_tmp = df.loc[:, idx_tmp].copy()
    df = df.loc[:, ~idx_tmp].copy()  # dataframe to process

    # corr can be computed using --> df.corr(method='pearson').abs() --> this is much slower than using just numpy!
    cc = np.corrcoef(df.values.T)
    corr = pd.DataFrame(cc, index=df.columns, columns=df.columns).abs()

    # Iteratively update the correlation matrix
    # Iterate as long as there are correlations above thres_xcorr (excluding autocorrelation values, i.e. on diagonal)
    cols_dropped = []
    while (~np.eye(len(corr), dtype=np.bool) * (corr >= th)).any().sum():
        # print('corr matrix: shape={}'.format(corr.shape))
        # mask relevant indexes --> where corr>=th
        th_mask = ~np.eye(len(corr), dtype=np.bool) * (corr >= th)
        # count occurance of relevant indexes for each col --> how many times a col has corr>th
        corr_count = th_mask.sum(axis=1)
        # get indexes of most relevant cols --> 'most relevant' = most occurances
        col_index = corr_count[corr_count == corr_count.max()].index
        # assign weight (sum of relevant corr values) to each col
        corr_weight = (th_mask * corr).sum(axis=1)

        # Among the most relevant cols (defined by col_index),
        # choose the col with the max corr weight to be dropped
        col_drop = corr_weight[col_index].idxmax()
        cols_dropped.append(col_drop)

        # Remove the col from the corr matrix
        corr = corr.loc[~(corr.index == col_drop), ~(corr.columns == col_drop)]

    # Update the original dataset
    # corr.index contains columns names that are left after removing high correlations columns
    df = df.loc[:, corr.index]

    # Concatenate processed df and df_tmp
    ## df = pd.concat([df, df_tmp], axis=1, ignore_index=False)

    if verbose:
        print(f"\n{col_len-len(df.columns)} cols were dropped based on high xcorr between cols (th={th}).")

    return df, corr, cols_dropped


def contingency_table(df, cols, to_plot=True, figsize=None, title=None, margins=False, normalize=False):
    """
    Args:
        df : df with categorical 
        cols (list) : list of column names for which to generate the contingency table
        to_plot : whether to plot the contingency table
        margins (bool) : arg for pd.crosstab; whether to include row/column margins (subtotals)
        normalize : arg for pd.crosstab; whether to normalize (‘all’, ‘index’, ‘columns’) or False
    Returns:
        table (df) : contingency table
    https://hamelg.blogspot.com/2015/11/python-for-data-analysis-part-19_17.html
    """
    assert len(cols) == 2, 'Exactly two column names should be in the list.'
    assert set(cols).issubset(set(df.columns)), "`cols` are not in df.columns."
    table = df[cols].copy()
    # table['ones'] = 1
    # table = pd.pivot_table(table, index=cols[0], columns=cols[1], values='ones',
    #                        aggfunc=np.sum, fill_value=0)
    table = pd.crosstab(index=df[cols[0]], columns=df[cols[1]],
                        margins=margins, normalize=normalize)
    table.index.name = None
    table.columns.name = None

    if to_plot:
        fontsize = 20
        if figsize is None:
            fontsize = 10
            sc_x, sc_y = 0.5, 0.5
            figsize = sc_x * table.shape[1], sc_y * table.shape[0]

        fig, ax = plt.subplots(figsize=figsize)
        # sns.set(font_scale=1.6)
        if normalize:
            ax = sns.heatmap(table, annot=True, linewidths=0.9, cmap='Greens', annot_kws={"size": fontsize})
        else:
            ax = sns.heatmap(table, annot=True, fmt='d', linewidths=0.9, cmap='Greens', annot_kws={"size": fontsize})
        colnames = table.columns.tolist()
        rownames = table.index.tolist()
        ax.set_xticklabels(colnames, rotation=80, fontsize=fontsize)
        ax.set_yticklabels(rownames, rotation=0, fontsize=fontsize)
        # print(table)

        if title:
            ax.set_title(title, fontsize=fontsize)

    return table


def plot_pca(df, components=[1, 2], figsize=(8, 5),
             color_vector=None, marker_vector=None,
             to_scale=False, title=None, verbose=True):
    """
    Apply PCA to input df.
    Args:
        color_vector : each element corresponds to a row in df. The unique elements will be colored
            with a different color.
        marker_vector : each element corresponds to a row in df. The unique elements will be marked
            with a different marker.
    Returns:
        pca_obj : object of sklearn.decomposition.PCA()
        pca : pca matrix
        fig : PCA plot figure handle

    https://stackoverflow.com/questions/12236566/setting-different-color-for-each-series-in-scatter-plot-on-matplotlib
    """
    if color_vector is not None:
        assert len(df) == len(color_vector), 'len(df) and len(color_vector) must be the same size.'
        n_colors = len(np.unique(color_vector))
        colors = iter(cm.rainbow(np.linspace(0, 1, n_colors)))

    if marker_vector is not None:
        assert len(df) == len(marker_vector), 'len(df) and len(marker_vector) shuold be the same size.'
        all_markers = ('o', 'v', 's', 'p', '^', '<', '>', '8', '*', 'h', 'H', 'D', 'd', 'P', 'X')
        markers = all_markers[:len(np.unique(marker_vector))]

    df = df.copy()

    # PCA
    if to_scale:
        xx = StandardScaler().fit_transform(df.values)
    else:
        xx = df.values

    n_components = max(components)
    pca_obj = PCA(n_components=n_components)
    pca = pca_obj.fit_transform(xx)
    pc0 = components[0] - 1
    pc1 = components[1] - 1

    # Start plotting
    fig, ax = plt.subplots(figsize=figsize)

    if (color_vector is not None) and (marker_vector is not None):
        for i, marker in enumerate(np.unique(marker_vector)):
            for color in np.unique(color_vector):
                # print(i, 'marker:', marker, 'color:', color)
                idx = (marker_vector == marker) & (color_vector == color)
                ax.scatter(pca[idx, pc0], pca[idx, pc1], alpha=0.7,
                            marker=markers[i],
                            edgecolors='black',
                            color=next(colors),
                            label='{}, {}'.format(marker, color))

    elif (color_vector is not None):
        for color in np.unique(color_vector):
            idx = (color_vector == color)
            ax.scatter(pca[idx, pc0], pca[idx, pc1], alpha=0.7,
                        marker='o',
                        edgecolors='black',
                        color=next(colors),
                        label='{}'.format(color))

    elif (marker_vector is not None):
        for i, marker in enumerate(np.unique(marker_vector)):
            idx = (marker_vector == marker)
            ax.scatter(pca[idx, pc0], pca[idx, pc1], alpha=0.7,
                        marker=markers[i],
                        edgecolors='black',
                        color='blue',
                        label='{}'.format(marker))

    else:
        ax.scatter(pca[:, pc0], pca[:, pc1], alpha=0.7,
                   marker='s', edgecolors='black', color='blue')

    if title:
        ax.set_title(title)
    ax.set_xlabel('PC'+str(components[0]))
    ax.set_ylabel('PC'+str(components[1]))
    ax.legend(loc='lower left', bbox_to_anchor=(1.01, 0.0), ncol=1,
              borderaxespad=0, frameon=True)
    plt.grid(True)

    if verbose:
        print('Explained variance by PCA components [{}, {}]:  [{:.5f}, {:.5f}]'.format(
            components[0], components[1],
            pca_obj.explained_variance_ratio_[pc0],
            pca_obj.explained_variance_ratio_[pc1]))

    return pca_obj, pca, fig


