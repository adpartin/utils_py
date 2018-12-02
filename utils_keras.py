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

import seaborn as sns

from keras import backend as K
from keras.utils import np_utils


def get_keras_performance_metrics(history):
    """ Extract names of all the recorded performance metrics from keras `history` variable
    for train and val sets. The performance metrics can be indentified as those that start
    with 'val'.
    """
    all_metrics = list(history.history.keys())  # all metrics including everything returned from callbacks
    pr_metrics = []  # performance metrics recorded for train and val such as 'loss', etc. (excluding callbacks)
    for m in all_metrics:
        if 'val' in m:
            pr_metrics.append('_'.join(m.split('_')[1:]))

    return pr_metrics


def add_another_y_axis(ax, x, y, color='g', marker='o', yscale='linear', y_axis_name=None):
    """ Adapted from:  https://matplotlib.org/devdocs/gallery/api/two_scales.html
    Args:
        ax (axis obj) : Axis to put two scales on
        x (array-like) : x-axis values for both datasets
        y (array-like) : Data for right hand scale
        color (str) : Color for added line
        marker (str) : Color for added line
        y_axis_name (str) : Name of the plotted value
    Returns:
        ax2 (axis obj) : New twin axis
    """
    legend_fontsize = 10
    fontsize = 12
    markersize = 5
    ax2 = ax.twinx()
    ax2.plot(x, y, color=color, marker=marker, alpha=0.6, markersize=markersize, label=y_axis_name)

    ax2.set_yscale(yscale)

    if y_axis_name:
        ax2.set_ylabel(y_axis_name, color=color, fontsize=fontsize)

    for tl in ax2.get_yticklabels():
        tl.set_color(color)

    # legend = ax2.legend(loc='best', prop={'size': legend_fontsize})

    ymin, ymax = np.min(y)/10.0, np.max(y)*10.0
    ax2.set_ylim([ymin, ymax])
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(axis='both', which='minor', labelsize=12)

    return ax2


def plot_keras_learning(history, figsize=(10, 8), savefig=True, img_name='learning_with_lr'):
    """ Plot the learning progress for all the recorded metrics. This function should be used with
    hold-out validation scheme since it allows to plot learning rate on a separate axis.
    Args:
        history (keras callbacks) : return callback object from keras model tranining model.fit()
    """
    import matplotlib.pyplot as plt
    ##plt.rcParams['figure.figsize'] = figsize
    legend_fontsize = 10
    fontsize = 12
    markersize = 5

    # Get the epochs vector and the recorded metrics during training
    epochs = np.asarray(history.epoch) + 1
    hh = history.history.copy()

    # Extract names of all recorded performance metrics for training and val sets
    pr_metrics = get_keras_performance_metrics(history)

    fig = plt.figure(figsize=figsize)
    for p, m in enumerate(pr_metrics):
        ax = fig.add_subplot(len(pr_metrics), 1, p + 1)

        metric_name = m
        metric_name_val = 'val_' + m

        plt.plot(epochs, hh[metric_name], 'bo', alpha=0.6, markersize=markersize, label=metric_name)
        plt.plot(epochs, hh[metric_name_val], 'ro', alpha=0.6, markersize=markersize, label=metric_name_val)
        plt.ylabel(metric_name, fontsize=fontsize)

        plt.grid(True)
        plt.xlim([0.5, len(epochs) + 0.5])
        plt.ylim([0, 1])
        legend = ax.legend(loc='best', prop={'size': legend_fontsize})
        frame = legend.get_frame()
        frame.set_facecolor('0.50')

        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=12)

        # Plot learning rate over epochs
        if 'lr' in hh.keys():
            _ = add_another_y_axis(ax=ax, x=epochs, y=hh['lr'], color='g', marker='o', yscale='log', y_axis_name='Learning Rate')

    ax.set_xlabel('Epochs', fontsize=fontsize)

    if savefig:
        plt.savefig(img_name, bbox_inches='tight')


def plot_keras_learning_kfold(hs, savefig=True, img_name='learn_kfold'):
    """ Plot the learning progress (averaged across k folds).
    Args:
        hs (dict of keras callbacks) : a callback object from keras model tranining model.fit()
    """
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = [10, 8]
    legend_font_size = 10
    fontsize = 12
    markersize = 5

    epochs = np.asarray(hs[0].epoch) + 1
    k_folds = len(hs)

    # Extract names of all recorded metrics for training and val sets
    pr_metrics = get_keras_performance_metrics(hs[0])

    # Plot
    for m in pr_metrics:
        metric_name = m
        metric_name_val = 'val_' + m

        # Compute the average of a metric across folds
        metric_avg = np.asarray([hs[fold].history[metric_name] for fold in hs]).sum(axis=0, keepdims=True) / k_folds
        metric_avg_val = np.asarray([hs[fold].history[metric_name_val] for fold in hs]).sum(axis=0, keepdims=True) / k_folds

        # Plot a metric for each fold vs epochs
        marker = ['b.', 'r^', 'kx', 'mv', 'gp', 'bs', 'r8', 'kD']
        fig = plt.figure()
        for i, metric in enumerate([metric_name, metric_name_val]):
            ax = fig.add_subplot(3, 1, i + 1)
            for fold in range(k_folds):
                plt.plot(epochs, hs[fold].history[metric], alpha=0.5, markersize=markersize, label='fold{}'.format(fold + 1))
            plt.ylabel(metric, fontsize=fontsize)
            plt.grid(True)
            plt.xlim([0.5, len(epochs) + 0.5])
            plt.ylim([0, 1])
            ax.tick_params(axis='both', which='major', labelsize=fontsize)
            ax.tick_params(axis='both', which='minor', labelsize=fontsize)
            plt.legend(loc='best', prop={'size': legend_font_size})


        # Plot the average of a metric across folds vs epochs
        ax = fig.add_subplot(3, 1, 3)
        plt.plot(epochs, metric_avg.flatten(), 'bo', alpha=0.6, markersize=markersize, label=metric_name)
        plt.plot(epochs, metric_avg_val.flatten(), 'rs', alpha=0.6, markersize=markersize, label=metric_name_val)
        plt.ylabel(metric_name+' avg over folds', fontsize=fontsize)
        plt.xlabel('epochs', fontsize=fontsize)
        plt.grid(True)
        plt.xlim([0.5, len(epochs) + 0.5])
        plt.ylim([0, 1])
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.tick_params(axis='both', which='minor', labelsize=fontsize)
        plt.legend(loc='best', prop={'size': legend_font_size})

        if savefig:
            plt.savefig(img_name + '_' + metric_name + '.png', bbox_inches='tight')


def print_results(train_scores, val_scores):
    """ Print training results collected with k-fold cv into file.
    Args:
        train_scores & val_scores (df) : each row value represents a train/val score for each epoch;
            the col names are the recorded metrics (e.g., loss, mae, r_square)
    """
    metrics = list(train_scores.columns)
    k_folds = len(train_scores)

    for i, m in enumerate(metrics):
        print("\n{:<13}Train,  Val".format(m))
        for f in range(len(val_scores)):
            print("  fold {}/{}:  {:=+5.3f}, {:=+5.3f}".format(f + 1, k_folds, train_scores.iloc[f, i],
                                                                               val_scores.iloc[f, i]))

    print("\n{:<15}Train,  Val".format(''))
    for i, m in enumerate(metrics):
        print("Mean {:<10}{:=+5.3f}, {:=+5.3f}".format(m, train_scores.iloc[:, i].sum() / k_folds,
                                                          val_scores.iloc[:, i].sum() / k_folds))


def save_results(train_scores, val_scores, model_name):
    """ Save training results collected with k-fold cv.
    Args:
        train_scores & val_scores (df) : each row value represents a train/val score for each epoch;
            the col names are the recorded metrics (e.g., loss, mae, r_square)
    """
    metrics = list(train_scores.columns)
    k_folds = len(train_scores)

    scores_fname = '{}.scores'.format(model_name)

    with open(scores_fname, 'w') as scores_file:
        for i, m in enumerate(metrics):
            scores_file.write('{:<13}Train,  Val\n'.format(m))
            for f in range(len(val_scores)):
                scores_file.write('  fold {}/{}:  {:=+5.3f}, {:=+5.3f}\n'.format(
                    f + 1, k_folds, train_scores.iloc[f, i], val_scores.iloc[f, i]))
            scores_file.write('\n')

        scores_file.write('{:<15}Train,  Val\n'.format(''))
        for i, m in enumerate(metrics):
            scores_file.write('Mean {:<10}{:=+5.3f}, {:=+5.3f}\n'.format(m,
                              train_scores.iloc[:, i].sum() / k_folds, val_scores.iloc[:, i].sum() / k_folds))


