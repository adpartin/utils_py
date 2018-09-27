from __future__ import absolute_import
from __future__ import division, print_function

import os
import time
import numpy as np
import pandas as pd


def make_dir(path):
    """ This is used with a python version where the `exist_ok` arg is not available in os.makedirs(). """
    if not os.path.exists(path=path):
        os.makedirs(path)


def read_file_by_chunks(path, chunksize=1000, sep='\t', non_numeric_cols=None):
    """ Reads a large file in chunks (e.g., RNA-Seq).
    Args:
        path : path to the data file
        non_numeric_cols : a list of column names which contain non numeric values
    """
    t0 = time.time()
    columns = pd.read_table(path, header=None, nrows=1, sep=sep, engine='c').iloc[0, :].tolist()  # read col names only
    types = OrderedDict((c, str if c in non_numeric_cols else np.float32) for c in columns)  # organize types in dict
    chunks = pd.read_table(path, chunksize=chunksize, dtype=types, sep=sep, engine='c')

    print('Loading dataframe by chunks...')
    chunk_list = []
    for i, chunk in enumerate(chunks):
        # print('Chunk {}'.format(i+1))
        chunk_list.append(chunk)

    # print('Loading time:  {:.2f} mins'.format((time.time() - t0)/60))

    df = pd.concat(chunk_list, ignore_index=True)
    return df



