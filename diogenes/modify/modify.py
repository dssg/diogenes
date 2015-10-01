from uuid import uuid4
from collections import Counter

import numpy as np
from sklearn import preprocessing

from diogenes.utils import (remove_cols, append_cols, distance,convert_to_sa, 
                            stack_rows, sa_from_cols, join)

"""This module provides a number of operations to modify structured arrays

A number of functions take a parameter called "arguments" of type list of
dict. Diogenes expects these parameters to be expressed in the following
format for functions operating on columns (choose_cols_where, 
remove_cols_where):

    [{'func': LAMBDA_1, 'vals': LAMBDA_ARGUMENTS_1},
     {'func': LAMBDA_2, 'vals': LAMBDA_ARGUMENTS_2},
     ...
     {'func': LAMBDA_N, 'vals': LAMBDA_ARGUMENTS_N}]

and in this format for functions operating on rows (choose_rows_where, 
remove_rows_where, where_all_are_true):

    [{'func': LAMBDA_1, 'vals': LAMBDA_ARGUMENTS_1, 'col_name': LAMBDA_COL_1},
     {'func': LAMBDA_2, 'vals': LAMBDA_ARGUMENTS_2, 'col_name': LAMBDA_COL_2},
     ...
     {'func': LAMBDA_N, 'vals': LAMBDA_ARGUMENTS_N, 'col_name': LAMBDA_COL_N}]

In either case, the user can think of arguments as a query to be matched 
against certain rows or columns. Some operation will then be performed on
the matched rows or columns. For example, in choose_rows_where, an array
will be returned that has only the rows that matched the query. In 
remove_cols_where, all columns that matched the query will be removed.

Each dictionary is a single directive. The value assigned to the 'func' key
is a function that returns a binary array signifying the rows or columns
that pass a certain check. The value assigned to the 'vals' key is an argument
to be passed to the function assigned to the 'func' key. For queries affecting
rows, the value assigned to the 'col_name' key is the column over which the
'func' function should be applied. For example, in order to pick all rows
for which the 'year' column is between 1990 and 2000, we would create the
directive:

    {'func': diogenes.modify.row_val_between, 'vals': [1990, 2000],
     'col_name': 'year'}

To pick columns where every cell in the column is 0, we would create the
directive:

    {'func': diogenes.modify.col_val_eq, 'vals': 0}

Ultimately, diogenes will pick the columns or rows for which all directives
in the passed list are True. For example, if we want to pick rows for which
the 'year' column is between 1990 and 2000. We use:

    arguments=[{'func': diogenes.modify.row_val_between, 'vals': [1990, 2000],
                'col_name': 'year'}]

If we want to pick rows for which the 'year' column is between 1990 and 2000
*and* the 'gender' column is 'F' we use:

    arguments=[{'func': diogenes.modify.row_val_between, 'vals': [1990, 2000],
                'col_name': 'year'},
               {'func': diogenes.modify.row_val_eq, 'vals': 'F', 
                'col_name': 'gender'}]

Note that arguments must always be a list of dict, so even if there is only
one directive it must be in a list.

"""

def choose_cols_where(M, arguments):
    """Returns a structured array containing only columns adhering to a query

    Parameters
    ----------
    M : numpy.ndarray
        Structured array 
    arguments : list of dict
        See module documentation

    Returns
    -------
    numpy.ndarray
        Structured array with only specified columns

    """
    to_keep = np.ones(len(M.dtype), dtype=bool)
    for arg_set in arguments:
        lambd, vals = (arg_set['func'], arg_set['vals'])
        to_keep = np.logical_and(to_keep, lambd(M,  vals))
    keep_col_names = [col_name for col_name,included in zip(M.dtype.names, to_keep) if included] 
    return M[keep_col_names]
    
def remove_cols_where(M, arguments):
    """Returns a structured array containing columns not adhering to a query

    Parameters
    ----------
    M : numpy.ndarray
        Structured array 
    arguments : list of dict
        See module documentation

    Returns
    -------
    numpy.ndarray
        Structured array without specified columns

    """
    to_remove = np.ones(len(M.dtype), dtype=bool)
    for arg_set in arguments:
        lambd, vals = (arg_set['func'], arg_set['vals'])
        to_remove = np.logical_and(to_remove, lambd(M,  vals))
    remove_col_names = [col_name for col_name,included in 
                        zip(M.dtype.names, to_remove) if included] 
    return remove_cols(M, remove_col_names)

def col_random(M, boundary):
    """Pick random columns

    To be used as a 'func' argument in choose_cols_where or remove_cols_where 
    (see module documentation)

    Parameters
    ----------
    M : numpy.ndarray
        structured array
    boundary : int
        number of columns to pick

    Returns
    -------
    numpy.ndarray
        boolean array: True if column picked, False if not.

    """
    num_col = len(M.dtypes.names)
    remove_these_columns = np.random.choice(num_col, boundary, replace=False)
    indices = np.ones(num_col, dtype=bool)
    indices[remove_these_column] = False
    return indices

def col_val_eq(M, boundary):
    """Pick columns where every cell equals specified value

    To be used as a 'func' argument in choose_cols_where or remove_cols_where 
    (see module documentation)

    Parameters
    ----------
    M : numpy.ndarray
        structured array
    boundary : number or str or bool
        if every cell==boundary, the column will be picked

    Returns
    -------
    numpy.ndarray
        boolean array: True if column picked, False if not.

    """
    return [np.all(M[col_name] == boundary) for col_name in M.dtype.names]

def col_val_eq_any(M, boundary=None):
    """Pick columns for which every cell has the same value

    To be used as a 'func' argument in choose_cols_where or remove_cols_where 
    (see module documentation)

    Parameters
    ----------
    M : numpy.ndarray
        structured array
    boundary : None
        ignored

    Returns
    -------
    numpy.ndarray
        boolean array: True if column picked, False if not.

    """
    return [np.all(M[col_name]==M[col_name][0]) for col_name in M.dtype.names]

def col_fewer_than_n_nonzero(M, boundary=2):
    """Pick columns that have fewer than a specified number of nonzeros

    To be used as a 'func' argument in choose_cols_where or remove_cols_where 
    (see module documentation)

    Parameters
    ----------
    M : numpy.ndarray
        structured array
    boundary : int
        If the number of nonzeros is at or above boundary nonzeros, the column
        will not be picked

    Returns
    -------
    numpy.ndarray
        boolean array: True if column picked, False if not.

    """
    return [len(np.where(M[col_name]!=0)[0])<boundary for col_name in M.dtype.names]

#write below diffently as lambda
def col_has_lt_threshold_unique_values(M, threshold):
    """Pick columns that have fewer than a specified number of unique values

    To be used as a 'func' argument in choose_cols_where or remove_cols_where 
    (see module documentation)

    Parameters
    ----------
    M : numpy.ndarray
        structured array
    boundary : int
        If the number of nonzeros is at or above boundary unique values, the column
        will not be picked

    Returns
    -------
    numpy.ndarray
        boolean array: True if column picked, False if not.

    """
    ret = []
    for name in M.dtype.names:
        col = M[name]
        d = Counter(col)
        vals = sort(d.values())
        ret.append(sum(vals[:-1]) < threshold) 
    return ret

def choose_rows_where(M, arguments):
    """Returns a structured array containing only rows adhering to a query

    Parameters
    ----------
    M : numpy.ndarray
        Structured array 
    arguments : list of dict
        See module documentation

    Returns
    -------
    numpy.ndarray
        Structured array with only specified rows

    """
    to_select = np.ones(M.size, dtype=bool)
    for arg_set in arguments:
        lambd, col_name, vals = (arg_set['func'], arg_set['col_name'],
                                    arg_set['vals'])
        to_select = np.logical_and(to_select, lambd(M, col_name, vals))
    return M[to_select]

def remove_rows_where(M, arguments):
    """Returns a structured array containing rows not adhering to a query

    Parameters
    ----------
    M : numpy.ndarray
        Structured array 
    arguments : list of dict
        See module documentation

    Returns
    -------
    numpy.ndarray
        Structured array without specified rows

    """
    to_remove = np.ones(M.size, dtype=bool)
    for arg_set in arguments:
        lambd, col_name, vals = (arg_set['func'], arg_set['col_name'],
                                    arg_set['vals'])
        to_remove = np.logical_and(to_remove, lambd(M, col_name, vals))
    return M[np.logical_not(to_remove)]

def where_all_are_true(M, arguments, generated_name=None):
    """Appends a boolean column to M which specifies which rows pass a query

    Parameters
    ----------
    M : numpy.ndarray
        Structured array 
    arguments : list of dict
        See module documentation
    generated_name : str
        Name to give new column. If not specified, and arbitrary name will
        be generated

    Returns
    -------
    numpy.ndarray
        Structured array with extra column

    """
    if generated_name is None:
        generated_name = str(uuid4())
    to_select = np.ones(M.size, dtype=bool)
    for arg_set in arguments:
        lambd, col_name, vals = (arg_set['func'], arg_set['col_name'],
                                    arg_set['vals'])
        to_select = np.logical_and(to_select, lambd(M, col_name, vals))
    return append_cols(M, to_select, generated_name)

def row_is_outlier(M, col_name, boundary=3.0):
    """Picks rows that are not within some a number of deviations of the mean

    To be used as a 'func' argument in choose_rows_where, remove_rows_where, 
    or where_all_are_true (see module documentation).

    Parameters
    ----------
    M : numpy.ndarray
        structured array
    col_name : str
        name of column to check
    boundary : float
        number of standard deviations from mean required to be considered
        an outlier

    Returns
    -------
    numpy.ndarray
        boolean array: True if row picked, False if not.

    """
    std = np.std(M[col_name])
    mean = np.mean(M[col_name])
    return np.logical_or(
            (mean - boundary * std) > M[col_name], 
            (mean + boundary * std) < M[col_name]) 
    
def row_val_eq(M, col_name, boundary):
    """Picks rows for which cell is equal to a specified value

    To be used as a 'func' argument in choose_rows_where, remove_rows_where, 
    or where_all_are_true (see module documentation).

    Parameters
    ----------
    M : numpy.ndarray
        structured array
    col_name : str
        name of column to check
    boundary : number
        value to which cell must be equal

    Returns
    -------
    numpy.ndarray
        boolean array: True if row picked, False if not.

    """
    return M[col_name] == boundary

def row_val_lt(M, col_name, boundary):
    """Picks rows for which cell is less than to a specified value

    To be used as a 'func' argument in choose_rows_where, remove_rows_where, 
    or where_all_are_true (see module documentation).

    Parameters
    ----------
    M : numpy.ndarray
        structured array
    col_name : str
        name of column to check
    boundary : number
        value which cell must be less than

    Returns
    -------
    numpy.ndarray
        boolean array: True if row picked, False if not.

    """
    return M[col_name] < boundary

def row_val_gt(M, col_name, boundary):
    """Picks rows for which cell is greater than to a specified value

    To be used as a 'func' argument in choose_rows_where, remove_rows_where, 
    or where_all_are_true (see module documentation).

    Parameters
    ----------
    M : numpy.ndarray
        structured array
    col_name : str
        name of column to check
    boundary : number
        value which cell must be greater than

    Returns
    -------
    numpy.ndarray
        boolean array: True if row picked, False if not.

    """
    return M[col_name] > boundary

def row_val_between(M, col_name, boundary):
    """Picks rows for which cell is between the specified values

    To be used as a 'func' argument in choose_rows_where, remove_rows_where, 
    or where_all_are_true (see module documentation).

    Parameters
    ----------
    M : numpy.ndarray
        structured array
    col_name : str
        name of column to check
    boundary : (number, number)
        To pick a row, the cell must be greater than or equal to boundary[0]
        and less than or equal to boundary[1]

    Returns
    -------
    numpy.ndarray
        boolean array: True if row picked, False if not.

    """
    return np.logical_and(boundary[0] <= M[col_name], M[col_name] <= boundary[1])


def row_is_within_region(M, col_names, boundary):
    """Picks rows for which cell is within a spacial region

    To be used as a 'func' argument in choose_rows_where, remove_rows_where, 
    or where_all_are_true (see module documentation).

    Parameters
    ----------
    M : numpy.ndarray
        structured array
    col_names : list of str
        pair of column names signifying x and y coordinates
    boundary : array of points
        shape which cell must be within

    Returns
    -------
    numpy.ndarray
        boolean array: True if row picked, False if not.

    """
    import matplotlib.path as mplPath
    bbPath = mplPath.Path(np.array(boundary))
    return [bbPath.contains_point(point) for point in M[col_names]]

def combine_cols(M, lambd, col_names, generated_name):
    """Create a new column that is the function of existing columns

    Parameters
    ----------
    lambd : list of np.array > np.array
        Function that takes a list of columns and produces a single
        column. 
    col_names : list of str
        Names of columns to combine
    generated_name : str
        Name for generated column
    """
    new_col = lambd(*[M[name] for name in col_names])
    return append_cols(M, new_col, generated_name)

@np.vectorize
def combine_sum(*args):
    """Returns the elementwise sum of columns

    Intended to be used as the lambd argument in combine_cols

    Parameters
    ----------
    args : list of numpy.ndarray

    Returns
    -------
    mumpy.ndarray
    """
    return sum(args)

@np.vectorize
def combine_mean(*args):
    """Returns the elementwise mean average of columns

    Intended to be used as the lambd argument in combine_cols

    Parameters
    ----------
    args : list of numpy.ndarray

    Returns
    -------
    mumpy.ndarray
    """
    return np.mean(args)
    
def label_encode(M):
    """Changes string cols to ints so that there is a 1-1 mapping between 
    strings and ints

    Parameters
    ----------
    M : numpy.ndarray
        structured array

    Returns
    -------
    numpy.ndarray

    """

    M = convert_to_sa(M)
    le = preprocessing.LabelEncoder()
    new_dtype = []
    result_arrays = []
    for (col_name, fmt) in M.dtype.descr:
        if 'S' in fmt:
            result_arrays.append(le.fit_transform(M[col_name]))
            new_dtype.append((col_name, int))
        else:
            result_arrays.append(M[col_name])
            new_dtype.append((col_name, fmt))
    return np.array(zip(*result_arrays), dtype=new_dtype)

def replace_missing_vals(M, strategy, missing_val=np.nan, constant=0):
    """Replace values signifying missing data with some substitute
    
    Parameters
    ----------
    M : numpy.ndarray
        structured array
    strategy : {'mean', 'median', 'most_frequent', 'constant'}
        method to use to replace missing data
    missing_val : value that M uses to represent missint data. i.e.
        numpy.nan for floats or -999 for integers
    constant : int
        If the 'constant' strategy is chosen, this is the value to
        replace missing_val with

    """
    # TODO support times, strings
    M = convert_to_sa(M)

    if strategy not in ['mean', 'median', 'most_frequent', 'constant']:
        raise ValueError('Invalid strategy')

    M_cp = M.copy()

    if strategy == 'constant':

        try:
            missing_is_nan = np.isnan(missing_val)
        except TypeError:
            # missing_val is not a float
            missing_is_nan = False

        if missing_is_nan: # we need to be careful about handling nan
            for col_name, col_type in M_cp.dtype.descr:
                if 'f' in col_type:
                    col = M_cp[col_name]
                    col[np.isnan(col)] = constant
            return M_cp        

        for col_name, col_type in M_cp.dtype.descr:
            if 'i' in col_type or 'f' in col_type:
                col = M_cp[col_name]
                col[col == missing_val] = constant
        return M_cp

    # we're doing one of the sklearn imputer strategies
    imp = preprocessing.Imputer(missing_values=missing_val, strategy=strategy, axis=1)
    for col_name, col_type in M_cp.dtype.descr:
        if 'f' in col_type or 'i' in col_type:
            # The Imputer only works on float and int columns
            col = M_cp[col_name]
            col[:] = imp.fit_transform(col)
    return M_cp

def generate_bin(col, num_bins):
    """Generates a column of categories, where each category is a bin.

    Parameters
    ----------
    col : np.ndarray
    
    Returns
    -------
    np.ndarray
    
    Examples
    --------
    >>> M = np.array([0.1, 3.0, 0.0, 1.2, 2.5, 1.7, 2])
    >>> generate_bin(M, 3)
    [0 3 0 1 2 1 2]

    """

    minimum = float(min(col))
    maximum = float(max(col))
    distance = float(maximum - minimum)
    return [int((x - minimum) / distance * num_bins) for x in col]

def normalize(col, mean=None, stddev=None, return_fit=False):
    """Generate a normalized column.
    
    Normalize both mean and std dev.
    
    Parameters
    ----------
    col : np.ndarray
    mean : float or None
        Mean to use for fit. If none, will use 0
    stddev : float or None
    return_fit : boolean
        If True, returns tuple of fitted col, mean, and standard dev of fit.
        If False, only returns fitted col
    Returns
    -------
    np.ndarray or (np.array, float, float)
    
    """
    # see infonavit for applying to different set than we fit on
    # https://github.com/dssg/infonavit-public/blob/master/pipeline_src/preprocessing.py#L99
    # Logic is from sklearn StandardScaler, but I didn't use sklearn because
    # I want to pass in mean and stddev rather than a fitted StandardScaler
    # https://github.com/scikit-learn/scikit-learn/blob/a95203b/sklearn/preprocessing/data.py#L276
    if mean is None:
        mean = np.mean(col)
    if stddev is None:
        stddev = np.std(col)
    res = (col - mean) / stddev
    if return_fit:
        return (res, mean, stddev)
    else:
        return res

def distance_from_point(lat_origin, lng_origin, lat_col, lng_col):
    """Generates a column of how far each record is from the origin
    
    Parameters
    ----------
    lat_origin : number
    lng_origin : number
    lat_col : np.ndarray
    lng_col : np.ndarray

    Returns
    -------
    np.ndarray

    """
    return distance(lat_origin, lng_origin, lat_col, lng_col)




