from uuid import uuid4
from collections import Counter

import numpy as np
from sklearn import preprocessing

from diogenes.utils import remove_cols, append_cols, distance,convert_to_sa

#convert below to a lambada
def col_random(M, number_to_select):
    num_col = len(M.dtypes.names)
    remove_these_columns = np.random.choice(num_col, number_to_select, replace=False)
    names = [col_names[i] for i in remove_these_columns]
    return names
    
def choose_cols_where(M, arguments):
    return
    
def remove_col_where(M, arguments):
    to_remove = np.ones(len(M.dtype), dtype=bool)
    for arg_set in arguments:
        lambd, vals = (arg_set['func'], arg_set['vals'])
        to_remove = np.logical_and(to_remove, lambd(M,  vals))
    remove_col_names = [col_name for col_name,included in zip(M.dtype.names, to_remove) if included] 
    return remove_cols(M, remove_col_names)

def col_val_eq(M, boundary):
    return [np.all(M[col_name] == boundary) for col_name in M.dtype.names]

def col_val_eq_any(M, boundary=None):
    return [np.all(M[col_name]==M[col_name][0]) for col_name in M.dtype.names]

def fewer_then_n_nonzero_in_col(M, boundary):
    return [len(np.where(M[col_name]!=0)[0])<2 for col_name in M.dtype.names]

def remove_rows_where(M, lamd, col_name, vals):
    to_remove = lamd(M, col_name, vals)
    to_keep = np.logical_not(to_remove)
    return M[to_keep]

#checks
# rewritten as lambda
def row_is_within_region(L, point):
    import matplotlib.path as mplPath
    bbPath = mplPath.Path(np.array(L))
    return bbPath.contains_point(point)
    


#write below diffently as lambda
def col_has_lt_threshold_unique_values(col, threshold):
    d = Counter(col)
    vals = sort(d.values())
    return ( sum(vals[:-1]) < threshold) 
    

def label_encode(M):
    """
    Changes string cols to integers so that there is a 1-1 mapping between 
    strings and ints
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



####Generate
def choose_rows_where(M, arguments, generated_name=None):
    if generated_name is None:
        generated_name = str(uuid4())
    to_select = np.ones(M.size, dtype=bool)
    for arg_set in arguments:
        lambd, col_name, vals = (arg_set['func'], arg_set['col_name'],
                                    arg_set['vals'])
        to_select = np.logical_and(to_select, lambd(M, col_name, vals))
    return append_cols(M, to_select, generated_name)

def row_is_outlier(M, col_name, boundary):
    std = np.std(M[col_name])
    mean = np.mean(M[col_name])
    return (np.logical_or( (mean-3*std)>M[col_name], (mean+3*std)<M[col_name]) )
    

def row_val_eq(M, col_name, boundary):
    return M[col_name] == boundary

def row_val_lt(M, col_name, boundary):
    return M[col_name] < boundary

def row_val_lt_TIME_EDITION(M, col_name, boundary):
    return M[col_name] < boundary

def row_val_gt(M, col_name, boundary):
    return M[col_name] > boundary

def row_val_between(M, col_name, boundary):
    return np.logical_and(boundary[0] <= M[col_name], M[col_name] <= boundary[1])



def generate_bin(col, num_bins):
    """Generates a column of categories, where each category is a bin.

    Parameters
    ----------
    col : np.array
    
    Returns
    -------
    np.array
    
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
    """
    
    Generate a normalized column.
    
    Normalize both mean and std dev.
    
    Parameters
    ----------
    col : np.array
    mean : float or None
        Mean to use for fit. If none, will use 0
    stddev : float or None
    return_fit : boolean
        If True, returns tuple of fitted col, mean, and standard dev of fit.
        If False, only returns fitted col
    Returns
    -------
    np.array or (np.array, float, float)
    
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
    """ Generates a column of how far each record is from the origin"""
    return distance(lat_origin, lng_origin, lat_col, lng_col)

@np.vectorize
def combine_sum(*args):
    return sum(args)

@np.vectorize
def combine_mean(*args):
    return np.mean(args)

def combine_cols(M, lambd, col_names, generated_name):
    new_col = lambd(*[M[name] for name in col_names])
    return append_cols(M, new_col, generated_name)



