#functions included here are those that you SHOULD be able to do in python syntax but can not.
import csv
import os
import itertools as it

import numpy as np
import pandas as pd
import numpy.lib.recfunctions as nprf
import matplotlib.mlab
from datetime import datetime
from dateutil.parser import parse

NOT_A_TIME = np.datetime64('NaT')

__special_csv_strings = {'': None,
                         'True': True,
                         'False': False} 

def __correct_csv_cell_type(cell):
    # Change strings in CSV to appropriate Python objects
    try:
        return __special_csv_strings[cell]
    except KeyError:
        pass
    try: 
        return int(cell)
    except ValueError:
        pass
    try:
        return float(cell)
    except ValueError:
        pass
    try:
        return parse(cell)
    except (TypeError, ValueError):
        pass
    return cell

def __open_csv_as_list(f, delimiter=',', header=True, col_names=None, return_col_names=False):
    # infers types
    reader = csv.reader(f,  delimiter=delimiter)
    if header:
        col_names = reader.next() # skip header
    data = [[__correct_csv_cell_type(cell) for cell in row] for
            row in reader]
    if col_names is None:
        col_names = ['f{}'.format(i) for i in xrange(len(data[0]))]
    if return_col_names:
        return data, col_names
    return data

def open_csv_as_sa(fin, delimiter=',', header=True, col_names=None):
    """Converts a csv to a structured array

    Parameters
    ----------
    fin : file-like object
        file-like object containing csv
    delimiter : str
        Character used to delimit csv fields
    header : bool
        If True, assumes the first line of the csv has column names
    col_names : list of str or None
        If header is False, this list will be used for column names

    Returns
    -------
    numpy.ndarray
        structured array corresponding to the csv

    If header is False and col_names is None, diogenes will assign
    arbitrary column names
    """
#    python_list, col_names = __open_csv_as_list(fin, delimiter, header, col_names, True)
#    return convert_to_sa(python_list, col_names)
    df = pd.read_csv(
            fin, 
            sep=delimiter, 
            header=0 if header else None,
            names=col_names,
            index_col=False)
    return df.to_records(index=False)

def utf_to_ascii(s):
    """Converts a unicode string to an ascii string"""
    # http://stackoverflow.com/questions/4299675/python-script-to-convert-from-utf-8-to-ascii
    if isinstance(s, unicode):
        return s.encode('ascii', 'replace')
    return s

@np.vectorize
def validate_time(date_text):
    """Returns boolean signifying whether a string is a valid datetime"""
    return __str_to_datetime(date_text) != NOT_A_TIME

def str_to_time(date_text):
    """Returns the datetime.datetime representation of a string

    Returns NOT_A_TIME if the string does not signify a valid datetime
    """
    return __str_to_datetime(date_text)

def transpose_dict_of_lists(dol):
    """Transforms a dictionary of lists into a list of dictionaries"""
    # http://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
    return (dict(it.izip(dol, x)) for 
            x in it.product(*dol.itervalues()))


def invert_dictionary(aDict):
    """Transforms a dict so that keys become values and values become keys"""
    return {v: k for k, v in aDict.items()}


TYPE_PRECEDENCE = {type(None): 0, 
                   bool: 100,
                   np.bool_: 101,
                   int: 200, 
                   long: 300,
                   np.int64: 301,
                   float: 400,
                   np.float64: 401,
                   str: 500,
                   np.string_: 501,
                   unicode: 600,
                   np.unicode_: 601,
                   datetime: 700,
                   np.datetime64: 701}

def __primitive_clean(cell, expected_type, alt):
    if cell == None:
        return alt
    try:
        return expected_type(cell)
    except (TypeError, ValueError):
        return alt

def __datetime_clean(cell):
    # Because, unlike primitives, we can't cast random objects to datetimes
    if isinstance(cell, datetime):
        return cell
    if isinstance(cell, basestring):
        return str_to_time(cell)
    return NOT_A_TIME

def __datetime64_clean(cell):
    try:
        # Not dealing with resolution. Everythin is us
        return np.datetime64(cell).astype('M8[us]')
    except (TypeError, ValueError):
        return NOT_A_TIME


CLEAN_FUNCTIONS = {type(None): lambda cell: '',
                   bool: lambda cell: __primitive_clean(cell, bool, False),
                   np.bool_: lambda cell: __primitive_clean(cell, np.bool_, 
                                                            np.bool_(False)),
                   int: lambda cell: __primitive_clean(cell, int, -999),
                   long: lambda cell: __primitive_clean(cell, long, -999L),
                   np.int64: lambda cell: __primitive_clean(cell, np.int64, 
                                                            np.int64(-999L)),
                   float: lambda cell: __primitive_clean(cell, float, np.nan),
                   np.float64: lambda cell: __primitive_clean(cell, np.float64, 
                                                              np.nan),
                   str: lambda cell: __primitive_clean(cell, str, ''),
                   np.string_: lambda cell: __primitive_clean(cell, np.string_,
                                                              np.string_('')),
                   unicode: lambda cell: __primitive_clean(cell, unicode, u''),
                   np.unicode_: lambda cell: __primitive_clean(
                       cell, 
                       np.unicode_,
                       np.unicode_('')),
                   datetime: __datetime_clean,
                   np.datetime64: __datetime64_clean}

STR_TYPE_LETTERS = {str: 'S',
                    np.string_: 'S',
                    unicode: 'U',
                    np.unicode_: 'U'}


def __str_to_datetime(s):
    # Invalid time if the string is too short
    # This prevents empty strings from being times
    # as well as odd short strings like 'a' 
    if len(s) < 6:
        return NOT_A_TIME
    # Invalid time if the string is just a number
    try: 
        float(s)
        return NOT_A_TIME
    except ValueError:
        pass
    # Invalid time if dateutil.parser.parse can't parse it
    try:
        return parse(s)
    except (TypeError, ValueError):
        return NOT_A_TIME

def __str_col_to_datetime(col):
    col_dtimes = [__str_to_datetime(s) for s in col]
    valid_dtimes = [dt for dt in col_dtimes if dt != NOT_A_TIME]
    # If there is even one valid datetime, we're calling this a datetime col
    return (bool(valid_dtimes), col_dtimes)

def cast_list_of_list_to_sa(L, col_names=None):
    """Transforms a list of lists to a numpy structured array

    Parameters
    ----------
    L : list of lists
        Signifies a table. Each inner list should have the same length
    col_names : list of str or None
        Names for columns. If unspecified, names will be arbitrarily chosen

    Returns
    -------
    numpy.ndarray
        Structured array

    """
    n_cols = len(L[0])
    if col_names is None:
        col_names = ['f{}'.format(i) for i in xrange(n_cols)]
    dtypes = []
    cleaned_cols = []
    for idx, col in enumerate(it.izip(*L)):
        dom_type = type(max(
            col, 
            key=lambda cell: TYPE_PRECEDENCE[type(cell)]))
        if dom_type in (bool, np.bool_, int, long, np.int64, float, 
                        np.float64):
            dtypes.append(dom_type)
            cleaned_cols.append(map(CLEAN_FUNCTIONS[dom_type], col))
        elif dom_type == datetime:
            dtypes.append('M8[us]')
            cleaned_cols.append(map(CLEAN_FUNCTIONS[dom_type], col))
        elif dom_type == np.datetime64:
            dtypes.append('M8[us]')
            cleaned_cols.append(map(CLEAN_FUNCTIONS[dom_type], col))
        elif dom_type in (str, unicode, np.string_, np.unicode_): 
            cleaned_col = map(CLEAN_FUNCTIONS[dom_type], col)
            is_datetime, dt_col = __str_col_to_datetime(cleaned_col)
            if is_datetime:
                dtypes.append('M8[us]')
                cleaned_cols.append(dt_col)
            else:
                max_len = max(
                        len(max(cleaned_col, 
                            key=lambda cell: len(dom_type(cell)))),
                        1)
                dtypes.append('|{}{}'.format(
                    STR_TYPE_LETTERS[dom_type],
                    max_len))
                cleaned_cols.append(cleaned_col)
        elif dom_type == type(None):
            # column full of None make it a column of empty strings
            dtypes.append('|S1')
            cleaned_cols.append([''] * len(col))
        else:
            raise ValueError(
                    'Type of col: {} could not be determined'.format(
                        col_names[idx]))

    return np.fromiter(it.izip(*cleaned_cols), 
                       dtype={'names': col_names, 
                              'formats': dtypes})

def convert_to_sa(M, col_names=None):
    """Converts an list of lists or a np ndarray to a Structured Arrray
    Parameters
    ----------
    M  : List of List or np.ndarray
       This is the Matrix M, that it is assumed is the basis for the ML algorithm
    col_names : list of str or None
        Column names for new sa. If M is already a structured array, col_names
        will be ignored. If M is not a structured array and col_names is None,
        names will be generated

    Returns
    -------
    np.ndarray
       structured array

    """
    if is_sa(M):
        return M

    if is_nd(M):
        return __cast_np_nd_to_sa(M, col_names=col_names)

    if isinstance(M, list):
        return cast_list_of_list_to_sa(M, col_names=col_names)
        # TODO make sure this function ^ ensures list of /lists/

    raise ValueError('Can\'t cast to sa')

__type_permissiveness_ranks = {'b': 0, 'M': 100, 'm': 100, 'i': 200, 'f': 300, 'S': 400}
def __type_permissiveness(dtype):
    # TODO handle other types
    return __type_permissiveness_ranks[dtype.kind] + dtype.itemsize

def np_dtype_is_homogeneous(A):
    """True iff dtype is nonstructured or every sub dtype is the same"""
    # http://stackoverflow.com/questions/3787908/python-determine-if-all-items-of-a-list-are-the-same-item
    if not is_sa(A):
        return True
    dtype = A.dtype
    first_dtype = dtype[0]
    return all(dtype[i] == first_dtype for i in xrange(len(dtype)))

def __cast_np_nd_to_sa(nd, dtype=None, col_names=None):
    """
    Returns a view of a numpy, single-type, 0, 1 or 2-dimensional array as a
    structured array
    Parameters
    ----------
    nd : numpy.ndarray
        The array to view
    dtype : numpy.dtype or None (optional)
        The type of the structured array. If not provided, or None, nd.dtype is
        used for all columns.
        If the dtype requested is not homogeneous and the datatype of each
        column is not identical nd.dtype, this operation may involve copying
        and conversion. Consequently, this operation should be avoided with
        heterogeneous or different datatypes.
    Returns
    -------
    A structured numpy.ndarray
    """
    if nd.ndim not in (0, 1, 2):
        raise TypeError('np_nd_to_sa only takes 0, 1 or 2-dimensional arrays')
    nd_dtype = nd.dtype
    if nd.ndim <= 1:
        nd = nd.reshape(nd.size, 1)
    if dtype is None:
        n_cols = nd.shape[1]
        if col_names is None: 
            col_names = map('f{}'.format, xrange(n_cols))
        dtype = np.dtype({'names': col_names,'formats': [nd_dtype for i in xrange(n_cols)]})
        return nd.reshape(nd.size).view(dtype)
    type_len = nd_dtype.itemsize
    #import pdb; pdb.set_trace()
    if all(dtype[i] == nd_dtype for i in xrange(len(dtype))):
        return nd.reshape(nd.size).view(dtype)
    #import pdb; pdb.set_trace()
    # if the user requests an incompatible type, we have to convert
    cols = (nd[:,i].astype(dtype[i]) for i in xrange(len(dtype))) 
    return np.array(it.izip(*cols), dtype=dtype)

def cast_np_sa_to_nd(sa):
    """
    
    Returns a view of a numpy structured array as a single-type 1 or
    2-dimensional array. If the resulting nd array would be a column vector,
    returns a 1-d array instead. If the resulting array would have a single
    entry, returns a 0-d array instead
    All elements are converted to the most permissive type. permissiveness
    is determined first by finding the most permissive type in the ordering:
    datetime64 < int < float < string
    then by selecting the longest typelength among all columns with with that
    type.
    If the sa does not have a homogeneous datatype already, this may require
    copying and type conversion rather than just casting. Consequently, this
    operation should be avoided for heterogeneous arrays
    Based on http://wiki.scipy.org/Cookbook/Recarray.

    Parameters
    ----------
    sa : numpy.ndarray
        The structured array to view

    Returns
    -------
    np.ndarray

    """
    if not is_sa(sa):
        return sa
    dtype = sa.dtype
    if len(dtype) == 1:
        if sa.size == 1:
            return sa.view(dtype=dtype[0]).reshape(())
        return sa.view(dtype=dtype[0]).reshape(len(sa))
    if np_dtype_is_homogeneous(sa):
        return sa.view(dtype=dtype[0]).reshape(len(sa), -1)
    # If type isn't homogeneous, we have to convert
    dtype_it = (dtype[i] for i in xrange(len(dtype)))
    most_permissive = max(dtype_it, key=__type_permissiveness)
    col_names = dtype.names
    cols = (sa[col_name].astype(most_permissive) for col_name in col_names)
    nd = np.column_stack(cols)
    return nd
    
def is_sa(M):
    """Returns True iff M is a structured array"""
    return is_nd(M) and M.dtype.names is not None

def is_nd(M):
    """Returns True iff M is a numpy.ndarray"""
    return isinstance(M, np.ndarray)

def distance(lat_1, lon_1, lat_2, lon_2):
    """Calculate the great circle distance between two points on earth 
    
    In Kilometers

    From:
    http://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points

    """
    # convert decimal degrees to radians 

    lon_1, lat_1, lon_2, lat_2 = map(np.radians, [lon_1, lat_1, lon_2, lat_2])

    # haversine formula 
    dlon = lon_2 - lon_1 
    dlat = lat_2 - lat_1 
    a = np.sin(dlat/2)**2 + np.cos(lat_1) * np.cos(lat_2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # 6371 Radius of earth in kilometers. Use 3956 for miles
    return c * r

def dist_less_than(lat_1, lon_1, lat_2, lon_2, threshold):
    """Tests whether distance between two points is less than a threshold

    Parameters
    ----------
    lat1 : float
    lon1 : float
    lat2 : float
    lon2 : float
    threshold : float
        max distance in kilometers

    Returns
    -------
    boolean 
    
    """
    return (distance(lat_1, lon_1, lat_2, lon_2) < threshold)

def stack_rows(*args):
    """Returns a structured array containing all the rows in its arguments
    
    Each argument must be a structured array with the same column names
    and column types. Similar to SQL UNION
    """
    return nprf.stack_arrays(args, usemask=False)

def sa_from_cols(cols):
    """Converts a list of columns to a structured array"""
    # TODO take col names
    return nprf.merge_arrays(cols, usemask=False)    

def append_cols(M, cols, col_names):
    """Append columns to an existing structured array

    Parameters
    ----------
    M : numpy.ndarray
        structured array
    cols : list of numpy.ndarray
    col_names : list of str
        names for new columns

    Returns
    -------
    numpy.ndarray
        structured array with new columns
    """
    return nprf.append_fields(M, col_names, data=cols, usemask=False)

def remove_cols(M, col_names):
    """Remove columns specified by col_names from structured array

    Parameters
    ----------
    M : numpy.ndarray
        structured array
    col_names : list of str
        names for columns to remove

    Returns
    -------
    numpy.ndarray
        structured array without columns
    """
    return nprf.drop_fields(M, col_names, usemask=False)

def __fill_by_descr(s):
    if 'b' in s:
        return False
    if 'i' in s:
        return -999
    if 'f' in s:
        return np.nan
    if 'S' in s:
        return ''
    if 'U' in s:
        return u''
    if 'M' in s or 'm' in s:
        return np.datetime64('NaT')
    raise ValueError('Unrecognized description {}'.format(s))

def join(left, right, how, left_on, right_on, suffixes=('_x', '_y')):
    """Does SQL-stype join between two numpy tables

    Supports equality join on an arbitrary number of columns

    Approximates Pandas DataFrame.merge
    http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.merge.html
    Implements a hash join 
    http://blogs.msdn.com/b/craigfr/archive/2006/08/10/687630.aspx

    Parameters
    ----------
    left : numpy.ndarray
        left structured array to join
    right : numpy.ndarray
        right structured array to join
    how : {'inner', 'outer', 'left', 'right'}
        As in SQL, signifies whether rows on one table should be included
        when they do not have matches in the other table.
    left_on : str or list or str
        names of column(s) for left table to join on. 
        If a list, the nth entry of left_on is joined to the nth entry of 
        right_on
    right_on : str or list or str
        names of column(s) for right table to join on
        If a list, the nth entry of left_on is joined to the nth entry of 
        right_on
    suffixes : (str, str)
        Suffixes to add to column names when left and right share column names

    Returns
    -------
    numpy.ndarray
        joined structured array

    """

    # left_on and right_on can both be strings or lists
    if isinstance(left_on, basestring):
        left_on = [left_on]
    if isinstance(right_on, basestring):
        right_on = [right_on]

    # assemble dtype for the merged array
    # Rules for naming columns in the new table, as inferred from Pandas:
    # 1. If a joined on column has the same name in both tables, it appears
    #    in the joined table once under that name (no suffix)
    # 2. Otherwise, every column from each table will appear in the joined
    #    table, whether they are joined on or not. If both tables share a 
    #    column name, the name will appear twice with suffixes. If a column
    #    name appears only in one table, it will appear without a suffix.
    frozenset_left_on = frozenset(left_on)
    frozenset_right_on = frozenset(right_on)
    frozenset_shared_on = frozenset_left_on.intersection(frozenset_right_on)
    shared_on = list(frozenset_shared_on)
    # get arrays without shared join columns
    left_names = left.dtype.names
    right_names = right.dtype.names
    frozenset_left_names = frozenset(left.dtype.names).difference(
            frozenset_shared_on)
    left_names = list(frozenset_left_names)
    frozenset_right_names = frozenset(right.dtype.names).difference(
            frozenset_shared_on)
    right_names = list(frozenset_right_names)
    left_no_idx = left[left_names]
    right_no_idx = right[right_names]
    left_names_w_suffix = [col_name + suffixes[0] if 
                           col_name in frozenset_right_names else
                           col_name for 
                           col_name in left_names]
    right_names_w_suffix = [col_name + suffixes[1] if 
                            col_name in frozenset_left_names else
                            col_name for 
                            col_name in right_names]
    col_names = (left_names_w_suffix + shared_on +  right_names_w_suffix)
    col_dtypes = ([left[left_col].dtype for left_col in left_names] +
                  [left[shared_on_col].dtype for shared_on_col in shared_on] +
                  [right[right_col].dtype for right_col in right_names])
    take_all_right_rows = how in ('outer', 'right')
    take_all_left_rows = how in ('outer', 'left')
    # data to fill in if we're doing an outer join and one of the sides is
    # missing
    left_fill = tuple([__fill_by_descr(dtype) for _, dtype in 
                       left_no_idx.dtype.descr])
    right_fill = tuple([__fill_by_descr(dtype) for _, dtype in 
                       right_no_idx.dtype.descr])

    # Make a hash of the first join column in the left table
    left_col = left[left_on[0]]
    hashed_col = {}
    for left_idx, left_cell in enumerate(left_col):
        try:
            rows = hashed_col[left_cell]
        except KeyError:
            rows = []
            hashed_col[left_cell] = rows
        rows.append(left_idx)

    # Pick out columns that we will be joining on beyond the 0th
    extra_left_cols = [left[left_on_name] for left_on_name in left_on[1:]]
    extra_right_cols = [right[right_on_name] for right_on_name in right_on[1:]]
    extra_contraint_cols = zip(extra_left_cols, extra_right_cols)

    rows_new_table = []
    right_col = right[right_on[0]]
    # keep track of used left rows so we can include all the rows if we're
    # doing a left or outer join
    left_rows_used = set()
    # Iterate through every row in the right table
    for right_idx, right_cell in enumerate(right_col):
        has_match = False
        # See if we have matches from the hashed col of the left table
        try:
            left_matches = hashed_col[right_cell]
            
            for left_idx in left_matches:
                # If all the constraints are met, we have a match
                if all([extra_left_col[left_idx] == extra_right_col[right_idx] 
                        for extra_left_col, extra_right_col in 
                        extra_contraint_cols]):
                    has_match = True
                    rows_new_table.append(
                            tuple(left_no_idx[left_idx]) + 
                            tuple([left[shared_on_col][left_idx] 
                                   for shared_on_col in shared_on]) +
                            tuple(right_no_idx[right_idx]))
                    left_rows_used.add(left_idx) 
        # No match found for this right row
        except KeyError:
            pass  
        # If we're doing a right or outer join and we didn't find a match, add
        # this row from the right table, filled with type-appropriate versions
        # of NULL from the left table
        if (not has_match) and take_all_right_rows:
            rows_new_table.append(left_fill + 
                    tuple([right[shared_on_col][right_idx] for shared_on_col in
                           shared_on]) + 
                    tuple(right_no_idx[right_idx]))

    # if we're doing a left or outer join, we have to add all rows from the 
    # left table, using type-appropriate versions of NULL for the right table
    if take_all_left_rows:    
        left_rows_unused = [i for i in xrange(len(left)) if i not in 
                            left_rows_used]
        for unused_left_idx in left_rows_unused:
            rows_new_table.append(
                    tuple(left_no_idx[unused_left_idx]) +
                    tuple([left[shared_on_col][unused_left_idx] 
                           for shared_on_col in shared_on]) +
                    right_fill)

    return np.array(rows_new_table, dtype={'names': col_names, 
                                           'formats': col_dtypes})

EPOCH = datetime.utcfromtimestamp(0)
def to_unix_time(dt):
    """Converts a datetime.datetime to seconds since epoch"""
    # TODO test this
    # from
    # http://stackoverflow.com/questions/6999726/how-can-i-convert-a-datetime-object-to-milliseconds-since-epoch-unix-time-in-p
    # and
    # http://stackoverflow.com/questions/29753060/how-to-convert-numpy-datetime64-into-datetime
    if isinstance(dt, np.datetime64):
        dt = dt.astype('O')
    if isinstance(dt, datetime):
        return (dt - EPOCH).total_seconds()
    return dt

def __sqlite_type(np_descr):
    if 'b' in np_descr:
        return 'BOOLEAN'
    if 'i' in np_descr:
        return 'INTEGER'
    if 'f' in np_descr:
        return 'REAL'
    if 'S' in np_descr:
        return 'TEXT'
    if 'M' in np_descr or 'm' in np_descr:
        return 'INTEGER'
    raise ValueError('No sqlite type found for np type: {}'.format(np_descr))

def __make_digestible_list_of_list(sa):
    res_cols = []
    for col_name, dtype in sa.dtype.descr:
        col = sa[col_name]
        if 'i' in dtype:
            res_cols.append([None if cell == -999 else cell for cell in
                             col])
        elif 'f' in dtype:
            res_cols.append([None if np.isnan(cell) else cell for cell in
                             col])
        elif 'm' in dtype or 'M' in dtype:
            res_cols.append([None if cell == NOT_A_TIME else
                             to_unix_time(cell) for cell in col])
        else:
            res_cols.append(col)
    return it.izip(*res_cols)

def csv_to_sql(conn, csv_path, table_name=None):
    """Converts a csv to a table in SQL
    
    Parameters
    ----------
    conn : sqlalchemy engine
        Connection to database
    csv_path : str
        Path to csv
    table_name : str or None
        Name of table to add to db. if None, will use the
        name of the csv with the .csv suffix stripped

    Returns
    -------
    str
        THe table name
    """


    # avoiding circular dependency
    from diogenes.read import open_csv
    if table_name is None:
        table_name = os.path.splitext(os.path.basename(csv_path))[0]
    sql_drop = 'DROP TABLE IF exists "{}"'.format(table_name)
    conn.execute(sql_drop)
    sa = open_csv(csv_path)
    col_names = sa.dtype.names
    sqlite_types = [__sqlite_type(np_descr) for _, np_descr in sa.dtype.descr]
    sql_create = 'CREATE TABLE "{}" ({})'.format(
            table_name,
            ', '.join(['{} {}'.format(col_name, sqlite_type) for
                       col_name, sqlite_type
                       in zip(col_names, sqlite_types)]))
    conn.execute(sql_create)
    data = __make_digestible_list_of_list(sa)
    sql_insert = 'INSERT INTO "{}" VALUES ({})'.format(
            table_name,
            ', '.join('?' * len(col_names)))
    for row in data:
        conn.execute(sql_insert, row)
    return table_name
