import os
import sys
import itertools as it
import numpy as np
import string
import diogenes.utils
from numpy.random import rand, seed
from contextlib import contextmanager
from StringIO import StringIO

TESTS_PATH = os.path.dirname(os.path.realpath(sys.argv[0]))
DATA_PATH = os.path.join(TESTS_PATH, 'data')
EIGHTS_PATH = os.path.join(TESTS_PATH, '..')

def path_of_data(filename):
    return os.path.join(DATA_PATH, filename)

def generate_test_matrix(n_rows, n_cols=1, n_classes=2, types=[], random_state=None):
    full_types = list(it.chain(types, it.repeat(float, n_cols - len(types))))
    np.random.seed(random_state)
    cols = []
    for col_type in full_types:
        if col_type is int:
            col = np.random.randint(100, size=rows)
        elif issubclass(col_type, basestring):
            col = np.random.choice(list(string.uppercase), size=n_rows)
        else:
            col = np.random.random(size=n_rows)
        cols.append(col)
    labels = np.random.randint(n_classes, size=n_rows)
    M = diogenes.utils.sa_from_cols(cols)
    return M, labels

def generate_correlated_test_matrix(n_rows):
    seed(0)
    M = rand(n_rows, 1)
    y = rand(n_rows) < M[:,0]
    return M, y

def array_equal(M1, M2, eps=1e-5, idx_col=None):
    """
    Test array equality. Similar to np.array_equal but with the following
    additional properties:

    * Floating point values tolerate a difference of eps
    * If two floating point cells are both nan, they are considered equal
    * Columns can be in different orders
    * If idx_col is not None, rows can be in different orders (rows will
      be uniquely identified by the value in the column specified by idx_col)
    """
    if M1.shape != M2.shape:
        return False
    frozenset_M1_names = frozenset(M1.dtype.names)
    frozenset_M2_names = frozenset(M2.dtype.names)
    if frozenset_M1_names != frozenset_M2_names:
        return False
    M2_reordered = M2[list(M1.dtype.names)]
    for (_, M1_dtype_str), (_, M2_dtype_str) in zip(M1.dtype.descr, 
                                                    M2_reordered.dtype.descr):
        if M1_dtype_str == M2_dtype_str:
            continue
        # Special case if one of them is 'O' and the other is 'S'
        if 'S' in M1_dtype_str and 'O' in M2_dtype_str:
            continue
        if 'O' in M1_dtype_str and 'S' in M2_dtype_str:
            continue
        import pdb; pdb.set_trace()
        return False
    if idx_col is not None:
        M1_idx_col = M1[idx_col]
        M2_idx_col = M2[idx_col]
        M1_idx_backindices = {}
        for row_num, idx in enumerate(M1_idx_col):
            M1_idx_backindices[idx] = row_num
        if len(M1_idx_backindices) != M1.shape[0]:
            raise ValueError('idx_col does not have unique indices in M1')
        M2_idx_backindices = {}
        for row_num, idx in enumerate(M2_idx_col):
            M2_idx_backindices[idx] = row_num
        if len(M2_idx_backindices) != M1.shape[0]:
            raise ValueError('idx_col does not have unique indices in M2')
        transitions = {}
        for idx in M1_idx_backindices:
            transitions[M1_idx_backindices[idx]] = M2_idx_backindices[idx]
        new_M2_order = [transitions[i] for i in xrange(M1.shape[0])]
        M2_reordered = M2_reordered[new_M2_order]
    print M1
    print M1.dtype
    print
    print M2_reordered
    print M2_reordered.dtype
    for col_name, col_type in M1.dtype.descr:
        M1_col = M1[col_name]
        M2_col = M2_reordered[col_name]
        if 'f' not in col_type:
            if not(np.array_equal(M1_col, M2_col)):
                return False
        else:
            if not (np.all(np.logical_or(
                    abs(M1_col - M2_col) < eps,
                    np.logical_and(np.isnan(M1_col), np.isnan(M2_col))))):
                return False
    return True

@contextmanager
def rerout_stdout():
    """
    print statements within the context are rerouted to a StringIO, which
    can be examined with the method that is yielded here.

    Examples
    --------
    >>> print 'This text appears in the console'
    This text appears in the console
    >>> with rerout_stdout() as get_rerouted_stdout:
    ...     print 'This text does not appear in the console'
    ...     # get_rerouted_stdout is a function that gets our rerouted output
    ...     assert(get_rerouted_stdout().strip() == 'This text does not appear in the console')
    >>> print 'This text also appears in the console'
    This text also appears in the console
    """
    # based on http://stackoverflow.com/questions/4219717/how-to-assert-output-with-nosetest-unittest-in-python
    saved_stdout = sys.stdout
    try:
        out = StringIO()
        sys.stdout = out
        yield out.getvalue
    finally:
        sys.stdout = saved_stdout

def print_in_box(heading, text):
    """ Prints text in a nice box. 

    Parameters
    ----------
    heading : str
    text : str or list of str
        if a list of str, each item of the list gets its own line
    """
    if isinstance(text, basestring):
        text = text.split('\n')
    str_len = max(len(heading), max([len(line) for line in text]))
    meta_fmt = ('{{border}}{{space}}'
                '{{{{content:{{fill}}{{align}}{str_len}}}}}'
                '{{space}}{{border}}\n').format(str_len=str_len)
    boundary = meta_fmt.format(
            fill='-', 
            align='^',
            border='+',
            space='-').format(
                    content='')
    heading_line = meta_fmt.format(
            fill='',
            align='^',
            border='|',
            space=' ').format(
                    content=heading)
    line_fmt = meta_fmt.format(
            fill='',
            align='<',
            border='|',
            space=' ')
    sys.stdout.write('\n')
    sys.stdout.write(boundary)
    sys.stdout.write(heading_line)
    sys.stdout.write(boundary.replace('-', '='))
    sys.stdout.write(''.join([line_fmt.format(content=line) for line in text]))
    sys.stdout.write(boundary)
    sys.stdout.write('\n')
