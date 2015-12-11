"""Tools to visualize data and display results"""

import os
import shutil
import StringIO
import cgi
import uuid
import abc
from datetime import datetime
from collections import Counter
import itertools as it

import numpy as np

from diogenes import utils
import matplotlib
if utils.on_headless_server():
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.dates
from matplotlib.pylab import boxplot 

from sklearn.grid_search import GridSearchCV
from sklearn.neighbors.kde import KernelDensity
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from sklearn.tree._tree import TREE_LEAF
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import pdfkit

from diogenes.grid_search import Experiment
from diogenes.utils import is_sa, is_nd, cast_np_sa_to_nd, convert_to_sa, cast_list_of_list_to_sa

if hasattr(plt, 'style'):
    # Make our plots pretty if matplotlib is new enough
    plt.style.use('ggplot')


def pprint_sa(M, row_labels=None, col_labels=None):
    """Prints a nicely formatted Structured array (or similar object) to console
    
    Parameters
    ----------
    M : numpy.ndarray or list of lists
        structured array or homogeneous array or list of lists to print
    row_labels : list or None
        labels to put in front of rows. Defaults to row number
    col_labels : list of str or None    
        names to label columns with. If M is a structured array, its column
        names will be used instead
        
    """
    M = utils.check_sa(M, col_names_if_converted=col_labels)
    if row_labels is None:
        row_labels = xrange(M.shape[0])
    col_labels = M.dtype.names
    # From http://stackoverflow.com/questions/9535954/python-printing-lists-as-tabular-data
    col_lens = [max(max([len('{}'.format(cell)) for cell in M[name]]), 
                len(name)) for name in col_labels]
    row_label_len = max([len('{}'.format(label)) for label in row_labels])
    row_format =('{{:>{}}} '.format(row_label_len) + 
                 ' '.join(['{{:>{}}}'.format(col_len) for col_len 
                           in col_lens]))
    print row_format.format("", *col_labels)
    for row_name, row in zip(row_labels, M):
        print row_format.format(row_name, *row)

__describe_cols_metrics = [('Count', len),
                           ('Mean', np.mean),
                           ('Standard Dev', np.std),
                           ('Minimum', min),
                           ('Maximum', max)]

__describe_cols_fill = [np.nan] * len(__describe_cols_metrics)

def describe_cols(M, verbose=True):
    """Returns summary statistics for a numpy array

    Parameters
    ----------
    M : numpy.ndarray
        structured array
       
    Returns
    -------
    numpy.ndarray
        structured array of summary statistics for M
       
    """ 
    M = utils.check_sa(M)           
    descr_rows = []
    for col_name, col_type in M.dtype.descr:
        if 'f' in col_type or 'i' in col_type:
            col = M[col_name]
            row = [col_name] + [func(col) for _, func in 
                                __describe_cols_metrics]
        else:
            row = [col_name] + __describe_cols_fill
        descr_rows.append(row)
    col_names = ['Column Name'] + [col_name for col_name, _ in 
                                   __describe_cols_metrics]
    ret = convert_to_sa(descr_rows, col_names=col_names)
    if verbose:
        pprint_sa(ret)
    return ret

def crosstab(col1, col2, verbose=True):
    """
    Makes a crosstab of col1 and col2. This is represented as a
    structured array with the following properties:

    1. The first column is the value of col1 being crossed
    2. The name of every column except the first is the value of col2 being
       crossed
    3. To find the number of cooccurences of x from col1 and y in col2,
       find the row that has 'x' in col1 and the column named 'y'. The 
       corresponding cell is the number of cooccurrences of x and y

    Parameters
    ----------
    col1 : np.ndarray
    col2 : np.ndarray

    Returns
    -------
    np.ndarray
        structured array

    """
    col1 = utils.check_col(col1, argument_name='col1')
    col2 = utils.check_col(col2, argument_name='col2')
    col1_unique = np.unique(col1)
    col2_unique = np.unique(col2)
    crosstab_rows = []
    for col1_val in col1_unique:
        loc_col1_val = np.where(col1==col1_val)[0]
        col2_vals = col2[loc_col1_val]
        cnt = Counter(col2_vals)
        counts = [cnt[col2_val] if cnt.has_key(col2_val) else 0 for col2_val 
                  in col2_unique]
        crosstab_rows.append(['{}'.format(col1_val)] + counts)
    col_names = ['col1_value'] + ['{}'.format(col2_val) for col2_val in 
                                  col2_unique]
    ret = convert_to_sa(crosstab_rows, col_names=col_names)
    if verbose:
        pprint_sa(ret)
    return ret


def plot_simple_histogram(col, verbose=True):
    """Makes a histogram of values in a column

    Parameters
    ----------
    col : np.ndarray
    verbose : boolean
        iff True, display the graph

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing plot

    """
    col = utils.check_col(col)
    hist, bins = np.histogram(col, bins=50)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    f = plt.figure()
    plt.bar(center, hist, align='center', width=width)
    if verbose:
        plt.show()
    return f

# all of the below take output from any func in grid_search or operate

def plot_prec_recall(labels, score, title='Prec/Recall', verbose=True):
    """Plot precision/recall curve

    Parameters
    ----------
    labels : np.ndarray
        vector of ground truth
    score : np.ndarray
        vector of scores assigned by classifier (i.e. 
        clf.pred_proba(...)[-1] in sklearn)
    title : str
        title of plot
    verbose : boolean
        iff True, display the graph
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing plot

    """
    labels = utils.check_col(labels, argument_name='labels')
    score = utils.check_col(score, argument_name='score')
    # adapted from Rayid's prec/recall code
    y_true = labels
    y_score = score
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
        y_true, 
        y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    fig = plt.figure()
    ax1 = plt.gca()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    plt.title(title)
    if verbose:
        fig.show()
    return fig

def plot_roc(labels, score, title='ROC', verbose=True):
    """Plot ROC curve

    Parameters
    ----------
    labels : np.ndarray
        vector of ground truth
    score : np.ndarray
        vector of scores assigned by classifier (i.e. 
        clf.pred_proba(...)[-1] in sklearn)
    title : str
        title of plot
    verbose : boolean
        iff True, display the graph
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing plot

    """
    labels = utils.check_col(labels, argument_name='labels')
    score = utils.check_col(score, argument_name='score')
    # adapted from Rayid's prec/recall code
    fpr, tpr, thresholds = roc_curve(labels, score)
    fpr = fpr
    tpr = tpr
    pct_above_per_thresh = []
    number_scored = len(score)
    for value in thresholds:
        num_above_thresh = len(score[score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)

    fig = plt.figure()
    ax1 = plt.gca()
    ax1.plot(pct_above_per_thresh, fpr, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('fpr', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, tpr, 'r')
    ax2.set_ylabel('tpr', color='r')
    plt.title(title)
    if verbose:
        fig.show()
    return fig

def plot_box_plot(col, title=None, verbose=True):
    """Makes a box plot for a feature
    
    Parameters
    ----------
    col : np.array
    title : str or None
        title of a plot
    verbose : boolean
        iff True, display the graph
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing plot
    
    """
    col = utils.check_col(col)

    fig = plt.figure()
    boxplot(col)
    if title:
        plt.title(title)
    #add col_name to graphn
    if verbose:
        plt.show()
    return fig

def get_top_features(clf, M=None, col_names=None, n=10, verbose=True):
    """Gets the top features for a fitted clf

    Parameters
    ----------
    clf : sklearn.base.BaseEstimator
        Fitted classifier with a feature_importances_ attribute
    M : numpy.ndarray or None
        Structured array corresponding to fitted clf. Used here to deterimine
        column names
    col_names : list of str or None
        List of column names corresponding to fitted clf.
    n : int
        Number of features to return
    verbose : boolean
        iff True, prints ranked features

    Returns
    -------
    numpy.ndarray
        structured array with top feature names and scores

    """
    if not isinstance(clf, BaseEstimator):
        raise ValueError('clf must be an instance of sklearn.Base.BaseEstimator')


    scores = clf.feature_importances_
    if col_names is None:
        if is_sa(M):
            col_names = M.dtype.names
        else:
            col_names = ['f{}'.format(i) for i in xrange(len(scores))]
    else:
        col_names = utils.check_col_names(col_names, n_cols = scores.shape[0])
    ranked_name_and_score = [(col_names[x], scores[x]) for x in 
                             scores.argsort()[::-1]]
    ranked_name_and_score = convert_to_sa(
            ranked_name_and_score[:n], 
            col_names=('feat_name', 'score'))
    if verbose:
        pprint_sa(ranked_name_and_score)
    return ranked_name_and_score

# TODO features form top % of clfs

def get_roc_auc(labels, score, verbose=True):
    """return area under ROC curve

    Parameters
    ----------
    labels : np.ndarray
        vector of ground truth
    score : np.ndarray
        vector of scores assigned by classifier (i.e. 
        clf.pred_proba(...)[-1] in sklearn)
    verbose : boolean
        iff True, prints area under the curve
        
    Returns
    -------
    float
        area under the curve

    """
    labels = utils.check_col(labels, argument_name='labels')
    score = utils.check_col(score, argument_name='score')
    auc_score = roc_auc_score(labels, score)
    if verbose:
        print 'ROC AUC: {}'.format(auc_score)
    return auc_score

def plot_correlation_matrix(M, verbose=True):
    """Plot correlation between variables in M
    
    Parameters
    ----------
    M : numpy.ndarray
        structured array
    verbose : boolean
        iff True, display the graph

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing plot
    
    """
    # http://glowingpython.blogspot.com/2012/10/visualizing-correlation-matrices.html
    # TODO work on structured arrays or not
    # TODO ticks are col names
    M = utils.check_sa(M)
    names = M.dtype.names
    M = cast_np_sa_to_nd(M)
    
    #set rowvar =0 for rows are items, cols are features
    cc = np.corrcoef(M, rowvar=0)
    
    fig = plt.figure()
    plt.pcolor(cc)
    plt.colorbar()
    plt.yticks(np.arange(0.5, M.shape[1] + 0.5), range(0, M.shape[1]))
    plt.xticks(np.arange(0.5, M.shape[1] + 0.5), range(0, M.shape[1]))
    if verbose:
        plt.show()
    return fig

def plot_correlation_scatter_plot(M, verbose=True):
    """Makes a grid of scatter plots representing relationship between variables
    
    Each scatter plot is one variable plotted against another variable
    
    Parameters
    ----------
    M : numpy.ndarray
        structured array
    verbose : boolean
        iff True, display the graph

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing plot
    
    """
    # TODO work for all three types that M might be
    # TODO ignore classification variables
    # adapted from the excellent 
    # http://stackoverflow.com/questions/7941207/is-there-a-function-to-make-scatterplot-matrices-in-matplotlib
    
    M = utils.check_sa(M)

    numdata = M.shape[0]
    numvars = len(M.dtype)
    names = M.dtype.names
    fig, axes = plt.subplots(numvars, numvars)
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for ax in axes.flat:
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # Set up ticks only on one side for the "edge" subplots...
        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')

    # Plot the M.
    for i, j in zip(*np.triu_indices_from(axes, k=1)):
        for x, y in [(i,j), (j,i)]: 
            axes[x,y].plot(M[M.dtype.names[x]], M[M.dtype.names[y]], '.')

    # Label the diagonal subplots...
    for i, label in enumerate(names):
        axes[i,i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                ha='center', va='center')

    # Turn on the proper x or y axes ticks.
    for i, j in zip(range(numvars), it.cycle((-1, 0))):
        axes[j,i].xaxis.set_visible(True)
        axes[i,j].yaxis.set_visible(True)
    if verbose:
        plt.show()
    return fig

def plot_kernel_density(col, verbose=True): 
    """Plots kernel density function of column

    From: 
    https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/

    Parameters
    ----------
    col : np.ndarray
    verbose : boolean
        iff True, display the graph

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing plot

    """
    #address pass entire matrix
    # TODO respect missing_val
    # TODO what does n do?
    col = utils.check_col(col)
    x_grid = np.linspace(min(col), max(col), 1000)

    grid = GridSearchCV(KernelDensity(), {'bandwidth': np.linspace(0.1,1.0,30)}, cv=20) # 20-fold cross-validation
    grid.fit(col[:, None])

    kde = grid.best_estimator_
    pdf = np.exp(kde.score_samples(x_grid[:, None]))

    fig, ax = plt.subplots()
    #fig = plt.figure()
    ax.plot(x_grid, pdf, linewidth=3, alpha=0.5, label='bw=%.2f' % kde.bandwidth)
    ax.hist(col, 30, fc='gray', histtype='stepfilled', alpha=0.3, normed=True)
    ax.legend(loc='upper left')
    ax.set_xlim(min(col), max(col))
    if verbose:
        plt.show()
    return fig

def plot_on_timeline(col, verbose=True):
    """Plots points on a timeline
    
    Parameters
    ----------
    col : np.array
    verbose : boolean
        iff True, display the graph

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing plot

    
    Returns
    -------
    matplotlib.figure.Figure
    """
    col = utils.check_col(col)
    # http://stackoverflow.com/questions/1574088/plotting-time-in-python-with-matplotlib
    if is_nd(col):
        col = col.astype(datetime)
    dates = matplotlib.dates.date2num(col)
    fig = plt.figure()
    plt.plot_date(dates, [0] * len(dates))
    if verbose:
        plt.show()
    return fig
    
def _feature_pair_report(pair_and_values,
                         description='pairs', 
                         measurement='value',
                         note=None,
                         n=10):
    print '-' * 80
    print description
    print '-' * 80
    print 'feature pair : {}'.format(measurement)
    for pair, value in it.islice(pair_and_values, n):
        print '{} : {}'.format(pair, value)
    if note is not None:
        print '* {}'.format(note)
    print


def feature_pairs_in_tree(dt):
    """Lists subsequent features sorted by importance

    Parameters
    ----------
    dt : sklearn.tree.DecisionTreeClassifer

    Returns
    -------
    list of list of tuple of int :
        Going from inside to out:

        1. Each int is a feature that a node split on
    
        2. If two ints appear in the same tuple, then there was a node
           that split on the second feature immediately below a node
           that split on the first feature

        3. Tuples appearing in the same inner list appear at the same
           depth in the tree

        4. The outer list describes the entire tree

    """
    if not isinstance(dt, DecisionTreeClassifier):
        raise ValueError('dt must be an sklearn.tree.DecisionTreeClassifier')
    t = dt.tree_
    feature = t.feature
    children_left = t.children_left
    children_right = t.children_right
    result = []
    if t.children_left[0] == TREE_LEAF:
        return result
    next_queue = [0]
    while next_queue:
        this_queue = next_queue
        next_queue = []
        results_this_depth = []
        while this_queue:
            node = this_queue.pop()
            left_child = children_left[node]
            right_child = children_right[node]
            if children_left[left_child] != TREE_LEAF:
                results_this_depth.append(tuple(sorted(
                    (feature[node], 
                     feature[left_child]))))
                next_queue.append(left_child)
            if children_left[right_child] != TREE_LEAF:
                results_this_depth.append(tuple(sorted(
                    (feature[node], 
                     feature[right_child]))))
                next_queue.append(right_child)
        result.append(results_this_depth)
    result.pop() # The last results are always empty
    return result
    

def feature_pairs_in_rf(rf, weight_by_depth=None, verbose=True, n=10):
    """Describes the frequency of features appearing subsequently in each tree
    in a random forest
    
    Parameters
    ----------
    rf : sklearn.ensemble.RandomForestClassifier
        Fitted random forest
    weight_by_depth : iterable or None
        Weights to give to each depth in the "occurences weighted by depth
        metric"

        weight_by_depth is a vector. The 0th entry is the weight of being at
        depth 0; the 1st entry is the weight of being at depth 1, etc.
        If not provided, wdiogenes are linear with negative depth. If 
        the provided vector is not as long as the number of depths, then 
        remaining depths are weighted with 0
    verbose : boolean
        iff True, prints metrics to console
    n : int
        Prints the top-n-scoring feature pairs to console if verbose==True

    Returns
    -------
    (collections.Counter, list of collections.Counter, dict, dict)
        A tuple with a number of metrics

        1. A Counter of cooccuring feature pairs at all depths
        2. A list of Counters of feature pairs. Element 0 corresponds to 
           depth 0, element 1 corresponds to depth 1 etc.
        3. A dict where keys are feature pairs and values are the average
           depth of those feature pairs
        4. A dict where keys are feature pairs and values are the number
           of occurences of those feature pairs weighted by depth
        
    """
    if not isinstance(rf, RandomForestClassifier):
        raise ValueError(
            'rf must be an sklearn.Ensemble.RandomForestClassifier')

    pairs_by_est = [feature_pairs_in_tree(est) for est in rf.estimators_]
    pairs_by_depth = [list(it.chain(*pair_list)) for pair_list in 
                      list(it.izip_longest(*pairs_by_est, fillvalue=[]))]
    pairs_flat = list(it.chain(*pairs_by_depth))
    depths_by_pair = {}
    for depth, pairs in enumerate(pairs_by_depth):
        for pair in pairs:
            try:
                depths_by_pair[pair] += [depth]
            except KeyError:
                depths_by_pair[pair] = [depth]
    counts_by_pair=Counter(pairs_flat)
    count_pairs_by_depth = [Counter(pairs) for pairs in pairs_by_depth]

    depth_len = len(pairs_by_depth)
    if weight_by_depth is None:
        weight_by_depth = [(depth_len - float(depth)) / depth_len for depth in
                           xrange(depth_len)]
    weight_filler = it.repeat(0.0, depth_len - len(weight_by_depth))
    wdiogenes = list(it.chain(weight_by_depth, weight_filler))
    
    average_depth_by_pair = {pair: float(sum(depths)) / len(depths) for 
                             pair, depths in depths_by_pair.iteritems()}

    weighted = {pair: sum([wdiogenes[depth] for depth in depths])
                for pair, depths in depths_by_pair.iteritems()}

    if verbose:
        print '=' * 80
        print 'RF Subsequent Pair Analysis'
        print '=' * 80
        print
        _feature_pair_report(
                counts_by_pair.most_common(), 
                'Overall Occurrences', 
                'occurrences',
                n=n)
        _feature_pair_report(
                sorted([item for item in average_depth_by_pair.iteritems()], 
                       key=lambda item: item[1]),
                'Average depth',
                'average depth',
                'Max depth was {}'.format(depth_len - 1),
                n=n)
        _feature_pair_report(
                sorted([item for item in weighted.iteritems()], 
                       key=lambda item: item[1]),
                'Occurrences weighted by depth',
                'sum weight',
                'Wdiogenes for depth 0, 1, 2, ... were: {}'.format(wdiogenes),
                n=n)

        for depth, pairs in enumerate(count_pairs_by_depth):
            _feature_pair_report(
                    pairs.most_common(), 
                    'Occurrences at depth {}'.format(depth), 
                    'occurrences',
                    n=n)


    return (counts_by_pair, count_pairs_by_depth, average_depth_by_pair, 
            weighted)



class ReportError(Exception):
    """Error generated by Report"""
    pass

class Report(object):
    """Creates pdf reports.

    Reports can either be associated with a particular 
    diogenes.grid_search.experiment.Experiment or it can simply be used as
    a way to concatenate figures, text, and tables

    Parameters
    ----------
    exp : diogenes.grid_search.experiment.Experiment or None
        Experiment used to make figures. Optional.
    report_path : path of the generated pdf

    """

    def __init__(self, exp=None, report_path='report.pdf'):
        self.__exp = exp
        if exp is not None:
            self.__back_indices = {trial: i for i, trial in enumerate(exp.trials)}
        self.__objects = []
        self.__tmp_folder = 'diogenes_temp'
        if not os.path.exists(self.__tmp_folder):
            os.mkdir(self.__tmp_folder)
        self.__html_src_path = os.path.join(self.__tmp_folder, 
                                            '{}.html'.format(uuid.uuid4()))
        self.__report_path = report_path

    def __html_escape(self, s):
        """Returns a string with all its html-averse characters html escaped"""
        return cgi.escape(s).encode('ascii', 'xmlcharrefreplace')
        
    def __html_format(self, fmt, *args, **kwargs):
        clean_args = [self.__html_escape(str(arg)) for arg in args]
        clean_kwargs = {key: self.__html_escape(str(kwargs[key])) for 
                        key in kwargs}
        return fmt.format(*clean_args, **clean_kwargs)

    def to_pdf(self, options={}, verbose=True):
        """Generates a pdf

        Parameters
        ----------
        options : dict
            options are pdfkit.from_url options. See 
            https://pypi.python.org/pypi/pdfkit
        verbose : bool
            iff True, gives output regarding pdf creation 

        Returns
        -------
        Path of generated pdf
        """

        if verbose:
            print 'Generating report...'
        with open(self.__html_src_path, 'w') as html_out:
            html_out.write(self.__get_header())
            html_out.write('\n'.join(self.__objects))
            html_out.write(self.__get_footer())
        if not verbose:
            options['quiet'] = ''
        pdfkit.from_url(self.__html_src_path, self.__report_path, 
                        options=options)
        report_path = self.get_report_path()
        if verbose:
            print 'Report written to {}'.format(report_path)
        return report_path

    def __np_to_html_table(self, sa, fout, show_shape=False):
        if show_shape:
            fout.write('<p>table of shape: ({},{})</p>'.format(
                len(sa),
                len(sa.dtype)))
        fout.write('<p><table>\n')
        header = '<tr>{}</tr>\n'.format(
            ''.join(
                    [self.__html_format(
                        '<th>{}</th>',
                        name) for 
                     name in sa.dtype.names]))
        fout.write(header)
        data = '\n'.join(
            ['<tr>{}</tr>'.format(
                ''.join(
                    [self.__html_format(
                        '<td>{}</td>',
                        cell) for
                     cell in row])) for
             row in sa])
        fout.write(data)
        fout.write('\n')
        fout.write('</table></p>')


    def get_report_path(self):
        """Returns path of generated pdf"""
        return os.path.abspath(self.__report_path)

    def __get_header(self):
        # Thanks to http://stackoverflow.com/questions/13516534/how-to-avoid-page-break-inside-table-row-for-wkhtmltopdf
        # For not page breaking in the middle of tables
        return ('<!DOCTYPE html>\n'
                '<html>\n'
                '<head>\n'
                '<style>\n'
                'table td, th {\n'
                '    border: 1px solid black;\n'
                '}\n'
                'table {\n'
                '    border-collapse: collapse;\n'
                '}\n'
                'tr:nth-child(even) {\n'
                '    background: #CCC\n'
                '}\n'
                'tr:nth-child(odd) {\n'
                '    background: white\n'
                '}\n'
                'table, tr, td, th, tbody, thead, tfoot {\n'
                '    page-break-inside: avoid !important;\n'
                '}\n' 
                '</style>\n'
                '</head>\n'
                '<body>\n')

    def add_subreport(self, subreport):    
        """Incorporates another Report into this one

        Parameters
        ----------
        subreport : Report
            report to add

        """
        self.__objects += subreport.__objects

    def __get_footer(self):
        return '\n</body>\n</html>\n'

    def add_heading(self, heading, level=2):
        """Adds a heading to the report

        Parameters
        ----------
        heading : str
            text of heading
        level : int
            heading level (1 corresponds to html h1, 2 corresponds to 
            html h2, etc)

        """
        self.__objects.append(self.__html_format(
            '<h{}>{}</h{}>',
            level,
            heading,
            level))

    def add_text(self, text):
        """Adds block of text to report"""
        self.__objects.append(self.__html_format(
                    '<p>{}</p>',
                    text))

    def add_table(self, M):
        """Adds structured array to report"""
        M = utils.check_sa(M)
        sio = StringIO.StringIO()
        self.__np_to_html_table(M, sio)
        self.__objects.append(sio.getvalue())

    def add_fig(self, fig):
        """Adds matplotlib.figure.Figure to report"""
        # So we don't get pages with nothing but one figure on them
        fig.set_figheight(5.0)
        filename = 'fig_{}.png'.format(str(uuid.uuid4()))
        path = os.path.join(self.__tmp_folder, filename)
        fig.savefig(path)
        self.__objects.append('<img src="{}">'.format(filename))

    def add_summary_graph(self, measure):
        """Adds a graph to report that summarizes across an Experiment

        Parameters
        ----------
        measure : str
            Function of Experiment to call. The function must return a dict of 
            Trial: score. Examples are 'average_score' and 'roc_auc'
        """

        if self.__exp is None:
            raise ReportError('No experiment provided for this report. '
                              'Cannot add summary graphs.')
        results = [(trial, score, self.__back_indices[trial]) for 
                   trial, score in getattr(self.__exp, measure)().iteritems()]
        results_sorted = sorted(
                results, 
                key=lambda result: result[1],
                reverse=True)
        y = [result[1] for result in results_sorted]
        x = xrange(len(results))
        fig = plt.figure()
        plt.bar(x, y)
        fig.set_size_inches(8, fig.get_size_inches()[1])
        maxy = max(y)
        for rank, result in enumerate(results_sorted):
            plt.text(rank, result[1], '{}'.format(result[2]))
        plt.ylabel(measure)
        self.add_fig(fig)
        plt.close()

    def add_summary_graph_roc_auc(self):
        """Adds a graph to report that summarizes roc_auc across Experiment"""
        self.add_summary_graph('roc_auc')

    def add_summary_graph_average_score(self):
        """Adds a graph to report that summarizes average_score across Experiment
        """
        self.add_summary_graph('average_score')

    def add_graph_for_best(self, func_name):
        """Adds a graph to report that gives performance of the best Trial

        Parameters
        ----------
        func_name : str
            Name of a function that can be run on a Trial that returns a 
            figure. For example 'roc_curve' or 'prec_recall_curve'
        """
        if self.__exp is None:
            raise ReportError('No experiment provided for this report. '
                              'Cannot add graph for best trial.')
        best_trial = max(
            self.__exp.trials, 
            key=lambda trial: trial.average_score())
        fig = getattr(best_trial, func_name)()
        self.add_fig(fig)
        self.add_text('Best trial is trial {} ({})]'.format(
            self.__back_indices[best_trial],
            best_trial))
        plt.close()

    def add_graph_for_best_roc(self):
        """Adds roc curve for best Trial in an experiment"""
        self.add_graph_for_best('roc_curve')

    def add_graph_for_best_prec_recall(self):
        """Adds prec/recall for best Trial in an experiment"""
        self.add_graph_for_best('prec_recall_curve')

    def add_legend(self):
        """
        Adds a legend that shows which trial number in a summary graph
        corresponds to which Trial
        """
        if self.__exp is None:
            raise ReportError('No experiment provided for this report. '
                              'Cannot add legend.')
        list_of_tuple = [(str(i), str(trial)) for i, trial in 
                         enumerate(self.__exp.trials)]
        table = cast_list_of_list_to_sa(list_of_tuple, col_names=('Id', 'Trial'))
        # display 10 at a time to give pdfkit an easier time with page breaks
        start_row = 0
        n_trials = len(list_of_tuple)
        while start_row < n_trials:
            self.add_table(table[start_row:start_row+9])
            start_row += 9 


