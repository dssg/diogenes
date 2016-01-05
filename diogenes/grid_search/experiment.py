"""Provides classes necessary for organizing an Experiment"""

import inspect 
import json
import copy
import abc
import datetime
import csv
import os
import itertools as it
from collections import Counter
from multiprocessing import cpu_count

import numpy as np
from joblib import Parallel, delayed

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from sklearn.base import BaseEstimator
from sklearn.cross_validation import _PartitionIterator

from diogenes.grid_search import subset as s_i
from diogenes.grid_search import partition_iterator as p_i
import diogenes.utils as utils
from diogenes.utils import remove_cols


def _run_trial(trial):
    return trial.run()

class Experiment(object):
    """Class to execute and organize grid searches.

    Several of the init arguments are of type list of dict. Experiment 
    expects these to be in a particular format:

    [{CLASS_SPECIFIER: CLASS_1, 
      CLASS_1_PARAM_1: [CLASS_1_PARAM_1_VALUE_1, CLASS_1_PARAM_1_VALUE_2, ...
                        CLASS_1_PARAM_1_VALUE_N],
      CLASS_1_PARAM_2: [CLASS_1_PARAM_2_VALUE_1, CLASS_1_PARAM_2_VALUE_2, ...
                        CLASS_1_PARAM_2_VALUE_N],
      ...
      CLASS_1_PARAM_M: [CLASS_1_PARAM_M_VALUE_1, CLASS_1_PARAM_M_VALUE_2, ...
                        CLASS_1_PARAM_M_VALUE_N]},
     {CLASS_SPECIFIER: CLASS_2, 
      CLASS_2_PARAM_1: [CLASS_2_PARAM_1_VALUE_1, CLASS_2_PARAM_1_VALUE_2, ...
                        CLASS_2_PARAM_1_VALUE_N],
      CLASS_2_PARAM_2: [CLASS_2_PARAM_2_VALUE_1, CLASS_2_PARAM_2_VALUE_2, ...
                        CLASS_2_PARAM_2_VALUE_N],
      ...
      CLASS_2_PARAM_M: [CLASS_2_PARAM_M_VALUE_1, CLASS_2_PARAM_M_VALUE_2, ...
                        CLASS_2_PARAM_M_VALUE_N]},
     ...
     {CLASS_SPECIFIER: CLASS_L, 
      CLASS_L_PARAM_1: [CLASS_L_PARAM_1_VALUE_1, CLASS_L_PARAM_1_VALUE_2, ...
                        CLASS_L_PARAM_1_VALUE_N],
      CLASS_L_PARAM_2: [CLASS_L_PARAM_2_VALUE_1, CLASS_L_PARAM_2_VALUE_2, ...
                        CLASS_L_PARAM_2_VALUE_N],
      ...
      CLASS_L_PARAM_M: [CLASS_L_PARAM_M_VALUE_1, CLASS_L_PARAM_M_VALUE_2, ...
                        CLASS_L_PARAM_M_VALUE_N]}]

    CLASS_SPECIFIER is a different string constant for each argument. for 
    clfs, it is 'clf'. For subsets, it is 'subset', and for cvs it is 'cv'.

    CLASS_* is a class object which will be used to either classify data
    (in clfs), take a subset of data (in subsets) or specify train/test
    splits (in cvs). In clfs, it should be a subclass
    of sklearn.base.BaseEstimator. In subsets, it should be a 
    subclass of diogenes.grid_search.subset.BaseSubsetIter. In cvs, it
    should be a subclass of sklearn.cross_validation._PartitionIterator

    CLASS_X_PARAM_* is an init argument of CLASS_X. For example, if CLASS_1
    is sklearn.ensemble.RandomForest, CLASS_1_PARAM_1 might be the string
    literal 'n_estimators' or the string literal 'n_features'

    CLASS_X_PARAM_Y_VALUE_* is a single value to try as the argument for
    CLASS_X_PARAM. For example, if CLASS_1_PARAM_1 is 'n_estimators',
    then CLASS_1_PARAM_1_VALUE_1 could be 10 and CLASS_1_PARAM_1_VALUE_1
    could be 100.

    When we run the Experiment with Experiment.run, Experiment will create
    a Trial for each element in the cartesian product of all parameters
    for each class. if we have {'clf' : RandomForestEstimator, 
    'n_estimators' : [10, 100], 'n_features': [3, 5]} Then there will be
    a Trial for each of RandomForestEstimator(n_estimators=10, n_features=3),
    RandomForestEstimator(n_estimators=100, n_features=3),
    RandomForestEstimator(n_estimators=10, n_features=5), and
    RandomForestEstimator(n_estimators=100, n_features=5).

    For examples of how to create these arguments, look at
    diogenes.grid_search.standard_clfs.py.

    Parameters
    ----------
    M : numpy.ndarray
        structured array corresponding to features to experiment on
    labels : numpy.ndarray
        vector of labels
    clfs : list of dict
        classifiers to run
    subsets : list of dict
        subsetting operations to perform
    cvs : list of dict
        directives to make train and test sets
    trials : list or Trial or None  
        If a number of Trials have already been run, and we just want to
        collect them into an Experiment rather than starting from scratch,
        Experiment can be initiazed with a list of already run Trials
    
    Attributes
    ----------
    trials : list of Trial
        Trials corresponding to this experiment.

    """
    def __init__(
            self, 
            M, 
            labels, 
            clfs=[{'clf': RandomForestClassifier}], 
            subsets=[{'subset': s_i.SubsetNoSubset}], 
            cvs=[{'cv': KFold}],
            trials=None):
        if M is not None:
            if utils.is_nd(M) and not utils.is_sa(M):
                # nd_array, short circuit the usual type checking and coersion
                if M.ndim != 2:
                    raise ValueError('Expected 2-dimensional array for M')
                self.M = M
                self.col_names = ['f{}'.format(i) for i in xrange(M.shape[1])]
                self.labels = utils.check_col(
                        labels, 
                        n_rows=M.shape[0], 
                        argument_name='labels')
            else:    
                # M is either a structured array or something that should
                # be converted
                (M, self.labels) = utils.check_consistent(
                        M, 
                        labels, 
                        col_argument_name='labels')
                self.col_names = M.dtype.names
                self.M = utils.cast_np_sa_to_nd(M)
        else:
            self.col_names = None
        if trials is None:
            clfs = utils.check_arguments(
                    clfs, 
                    {'clf': lambda clf: issubclass(clf, BaseEstimator)},
                    optional_keys_take_lists=True,
                    argument_name='clfs')
            subsets = utils.check_arguments(
                    subsets,
                    {'subset': lambda subset: issubclass(subset, s_i.BaseSubsetIter)},
                    optional_keys_take_lists=True,
                    argument_name='subsets')
            cvs = utils.check_arguments(
                    cvs,
                    {'cv': lambda cv: issubclass(cv, _PartitionIterator)},
                    optional_keys_take_lists=True,
                    argument_name='cvs')
        self.clfs = clfs
        self.subsets = subsets
        self.cvs = cvs
        self.trials = trials

    def __repr__(self):
        return 'Experiment(clfs={}, subsets={}, cvs={})'.format(
                self.clfs, 
                self.subsets, 
                self.cvs)

        
    def __run_all_trials(self, trials):
        # TODO parallelize on Runs too
        return Parallel(n_jobs=cpu_count())(delayed(_run_trial)(t) 
                                           for t in trials)
        #return [_run_trial(t) for t in trials]

    def __copy(self, trials):
        return Experiment(
                self.M, 
                self.labels,
                self.clfs,
                self.subsets,
                self.cvs,
                trials)

    def __transpose_dict_of_lists(self, dol):
        return utils.transpose_dict_of_lists(dol)

    def slice_on_dimension(self, dimension, value):
        """Select Trials where dimension == value

        Parameters
        ----------
        dimension : {CLF, CLF_PARAMS, SUBSET, SUBSET_PARAMS, CV, CV_PARAMS}
            dimension to slice on:

            CLF
                select Trials where the classifier == value
            CLF_PARAMS
                select Trials where the classifier parameters == value
            SUBSET
                select Trials where the subset iterator == value
            SUBSET_PARAMS
                select Trials where the subset iterator parameters == value
            CV
                select Trials where the cross-validation partition iterator ==
                value
            CV_PARAMS 
                select Trials where the partition iterator params == value
        value : ?
            Value to match

        Returns
        -------
        Experiment
            containing only the specified Trials
        """
        self.run()
        return self.__copy([trial for trial in self.trials if 
                            trial[dimension] == value])

    def iterate_over_dimension(self, dimension):
        """Iterates over sets of Trials with respect to dimension

        For example, if we iterate across CLF, each iteration will
        include all the Trials that use a given CLF. If the experiment
        has RandomForestClassifier and SVC trials, then  one iteration
        will have all trials with RandomForestClassifer and the other 
        iteration will have all trials with SVC

        Parameters
        ----------
        dimension : {CLF, CLF_PARAMS, SUBSET, SUBSET_PARAMS, CV, CV_PARAMS}
            dimension to iterate over:

            CLF
                iterate over Trials on classifier
            CLF_PARAMS
                iterate over Trials on classifier parameters
            SUBSET
                iterate over Trials on subset iterator
            SUBSET_PARAMS
                iterate over Trials on subset iterator parameters
            CV
                iterate over Trials on cross-validation partition iterator
            CV_PARAMS 
                iterate over Trials on partition iterator
        
        Returns
        -------
        iterator of (?, Experiment)
            The first element of the tuple is the value of dimension that
            all trials in the second element of the tuple has. 

            The second element of the tuple is an Experiment contain all
            trials where the given dimension is equal to the value in the
            first element of the tuple.
        """

        by_dim = {}
        for trial in self.trials:
            val_of_dim = trial[dimension]
            try:
                by_dim[val_of_dim].append(trial)
            except KeyError:
                by_dim[val_of_dim] = [trial]
        for val_of_dim, trials_this_dim in by_dim.iteritems():
            yield (val_of_dim, self.__copy(trials_this_dim))
            

    def slice_by_best_score(self, dimension):
        """Returns trials that have the best score across dimension

        Parameters
        ----------
        dimension : {CLF, CLF_PARAMS, SUBSET, SUBSET_PARAMS, CV, CV_PARAMS}
            dimension to find best trials over

            CLF
                find best Trials over classifier
            CLF_PARAMS
                find best Trials over classifier parameters
            SUBSET
                find best Trials over subset iterator
            SUBSET_PARAMS
                find best Trials over subset iterator parameters
            CV
                find best Trials over cross-validation partition iterator
            CV_PARAMS 
                find best Trials over partition iterator
        
        Returns
        -------
        Experiment
            With only trials that have the best scores over the selected
            dimension

        """
        self.run()
        categories = {}
        other_dims = list(dimensions)
        other_dims.remove(dimension)
        for trial in self.trials:
            # http://stackoverflow.com/questions/5884066/hashing-a-python-dictionary
            key = repr([trial[dim] for dim in other_dims])
            try:
                categories[key].append(trial)
            except KeyError:
                categories[key] = [trial]
        result = []
        for key in categories:
            result.append(max(
                categories[key], 
                key=lambda trial: trial.average_score()))
        return self.__copy(result)

    def has_run(self):
        """Returns boolean specifying whether this experiment has been run"""
        return self.trials is not None

    def run(self):
        """Runs the experiment. Fits all classifiers
        
        Returns
        -------
        list of Trial
            Trials with fitted classifiers
        """
        if self.has_run():
            return self.trials
        trials = []
        for clf_args in self.clfs:
            clf = clf_args['clf']
            all_clf_ps = clf_args.copy()
            del all_clf_ps['clf']
            for clf_params in self.__transpose_dict_of_lists(all_clf_ps):
                for subset_args in self.subsets:
                    subset = subset_args['subset']
                    all_sub_ps = subset_args.copy()
                    del all_sub_ps['subset']
                    for subset_params in self.__transpose_dict_of_lists(all_sub_ps):
                        for cv_args in self.cvs:
                            cv = cv_args['cv']
                            all_cv_ps = cv_args.copy()
                            del all_cv_ps['cv']
                            for cv_params in self.__transpose_dict_of_lists(all_cv_ps):
                                trial = Trial(
                                    M=self.M,
                                    labels=self.labels,
                                    col_names=self.col_names,
                                    clf=clf,
                                    clf_params=clf_params,
                                    subset=subset,
                                    subset_params=subset_params,
                                    cv=cv,
                                    cv_params=cv_params)
                                trials.append(trial)
        trials = self.__run_all_trials(trials)
        self.trials = trials
        return trials

    def average_score(self):
        """Get average score for all Trials in this experiment

        Returns
        -------
        dict of Trial: float
            Provides the average score of each trial
        """

        self.run()
        return {trial: trial.average_score() for trial in self.trials}

    def roc_auc(self):
        """Get average area under the roc curve for all Trials in this experiment

        Returns
        -------
        dict of Trial: float
            Provides the average area under the roc curve of each trial
        """
        self.run()
        return {trial: trial.roc_auc() for trial in self.trials}

    @staticmethod
    def csv_header():
        """Returns the header required to make a csv"""
        return Trial.csv_header()

    def make_report(
        self, 
        report_file_name='report.pdf',
        dimension=None,
        return_report_object=False,
        verbose=True):
        """Creates a pdf report of this experiment

        Parameters
        ----------
        report_file_name : str
            path of file for report output
        dimension : {CLF, CLF_PARAMS, SUBSET, SUBSET_PARAMS, CV, CV_PARAMS, None}
            If not None, will make a subreport for each unique value of
            dimension
        return_report_object : boolean
            Iff True, this function returns the report file name and the
            diogenes.display.Report object. Otherwise, just returns the report 
            file name.
        verbose : boolean
            iff True, gives output about report generation

        Returns
        -------
        str or (str, diogenes.display.Report) 
            If return_report_object is False, returns the file name of the
            generated report. Else, returns a tuple of the filename of the
            generated report as well as the Report object representing the
            report

        """
            
        # TODO make this more flexible
        from ..display import Report
        self.run()
        if dimension is None:
            dim_iter = [(None, self)]
        else:
            dim_iter = self.iterate_over_dimension(dimension)
        rep = Report(self, report_file_name)
        rep.add_heading('Eights Report {}'.format(datetime.datetime.now()), 1)
        for val_of_dim, sub_exp in dim_iter:
            sub_rep = Report(sub_exp)
            if val_of_dim is not None:
                sub_rep.add_heading('Subreport for {} = {}'.format(
                    dimension_descr[dimension],
                    val_of_dim), 1)
            sub_rep.add_heading('Roc AUCs', 3)
            sub_rep.add_summary_graph_roc_auc()
            sub_rep.add_heading('Average Scores', 3)
            sub_rep.add_summary_graph_average_score()
            sub_rep.add_heading('ROC for best trial', 3)
            sub_rep.add_graph_for_best_roc()
            sub_rep.add_heading('Prec recall for best trial', 3)
            sub_rep.add_graph_for_best_prec_recall()
            sub_rep.add_heading('Legend', 3)
            sub_rep.add_legend()
            rep.add_subreport(sub_rep)
        returned_report_file_name = rep.to_pdf(verbose=verbose)
        if return_report_object:
            return (returned_report_file_name, rep)
        return returned_report_file_name

    def make_csv(self, file_name='report.csv'):
        """Creates a csv summarizing the experiment

        Parameters
        ----------
        file_name : str
            path of csv to be generated

        Returns
        -------
        str
            path of generated csv

        """
        self.run()
        with open(file_name, 'w') as fout:
            writer = csv.writer(fout)
            writer.writerow(self.csv_header())
            for trial in self.trials:
                writer.writerows(trial.csv_rows())
        return os.path.abspath(file_name)

CLF, CLF_PARAMS, SUBSET, SUBSET_PARAMS, CV, CV_PARAMS = range(6)
dimensions = (CLF, CLF_PARAMS, SUBSET, SUBSET_PARAMS, CV, CV_PARAMS)
dimension_descr = {CLF: 'classifier',
                   CLF_PARAMS: 'classifier parameters',
                   SUBSET: 'subset type',
                   SUBSET_PARAMS: 'subset parameters',
                   CV: 'cross-validation method',
                   CV_PARAMS: 'cross-validation parameters'}
    
# TODO these really, really need to be dynamically generated based on the experiment
all_subset_notes = sorted(['sample_num', 'rows', 'prop_positive', 
                           'excluded_col', 'max_grade', 'train_interval_start',
                           'train_interval_end', 'test_interval_start', 
                           'test_interval_end',
                           'train_label_interval_start', 'train_label_interval_end',
                           'test_label_interval_start', 'test_label_interval_end'])

all_subset_notes_backindex = {name: i for i, name in 
                              enumerate(all_subset_notes)}

all_cv_notes = sorted(['train_start', 'train_end', 'test_start', 
                        'test_end', 'fold']) 

all_cv_notes_backindex = {name: i for i, name in 
                              enumerate(all_cv_notes)}

class Run(object):
    """Object encapsulating a single fitted classifier and specific data

    Parameters
    ----------
    M : numpy.ndarray
        Homogeneous (not structured) array. The array of features.
        If M_test is None, M contains both train and test sets. If
        M_test is not None, M contains only the training set.
    labels : numpy.ndarray
        Array of labels. If labels_test is None, labels contains both train and
        test sets. If labels_test is not None, M contains only the training
        set.
    col_names : list of str
        Names of features
    clf : sklearn.base.BaseEstimator
        clf fitted with testing data
    train_indices : np.ndarray or None
        If M_test and labels_test are None, The indices of M and labels that 
        comprise the training set
    test_indices : np.ndarray or None
        If M_test and labels_test are None, The indices of M and labels that 
        comprise the testing set
    sub_col_names : list of str
        If subset takes a subset of columns, these are the column names
        involved in this subset
    sub_col_inds : np.ndarray
        If subset takes a subset of columns, these are the indices of the
        columns involved in this subset
    subset_note : dict of str : ?
        Extra information about this Run provided by the subsetter
    cv_note : dict of str : ?
        Extra information about this run provided by the partition iterator
    M_test : np.ndarray or None
        If not None, the features in the test set
    labels_test : np.ndarray or None
        If not None, the labels in the test set

    Attributes
    ----------
    M : numpy.ndarray
        Homogeneous (not structured) array. The array of features.
        If M_test is None, M contains both train and test sets. If
        M_test is not None, M contains only the training set.
    labels : numpy.ndarray
        Array of labels. If labels_test is None, labels contains both train and
        test sets. If labels_test is not None, M contains only the training
        set.
    col_names : list of str
        Names of features
    clf : sklearn.base.BaseEstimator
        clf fitted with testing data
    train_indices : np.ndarray or None
        If M_test and labels_test are None, The indices of M and labels that 
        comprise the training set
    test_indices : np.ndarray or None
        If M_test and labels_test are None, The indices of M and labels that 
        comprise the testing set
    sub_col_names : list of str
        If subset takes a subset of columns, these are the column names
        involved in this subset
    sub_col_inds : np.ndarray
        If subset takes a subset of columns, these are the indices of the
        columns involved in this subset
    subset_note : dict of str : ?
        Extra information about this Run provided by the subsetter
    cv_note : dict of str : ?
        Extra information about this run provided by the partition iterator
    M_test : np.ndarray or None
        If not None, the features in the test set
    labels_test : np.ndarray or None
        If not None, the labels in the test set

    """

    def __init__(
        self,
        M,
        labels,
        col_names,
        clf,
        train_indices,
        test_indices,
        sub_col_names,
        sub_col_inds,
        subset_note,
        cv_note,
        M_test=None,
        labels_test=None):
        self.M = M
        self.labels = labels
        self.col_names = col_names
        self.sub_col_names = sub_col_names
        self.sub_col_inds = sub_col_inds
        self.clf = clf
        self.test_indices = test_indices
        self.train_indices = train_indices
        self.subset_note = subset_note
        self.cv_note = cv_note
        self.M_test = M_test
        self.labels_test = labels_test

    def __repr__(self):
        return 'Run(clf={}, subset_note={}, cv_note={})'.format(
                self.clf, self.subset_note, self.cv_note)

    def __test_M(self):
        if self.M_test is not None:
            return self.M_test
        return self.M[np.ix_(self.test_indices, self.sub_col_inds)]

    def __test_labels(self):
        if self.labels_test is not None:
            return self.labels_test
        return self.labels[self.test_indices]

    def __pred_proba(self):
        try:
            return self.clf.predict_proba(self.__test_M())[:,-1]
        except AttributeError:
            return np.zeros((self.__test_M().shape[0]))

    def __predict(self):
        return self.clf.predict(self.__test_M())

    @staticmethod
    def csv_header():
        """Returns a portion of the header necessary in constructing the csv"""
        return (['subset_note_' + name for name in all_subset_notes] + 
                ['cv_note_' + name for name in all_cv_notes] + 
                ['f1_score', 'roc_auc', 'prec@1%', 'prec@2%', 'prec@5%', 
                 'prec@10%', 'prec@20%'] + 
                ['feature_ranked_{}'.format(i) for i in xrange(10)] +
                ['feature_score_{}'.format(i) for i in xrange(10)])

    def __subset_note_list(self):
        notes = [''] * len(all_subset_notes)
        for name, val in self.subset_note.iteritems():
            notes[all_subset_notes_backindex[name]] = str(val)
        return notes

    def __cv_note_list(self):
        notes = [''] * len(all_cv_notes)
        for name, val in self.cv_note.iteritems():
            notes[all_cv_notes_backindex[name]] = str(val)
        return notes

    def __feat_import(self):
        col_names, scores = self.sorted_top_feat_importance(10)
        return list(
                it.chain(it.chain(
                    col_names, 
                    it.repeat('', 10 - len(col_names)),
                    scores)))

    def csv_row(self):
        """This Run's portion of its row in produces csv"""
        return (self.__subset_note_list() +
                self.__cv_note_list() + 
                [self.f1_score()] + 
                [self.roc_auc()] +
                self.precision_at_thresholds([.01, .02, .05, .10,
                                              .20]).tolist() +
                self.__feat_import())

    def score(self):
        """Returns score of fitted clf"""
        return self.clf.score(self.__test_M(), self.__test_labels())

    def roc_curve(self):
        """Returns matplotlib.figure.Figure of ROC curve"""
        from ..display import plot_roc
        return plot_roc(
                self.__test_labels(), 
                self.__pred_proba(), 
                verbose=False)

    def prec_recall_curve(self):
        """Returns matplotlib.figure.Figure of precision/recall curve"""
        from ..display import plot_prec_recall
        return plot_prec_recall(self.__test_labels(), self.__pred_proba(), 
                                verbose=False)
   
    def sorted_top_feat_importance(self, n = 25):
        """Returns top feature importances

        Parameters
        ----------
        n : int
            number of feature importances to return
        
        Returns
        -------
        [list of str, list of floats]
            names and scores of top features

        """
        if not hasattr(self.clf, 'feature_importances_'):
            return [[], []]
        feat_imp = self.clf.feature_importances_
        n = min(n, len(feat_imp))
        ind = np.argpartition(feat_imp, -n)[-n:]
        top_cols = ind[np.argsort(feat_imp[ind])][::-1]
        top_vals = feat_imp[top_cols]
        col_names = [self.sub_col_names[idx] for idx in top_cols]
        return [col_names, top_vals]

    def f1_score(self):
        """Returns f1 score"""
        return f1_score(self.__test_labels(), self.__predict())

    def precision_at_thresholds(self, query_thresholds):
        """Returns precision at given thresholds
        
        Parameters
        ----------
        query_thresholds : list of float
            for each element, 0 <= thresh <= 1

        Returns
        -------
        list of float

        """
        labels_true = self.__test_labels()
        labels_score = self.__pred_proba()
        prec, _, thresh = precision_recall_curve(
                labels_true, 
                labels_score)

        # Adopted from Rayid's code
        precision_curve = prec[:-1]
        pct_above_per_thresh = []
        number_scored = len(labels_score)
        for value in thresh:
            num_above_thresh = len(labels_score[labels_score>=value])
            pct_above_thresh = num_above_thresh / float(number_scored)
            pct_above_per_thresh.append(pct_above_thresh)
        pct_above_per_thresh = np.array(pct_above_per_thresh)
        # Add point at 0% above thresh, 1, precision
        pct_above_per_thresh = np.append(pct_above_per_thresh, 0)
        precision_curve = np.append(precision_curve, 1)

        # TODO something better than linear interpolation
        return np.interp(
                query_thresholds, 
                pct_above_per_thresh[::-1],
                precision_curve[::-1])

    def roc_auc(self):
        """Returns area under ROC curve"""
        try:
            return roc_auc_score(self.__test_labels(), self.__pred_proba())
        # If there was only one class
        except ValueError:
            return 0


# TODO other clfs
all_clf_params = sorted(
        list(
            frozenset(
                it.chain(
                    *[clf().get_params().keys() for clf in 
                      (AdaBoostClassifier,
                       RandomForestClassifier,
                       LogisticRegression,
                       DecisionTreeClassifier,
                       SVC,
                       DummyClassifier)]))))
                                        
all_clf_params_backindex = {param: i for i, param in enumerate(all_clf_params)}

#TODO these really need to be dynamically generated based on the experiment
all_subset_params = sorted(['subset_size', 'n_subsets', 'num_rows', 
                            'proportions_positive', 'cols_to_exclude',
                            'max_grades', 'random_state'])

all_subset_params_backindex = {param: i for i, param in 
                               enumerate(all_subset_params)}

all_cv_params = sorted(['n_folds', 'indices', 'shuffle', 'random_state',
                        'train_start', 'train_window_size',
                        'test_start', 'test_window_size', 
                        'inc_value', 'expanding_train', 'guide_col_name',
                        'col_name'])
                        
all_cv_params_backindex = {param: i for i, param in 
                           enumerate(all_cv_params)}

class Trial(object):
    """Object encapsulating all Runs for a given configuration
    
    Parameters
    ----------
    M : numpy.ndarray
        Homogeneous (not structured) array of features
    labels : numpy.ndarray
        Array of labels
    col_names : list of str
        Names of features
    clf : sklearn.base.BaseEstimator class
        Classifier for this trial
    clf_params : dict of str : ?
        init parameters for clf
    subset : diogenes.grid_search.subset.BaseSubsetIter class
        class of object making subsets
    subset_params : dict of str : ?
        init parameters for subset
    cv : sklearn.cross_validation._PartitionIterator class
        class used to product train and test sets
    cv_params : dict of str : ?
        init parameters of cv
    runs : list of list of Run or None
        if not None, can initialize this trial with a list of Runs
        that have already been created.

    Attributes
    ----------
    M : numpy.ndarray
        Homogeneous (not structured) array of features
    labels : numpy.ndarray
        Array of labels
    col_names : list of str
        Names of features
    clf : sklearn.base.BaseEstimator class
        Classifier for this trial
    clf_params : dict of str : ?
        init parameters for clf
    subset : diogenes.grid_search.subset.BaseSubsetIter class
        class of object making subsets
    subset_params : dict of str : ?
        init parameters for subset
    cv : sklearn.cross_validation._PartitionIterator class
        class used to product train and test sets
    cv_params : dict of str : ?
        init parameters of cv
    runs : list or Run or None
        Runs in this Trial. The outer list signifies different subsets. The
        outer lists signify different train/test splits using the same subset

    """

    def __init__(
        self, 
        M,
        labels,
        col_names,
        clf=RandomForestClassifier,
        clf_params={},
        subset=s_i.SubsetNoSubset,
        subset_params={},
        cv=p_i.NoCV,
        cv_params={},
        runs=None):
        self.M = M
        self.labels = labels
        self.col_names = col_names 
        self.runs = runs
        self.clf = clf
        self.clf_params = clf_params
        self.subset = subset
        self.subset_params = subset_params
        self.cv = cv
        self.cv_params = cv_params
        self.__by_dimension = {CLF: self.clf,
                               CLF_PARAMS: self.clf_params,
                               SUBSET: self.subset,
                               SUBSET_PARAMS: self.subset_params,
                               CV: self.cv,
                               CV_PARAMS: self.cv_params}
        self.__cached_ave_score = None
        self.repr = ('Trial(clf={}, clf_params={}, subset={}, '
                     'subset_params={}, cv={}, cv_params={})').format(
                        self.clf,
                        self.clf_params,
                        self.subset,
                        self.subset_params,
                        self.cv,
                        self.cv_params)
        self.hash = hash(self.repr)


    def __hash__(self):
        return self.hash

    def __repr__(self):
        return self.repr
    def __getitem__(self, arg):
        return self.__by_dimension[arg]

    def has_run(self):
        """Returns True iff this Trial has been run"""
        return self.runs is not None

    @staticmethod
    def __indices_of_sublist(full_list, sublist):
        set_sublist = frozenset(sublist)
        return [i for i, full_list_item in enumerate(full_list) if
                full_list_item in set_sublist]

    def run(self):
        """Run the Trial"""
        if self.has_run():
            return self.runs
        runs = []
        for subset, sub_col_names, subset_note in self.subset(
                self.labels, 
                self.col_names, 
                **self.subset_params):
            runs_this_subset = []
            labels_sub = self.labels[subset]
            sub_col_inds = self.__indices_of_sublist(
                self.col_names, 
                sub_col_names)
            # np.ix_ from http://stackoverflow.com/questions/30176268/error-when-indexing-with-2-dimensions-in-numpy
            M_sub = self.M[np.ix_(subset, sub_col_inds)]
            cv_cls = self.cv
            cv_kwargs = copy.deepcopy(self.cv_params)
            expected_cv_kwargs = inspect.getargspec(cv_cls.__init__).args
            if 'n' in expected_cv_kwargs:
                cv_kwargs['n'] = labels_sub.shape[0]
            if 'y' in expected_cv_kwargs:
                cv_kwargs['y'] = labels_sub
            if 'labels' in expected_cv_kwargs:
                cv_kwargs['labels'] = labels_sub
            if 'M' in expected_cv_kwargs:
                cv_kwargs['M'] = M_sub
            if 'col_names' in expected_cv_kwargs:
                cv_kwargs['col_names'] = sub_col_names
            cv_inst = cv_cls(**cv_kwargs)
            for fold_idx, (train, test) in enumerate(cv_inst):
                if hasattr(cv_inst, 'cv_note'):
                    cv_note = cv_inst.cv_note()
                else:
                    cv_note = {'fold': fold_idx}
                clf_inst = self.clf(**self.clf_params)
                clf_inst.fit(M_sub[train], labels_sub[train])
                test_indices = subset[test]
                train_indices = subset[train]
                runs_this_subset.append(Run(
                    self.M, 
                    self.labels, 
                    self.col_names, 
                    clf_inst, 
                    train_indices, 
                    test_indices,
                    sub_col_names, 
                    sub_col_inds,
                    subset_note, 
                    cv_note))
            runs.append(runs_this_subset)    
        self.runs = runs
        return self

    @staticmethod
    def csv_header():
        """Returns portion of header used when creating csv"""
        return (['clf'] + ['clf_' + name for name in all_clf_params] +  
                ['subset'] + ['subset_' + name for name in all_subset_params] +
                ['cv'] + ['cv_' + name for name in all_cv_params] +
                Run.csv_header())

    def __clf_param_list(self):
        param_vals = [''] * len(all_clf_params)
        for name, val in self.clf_params.iteritems():
            param_vals[all_clf_params_backindex[name]] = str(val)
        return param_vals

    def __subset_param_list(self):
        param_vals = [''] * len(all_subset_params)
        for name, val in self.subset_params.iteritems():
            param_vals[all_subset_params_backindex[name]] = str(val)
        return param_vals

    def __cv_param_list(self):
        param_vals = [''] * len(all_cv_params)
        for name, val in self.cv_params.iteritems():
            param_vals[all_cv_params_backindex[name]] = str(val)
        return param_vals

    def csv_rows(self):
        """Returns portions of rows used in creating csv"""
        return [[str(self.clf)] + self.__clf_param_list() + 
                [str(self.subset)] + self.__subset_param_list() + 
                [str(self.cv)] + self.__cv_param_list() + 
                 run.csv_row() for run in self.runs_flattened()]

    def average_score(self):
        """Returns average score accross all Runs"""
        if self.__cached_ave_score is not None:
            return self.__cached_ave_score
        self.run()
        ave_score = np.mean([run.score() for run in self.runs_flattened()])
        self.__cached_ave_score = ave_score
        return ave_score
    
    def median_run(self):
        """Returns Run with median score"""
        # Give or take
        #runs_with_score = [(run.score(), run) for run in self.runs]
        runs_with_score = [(run.score(), run) for run in it.chain(*self.runs)]
        runs_with_score.sort(key=lambda t: t[0])
        return runs_with_score[len(runs_with_score) / 2][1]

    def runs_flattened(self):
        """Returns list of all Runs (rather than list of list of Runs)"""
        return [run for run in it.chain(*self.runs)]

    # TODO These should all be average across runs rather than picking the 
    # median

    def roc_curve(self):
        """Returns matplotlib.figure.Figure of roc_curve of median run"""
        return self.median_run().roc_curve()

    def roc_auc(self):
        """Returns area under roc curve of median run"""
        return self.median_run().roc_auc()

    def prec_recall_curve(self):
        """Returns matplotlib.figure.Figure of precision/recall curve 
        
        (of median run)
        
        """
        return self.median_run().prec_recall_curve()

