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

from diogenes.grid_search import subset
from diogenes.grid_search import partition_iterator as p_i
import diogenes.utils as utils
from diogenes.utils import remove_cols

def _run_trial(trial):
    return trial.run()

class Experiment(object):
    def __init__(
            self, 
            M, 
            y, 
            clfs=[{'clf': RandomForestClassifier}], 
            subsets=[{'subset': subset.SubsetNoSubset}], 
            cvs=[{'cv': KFold}],
            trials=None):
        if utils.is_sa(M):
            self.col_names = M.dtype.names
            self.M = utils.cast_np_sa_to_nd(M)
        else: # assuming an nd_array
            self.M = M
            self.col_names = ['f{}'.format(i) for i in xrange(M.shape[1])]
        self.y = y
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
                self.y,
                self.clfs,
                self.subsets,
                self.cvs,
                trials)

    def __transpose_dict_of_lists(self, dol):
        # http://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
        return (dict(it.izip(dol, x)) for 
                x in it.product(*dol.itervalues()))

    def slice_on_dimension(self, dimension, value, trials=None):
        self.run()
        return self.__copy([trial for trial in self.trials if 
                            trial[dimension] == value])

    def iterate_over_dimension(self, dimension):
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
        return self.trials is not None

    def run(self):
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
                                    y=self.y,
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
        self.run()
        return {trial: trial.average_score() for trial in self.trials}

    def roc_auc(self):
        self.run()
        return {trial: trial.roc_auc() for trial in self.trials}

    @staticmethod
    def csv_header():
        return Trial.csv_header()

    def make_report(
        self, 
        report_file_name='report.pdf',
        dimension=None,
        return_report_object=False,
        verbose=True):
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
        self.run()
        with open(file_name, 'w') as fout:
            writer = csv.writer(fout)
            writer.writerow(self.csv_header())
            for trial in self.trials:
                writer.writerows(trial.csv_rows())
        return os.path.abspath(file_name)



# TODO By and large, we shouldn't be using SKLearn's internal classes.

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
                           'excluded_col', 'max_grade'])

all_subset_notes_backindex = {name: i for i, name in 
                              enumerate(all_subset_notes)}

all_cv_notes = sorted(['train_start', 'train_end', 'test_start', 
                        'test_end', 'fold']) 

all_cv_notes_backindex = {name: i for i, name in 
                              enumerate(all_cv_notes)}

class Run(object):
    def __init__(
        self,
        M,
        y,
        col_names,
        clf,
        train_indices,
        test_indices,
        sub_col_names,
        sub_col_inds,
        subset_note,
        cv_note):
        self.M = M
        self.y = y
        self.col_names = col_names
        self.sub_col_names = sub_col_names
        self.sub_col_inds = sub_col_inds
        self.clf = clf
        self.test_indices = test_indices
        self.train_indices = train_indices
        self.subset_note = subset_note
        self.cv_note = cv_note

    def __repr__(self):
        return 'Run(clf={}, subset_note={}, cv_note={})'.format(
                self.clf, self.subset_note, self.cv_note)

    def __test_M(self):
        return self.M[np.ix_(self.test_indices, self.sub_col_inds)]

    def __test_y(self):
        return self.y[self.test_indices]

    def __pred_proba(self):
        try:
            return self.clf.predict_proba(self.__test_M())[:,-1]
        except AttributeError:
            return np.zeros((self.__test_M().shape[0]))

    def __predict(self):
        return self.clf.predict(self.__test_M())

    @staticmethod
    def csv_header():
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
        return (self.__subset_note_list() +
                self.__cv_note_list() + 
                [self.f1_score()] + 
                [self.roc_auc()] +
                self.precision_at_thresholds([.01, .02, .05, .10,
                                              .20]).tolist() +
                self.__feat_import())

    def score(self):
        return self.clf.score(self.__test_M(), self.__test_y())

    def roc_curve(self):
        from ..display import plot_roc
        return plot_roc(self.__test_y(), self.__pred_proba(), verbose=False)

    def prec_recall_curve(self):
        from ..display import plot_prec_recall
        return plot_prec_recall(self.__test_y(), self.__pred_proba(), 
                                verbose=False)
   
    def sorted_top_feat_importance(self, n = 25):
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
        return f1_score(self.__test_y(), self.__predict())

    def precision_at_thresholds(self, query_thresholds):
        """
        Parameters
        query_thresholds : float
            0 <= thresh <= 1
        """
        y_true = self.__test_y()
        y_score = self.__pred_proba()
        prec, _, thresh = precision_recall_curve(
                y_true, 
                y_score)

        # Adopted from Rayid's code
        precision_curve = prec[:-1]
        pct_above_per_thresh = []
        number_scored = len(y_score)
        for value in thresh:
            num_above_thresh = len(y_score[y_score>=value])
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
        try:
            return roc_auc_score(self.__test_y(), self.__pred_proba())
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
                        'inc_value', 'expanding_train', 'col_name',
                        'col_name'])
                        
all_cv_params_backindex = {param: i for i, param in 
                           enumerate(all_cv_params)}

class Trial(object):
    def __init__(
        self, 
        M,
        y,
        col_names,
        clf=RandomForestClassifier,
        clf_params={},
        subset=subset.SubsetNoSubset,
        subset_params={},
        cv=p_i.NoCV,
        cv_params={}):
        self.M = M
        self.y = y
        self.col_names = col_names 
        self.runs = None
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
        return self.runs is not None

    @staticmethod
    def __indices_of_sublist(full_list, sublist):
        set_sublist = frozenset(sublist)
        return [i for i, full_list_item in enumerate(full_list) if
                full_list_item in set_sublist]

    def run(self):
        if self.has_run():
            return self.runs
        runs = []
        for subset, sub_col_names, subset_note in self.subset(
                self.y, 
                self.col_names, 
                **self.subset_params):
            runs_this_subset = []
            y_sub = self.y[subset]
            sub_col_inds = self.__indices_of_sublist(
                self.col_names, 
                sub_col_names)
            # np.ix_ from http://stackoverflow.com/questions/30176268/error-when-indexing-with-2-dimensions-in-numpy
            M_sub = self.M[np.ix_(subset, sub_col_inds)]
            cv_cls = self.cv
            cv_kwargs = copy.deepcopy(self.cv_params)
            expected_cv_kwargs = inspect.getargspec(cv_cls.__init__).args
            if 'n' in expected_cv_kwargs:
                cv_kwargs['n'] = y_sub.shape[0]
            if 'y' in expected_cv_kwargs:
                cv_kwargs['y'] = y_sub
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
                clf_inst.fit(M_sub[train], y_sub[train])
                test_indices = subset[test]
                train_indices = subset[train]
                runs_this_subset.append(Run(
                    self.M, 
                    self.y, 
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
        return [[str(self.clf)] + self.__clf_param_list() + 
                [str(self.subset)] + self.__subset_param_list() + 
                [str(self.cv)] + self.__cv_param_list() + 
                 run.csv_row() for run in self.runs_flattened()]

    def average_score(self):
        if self.__cached_ave_score is not None:
            return self.__cached_ave_score
        self.run()
        M = self.M
        y = self.y
        ave_score = np.mean([run.score() for run in self.runs_flattened()])
        self.__cached_ave_score = ave_score
        return ave_score
    
    def median_run(self):
        # Give or take
        #runs_with_score = [(run.score(), run) for run in self.runs]
        runs_with_score = [(run.score(), run) for run in it.chain(*self.runs)]
        runs_with_score.sort(key=lambda t: t[0])
        return runs_with_score[len(runs_with_score) / 2][1]

    def runs_flattened(self):
        return [run for run in it.chain(*self.runs)]

    # TODO These should all be average across runs rather than picking the 
    # median

    def roc_curve(self):
        return self.median_run().roc_curve()

    def roc_auc(self):
        return self.median_run().roc_auc()

    def prec_recall_curve(self):
        return self.median_run().prec_recall_curve()




    

