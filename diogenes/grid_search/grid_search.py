import inspect 
import json
import copy
import abc
import datetime
import itertools as it
import numpy as np
import csv
import os

from collections import Counter


from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.cross_validation import _PartitionIterator

from joblib import Parallel, delayed
from multiprocessing import cpu_count

from .grid_search_helper import *
import diogenes.utils as utils


from sklearn.ensemble import (AdaBoostClassifier, 
                              RandomForestClassifier,
                              ExtraTreesClassifier,
                              GradientBoostingClassifier)
from sklearn.linear_model import (LogisticRegression, 
                                  RidgeClassifier, 
                                  SGDClassifier, 
                                  Perceptron, 
                                  PassiveAggressiveClassifier)
from sklearn.cross_validation import (StratifiedKFold, 
                                      KFold)
from sklearn.naive_bayes import (BernoulliNB, 
                                 MultinomialNB,
                                 GaussianNB)
from sklearn.neighbors import(KNeighborsClassifier, 
                              NearestCentroid)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier

from ..utils import remove_cols


std_clfs = [{'clf': AdaBoostClassifier, 'n_estimators': [20,50,100]}, 
            {'clf': RandomForestClassifier, 
             'n_estimators': [10,30,50],
             'max_features': ['sqrt','log2'],
             'max_depth': [None,4,7,15],
             'n_jobs':[1]}, 
            {'clf': LogisticRegression, 
             'C': [1.0,2.0,0.5,0.25],
             'penalty': ['l1','l2']}, 
            {'clf': DecisionTreeClassifier, 
             'max_depth': [None,4,7,15,25]},
            {'clf': SVC, 'kernel': ['linear','rbf'], 
             'probability': [True]},
            {'clf': DummyClassifier, 
             'strategy': ['stratified','most_frequent','uniform']}]

DBG_std_clfs = [{'clf': AdaBoostClassifier, 'n_estimators': [20]}, 
            {'clf': RandomForestClassifier, 
             'n_estimators': [10],
             'max_features': ['sqrt'],
             'max_depth': [None],
             'n_jobs':[1]}, 
            {'clf': LogisticRegression, 
             'C': [1.0],
             'penalty': ['l1']}, 
            {'clf': DecisionTreeClassifier, 
             'max_depth': [None]},
            {'clf': DummyClassifier, 
             'strategy': ['stratified','most_frequent']}]


rg_clfs= [{'clf': RandomForestClassifier,
           'n_estimators': [1,10,100,1000,10000], 
           'max_depth': [1,5,10,20,50,100], 
           'max_features': ['sqrt','log2'],
           'min_samples_split': [2,5,10],
           'n_jobs': [1]},
          {'clf': LogisticRegression,
           'penalty': ['l1','l2'], 
           'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
          {'clf': SGDClassifier,
           'loss':['hinge','log','perceptron'], 
           'penalty':['l2','l1','elasticnet']},
          {'clf': ExtraTreesClassifier,
           'n_estimators': [1,10,100,1000,10000], 
           'criterion' : ['gini', 'entropy'],
           'max_depth': [1,5,10,20,50,100], 
           'max_features': ['sqrt','log2'],
           'min_samples_split': [2,5,10],
           'n_jobs': [1]},
          {'clf': AdaBoostClassifier,
           'algorithm' :['SAMME', 'SAMME.R'], 
            'n_estimators': [1,10,100,1000,10000],
            'base_estimator': [DecisionTreeClassifier(max_depth=1)]},
          {'clf': GradientBoostingClassifier,
           'n_estimators': [1,10,100,1000,10000], 
           'learning_rate' : [0.001,0.01,0.05,0.1,0.5],
           'subsample' : [0.1,0.5,1.0], 
            'max_depth': [1,3,5,10,20,50,100]},
          {'clf': GaussianNB },
          {'clf': DecisionTreeClassifier,
           'criterion': ['gini', 'entropy'],
           'max_depth': [1,5,10,20,50,100], 
           'max_features': ['sqrt','log2'],
           'min_samples_split': [2,5,10]},
          {'clf':SVC,
           'C': [0.00001,0.0001,0.001,0.01,0.1,1,10],
           'kernel': ['linear'],
           'probability': [True]},
          {'clf': KNeighborsClassifier,
           'n_neighbors':[1,5,10,25,50,100],
           'wdiogenes': ['uniform','distance'],
           'algorithm':['auto','ball_tree','kd_tree']}]      

DBG_rg_clfs= [{'clf': RandomForestClassifier,
            'n_estimators': [1], 
            'max_depth': [1], 
            'max_features': ['sqrt'],
            'min_samples_split': [2],
            'n_jobs': [1]},
           {'clf': LogisticRegression,
            'penalty': ['l1'], 
            'C': [0.00001]},
           {'clf': SGDClassifier,
            'loss':['log'], # hinge doesn't have predict_proba
            'penalty':['l2']},
           {'clf': ExtraTreesClassifier,
            'n_estimators': [1], 
            'criterion' : ['gini'],
            'max_depth': [1], 
            'max_features': ['sqrt'],
            'min_samples_split': [2],
            'n_jobs': [1]},
           {'clf': AdaBoostClassifier,
            'algorithm' :['SAMME'],
            'n_estimators': [1],
            'base_estimator': [DecisionTreeClassifier(max_depth=1)]},
           {'clf': GradientBoostingClassifier,
            'n_estimators': [1],
            'learning_rate' : [0.001],
            'subsample' : [0.1],
            'max_depth': [1]},
           {'clf': GaussianNB },
           {'clf': DecisionTreeClassifier,
            'criterion': ['gini'],
            'max_depth': [1],
            'max_features': ['sqrt'],
            'min_samples_split': [2]},
           {'clf':SVC,
            'C': [0.00001],
            'kernel': ['linear'],
            'probability': [True]},
           {'clf': KNeighborsClassifier,
            'n_neighbors':[1],
            'wdiogenes': ['uniform'],
            'algorithm':['auto']}]     
    
alt_clfs = [{'clf': RidgeClassifier, 'tol':[1e-2], 'solver':['lsqr']},
            {'clf': SGDClassifier, 'alpha':[.0001], 'n_iter':[50],'penalty':['l1', 'l2', 'elasticnet']},
            {'clf': Perceptron, 'n_iter':[50]},
            {'clf': PassiveAggressiveClassifier, 'n_iter':[50]},
            {'clf': BernoulliNB, 'alpha':[.01]},
            {'clf': MultinomialNB, 'alpha':[.01]},
            {'clf': KNeighborsClassifier, 'n_neighbors':[10]},
            {'clf': NearestCentroid}]



def _run_trial(trial):
    return trial.run()

class Experiment(object):
    def __init__(
            self, 
            M, 
            y, 
            clfs=[{'clf': RandomForestClassifier}], 
            subsets=[{'subset': SubsetNoSubset}], 
            cvs=[{'cv': NoCV}],
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
        #return Parallel(n_jobs=cpu_count())(delayed(_run_trial)(t) 
        #                                   for t in trials)
        return [_run_trial(t) for t in trials]

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


def random_subset_of_columns(M, number_to_select):
    num_col = len(M.dtypes.names)
    remove_these_columns = np.random.choice(num_col, number_to_select, replace=False)
    names = [col_names[i] for i in remove_these_columns]
    return names
    



    

