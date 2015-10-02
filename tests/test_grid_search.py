import unittest
import cPickle
import os

import numpy as np

from sklearn.svm import SVC 
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold

import diogenes.grid_search as per
import diogenes.display as dsp

import utils_for_tests as uft

REPORT_PATH = uft.path_of_data('test_grid_search.pdf')
REFERENCE_REPORT_PATH = uft.path_of_data('test_grid_search_ref.pdf')
REFERENCE_PKL_PATH = uft.path_of_data('test_grid_search')

class TestGridSearch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.report = dsp.Report(report_path=REPORT_PATH)

    @classmethod
    def tearDownClass(cls):
        report_path = cls.report.to_pdf(verbose=False)
        uft.print_in_box(
                'Test Perambulate visual regression tests',
                ['graphical output available at:',
                 report_path,
                 'Reference available at:',
                 REFERENCE_REPORT_PATH])

    def __pkl_store(self, obj, key):
        with open(os.path.join(REFERENCE_PKL_PATH, key + '.pkl'), 'w') as pkl:
            cPickle.dump(obj, pkl)

    def __get_ref_pkl(self, key):
        with open(os.path.join(REFERENCE_PKL_PATH, key + '.pkl')) as pkl:
            return cPickle.load(pkl)

    def __compare_to_ref_pkl(self, result, key):
        ref = self.__get_ref_pkl(key)
        self.assertEqual(ref, result) 
        
    def test_run_experiment(self):
        iris = datasets.load_iris()
        y = iris.target
        M = iris.data
        clfs = [{'clf': RandomForestClassifier, 'random_state': [0]}]
        subsets = [{'subset': per.SubsetRandomRowsActualDistribution, 
                    'subset_size': [20, 40, 60, 80, 100],
                    'random_state': [0]}]
        cvs = [{'cv': StratifiedKFold}]
        exp = per.Experiment(M, y, clfs, subsets, cvs)
        result = {str(key) : val for key, val in 
                  exp.average_score().iteritems()}
        self.__compare_to_ref_pkl(result, 'run_experiment')

    def test_std_clfs(self):
        M, y = uft.generate_test_matrix(100, 5, 2, random_state=0)
        cvs = [{'cv': StratifiedKFold}]
        for label, clfs in zip(('std',), (per.DBG_std_clfs,)):
            exp = per.Experiment(M, y, clfs=clfs, cvs=cvs)
            exp.run()
            result = {str(trial) for trial in exp.trials}
            self.__compare_to_ref_pkl(
                    result, 
                    'test_operate_{}'.format(label))
                    
    def test_slice_on_dimension(self):
        iris = datasets.load_iris()
        y = iris.target
        M = iris.data
        clfs = [{'clf': RandomForestClassifier, 
                 'n_estimators': [10, 100], 
                 'max_depth': [1, 10],
                 'random_state': [0]}, 
                 {'clf': SVC, 'kernel': ['linear', 'rbf'], 
                  'random_state': [0]}]        
        subsets = [{'subset': per.SubsetRandomRowsActualDistribution, 
                    'subset_size': [20, 40, 60, 80, 100],
                    'random_state': [0]}]
        cvs = [{'cv': StratifiedKFold}]
        exp = per.Experiment(M, y, clfs, subsets, cvs)
        result = [str(trial) for trial in exp.slice_on_dimension(
                per.CLF, 
                RandomForestClassifier).trials]
        self.__compare_to_ref_pkl(result, 'slice_on_dimension_clf')
        result = [str(trial) for trial  in exp.slice_on_dimension(
                per.SUBSET_PARAMS, 
                {'subset_size': 60}).trials]
        self.__compare_to_ref_pkl(result, 'slice_on_dimension_subset_params')

    def test_slice_by_best_score(self):
        iris = datasets.load_iris()
        y = iris.target
        M = iris.data
        clfs = [{'clf': RandomForestClassifier, 
                 'n_estimators': [10, 100], 
                 'max_depth': [1, 10],
                 'random_state': [0]}, 
                 {'clf': SVC, 'kernel': ['linear', 'rbf'],
                  'random_state': [0]}]        
        subsets = [{'subset': per.SubsetRandomRowsActualDistribution, 
                    'subset_size': [20, 40],
                    'random_state': [0]}]
        cvs = [{'cv': StratifiedKFold}]
        exp = per.Experiment(M, y, clfs, subsets, cvs)
        exp.run()
        result = {str(trial): trial.average_score() for trial in 
                  exp.slice_by_best_score(per.CLF_PARAMS).trials}
        self.__compare_to_ref_pkl(result, 'slice_by_best_score')

    def test_make_csv(self):
        M, y = uft.generate_test_matrix(1000, 5, 2, random_state=0)
        clfs = [{'clf': RandomForestClassifier, 
                 'n_estimators': [10, 100], 
                 'max_depth': [5, 25],
                 'random_state': [0]},
                {'clf': SVC, 
                 'kernel': ['linear', 'rbf'], 
                 'probability': [True],
                 'random_state': [0]}]        
        subsets = [{'subset': per.SubsetSweepNumRows, 
                    'num_rows': [[100, 200]],
                    'random_state': [0]}]
        cvs = [{'cv': StratifiedKFold, 
                'n_folds': [2, 3]}]
        exp = per.Experiment(M, y, clfs=clfs, subsets=subsets, cvs=cvs)
        result_path = exp.make_csv()

    def test_report_simple(self):
        M, y = uft.generate_test_matrix(100, 5, 2, random_state=0)
        clfs = [{'clf': RandomForestClassifier, 
                 'n_estimators': [10, 100, 1000],
                 'random_state': [0]}]
        cvs = [{'cv': StratifiedKFold}]
        exp = per.Experiment(M, y, clfs=clfs, cvs=cvs)
        _, rep = exp.make_report(return_report_object=True, verbose=False)
        self.report.add_heading('test_report_simple', 1)
        self.report.add_subreport(rep)

    def test_report_complex(self):
        M, y = uft.generate_test_matrix(100, 5, 2)
        clfs = [{'clf': RandomForestClassifier, 
                 'n_estimators': [10, 100], 
                 'max_depth': [1, 10],
                 'random_state': [0]}, 
                 {'clf': SVC, 
                  'kernel': ['linear', 'rbf'], 
                  'probability': [True],
                  'random_state': [0]}]        
        subsets = [{'subset': per.SubsetRandomRowsActualDistribution, 
                    'subset_size': [20, 40, 60, 80, 100],
                    'random_state': [0]}]
        cvs = [{'cv': StratifiedKFold}]
        exp = per.Experiment(M, y, clfs, subsets, cvs)
        _, rep = exp.make_report(dimension=per.CLF, return_report_object=True, 
                                 verbose=False)
        self.report.add_heading('test_report_complex', 1)
        self.report.add_subreport(rep)

    def test_subsetting(self):
        M, y = uft.generate_test_matrix(1000, 5, 2, random_state=0)
        subsets = [{'subset': per.SubsetRandomRowsEvenDistribution, 
                    'subset_size': [20],
                    'random_state': [0]},
                   {'subset': per.SubsetRandomRowsActualDistribution, 
                    'subset_size': [20],
                    'random_state': [0]},
                   {'subset': per.SubsetSweepNumRows, 
                    'num_rows': [[10, 20, 30]],
                    'random_state': [0]},
                   {'subset': per.SubsetSweepVaryStratification, 
                    'proportions_positive': [[.5, .75, .9]],
                    'subset_size': [10],
                    'random_state': [0]}]
        exp = per.Experiment(M, y, subsets=subsets)
        exp.run()
        result = {str(trial) : frozenset([str(run) for run in trial.runs]) for 
                  trial in exp.trials}
        self.__compare_to_ref_pkl(result, 'test_subsetting')

    def test_sliding_windows(self):
        M = np.array([(0, 2003),
                      (1, 1997),
                      (2, 1998),
                      (3, 2003),
                      (4, 2002),
                      (5, 2000),
                      (6, 2000),
                      (7, 2001),
                      (8, 1997),
                      (9, 2005), 
                      (10, 2005)], dtype=[('id', int), ('year', int)])
        y = np.array([True, False, True, False, True, False, True, False,
                      True, False])
        clfs = [{'clf': RandomForestClassifier,
                 'random_state': [0]}]
        cvs = [{'cv': per.SlidingWindowIdx, 
                'train_start': [0], 
                'train_window_size': [2], 
                'test_start': [2], 
                'test_window_size': [2],
                'inc_value': [2]},
                {'cv': per.SlidingWindowValue, 
                 'train_start': [1997], 
                 'train_window_size': [2],
                 'test_start': [1999], 
                 'test_window_size': [2],
                 'inc_value': [2], 
                 'guide_col_name': ['year']}]
        exp = per.Experiment(M, y, clfs=clfs, cvs=cvs)
        exp.run()
        result = {str(trial) : frozenset([str(run) for run in trial.runs]) for 
                  trial in exp.trials}
        self.__pkl_store(result, 'test_sliding_windows')
        self.__compare_to_ref_pkl(result, 'test_sliding_windows')

if __name__ == '__main__':
    unittest.main()
	

