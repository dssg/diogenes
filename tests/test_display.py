import unittest
from datetime import datetime
from collections import Counter
from diogenes.display.display import plot_box_plot,plot_simple_histogram

import diogenes.display as dsp
from diogenes.display.display import feature_pairs_in_tree
from diogenes.display.display import feature_pairs_in_rf
from diogenes.display.display import table
from diogenes.display.display import crosstab
from diogenes.display.display import describe_cols
from diogenes import utils
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
import utils_for_tests as uft
import numpy as np
import matplotlib.pyplot as plt


REPORT_PATH=uft.path_of_data('test_display.pdf')
SUBREPORT_PATH=uft.path_of_data('test_display_sub.pdf')
REFERENCE_REPORT_PATH=uft.path_of_data('test_display_ref.pdf')

class TestDisplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = dsp.Report(report_path=REPORT_PATH)

    @classmethod
    def tearDownClass(cls):
        report_path = cls.report.to_pdf(verbose=False)
        uft.print_in_box(
            'Test display visual regression tests',
            ['graphical output available at:',
             report_path,
             'Reference available at:',
             REFERENCE_REPORT_PATH])

    def add_fig_to_report(self, fig, heading):
        self.report.add_heading(heading)
        self.report.add_fig(fig)

    def test_pprint_sa(self):
        M = [(1, 2, 3), (4, 5, 6), (7, 8, 'STRING')]
        ctrl = """
  f0 f1     f2
0  1  2      3
1  4  5      6
2  7  8 STRING
        """.strip()
        with uft.rerout_stdout() as get_stdout:
            dsp.pprint_sa(M)
            self.assertEqual(get_stdout().strip(), ctrl)
        M = np.array([(1000, 'Bill'), (2000, 'Sam'), (3000, 'James')],
                     dtype=[('number', float), ('name', 'S5')])
        row_labels = [name[0] for name in M['name']]
        ctrl = """
  number  name
B 1000.0  Bill
S 2000.0   Sam
J 3000.0 James
        """.strip()
        with uft.rerout_stdout() as get_stdout:
            dsp.pprint_sa(M, row_labels=row_labels)
            self.assertEqual(get_stdout().strip(), ctrl)


    def test_describe_cols(self):
        test_list = [[1, 2],[2, 3],[3, 4],[4, 5],[5, 6],[6, 7]]
        test_nd = np.array(test_list)
        test_sa = np.array([(1, 2, 'a'), (2, 3, 'b'), (3, 4, 'c'), (4, 5, 'd'), 
                            (5, 6, 'e'), (6, 7, 'f')], 
                           dtype=[('id', int), ('val', float), ('name', 'S1')])
        ctrl_list = np.array([('f0', 6, 3.5, 1.707825127659933, 1, 6),
                              ('f1', 6, 4.5, 1.707825127659933, 2, 7)],
                             dtype=[('Column Name', 'S2'), ('Count', int),
                                    ('Mean', float), ('Standard Dev', float),
                                    ('Minimum', int), ('Maximum', int)])
        ctrl_printout = """
  Column Name Count Mean  Standard Dev Minimum Maximum
0          f0     6  3.5 1.70782512766       1       6
1          f1     6  4.5 1.70782512766       2       7
        """.strip()
        with uft.rerout_stdout() as get_stdout:
            self.assertTrue(uft.array_equal(ctrl_list, 
                                            describe_cols(
                                                test_list)))
            self.assertEqual(get_stdout().strip(), ctrl_printout)
        self.assertTrue(uft.array_equal(ctrl_list, 
                                        describe_cols(
                                            test_nd, verbose=False)))
        ctrl_sa = np.array([('id', 6, 3.5, 1.707825127659933, 1, 6),
                            ('val', 6, 4.5, 1.707825127659933, 2, 7),
                            ('name', np.nan, np.nan, np.nan, np.nan, np.nan)],
                           dtype=[('Column Name', 'S4'), ('Count', float),
                                  ('Mean', float), ('Standard Dev', float),
                                  ('Minimum', float), ('Maximum', float)])
        self.assertTrue(uft.array_equal(ctrl_sa, 
                                        describe_cols(
                                            test_sa,
                                            verbose=False)))

    def test_table(self):
        data = np.array(['a', 'b', 'a', 'b', 'b', 'b', 'b', 'a', 'c', 'c', 
                         'b', 'c', 'a'], dtype='O')
        ctrl_sa = np.array(
                [('a', 4), ('b', 6), ('c', 3)],
                dtype=[('col_name', 'S1'), ('count', int)])           
        ctrl_printout = """
  col_name count
0        a     4
1        b     6
2        c     3
        """.strip()
        with uft.rerout_stdout() as get_stdout:
            self.assertTrue(uft.array_equal(ctrl_sa, 
                                            table(data)))
            self.assertEqual(get_stdout().strip(), ctrl_printout)

    def test_crosstab(self):
        l1= [1, 2, 7, 7, 2, 1, 2, 1, 1]
        l2= [1, 3, 2, 6, 6, 3, 6, 4, 4]
        correct = np.array([('1', 1, 0, 1, 2, 0),
                            ('2', 0, 0, 1, 0, 2),
                            ('7', 0, 1, 0, 0, 1)],
                           dtype=[('col1_value', 'S1'),
                                  ('1', int),
                                  ('2', int),
                                  ('3', int),
                                  ('4', int),
                                  ('6', int)])
        correct_printout = """
  col1_value 1 2 3 4 6
0          1 1 0 1 2 0
1          2 0 0 1 0 2
2          7 0 1 0 0 1
        """.strip()
        with uft.rerout_stdout() as get_stdout:
            self.assertTrue(np.array_equal(correct, crosstab(l1,l2)))
            self.assertEqual(get_stdout().strip(), correct_printout)

    def test_plot_simple_histogram(self):
        np.random.seed(0)
        data = np.random.normal(size=(1000,))
        fig = plot_simple_histogram(data, verbose=False)
        self.add_fig_to_report(fig, 'plot_simple_histogram')
        data = np.array(['a', 'b', 'a', 'b', 'b', 'b', 'b', 'a', 'c', 'c', 
                         'b', 'c', 'a'], dtype='O')
        fig = plot_simple_histogram(data, verbose=False)
        self.add_fig_to_report(fig, 'plot_simple_histogram_categorical')

    def test_plot_prec_recall(self):
        M, labels = uft.generate_correlated_test_matrix(1000)
        M_train, M_test, labels_train, labels_test = train_test_split(
                M, 
                labels)
        clf = RandomForestClassifier(random_state=0)
        clf.fit(M_train, labels_train)
        score = clf.predict_proba(M_test)[:,-1]
        fig = dsp.plot_prec_recall(labels_test, score, verbose=False)
        self.add_fig_to_report(fig, 'plot_prec_recall')

    def test_plot_roc(self):
        M, labels = uft.generate_correlated_test_matrix(1000)
        M_train, M_test, labels_train, labels_test = train_test_split(
                M, 
                labels)
        clf = RandomForestClassifier(random_state=0)
        clf.fit(M_train, labels_train)
        score = clf.predict_proba(M_test)[:,-1]
        fig = dsp.plot_roc(labels_test, score, verbose=False)
        self.add_fig_to_report(fig, 'plot_roc')

    def test_plot_box_plot(self):
        np.random.seed(0)
        data = np.random.normal(size=(1000,))
        fig = plot_box_plot(data, title='box', verbose=False)
        self.add_fig_to_report(fig, 'plot_box_plot')

    def test_get_top_features(self):
        M, labels = uft.generate_test_matrix(1000, 15, random_state=0)
        M = utils.cast_np_sa_to_nd(M)
        M_train, M_test, labels_train, labels_test = train_test_split(
                M, 
                labels)
        clf = RandomForestClassifier(random_state=0)
        clf.fit(M_train, labels_train)
        res = dsp.get_top_features(clf, M, verbose=False)
        ctrl = utils.convert_to_sa(
                [('f5',  0.0773838526068), 
                 ('f13',   0.0769596713039),
                 ('f8',  0.0751584839431),
                 ('f6',  0.0730815879102),
                 ('f11',   0.0684456133071),
                 ('f9',  0.0666747414603),
                 ('f10',   0.0659621889608),
                 ('f7',  0.0657988099065),
                 ('f2',  0.0634000069218),
                 ('f0',  0.0632912268319)],
                col_names=('feat_name', 'score'))
        self.assertTrue(uft.array_equal(ctrl, res))
        res = dsp.get_top_features(clf, col_names=['f{}'.format(i) for i in xrange(15)], verbose=False)
        self.assertTrue(uft.array_equal(ctrl, res))

    def test_get_roc_auc(self):
        M, labels = uft.generate_correlated_test_matrix(1000)
        M_train, M_test, labels_train, labels_test = train_test_split(
                M, 
                labels)
        clf = RandomForestClassifier(random_state=0)
        clf.fit(M_train, labels_train)
        score = clf.predict_proba(M_test)[:,-1]
        self.assertTrue(np.allclose(
            dsp.get_roc_auc(labels_test, score, verbose=False),
            roc_auc_score(labels_test, score)))

    def test_plot_correlation_matrix(self):
        col1 = range(10)
        col2 = [cell * 3 + 1 for cell in col1]
        col3 = [1, 5, 8, 4, 1, 8, 5, 9, 0, 1]
        sa = utils.convert_to_sa(
                zip(col1, col2, col3), 
                col_names=['base', 'linear_trans', 'no_correlation'])
        fig = dsp.plot_correlation_matrix(sa, verbose=False)
        self.add_fig_to_report(fig, 'plot_correlation_matrix')

    def test_plot_correlation_scatter_plot(self):
        col1 = range(10)
        col2 = [cell * 3 + 1 for cell in col1]
        col3 = [1, 5, 8, 4, 1, 8, 5, 9, 0, 1]
        sa = utils.convert_to_sa(
                zip(col1, col2, col3), 
                col_names=['base', 'linear_trans', 'no_correlation'])
        fig = dsp.plot_correlation_scatter_plot(sa, verbose=False)
        self.add_fig_to_report(fig, 'plot_correlation_scatter_plot')

    def test_plot_kernel_density(self):
        np.random.seed(0)
        data = np.random.normal(size=(1000,))
        fig = dsp.plot_kernel_density(data, verbose=False)
        self.add_fig_to_report(fig, 'plot_kernel_density')


    def test_plot_on_timeline(self):
        dates = [datetime(2015, 1, 1),
                 datetime(2015, 2, 1),
                 datetime(2015, 6, 1),
                 datetime(2015, 6, 15),
                 datetime(2015, 9, 2),
                 datetime(2016, 1, 5)]
        fig1 = dsp.plot_on_timeline(dates, verbose=False)
        self.add_fig_to_report(fig1, 'plot_on_timeline_1')
        dates = np.array(dates, dtype='M8[us]')
        fig2 = dsp.plot_on_timeline(dates, verbose=False)
        self.add_fig_to_report(fig1, 'plot_on_timeline_2')

    def test_report(self):
        subrep = dsp.Report(report_path=SUBREPORT_PATH)
        self.assertEqual(subrep.get_report_path(), SUBREPORT_PATH)
        subrep.add_heading('Subreport', level=3)
        subrep.add_text(
            (u'Sample text.\n'
             u'<p>HTML tags should render literally</p>\n'))
        subrep.add_heading('Sample table', level=4)
        sample_table  = np.array(
            [(1, datetime(2015, 1, 1), 'New Years Day'),
             (2, datetime(2015, 2, 14), 'Valentines Day'),
             (3, datetime(2015, 3, 15), 'The Ides of March')],
            dtype=[('idx', int), ('day', 'M8[us]'), ('Name', 'S17')])
        subrep.add_table(sample_table)
        sample_fig = plt.figure()
        plt.plot([1, 2, 3], [1, 2, 3])
        plt.title('Sample fig')
        subrep.add_heading('Sample figure', level=4)
        subrep.add_fig(sample_fig)
        self.report.add_heading('report')
        self.report.add_subreport(subrep)
        
    def test_feature_pairs_in_tree(self):
        iris = datasets.load_iris()
        rf = RandomForestClassifier(random_state=0)
        rf.fit(iris.data, iris.target)
        dt = rf.estimators_[0]
        result = feature_pairs_in_tree(dt)
        ctrl = [[(2, 3)], [(2, 3), (0, 2)], [(0, 2), (1, 3)]]
        self.assertEqual(result, ctrl)

    def test_feature_pairs_in_rf(self):
        iris = datasets.load_iris()
        rf = RandomForestClassifier(random_state=0)
        rf.fit(iris.data, iris.target)
        results = feature_pairs_in_rf(rf, [1, 0.5], verbose=False)
        # TODO make sure these results are actually correct
        ctrl_cts_by_pair = Counter(
            {(2, 3): 16, (0, 2): 14, (0, 3): 12, (3, 3): 7, (2, 2): 6, 
             (0, 1): 4, (1, 2): 3, (1, 3): 3, (0, 0): 2, (1, 1): 1})
        ctrl_ct_pairs_by_depth = [
            Counter({(2, 3): 3, (0, 3): 3, (3, 3): 2, (2, 2): 2, (0, 1): 1, 
                     (0, 0): 1}), 
            Counter({(0, 2): 7, (2, 3): 5, (3, 3): 2, (2, 2): 2, (0, 3): 2, 
                     (1, 1): 1}), 
            Counter({(2, 3): 5, (0, 2): 5, (2, 2): 2, (0, 3): 2, (1, 2): 1, 
                     (0, 1): 1, (1, 3): 1, (3, 3): 1, (0, 0): 1}), 
            Counter({(0, 3): 3, (1, 2): 2, (2, 3): 2, (0, 1): 1, (1, 3): 1, 
                     (3, 3): 1}), 
            Counter({(0, 1): 1, (1, 3): 1, (3, 3): 1, (0, 2): 1}), 
            Counter({(0, 3): 1, (2, 3): 1, (0, 2): 1}), 
            Counter({(0, 3): 1})]
        ctrl_av_depth_by_pair = {
            (0, 1): 2.25, (1, 2): 2.6666666666666665, (0, 0): 1.0, 
            (3, 3): 1.5714285714285714, (0, 2): 1.8571428571428572, 
            (1, 3): 3.0, (2, 3): 1.625, (2, 2): 1.0, 
            (0, 3): 2.1666666666666665, (1, 1): 1.0}
        ctrl_weighted= {
            (0, 1): 1.0, (1, 2): 0.0, (0, 0): 1.0, (3, 3): 3.0, (0, 2): 3.5, 
            (1, 3): 0.0, (2, 3): 5.5, (2, 2): 3.0, (0, 3): 4.0, (1, 1): 0.5}
        for result, ctrl in zip(
            results, 
            (ctrl_cts_by_pair, ctrl_ct_pairs_by_depth,
             ctrl_av_depth_by_pair, ctrl_weighted)):
            self.assertEqual(result, ctrl)       

if __name__ == '__main__':
    unittest.main()
