import unittest
import itertools as it
from datetime import datetime
import numpy as np

import utils_for_tests as uft
from diogenes import array_emitter
from diogenes.grid_search.standard_clfs import DBG_std_clfs
from diogenes.utils import append_cols

class TestArrayEmitter(unittest.TestCase):

    def test_basic(self):
        db_file = uft.path_of_data('rg_students.db')
        conn_str = 'sqlite:///{}'.format(db_file)
        ae = array_emitter.ArrayEmitter()
        ae = ae.get_rg_from_sql(conn_str, 'rg_students')
        ae = ae.set_aggregation('absences', 'MAX')
        ae = ae.set_interval(2005, 2007)
        ae = ae.set_label_feature('graduated')
        ae = ae.set_label_interval(2009, 2009)
        res = ae.emit_M()
        ctrl = np.array([(0, 2.2, 3.95, 8.0, 1.0),
                         (1, 3.45, np.nan, 0.0, 0.0),
                         (2, 3.4, np.nan, 96.0, np.nan)],
                        dtype=[('id', '<i8'), ('math_gpa_AVG', '<f8'), 
                               ('english_gpa_AVG', '<f8'), 
                               ('absences_MAX', '<f8'),
                               ('graduated_AVG', '<f8')])
        self.assertTrue(uft.array_equal(res, ctrl))

    def test_multiple_aggr(self):
        db_file = uft.path_of_data('rg_students.db')
        conn_str = 'sqlite:///{}'.format(db_file)
        ae = array_emitter.ArrayEmitter()
        ae = ae.get_rg_from_sql(conn_str, 'rg_students')
        ae = ae.set_default_aggregation(['AVG', 'MIN', 'MAX', 'COUNT'])
        ae = ae.set_aggregation('absences', ['MIN', 'MAX'])
        ae = ae.set_aggregation('graduated', 'MAX')
        ae = ae.set_interval(2005, 2007)
        ae = ae.set_label_feature('graduated')
        ae = ae.set_label_interval(2009, 2009)
        res = ae.emit_M()
        ctrl = np.array([(0, 2.2, 2.1, 2.3, 2, 3.95, 3.9, 4.0, 2, 7.0, 8.0, 1.0),
                         (1, 3.45, 3.4, 3.5, 2, np.nan, np.nan, np.nan, np.nan, 
                          0.0, 0.0, 0.0),
                         (2, 3.4, 3.4, 3.4, 1.0, np.nan, np.nan, np.nan, np.nan, 
                          14.0, 96.0, np.nan)],
                        dtype=[('id', '<i8'), 
                               ('math_gpa_AVG', '<f8'), 
                               ('math_gpa_MIN', '<f8'), 
                               ('math_gpa_MAX', '<f8'), 
                               ('math_gpa_COUNT', '<i8'), 
                               ('english_gpa_AVG', '<f8'), 
                               ('english_gpa_MIN', '<f8'), 
                               ('english_gpa_MAX', '<f8'), 
                               ('english_gpa_COUNT', '<f8'), 
                               ('absences_MIN', '<f8'),
                               ('absences_MAX', '<f8'),
                               ('graduated_MAX', '<f8')])
        self.assertTrue(uft.array_equal(res, ctrl))

    def test_complex_date(self):
        db_file = uft.path_of_data('rg_complex_dates.db')
        conn_str = 'sqlite:///{}'.format(db_file)
        ae = array_emitter.ArrayEmitter(convert_to_unix_time=True)
        ae = ae.set_aggregation('bounded', 'SUM')
        ae = ae.set_aggregation('no_start', 'SUM')
        ae = ae.set_aggregation('no_stop', 'SUM')
        ae = ae.set_aggregation('unbounded', 'SUM')
        ae = ae.get_rg_from_sql(conn_str, 'rg_complex_dates', 
                                feature_col='feature')
        res1 = ae.set_interval(
            datetime(2010, 1, 1), 
            datetime(2010, 6, 30)).emit_M()
        res2 = ae.set_interval(
            datetime(2010, 7, 1), 
            datetime(2010, 12, 31)).emit_M()
        res3 = ae.set_interval(
            datetime(2010, 1, 1), 
            datetime(2010, 12, 31)).emit_M()
        ctrl_dtype = [('id', '<i8'), ('bounded_SUM', '<f8'), 
                      ('no_start_SUM', '<f8'), ('no_stop_SUM', '<f8'), 
                      ('unbounded_SUM', '<f8')]
        ctrl1_dat = [(0, 1.0, 100.0, 100000.0, 1000000.0),
                     (1, 0.01, 0.001, 1e-06, 1e-07), 
                     (2, np.nan, np.nan, np.nan, 2e-08)]
        ctrl2_dat = [(0, 10.0, 1000.0, 10000.0, 1000000.0),
                     (1, 0.1, 0.0001, 1e-05, 1e-07),
                     (2, np.nan, np.nan, np.nan, 2e-08)]
        ctrl3_dat = [(0, 11.0, 1100.0, 110000.0, 1000000.0),
                     (1, 0.11, 0.0011, 1.1e-05, 1e-07),
                     (2, np.nan, np.nan, np.nan, 2e-08)]
        for res, ctrl_dat in zip((res1, res2, res3), (ctrl1_dat, ctrl2_dat, 
                                                      ctrl3_dat)):
            self.assertTrue(uft.array_equal(
                res, 
                np.array(ctrl_dat, dtype=ctrl_dtype)))  

    def test_from_csv(self):
        db_file = uft.path_of_data('rg_complex_dates.csv')
        ae = array_emitter.ArrayEmitter()
        ae = ae.set_aggregation('bounded', 'SUM')
        ae = ae.set_aggregation('no_start', 'SUM')
        ae = ae.set_aggregation('no_stop', 'SUM')
        ae = ae.set_aggregation('unbounded', 'SUM')
        ae = ae.get_rg_from_csv(db_file, feature_col='feature',
                                parse_datetimes=['start', 'stop'])
        res1 = ae.set_interval(
            datetime(2010, 1, 1), 
            datetime(2010, 6, 30)).emit_M()
        res2 = ae.set_interval(
            datetime(2010, 7, 1), 
            datetime(2010, 12, 31)).emit_M()
        res3 = ae.set_interval(
            datetime(2010, 1, 1), 
            datetime(2010, 12, 31)).emit_M()
        ctrl_dtype = [('id', '<i8'), ('bounded_SUM', '<f8'), 
                      ('no_start_SUM', '<f8'), ('no_stop_SUM', '<f8'), 
                      ('unbounded_SUM', '<f8')]
        ctrl1_dat = [(0, 1.0, 100.0, 100000.0, 1000000.0),
                     (1, 0.01, 0.001, 1e-06, 1e-07), 
                     (2, np.nan, np.nan, np.nan, 2e-08)]
        ctrl2_dat = [(0, 10.0, 1000.0, 10000.0, 1000000.0),
                     (1, 0.1, 0.0001, 1e-05, 1e-07),
                     (2, np.nan, np.nan, np.nan, 2e-08)]
        ctrl3_dat = [(0, 11.0, 1100.0, 110000.0, 1000000.0),
                     (1, 0.11, 0.0011, 1.1e-05, 1e-07),
                     (2, np.nan, np.nan, np.nan, 2e-08)]
        for res, ctrl_dat in zip((res1, res2, res3), (ctrl1_dat, ctrl2_dat, 
                                                      ctrl3_dat)):
            self.assertTrue(uft.array_equal(
                res, 
                np.array(ctrl_dat, dtype=ctrl_dtype)))  

    def test_select_rows_in_M(self):
        db_file = uft.path_of_data('rg_select_rows_in_M.db')
        conn_str = 'sqlite:///{}'.format(db_file)
        ae = array_emitter.ArrayEmitter()
        ae = ae.get_rg_from_sql(conn_str, 'select_rows_in_M')
        ae = ae.set_default_aggregation('SUM')
        ae_1 = ae.set_interval(2005, 2006)
        ae_1 = ae_1.select_rows_in_M('cohort_SUM = 2009')
        ae_2 = ae.set_interval(2005, 2007)
        ae_2 = ae_2.select_rows_in_M('cohort_SUM = 2010')
        ae_1_1 = ae_1.select_rows_in_M('took_ap_compsci_SUM')
        ae_1_2 = ae_1.select_rows_in_M('NOT took_ap_compsci_SUM')
        ae_2_1 = ae_2.select_rows_in_M('took_ap_compsci_SUM')
        ae_2_2 = ae_2.select_rows_in_M('NOT took_ap_compsci_SUM')
        ctrl_dtype = [('id', '<i8'), ('math_gpa_SUM', '<f8'), 
                      ('english_gpa_SUM', '<f8'), ('absences_SUM', '<f8'), 
                      ('cohort_SUM', '<f8'), ('took_ap_compsci_SUM', '<f8')]
        ctrl_data = [[(0, 1.0, 1.0, 1.0, 2009.0, 1.0)],
                     [(2, 3.0, 3.0, 3.0, 2009.0, 0.0)],
                     [(1, 2.2, 2.2, 2.2, 2010.0, 1.0)],
                     [(3, 4.4, 4.4, 4.4, 2010.0, 0.0)]]
        for ae_sel, dat in zip((ae_1_1, ae_1_2, ae_2_1, ae_2_2), ctrl_data):
            ctrl = np.array(dat, dtype=ctrl_dtype)
            res = ae_sel.emit_M()
            self.assertTrue(ctrl, res)

    def test_subset_over(self):
        db_file = uft.path_of_data('rg_subset_over.db')
        conn_str = 'sqlite:///{}'.format(db_file)
        ae = array_emitter.ArrayEmitter()
        ae = ae.get_rg_from_sql(conn_str, 'subset_over')
        ae = ae.set_default_aggregation('SUM')
        exp = ae.subset_over(
            label_col='label',
            interval_train_window_start=2004,
            interval_train_window_end=2005,
            interval_test_window_start=2006,
            interval_test_window_end=2007,
            interval_inc_value=1,
            interval_expanding=False,
            row_M_col_name='cohort',
            row_M_train_window_start=2008,
            row_M_train_window_end=2008,
            row_M_test_window_start=2009,
            row_M_test_window_end=2009,
            row_M_inc_value=1,
            row_M_expanding=False,
            clfs=DBG_std_clfs)
        exp.make_report(verbose=False)
        exp.make_csv()

    def test_subset_over_label_windows(self):
        db_file = uft.path_of_data('rg_label_windows.db')
        conn_str = 'sqlite:///{}'.format(db_file)
        ae = array_emitter.ArrayEmitter()
        ae = ae.get_rg_from_sql(conn_str, 'label_windows')
        ae = ae.set_default_aggregation('SUM')
        exp = ae.subset_over(
            label_col='inspection',
            interval_train_window_start=2000,
            interval_train_window_end=2001,
            interval_test_window_start=2002,
            interval_test_window_end=2003,
            interval_inc_value=1,
            interval_expanding=False,
            label_interval_train_window_start=2007,
            label_interval_train_window_end=2007,
            label_interval_test_window_start=2009,
            label_interval_test_window_end=2009,
            label_interval_inc_value=1,
            label_interval_expanding=False)
        exp.make_csv('label_window.csv')

    def test_feature_gen_lambda(self):

        def feature_gen(M, labels, test_or_train, interval_start, interval_end, 
                        label_interval_start, label_interval_end,
                        row_M_start, row_M_end):
            return (append_cols(M, M['relevent_feature'] * 2 if 
                        test_or_train == 'test' 
                        else M['relevent_feature'] * 3, 'mult'),
                    labels)
        db_file = uft.path_of_data('rg_subset_over.db')
        conn_str = 'sqlite:///{}'.format(db_file)
        ae = array_emitter.ArrayEmitter()
        ae = ae.get_rg_from_sql(conn_str, 'subset_over')
        ae = ae.set_default_aggregation('SUM')
        exp = ae.subset_over(
            label_col='label',
            interval_train_window_start=2004,
            interval_train_window_end=2005,
            interval_test_window_start=2006,
            interval_test_window_end=2007,
            interval_inc_value=1,
            interval_expanding=False,
            row_M_col_name='cohort',
            row_M_train_window_start=2008,
            row_M_train_window_end=2008,
            row_M_test_window_start=2009,
            row_M_test_window_end=2009,
            row_M_inc_value=1,
            row_M_expanding=False,
            clfs=DBG_std_clfs,
            feature_gen_lambda=feature_gen)
        for run in it.chain.from_iterable(
                [trial.runs_flattened() for trial in exp.trials]):
            relevent_idx = run.col_names.index('relevent_feature')
            mult_idx = run.col_names.index('mult')
            self.assertTrue(
                    np.allclose(run.M[:,relevent_idx] * 3, run.M[:,mult_idx]))
            self.assertTrue(
                    np.allclose(run.M_test[:,relevent_idx] * 2, 
                                run.M_test[:,mult_idx]))


if __name__ == '__main__':
    unittest.main()


