import unittest
from datetime import datetime
import numpy as np

import utils_for_tests as uft
from eights import array_emitter

class TestArrayEmitter(unittest.TestCase):

    def test_basic(self):
        db_file = uft.path_of_data('rg_students.db')
        conn_str = 'sqlite:///{}'.format(db_file)
        ae = array_emitter.ArrayEmitter()
        ae = ae.get_rg_from_sql(conn_str, 'rg_students')
        ae = ae.set_aggregation('absences', 'MAX')
        ae = ae.set_interval(2005, 2007)
        res = ae.emit_M()
        ctrl = np.array([(0, 2.2, 3.95, 8.0),
                         (1, 3.45, np.nan, 0.0),
                         (2, 3.4, np.nan, 96.0)],
                        dtype=[('id', '<i8'), ('math_gpa', '<f8'), 
                               ('english_gpa', '<f8'), 
                               ('absences', '<f8')])
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
        ctrl_dtype = [('id', '<i8'), ('bounded', '<f8'), 
                      ('no_start', '<f8'), ('no_stop', '<f8'), 
                      ('unbounded', '<f8')]
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
        ae = ae.get_rg_from_csv(db_file, feature_col='feature')
        res1 = ae.set_interval(
            datetime(2010, 1, 1), 
            datetime(2010, 6, 30)).emit_M()
        res2 = ae.set_interval(
            datetime(2010, 7, 1), 
            datetime(2010, 12, 31)).emit_M()
        res3 = ae.set_interval(
            datetime(2010, 1, 1), 
            datetime(2010, 12, 31)).emit_M()
        ctrl_dtype = [('id', '<i8'), ('bounded', '<f8'), 
                      ('no_start', '<f8'), ('no_stop', '<f8'), 
                      ('unbounded', '<f8')]
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
        ae_1 = ae_1.select_rows_in_M('cohort = 2009')
        ae_2 = ae.set_interval(2005, 2007)
        ae_2 = ae_2.select_rows_in_M('cohort = 2010')
        ae_1_1 = ae_1.select_rows_in_M('took_ap_compsci')
        ae_1_2 = ae_1.select_rows_in_M('NOT took_ap_compsci')
        ae_2_1 = ae_2.select_rows_in_M('took_ap_compsci')
        ae_2_2 = ae_2.select_rows_in_M('NOT took_ap_compsci')
        ctrl_dtype = [('id', '<i8'), ('math_gpa', '<f8'), 
                      ('english_gpa', '<f8'), ('absences', '<f8'), 
                      ('cohort', '<f8'), ('took_ap_compsci', '<f8')]
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
        for train_M, train_y, test_M, test_y in ae.subset_over(
            label_feat='label',
            interval_train_window_start=2004,
            interval_train_window_size=1,
            interval_test_window_start=2006,
            interval_test_window_size=1,
            interval_inc_value=1,
            interval_expanding=False,
            row_M_col_name='cohort',
            row_M_train_window_start=2008,
            row_M_train_window_size=0,
            row_M_test_window_start=2009,
            row_M_test_window_size=0,
            row_M_inc_value=1,
            row_M_expanding=False):
            print '-'*80
            print train_M
            print train_M.dtype
            print train_y
            print train_y.dtype
            print
            print test_M
            print test_M.dtype
            print test_y
            print test_y.dtype
            print

if __name__ == '__main__':
    unittest.main()


