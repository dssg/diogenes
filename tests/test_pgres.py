import unittest
import os
import subprocess

import utils_for_tests as uft
import numpy as np

from diogenes import read
from diogenes import array_emitter

class TestPgres(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skip = False
        try:
            pgres_host = os.environ['DIOGENES_PGRES_TEST_HOST']
            pgres_db = os.environ['DIOGENES_PGRES_TEST_DB']
            pgres_user = os.environ['DIOGENES_PGRES_TEST_USER']
            pgres_pw = os.environ['DIOGENES_PGRES_TEST_PW']
        except KeyError:
            print ('TestPgres requires the following environmental variables '
                   'to be defined:\n'
                   '* DIOGENES_PGRES_TEST_HOST\n'
                   '* DIOGENES_PGRES_TEST_DB\n'
                   '* DIOGENES_PGRES_TEST_USER\n'
                   '* DIOGENES_PGRES_TEST_PW\n'
                   '* DIOGENES_PGRES_TEST_PORT (optional)\n'
                   'At least one of these is not defined. Skipping TestPgres')
            cls.skip = True
            return
        try:
            pgres_port = os.environ['DIOGENES_PGRES_TEST_PORT']
        except KeyError:
            pgres_port = '5432'
        os.environ['PGPASSWORD'] = pgres_pw
        if subprocess.call(['psql', '-h', pgres_host, '-d', pgres_db, '-U',
                            pgres_user, '-p', pgres_port, '-f', 
                            uft.path_of_data('populate_pgres.sql')]):
            print 'Could not populate database. Skipping TestPgres'
            cls.skip = True
            return
        cls.conn_str = 'postgresql://{}:{}@{}:{}/{}'.format(
                pgres_user,
                pgres_pw,
                pgres_host,
                pgres_port,
                pgres_db)

    def setUp(self):
        if self.skip:
            self.skipTest('Valid PGres environment not found')

    def test_basic_query(self):
        conn = read.connect_sql(self.conn_str)
        self.assertTrue(conn.psql_optimized)
        sa = conn.execute('SELECT * FROM employees')
        ctrl = np.array([(1, u'Arthur', u'King', 40000.0, 2.1, 10),
                         (2, u'Jones', u'James', 1000000.0, 1.9, 2),
                         (3, u'The Moabite', u'Ruth', 50000.0, 1.8, 6)],
                        dtype=[('id', '<i8'), ('last_name', 'S11'), 
                               ('first_name', 'S5'), 
                               ('salary', '<f8'), ('height', '<f8'), 
                               ('usefulness', '<i8')])
        self.assertTrue(np.array_equal(sa, ctrl))

    def test_array_emitter(self):
        db_file = uft.path_of_data('rg_complex_dates.db')
        ae = array_emitter.ArrayEmitter(convert_to_unix_time=True)
        ae = ae.set_aggregation('bounded', 'SUM')
        ae = ae.set_aggregation('no_start', 'SUM')
        ae = ae.set_aggregation('no_stop', 'SUM')
        ae = ae.set_aggregation('unbounded', 'SUM')
        ae = ae.get_rg_from_sql(self.conn_str, 'rg_complex_dates', 
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

if __name__ == '__main__':
    unittest.main()
