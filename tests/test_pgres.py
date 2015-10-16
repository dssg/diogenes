import unittest
import os
import subprocess

import utils_for_tests as uft
import numpy as np

from diogenes import read

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

if __name__ == '__main__':
    unittest.main()
