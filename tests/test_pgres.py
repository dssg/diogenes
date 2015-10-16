import unittest
import os
import subprocess

import utils_for_tests as uft

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
        if subprocess.call('psql', '-h', pgres_host, '-d', pgres_db, '-U',
                           pgres_user, '-p', pgres_port, '-f', 
                           uft.path_of_data('populate_pgres.sql')):
            print 'Could not populate database. Skipping TestPgres'
            cls.skip = True
            return
        cls.conn_str = 'postgresql://{}:{}@{}:{}/{}'.format(
                pgres_user,
                pgres_pw,
                pgres_host,
                pgres_port,
                pgres_db)


    @classmethod
    def tearDownClass(cls):
        if cls.existing_pg_pw is not None:
            os.environ['PGPASSWORD'] = cls.existing_pg_pw

    def setUp(self):
        if self.skip:
            self.skipTest('Valid PGres environment not found')


if __name__ == '__main__':
    unittest.main()
