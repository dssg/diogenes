import unittest
import numpy as np
from numpy.random import rand

from diogenes.display.display import describe_cols

import diogenes.read as read
from diogenes import utils

import utils_for_tests 

from collections import Counter
class TestRead(unittest.TestCase):

    def test_open_csv(self):
        csv_file = utils_for_tests.path_of_data("mixed.csv")
        correct = np.array([(0, 'Jim', 5.6), (1, 'Jill', 5.5)],dtype=[('id', '<i8'), ('name', 'O'), ('height', '<f8')])
        res = read.open_csv(csv_file)
        self.assertTrue(np.array_equal(res, correct))

    def test_open_csv_url(self): 
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
        sa = read.open_csv_url(url, delimiter=';')
        ctrl_dtype = [('fixed acidity', '<f8'), 
                      ('volatile acidity', '<f8'), 
                      ('citric acid', '<f8'), 
                      ('residual sugar', '<f8'), 
                      ('chlorides', '<f8'), 
                      ('free sulfur dioxide', '<f8'), 
                      ('total sulfur dioxide', '<f8'), 
                      ('density', '<f8'), 
                      ('pH', '<f8'), 
                      ('sulphates', '<f8'), 
                      ('alcohol', '<f8'), 
                      ('quality', '<i8')]
        self.assertEqual(sa.dtype, ctrl_dtype)

    def test_connect_sql(self):
        conn_str = 'sqlite:///{}'.format(utils_for_tests.path_of_data('small.db'))
        conn = read.connect_sql(conn_str)
        sa = conn.execute('SELECT * FROM employees')
        ctrl = np.array([(1, u'Arthur', u'King', 40000.0, 2.1, 10),
                         (2, u'Jones', u'James', 1000000.0, 1.9, 2),
                         (3, u'The Moabite', u'Ruth', 50000.0, 1.8, 6)],
                        dtype=[('id', '<i8'), ('last_name', '<U11'), 
                               ('first_name', '<U5'), 
                               ('salary', '<f8'), ('height', '<f8'), 
                               ('usefulness', '<i8')])
        self.assertTrue(np.array_equal(sa, ctrl))

if __name__ == '__main__':
    unittest.main()


