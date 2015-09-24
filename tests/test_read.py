import unittest
import numpy as np
from numpy.random import rand

from diogenes.display.display import describe_cols

from diogenes.read.read import plot_correlation_matrix
from diogenes.read.read import plot_correlation_scatter_plot
from diogenes.read.read import convert_to_sa
from diogenes.read.read import plot_kernel_density
from diogenes.read.read import connect_sql
from diogenes.read.read import cast_list_of_list_to_sa
from diogenes.read.read import open_csv_as_structured_array, describe_cols,open_csv_as_list,cast_list_of_list_to_sa, crosstab
from diogenes import utils

import utils_for_tests 

from collections import Counter
class TestInvestigate(unittest.TestCase):

    #1
    def test_open_csv_as_list(self):
        csv_file = utils_for_tests.path_of_data("mixed.csv")
        correct = [[0, 'Jim', 5.6], [1, 'Jill', 5.5]]
        self.assertEqual(open_csv_as_list(csv_file),correct)

    #2    
    def test_open_csv_as_structured_array(self):
        csv_file = utils_for_tests.path_of_data("mixed.csv")
        correct = np.array([(0, 'Jim', 5.6), (1, 'Jill', 5.5)],dtype=[('id', '<i8'), ('name', 'S4'), ('height', '<f8')])
        self.assertTrue(np.array_equal(open_csv_as_structured_array(csv_file),correct))

    #3    
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
        self.assertTrue(utils_for_tests.array_equal(ctrl_list, 
                                                    describe_cols(test_list)))
        self.assertTrue(utils_for_tests.array_equal(ctrl_list, 
                                                    describe_cols(test_nd)))
        ctrl_sa = np.array([('id', 6, 3.5, 1.707825127659933, 1, 6),
                            ('val', 6, 4.5, 1.707825127659933, 2, 7),
                            ('name', np.nan, np.nan, np.nan, np.nan, np.nan)],
                           dtype=[('Column Name', 'S4'), ('Count', float),
                                  ('Mean', float), ('Standard Dev', float),
                                  ('Minimum', float), ('Maximum', float)])
        self.assertTrue(utils_for_tests.array_equal(ctrl_sa, 
                                                    describe_cols(test_sa)))

    #4
    def test_cast_list_of_list_to_sa(self):
        test = [[1,2.,'a'],[2,4.,'b'],[4,5.,'g']]
        names = ['ints','floats','strings']
        correct_1 = np.array([(1, 2.0, 'a'), (2, 4.0, 'b'), (4, 5.0, 'g')],dtype=[('f0', '<i8'), ('f1', '<f8'), ('f2', 'S1')])
        correct_2 = np.array([(1, 2.0, 'a'), (2, 4.0, 'b'), (4, 5.0, 'g')], dtype=[('ints', '<i8'), ('floats', '<f8'), ('strings', 'S1')])
        self.assertTrue(np.array_equal(correct_1, cast_list_of_list_to_sa(test)))
        self.assertTrue(np.array_equal(correct_2, cast_list_of_list_to_sa(test, names)))

    #5
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
        self.assertTrue(np.array_equal(correct, crosstab(l1,l2)))
        
    def test_plot_correlation_scatter_plot(self):
        data = rand(100, 3)
        fig = plot_correlation_scatter_plot(data, verbose=False) 
        

    
if __name__ == '__main__':
    unittest.main()


