import unittest
import numpy as np
from collections import Counter

from diogenes.utils import remove_cols,cast_list_of_list_to_sa

import utils_for_tests 

from diogenes.modify import label_encode
from diogenes.modify import replace_missing_vals

from diogenes.modify import (col_has_all_same_val)

from diogenes.modify import (remove_col_where,
                                      all_equal_to,
                                      all_same_value,
                                      fewer_then_n_nonzero_in_col,
                                      remove_rows_where,
                                      val_eq)
import unittest
import numpy as np
from numpy.random import rand
import diogenes.read
import diogenes.utils
from diogenes.modify import generate_bin
from diogenes.modify import normalize
from diogenes.modify import distance_from_point
from diogenes.modify import where_all_are_true, val_eq, val_lt, val_between
from diogenes.modify import combine_sum, combine_mean, combine_cols



class TestModify(unittest.TestCase):
    def test_are_all_col_equal(self):
        M = cast_list_of_list_to_sa(
            [[1,2,3], [1,3,4], [1,4,5]],
            col_names=['height','weight', 'age'])
        
        arguments = [{'func': all_equal_to,  'vals': 1}]  
        M = remove_col_where(M, arguments)  
        correct = cast_list_of_list_to_sa(
            [[2,3], [3,4], [4,5]],
            col_names=['weight', 'age'])    
        self.assertTrue(np.array_equal(M, correct))
        
    def test_all_same_value(self):
        M = cast_list_of_list_to_sa(
            [[1,2,3], [1,3,4], [1,4,5]],
            col_names=['height','weight', 'age'])
        arguments = [{'func': all_same_value,  'vals': None}]  
        M = remove_col_where(M, arguments)  
        correct = cast_list_of_list_to_sa(
            [[2,3], [3,4], [4,5]],
            col_names=['weight', 'age'])    
        self.assertTrue(np.array_equal(M, correct)) 
        
    def test_fewer_then_n_nonzero_in_col(self):
        M = cast_list_of_list_to_sa(
            [[0,2,3], [0,3,4], [1,4,5]],
            col_names=['height','weight', 'age'])
        arguments = [{'func': fewer_then_n_nonzero_in_col,  'vals': 2}]  
        M = remove_col_where(M, arguments)  
        correct = cast_list_of_list_to_sa(
            [[2,3], [3,4], [4,5]],
            col_names=['weight', 'age'])   
        self.assertTrue(np.array_equal(M, correct))    
                   
    def test_remove_row(self):
        M = cast_list_of_list_to_sa(
            [[0,2,3], [0,3,4], [1,4,5]],
            col_names=['height','weight', 'age'])
        arguments = [{'func': fewer_then_n_nonzero_in_col,  'vals': 2}]  
        M = remove_rows_where(M, val_eq, 'weight', 3)
        correct = cast_list_of_list_to_sa(
             [[0, 2, 3], [1, 4, 5]],
            col_names=['height','weight', 'age'])   
        self.assertTrue(np.array_equal(M, correct))   
    
    def test_label_encoding(self):
        M = np.array(
            [('a', 0, 'Martin'),
             ('b', 1, 'Tim'),
             ('b', 2, 'Martin'),
             ('c', 3, 'Martin')],
            dtype=[('letter', 'S1'), ('idx', int), ('name', 'S6')])
        ctrl = np.array(
            [(0, 0, 0),
             (1, 1, 1),
             (1, 2, 0),
             (2, 3, 0)],
            dtype=[('letter', int), ('idx', int), ('name', int)])
        self.assertTrue(np.array_equal(ctrl, label_encode(M)))
        
    def test_replace_missing_vals(self):
        M = np.array([('a', 0, 0.0, 0.1),
                      ('b', 1, 1.0, np.nan),
                      ('', -999, np.nan, 0.0),
                      ('d', 1, np.nan, 0.2),
                      ('', -999, 2.0, np.nan)],
                     dtype=[('str', 'S1'), ('int', int), ('float1', float),
                            ('float2', float)])

        ctrl = M.copy()
        ctrl['float1'] = np.array([0.0, 1.0, -1.0, -1.0, 2.0])
        ctrl['float2'] = np.array([0.1, -1.0, 0.0, 0.2, -1.0])
        res = replace_missing_vals(M, 'constant', constant=-1.0)
        self.assertTrue(np.array_equal(ctrl, res))

        ctrl = M.copy()
        ctrl['int'] = np.array([100, 1, -999, 1, -999])
        ctrl['float1'] = np.array([100, 1.0, np.nan, np.nan, 2.0])
        ctrl['float2'] = np.array([0.1, np.nan, 100, 0.2, np.nan])
        res = replace_missing_vals(M, 'constant', missing_val=0, constant=100)
        self.assertTrue(utils_for_tests.array_equal(ctrl, res))

        ctrl = M.copy()
        ctrl['int'] = np.array([0, 1, 1, 1, 1])
        res = replace_missing_vals(M, 'most_frequent', missing_val=-999)
        self.assertTrue(utils_for_tests.array_equal(ctrl, res))

        ctrl = M.copy()
        ctrl['float1'] = np.array([0.0, 1.0, 1.0, 1.0, 2.0])
        ctrl['float2'] = np.array([0.1, 0.1, 0.0, 0.2, 0.1])
        res = replace_missing_vals(M, 'mean', missing_val=np.nan)
        self.assertTrue(utils_for_tests.array_equal(ctrl, res))
        

    def test_generate_bin(self):
        M = [1, 1, 1, 3, 3, 3, 5, 5, 5, 5, 2, 6]
        ctrl = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 0, 3]
        self.assertTrue(np.array_equal(ctrl, generate_bin(M, 3)))
        M = np.array([0.1, 3.0, 0.0, 1.2, 2.5, 1.7, 2])
        ctrl = [0, 3, 0, 1, 2, 1, 2]
        self.assertTrue(np.array_equal(ctrl, generate_bin(M, 3)))
    

    def test_where_all_are_true(self):
        M = [[1,2,3], [2,3,4], [3,4,5]]
        col_names = ['heigh','weight', 'age']
        lables= [0,0,1]
        M = diogenes.utils.cast_list_of_list_to_sa(
            M,
            col_names=col_names)

        arguments = [{'func': val_eq, 'col_name': 'heigh', 'vals': 1},
                     {'func': val_lt, 'col_name': 'weight', 'vals': 3},
                     {'func': val_between, 'col_name': 'age', 'vals': 
                      (3, 4)}]

        res = where_all_are_true(
            M, 
            arguments,
            'eq_to_stuff')
        ctrl = np.array(
            [(1, 2, 3, True), (2, 3, 4, False), (3, 4, 5, False)], 
            dtype=[('heigh', '<i8'), ('weight', '<i8'), ('age', '<i8'),
                   ('eq_to_stuff', '?')])
                   
        self.assertTrue(np.array_equal(res, ctrl))

    def test_combine_cols(self):
        M = np.array(
                [(0, 1, 2), (3, 4, 5), (6, 7, 8)], 
                dtype=[('f0', float), ('f1', float), ('f2', float)])
        ctrl = np.array(
                [(0, 1, 2, 1, 1.5), (3, 4, 5, 7, 4.5), (6, 7, 8, 13, 7.5)], 
                dtype=[('f0', float), ('f1', float), ('f2', float), 
                       ('sum', float), ('avg', float)])
        M = combine_cols(M, combine_sum, ('f0', 'f1'), 'sum')
        M = combine_cols(M, combine_mean, ('f1', 'f2'), 'avg')
        self.assertTrue(np.array_equal(M, ctrl))

    def test_normalize(self):
        col = np.array([-2, -1, 0, 1, 2])
        res, mean, stddev = normalize(col, return_fit=True)
        self.assertTrue(np.allclose(np.std(res), 1.0))
        self.assertTrue(np.allclose(np.mean(res), 0.0))
        col = np.arange(10)
        res = normalize(col, mean=mean, stddev=stddev)
        self.assertTrue(np.allclose(res, (col - mean) / stddev))

    def test_distance_from_point(self):
        # Coords according to https://tools.wmflabs.org/geohack/ 
        # Paris
        lat_origin = 48.8567
        lng_origin = 2.3508

        # New York, Beijing, Jerusalem
        lat_col = [40.7127, 39.9167, 31.7833]
        lng_col = [-74.0059, 116.3833, 35.2167]

        # According to http://www.movable-type.co.uk/scripts/latlong.html
        # (Rounds to nearest km)
        ctrl = np.array([5837, 8215, 3331])
        res = distance_from_point(lat_origin, lng_origin, lat_col, lng_col)

        # get it right within 1km
        self.assertTrue(np.allclose(ctrl, res, atol=1, rtol=0))


if __name__ == '__main__':
    unittest.main()

