import unittest
import numpy as np
from collections import Counter

from diogenes.utils import remove_cols,cast_list_of_list_to_sa

import utils_for_tests 

import unittest
import numpy as np
from numpy.random import rand
import diogenes.read
import diogenes.utils
from diogenes.modify import remove_cols_where
from diogenes.modify import col_val_eq
from diogenes.modify import col_val_eq_any
from diogenes.modify import col_fewer_than_n_nonzero
from diogenes.modify import where_all_are_true 
from diogenes.modify import choose_rows_where 
from diogenes.modify import remove_rows_where
from diogenes.modify import row_val_eq
from diogenes.modify import row_val_lt 
from diogenes.modify import row_val_between
from diogenes.modify import combine_cols
from diogenes.modify import combine_sum
from diogenes.modify import combine_mean 
from diogenes.modify import label_encode
from diogenes.modify import generate_bin
from diogenes.modify import normalize
from diogenes.modify import replace_missing_vals
from diogenes.modify import distance_from_point

class TestModify(unittest.TestCase):
    def test_col_val_eq(self):
        M = cast_list_of_list_to_sa(
            [[1,2,3], [1,3,4], [1,4,5]],
            col_names=['height','weight', 'age'])
        
        arguments = [{'func': col_val_eq,  'vals': 1}]  
        M = remove_cols_where(M, arguments)  
        correct = cast_list_of_list_to_sa(
            [[2,3], [3,4], [4,5]],
            col_names=['weight', 'age'])    
        self.assertTrue(np.array_equal(M, correct))
        
    def test_col_val_eq_any(self):
        M = cast_list_of_list_to_sa(
            [[1,2,3], [1,3,4], [1,4,5]],
            col_names=['height','weight', 'age'])
        arguments = [{'func': col_val_eq_any,  'vals': None}]  
        M = remove_cols_where(M, arguments)  
        correct = cast_list_of_list_to_sa(
            [[2,3], [3,4], [4,5]],
            col_names=['weight', 'age'])    
        self.assertTrue(np.array_equal(M, correct)) 
        
    def test_col_fewer_than_n_nonzero(self):
        M = cast_list_of_list_to_sa(
            [[0,2,3], [0,3,4], [1,4,5]],
            col_names=['height','weight', 'age'])
        arguments = [{'func': col_fewer_than_n_nonzero,  'vals': 2}]  
        M = remove_cols_where(M, arguments)  
        correct = cast_list_of_list_to_sa(
            [[2,3], [3,4], [4,5]],
            col_names=['weight', 'age'])   
        self.assertTrue(np.array_equal(M, correct))    
                   
    def test_label_encoding(self):
        M = np.array(
            [('a', 0, 'Martin'),
             ('b', 1, 'Tim'),
             ('b', 2, 'Martin'),
             ('c', 3, 'Martin')],
            dtype=[('letter', 'O'), ('idx', int), ('name', 'O')])
        ctrl = np.array(
            [(0, 0, 0),
             (1, 1, 1),
             (1, 2, 0),
             (2, 3, 0)],
            dtype=[('letter', int), ('idx', int), ('name', int)])
        ctrl_classes = {'letter': np.array(['a', 'b', 'c']),
                        'name': np.array(['Martin', 'Tim'])}
        new_M, classes = label_encode(M)
        self.assertTrue(np.array_equal(ctrl, new_M))
        self.assertEqual(ctrl_classes.keys(), classes.keys())
        for key in ctrl_classes:
            self.assertTrue(np.array_equal(ctrl_classes[key], classes[key]))
        
    def test_replace_missing_vals(self):
        M = np.array([('a', 0, 0.0, 0.1),
                      ('b', 1, 1.0, np.nan),
                      ('', -999, np.nan, 0.0),
                      ('d', 1, np.nan, 0.2),
                      ('', -999, 2.0, np.nan)],
                     dtype=[('str', 'O'), ('int', int), ('float1', float),
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

        arguments = [{'func': row_val_eq, 'col_name': 'heigh', 'vals': 1},
                     {'func': row_val_lt, 'col_name': 'weight', 'vals': 3},
                     {'func': row_val_between, 'col_name': 'age', 'vals': 
                      (3, 4)}]

        res = where_all_are_true(
            M, 
            arguments)

        ctrl = np.array([True, False, False])
                   
        self.assertTrue(np.array_equal(res, ctrl))

    def test_choose_rows_where(self):
        M = [[1,2,3], [2,3,4], [3,4,5]]
        col_names = ['heigh','weight', 'age']
        lables= [0,0,1]
        M = diogenes.utils.cast_list_of_list_to_sa(
            M,
            col_names=col_names)

        arguments = [{'func': row_val_eq, 'col_name': 'heigh', 'vals': 1},
                     {'func': row_val_lt, 'col_name': 'weight', 'vals': 3},
                     {'func': row_val_between, 'col_name': 'age', 'vals': 
                      (3, 4)}]

        res = choose_rows_where(
            M, 
            arguments)

        ctrl = cast_list_of_list_to_sa([[1,2,3]],col_names=['heigh','weight', 'age'])
                   
        self.assertTrue(np.array_equal(res, ctrl))

    def test_remove_rows_where(self):
        M = [[1,2,3], [2,3,4], [3,4,5]]
        col_names = ['heigh','weight', 'age']
        lables= [0,0,1]
        M = diogenes.utils.cast_list_of_list_to_sa(
            M,
            col_names=col_names)

        arguments = [{'func': row_val_eq, 'col_name': 'heigh', 'vals': 1},
                     {'func': row_val_lt, 'col_name': 'weight', 'vals': 3},
                     {'func': row_val_between, 'col_name': 'age', 'vals': 
                      (3, 4)}]

        res = remove_rows_where(
            M, 
            arguments)

        ctrl = cast_list_of_list_to_sa([[2,3,4],[3,4,5]],col_names=['heigh','weight', 'age'])
                   
        self.assertTrue(np.array_equal(res, ctrl))

    def test_combine_cols(self):
        M = np.array(
                [(0, 1, 2), (3, 4, 5), (6, 7, 8)], 
                dtype=[('f0', float), ('f1', float), ('f2', float)])
        ctrl_sum = np.array([1, 7, 13]) 
        ctrl_mean = np.array([1.5, 4.5, 7.5])
        res_sum = combine_cols(M, combine_sum, ('f0', 'f1'))
        res_mean = combine_cols(M, combine_mean, ('f1', 'f2'))
        self.assertTrue(np.array_equal(res_sum, ctrl_sum))
        self.assertTrue(np.array_equal(res_mean, ctrl_mean))

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

