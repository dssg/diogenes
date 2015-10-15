import unittest
from diogenes import utils
import utils_for_tests
from datetime import datetime

import numpy as np
import pandas as pd

class TestUtils(unittest.TestCase):
    def __sa_check(self, sa1, sa2):
        # This works even if both rows and columns are in different
        # orders in the two arrays
        frozenset_sa1_names = frozenset(sa1.dtype.names)
        frozenset_sa2_names = frozenset(sa2.dtype.names)
        self.assertEqual(frozenset_sa1_names,
                         frozenset_sa2_names)
        sa2_reordered = sa2[list(sa1.dtype.names)]
        sa1_set = {tuple(row) for row in sa1}
        sa2_set = {tuple(row) for row in sa2_reordered}
        self.assertEqual(sa1_set, sa2_set)


    def test_utf_to_ascii(self):
        s = u'\u03BBf.(\u03BBx.f(x x)) (\u05DC.f(x x))'
        ctrl = '?f.(?x.f(x x)) (?.f(x x))'
        res = utils.utf_to_ascii(s)
        self.assertTrue(isinstance(res, str))
        self.assertEqual(ctrl, res)

    def test_validate_time(self):
        trials = [('2014-12-12', True),
                  ('1/2/1999 8:23PM', True),
                  ('1988-08-15T13:43:01.123', True),
                  ('2014-14-12', False), # invalid month
                  ('2012', False), # Just a number
                  ('a', False), # dateutil interprets this as now
                 ]
        
        for (s, ctrl) in trials:
            self.assertEqual(utils.validate_time(s), ctrl)

    def test_str_to_time(self):
        trials = [('2014-12-12', datetime(2014, 12, 12)),
                  ('1/2/1999 8:23PM', datetime(1999, 1, 2, 20, 23)),
                  ('1988-08-15T13:43:01.123', 
                   datetime(1988, 8, 15, 13, 43, 1, 123000)),
                 ]

        for (s, ctrl) in trials:
            self.assertEqual(utils.str_to_time(s), ctrl)

    def test_cast_list_of_list_to_sa2(self):
        L = [[None, None, None],
             ['a',  5,    None],
             ['ab', 'x',  None]]
        ctrl = np.array(
                [('', '', ''), 
                 ('a', '5', ''),
                 ('ab', 'x', '')],
                dtype=[('f0', 'S2'),
                       ('f1', 'S1'),
                       ('f2', 'S1')])
        conv = utils.cast_list_of_list_to_sa(L)
        self.assertTrue(np.array_equal(conv, ctrl))                 
        L = [[None, u'\u05dd\u05d5\u05dc\u05e9', 4.0, 7],
             [2, 'hello', np.nan, None],
             [4, None, None, 14L]]
        ctrl = np.array(
                [(-999, u'\u05dd\u05d5\u05dc\u05e9', 4.0, 7),
                 (2, u'hello', np.nan, -999L),
                 (4, u'', np.nan, 14L)],
                dtype=[('int', int), ('ucode', 'U5'), ('float', float),
                       ('long', long)])
        conv = utils.cast_list_of_list_to_sa(
                L, 
                col_names=['int', 'ucode', 'float', 'long'])
        self.assertTrue(utils_for_tests.array_equal(ctrl, conv))

    def test_cast_list_of_list_to_sa1(self):
        test = [[1,2.,'a'],[2,4.,'b'],[4,5.,'g']]
        names = ['ints','floats','strings']
        correct_1 = np.array([(1, 2.0, 'a'), (2, 4.0, 'b'), (4, 5.0, 'g')],dtype=[('f0', '<i8'), ('f1', '<f8'), ('f2', 'S1')])
        correct_2 = np.array([(1, 2.0, 'a'), (2, 4.0, 'b'), (4, 5.0, 'g')], dtype=[('ints', '<i8'), ('floats', '<f8'), ('strings', 'S1')])
        self.assertTrue(np.array_equal(correct_1, utils.cast_list_of_list_to_sa(test)))
        self.assertTrue(np.array_equal(correct_2, utils.cast_list_of_list_to_sa(test, names)))

    def test_convert_to_sa(self):
        # already a structured array
        sa = np.array([(1, 1.0, 'a', datetime(2015, 01, 01)),
                       (2, 2.0, 'b', datetime(2016, 01, 01))],
                      dtype=[('int', int), ('float', float), ('str', 'S1'),
                             ('date', 'M8[s]')])
        self.assertTrue(np.array_equal(sa, utils.convert_to_sa(sa)))

        # homogeneous array no col names provided
        nd = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        ctrl = np.array([(1, 2, 3), (4, 5, 6), (7, 8, 9)],
                        dtype=[('f0', int), ('f1', int), ('f2', int)])
        self.assertTrue(np.array_equal(ctrl, utils.convert_to_sa(nd)))

        # homogeneous array with col names provided
        nd = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        ctrl = np.array([(1, 2, 3), (4, 5, 6), (7, 8, 9)],
                        dtype=[('i0', int), ('i1', int), ('i2', int)])
        self.assertTrue(np.array_equal(ctrl, utils.convert_to_sa(
            nd,
            col_names=['i0', 'i1', 'i2'])))

        # list of lists no col name provided
        lol = [[1, 1, None],
               ['abc', 2, 3.4]]
        ctrl = np.array([('1', 1, np.nan),
                         ('abc', 2, 3.4)],
                        dtype=[('f0', 'S3'), ('f1', int), ('f2', float)])
        res = utils.convert_to_sa(lol)
        self.assertTrue(utils_for_tests.array_equal(ctrl, res))

        # list of lists with col name provided
        lol = [['hello', 1.2, datetime(2012, 1, 1), None],
               [1.3, np.nan, None, '2013-01-01'],
               [1.4, 1.5, '2014-01-01', 'NO_SUCH_RECORD']]
        ctrl = np.array([('hello', 1.2, datetime(2012, 1, 1), utils.NOT_A_TIME),
                         ('1.3', np.nan, utils.NOT_A_TIME, datetime(2013, 1, 1)),
                         ('1.4', 1.5, datetime(2014, 1, 1), utils.NOT_A_TIME)],
                        dtype=[('i0', 'S5'), ('i1', float), ('i2', 'M8[us]'),
                               ('i3', 'M8[us]')])
        res = utils.convert_to_sa(lol, col_names = ['i0', 'i1', 'i2', 'i3'])
        self.assertTrue(utils_for_tests.array_equal(ctrl, res))

    def test_np_dtype_is_homogeneous(self):
        sa = np.array([(1, 'a', 2)], dtype=[('f0', int), ('f1', 'S1'), 
                                            ('f2', int)])
        self.assertFalse(utils.np_dtype_is_homogeneous(sa))

        sa = np.array([('aa', 'a')], dtype=[('f0', 'S2'), ('f1', 'S1')])
        self.assertFalse(utils.np_dtype_is_homogeneous(sa))

        sa = np.array([(1, 2, 3)], dtype=[('f0', int), ('f1', int),
                                          ('f2', int)])
        self.assertTrue(utils.np_dtype_is_homogeneous(sa))


    def test_sa_to_nd(self):
        dtype = np.dtype({'names': map('f{}'.format, xrange(3)),
                          'formats': [float] * 3})
        sa = np.array([(-1.0, 2.0, -1.0), (0.0, -1.0, 2.0)], dtype=dtype)
        control = np.array([[-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]],
                           dtype=float)
        result = utils.cast_np_sa_to_nd(sa)
        self.assertTrue(np.array_equal(result, control))

    def test_is_sa(self):
        nd = np.array([[1, 2, 3], [4, 5, 6]], dtype=int)
        dtype = np.dtype({'names': map('f{}'.format, xrange(3)),
                          'formats': [float] * 3})
        sa = np.array([(-1.0, 2.0, -1.0), (0.0, -1.0, 2.0)], dtype=dtype)
        self.assertFalse(utils.is_sa(nd))
        self.assertTrue(utils.is_sa(sa))

    def test_is_nd(self):
        nd = np.array([[1, 2, 3], [4, 5, 6]], dtype=int)
        dtype = np.dtype({'names': map('f{}'.format, xrange(3)),
                          'formats': [float] * 3})
        sa = np.array([(-1.0, 2.0, -1.0), (0.0, -1.0, 2.0)], dtype=dtype)
        self.assertTrue(utils.is_nd(nd))
        self.assertTrue(utils.is_nd(sa))

    def test_distance(self):
        # Coords according to https://tools.wmflabs.org/geohack/ 
        # Minneapolis
        lat1 = 44.98
        lng1 = -93.27
        
        # Chicago
        lat2 = 41.84
        lng2 = -87.68

        # Sao Paulo
        lat3 = -23.55
        lng3 = -46.63

        # distances from http://www.movable-type.co.uk/scripts/latlong.html
        # (Rounds to nearest km)

        self.assertTrue(np.allclose(utils.distance(lat1, lng1, lat2, lng2), 
                                                   570.6, atol=1, rtol=0))
        self.assertTrue(np.allclose(utils.distance(lat1, lng1, lat3, lng3), 
                                                   8966, atol=1, rtol=0))

    def test_dist_less_than(self):
        # Minneapolis
        lat1 = 44.98
        lng1 = -93.27
        
        # Chicago
        lat2 = 41.84
        lng2 = -87.68
        
        self.assertTrue(utils.dist_less_than(lat1, lng1, lat2, lng2, 600))
        self.assertFalse(utils.dist_less_than(lat1, lng1, lat2, lng2, 500))

    def test_stack_rows(self):
        dtype = [('id', int), ('name', 'O')]
        M1 = np.array([(1, 'a'), (2, 'b')], dtype=dtype)
        M2 = np.array([(3, 'c'), (4, 'd'), (5, 'e')], dtype=dtype)
        ctrl = np.array([(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd'), (5, 'e')],
                        dtype=dtype)
        res = utils.stack_rows(M1, M2)
        self.assertTrue(np.array_equal(ctrl, res))

    def test_from_cols(self):
        col1 = np.array([1, 2, 3])
        col2 = np.array([4.0, 5.0, 6.0])
        ctrl = np.array(
            [(1, 4.0), (2, 5.0), (3, 6.0)], 
            dtype=[('f0', int), ('f1', float)])
        res = utils.sa_from_cols([col1, col2])
        self.assertTrue(np.array_equal(ctrl, res))
        col1 = np.array(['ab', 'cd', 'ef'])
        col2 = np.array([1, 2, 3])
        col3 = np.array([1.0, 2.0, 3.0])
        ctrl = np.array(
                [('ab', 1, 1.0), ('cd', 2, 2.0), ('ef', 3, 3.0)],
                dtype=[('str', 'S2'), ('int', int), ('float', float)])
        res = utils.sa_from_cols(
                [col1, col2, col3], 
                col_names=['str', 'int', 'float'])
        self.assertTrue(np.array_equal(ctrl, res))

    def test_append_cols(self):
        M = np.array([(1, 'a'), (2, 'b')], dtype=[('int', int), ('str', 'S1')])
        col1 = np.array([1.0, 2.0])
        col2 = np.array([datetime(2015, 12, 12), datetime(2015, 12, 13)],
                        dtype='M8[us]')
        
        ctrl = np.array(
            [(1, 'a', 1.0), (2, 'b', 2.0)], 
            dtype=[('int', int), ('str', 'S1'), ('float', float)])
        res = utils.append_cols(M, col1, 'float')
        self.assertTrue(np.array_equal(ctrl, res))

        ctrl = np.array(
            [(1, 'a', 1.0, datetime(2015, 12, 12)), 
             (2, 'b', 2.0, datetime(2015, 12, 13))], 
            dtype=[('int', int), ('str', 'S1'), ('float', float),
                   ('dt', 'M8[us]')])
        res = utils.append_cols(M, [col1, col2], ['float', 'dt'])
        self.assertTrue(np.array_equal(ctrl, res))

    def test_remove_cols(self):
        M = np.array(
            [(1, 'a', 1.0, datetime(2015, 12, 12)), 
             (2, 'b', 2.0, datetime(2015, 12, 13))], 
            dtype=[('int', int), ('str', 'O'), ('float', float),
                   ('dt', 'M8[us]')])

        ctrl = np.array(
            [(1, 'a', 1.0), (2, 'b', 2.0)], 
            dtype=[('int', int), ('str', 'O'), ('float', float)])
        res = utils.remove_cols(M, 'dt')
        self.assertTrue(np.array_equal(ctrl, res))

        ctrl = np.array([(1, 'a'), (2, 'b')], dtype=[('int', int), 
                                                     ('str', 'O')])
        res = utils.remove_cols(M, ['dt', 'float'])        
        self.assertTrue(np.array_equal(ctrl, res))

    def test_join(self):
        # test basic inner join
        a1 = np.array([(0, 'Lisa', 2),
                       (1, 'Bill', 1),
                       (2, 'Fred', 2),
                       (3, 'Samantha', 2),
                       (4, 'Augustine', 1),
                       (5, 'William', 0)], dtype=[('id', int),
                                                  ('name', 'S9'),
                                                  ('dept_id', int)])
        a2 = np.array([(0, 'accts receivable'),
                       (1, 'accts payable'),
                       (2, 'shipping')], dtype=[('id', int),
                                                ('name', 'S16')])
        ctrl = pd.DataFrame(a1).merge(
                    pd.DataFrame(a2),
                    left_on='dept_id',
                    right_on='id').to_records(index=False)
        res = utils.join(a1, a2, 'inner', 'dept_id', 'id')
        self.__sa_check(ctrl, res)

        # test column naming rules
        a1 = np.array([(0, 'a', 1, 2, 3)], dtype=[('idx0', int),
                                    ('name', 'S1'),
                                    ('a1_idx1', int),
                                    ('idx2', int),
                                    ('idx3', int)])
        a2 = np.array([(0, 'b', 1, 2, 3)], dtype=[('idx0', int),
                                            ('name', 'S1'),
                                            ('a2_idx1', int),
                                            ('idx2', int),
                                            ('idx3', int)])
        pd1 = pd.DataFrame(a1)
        pd2 = pd.DataFrame(a2)
        ctrl = pd1.merge(
                pd2, 
                left_on=['idx0', 'a1_idx1', 'idx2'], 
                right_on=['idx0', 'a2_idx1', 'idx2'],
                suffixes=['_left', '_right']).to_records(index=False)
        res = utils.join(
                a1,
                a2, 
                'inner',
                left_on=['idx0', 'a1_idx1', 'idx2'], 
                right_on=['idx0', 'a2_idx1', 'idx2'],
                suffixes=['_left', '_right'])
        self.__sa_check(ctrl, res)

        # outer joins
        a1 = np.array(
            [(0, 'a1_0', 0),
             (1, 'a1_1', 1),
             (1, 'a1_2', 2),
             (2, 'a1_3', 3),
             (3, 'a1_4', 4)], 
            dtype=[('key', int), ('label', 'S4'), ('idx', int)])
        a2 = np.array(
            [(0, 'a2_0', 0),
             (1, 'a2_1', 1),
             (2, 'a2_2', 2),
             (2, 'a2_3', 3),
             (4, 'a2_4', 4)], 
            dtype=[('key', int), ('label', 'S4'), ('idx', int)])
        #for how in ('inner', 'left', 'right', 'outer'):
        merged_dtype = [('key', int), ('label_x', 'S4'), ('idx_x', int),
                        ('label_y', 'S4'), ('idx_y', int)]
        merge_algos = ('inner', 'left', 'right', 'outer')
        merged_data = [[(0, 'a1_0', 0, 'a2_0', 0),
                        (1, 'a1_1', 1, 'a2_1', 1),
                        (1, 'a1_2', 2, 'a2_1', 1),
                        (2, 'a1_3', 3, 'a2_2', 2),
                        (2, 'a1_3', 3, 'a2_3', 3)],
                       [(0, 'a1_0', 0, 'a2_0', 0),
                        (1, 'a1_1', 1, 'a2_1', 1),
                        (1, 'a1_2', 2, 'a2_1', 1),
                        (2, 'a1_3', 3, 'a2_2', 2),
                        (2, 'a1_3', 3, 'a2_3', 3),
                        (3, 'a1_4', 4, '', -999)], 
                       [(0, 'a1_0', 0, 'a2_0', 0),
                        (1, 'a1_1', 1, 'a2_1', 1),
                        (1, 'a1_2', 2, 'a2_1', 1),
                        (2, 'a1_3', 3, 'a2_2', 2),
                        (2, 'a1_3', 3, 'a2_3', 3),
                        (4, '', -999, 'a2_4', 4)], 
                       [(0, 'a1_0', 0, 'a2_0', 0),
                        (1, 'a1_1', 1, 'a2_1', 1),
                        (1, 'a1_2', 2, 'a2_1', 1),
                        (2, 'a1_3', 3, 'a2_2', 2),
                        (2, 'a1_3', 3, 'a2_3', 3),
                        (4, '', -999, 'a2_4', 4), 
                        (3, 'a1_4', 4, '', -999)]] 
        for how, data in zip(merge_algos, merged_data):
            res = utils.join(
                    a1,
                    a2, 
                    how,
                    left_on='key',
                    right_on='key')
            ctrl = np.array(data, dtype=merged_dtype)
            self.__sa_check(ctrl, res)


if __name__ == '__main__':
    unittest.main()
