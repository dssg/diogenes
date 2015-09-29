
class TestGenerate(unittest.TestCase):

    def test_generate_bin(self):
        M = [1, 1, 1, 3, 3, 3, 5, 5, 5, 5, 2, 6]
        ctrl = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 0, 3]
        self.assertTrue(np.array_equal(ctrl, generate_bin(M, 3)))
        M = np.array([0.1, 3.0, 0.0, 1.2, 2.5, 1.7, 2])
        ctrl = [0, 3, 0, 1, 2, 1, 2]
        self.assertTrue(np.array_equal(ctrl, generate_bin(M, 3)))
    

    def test_choose_rows_where(self):
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

        res = choose_rows_where(
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
