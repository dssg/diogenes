# from sklearn tutorial

from sklearn import datasets
from sklearn.cross_validation import train_test_split
import unittest
class TestSKLearnParity(unittest.TestCase):

    def test_sklearn_parity(self):
        iris = datasets.load_iris()

        y = iris.target
        M = iris.data
        # Converts 2-dimensional homogeneous array to structured array
        M = cast_np_nd_to_sa(M)

        def between(M, col_name, args):
            col = M[col_name]
            return np.logical_and(args[0] < col, col < args[1])
        def gt(M, col_name, args):
            col = M[col_name]
            return col > args[0]    
        # Create a boolean column where f2 is between 4 and 5 and f3 > 1.9
        where_col = where(
                            M, 
                            [between, gt],
                            ['f2', 'f3'],
                            [(4, 5), (`1.9`,)])

        # Append the column we just made to M
        M = sa_append(M, where_col)
    
        # remove columns f2 and f3
        M = sa_remove_col(M, ['f2', 'f3'])

        M_train, M_test, y_train, y_test = train_test_split(M, y)
    
        os.remove('report.pdf')

        # run classifiers over our modified data and generate a report
        run_std_classifiers(M_train, M_test, y_train, y_test, 'report.pdf')
    
        self.assertTrue(os.path.isfile('report.pdf'))    

