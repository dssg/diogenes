
import unittest
import utils_for_tests

class TestUtilsForTests(unittest.TestCase):
    def test_generate_matrix(self):
        M, y = utils_for_tests.generate_test_matrix(100, 5, 3, [float, str, int])
        print M
        print y

if __name__ == '__main__':
    unittest.main()
