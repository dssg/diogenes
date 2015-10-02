import numpy as np

from sklearn.cross_validation import _PartitionIterator
# TODO By and large, we shouldn't be using SKLearn's internal classes.

"""Custom objects used to produce train/test splits. 

Also, anything in sklearn.cross_validation should work in Experiments

"""

class NoCV(_PartitionIterator):
    """Partition iterator that just returns the entire set as the training set

    Parameters
    ----------
    n : int
        The number of rows in the data
    """
    def _iter_test_indices(self):
        yield np.array([], dtype=int)

class SlidingWindowIdx(_PartitionIterator):
    """Partition iterator that iterates across indices of array

    Has a moving window of indices for the training set and a moving
    window of indices for the testing set.

    Parameters
    ----------
    n : int
        Number of rows in the data
    train_start : int
        start index of training window
    train_window_size : int
        number of rows in the (initial) training window
    test_start : int
        start index of testing window
    test_window_size : int
        number of rows in testing window
    inc_value : int
        number of rows to increment train and test sets in each iteration
    expanding_train : bool
        If True, the end of the train window moves forward with each
        iteration, but the beginning of the train window does not. 
        Consequently, more rows are added to the training set with each
        iteration

        If False, the beginning and end of the train window both move
        forward, so the training set remains the same size.

    """

    def __init__(self, n, train_start, train_window_size, test_start, 
                 test_window_size, inc_value, expanding_train=False):
       super(SlidingWindowIdx, self).__init__(n)
       self.__n = n
       self.__train_start = train_start
       self.__train_window_size = train_window_size
       self.__train_end = train_start + train_window_size - 1
       self.__test_start = test_start
       self.__test_window_size = test_window_size
       self.__test_end = test_start + test_window_size - 1
       self.__inc_value = inc_value
       self.__expanding_train = expanding_train

    def cv_note(self):
        """dict of str providing extra info about the current iteration"""
        return {'train_start': self.__train_start,
                'train_end': self.__train_end,
                'test_start': self.__test_start,
                'test_end': self.__test_end}
                
    def _iter_test_indices(self):
        inc_value = self.__inc_value
        while self.__test_end < self.__n:
            yield np.arange(self.__test_start, self.__test_end + 1)
            if not self.__expanding_train:
                self.__train_start += inc_value
            self.__train_end += inc_value
            self.__test_start += inc_value
            self.__test_end += inc_value

    def __iter__(self):
        # _PartitionIterator assumes we're training on everything we're not
        # testing. We have to patch it's __iter__ so that isn't the case
        for train_index, test_index in super(
            SlidingWindowIdx, self).__iter__():
            yield (np.arange(self.__train_start, self.__train_end + 1), 
                   test_index)

class SlidingWindowValue(_PartitionIterator):
    """Partition iterator that iterates across values of a column in an array

    Has a moving window of indices for the training set and a moving
    window of indices for the testing set.

    Parameters
    ----------
    M : numpy.ndarray
        homogeneous (not structured) array. Feature array from which to draw
        train and test sets
    col_names : list of str
        names of features in M
    guide_col_name : str
        name of feature to use to determine train and test sets
    train_start : number
        start value for guide_col_name in training window
    train_window_size : number
        size of the (initial) training window
    test_start : number
        start value for guide_col_name in testing window
    test_window_size : number
        size in testing window
    inc_value : number
        value to increment train and test sets in each iteration
    expanding_train : bool
        If True, the end of the train window moves forward with each
        iteration, but the beginning of the train window does not. 
        Consequently, more rows are added to the training set with each
        iteration

        If False, the beginning and end of the train window both move
        forward, so the training window remains the same size.

    """
    def __init__(self, M, col_names, guide_col_name, train_start, 
                 train_window_size, test_start, 
                 test_window_size, inc_value, expanding_train=False):
        y = M[:,col_names.index(guide_col_name)]
        n = y.shape[0] 
        super(SlidingWindowValue, self).__init__(n)
        self.__y = y
        self.__n = n
        self.__train_start = train_start
        self.__train_window_size = train_window_size
        self.__train_end = train_start + train_window_size - 1
        self.__test_start = test_start
        self.__test_window_size = test_window_size
        self.__test_end = test_start + test_window_size - 1
        self.__inc_value = inc_value
        self.__expanding_train = expanding_train

    def cv_note(self):
        """dict of str providing extra info about the current iteration"""
        return {'train_start': self.__train_start,
                'train_end': self.__train_end,
                'test_start': self.__test_start,
                'test_end': self.__test_end}
                
    def _iter_test_indices(self):
        inc_value = self.__inc_value
        y = self.__y
        self.__test_mask = np.logical_and(
            y >= self.__test_start,
            y <= self.__test_end)
        self.__train_mask = np.logical_and(
            y >= self.__train_start,
            y <= self.__train_end)
        while np.any(self.__test_mask):
            yield self.__test_mask.nonzero()[0]
            if not self.__expanding_train:
                self.__train_start += inc_value
            self.__train_end += inc_value
            self.__test_start += inc_value
            self.__test_end += inc_value
            self.__test_mask = np.logical_and(
                y >= self.__test_start,
                y <= self.__test_end)
            self.__train_mask = np.logical_and(
                y >= self.__train_start,
                y <= self.__train_end)

    def __iter__(self):
        # _PartitionIterator assumes we're training on everything we're not
        # testing. We have to patch it's __iter__ so that isn't the case
        for train_index, test_index in super(
            SlidingWindowValue, self).__iter__():
            yield (self.__train_mask.nonzero()[0], test_index)

