import abc
from collections import Counter
from random import sample, seed, setstate, getstate

import numpy as np

"""This module provides different ways to take subsets of data"""

class BaseSubsetIter(object):
    __metaclass__ = abc.ABCMeta

    """Base class for subsetters. 

    supports iteration where each iteration returns a tuple:

    (np.ndarray, list of str, dict of str : ?)

    The first element is the indices in the subset

    The second element is the features in the subset

    The third elements are extra notes about the subset
    
    Parameters 
    ----------
    y : np.ndarray
        column to use to determine subsets
    col_names : list of str
        names of all features in data set

    """
    def __init__(self, y, col_names):
        self._y = y
        self._col_names = col_names
    
    @abc.abstractmethod
    def __iter__(self):
        # Indices, columns, notes
        yield (np.array([], dtype=int), [], {})

    @abc.abstractmethod
    def __repr__(self):
        return 'BaseSubsetIter()'

class SubsetNoSubset(BaseSubsetIter):
    """Generates a single subset consisting of all data"""
    def __iter__(self):
        yield (np.arange(self._y.shape[0]), self._col_names, {})

    def __repr__(self):
        return 'SubsetNoSubset()'

class SubsetRandomRowsActualDistribution(BaseSubsetIter):
    """Generates subsets reflecting actual distribution of labels

    Parameters
    ----------
    y : np.ndarray
        labels in data
    col_names : list of str
        names of all features in data
    subset_size : int
        number of rows in each subset
    n_subsets : int
        number of subsets to pick
    random_state : int
        random seed

    """
    
        
    def __init__(self, y, col_names, subset_size, n_subsets=3, 
                 random_state=None):
        super(SubsetRandomRowsActualDistribution, self).__init__(y, col_names)
        self.__subset_size = subset_size
        self.__n_subsets = n_subsets
        self.__random_state_seed = random_state

    def __iter__(self):
        y = self._y
        subset_size = self.__subset_size
        n_subsets = self.__n_subsets
        count = Counter(y)
        size_space = float(sum(count.values()))
        proportions = {key: count[key] / size_space for key in count}
        n_choices = {key: int(proportions[key] * subset_size) for 
                     key in proportions}
        indices = {key: np.where(y == key)[0] for key in count}
        seed(self.__random_state_seed)
        random_state = getstate()
        for i in xrange(n_subsets):
            setstate(random_state)
            samples = {key: sample(indices[key], n_choices[key]) for key in count}
            random_state = getstate()
            all_indices = np.sort(np.concatenate(samples.values()))
            yield (all_indices, self._col_names, {'sample_num': i})

    def __repr__(self):
        return ('SubsetRandomRowsActualDistribution('
                'subset_size={}, n_subsets={}, random_state={})').format(
                self.__subset_size,
                self.__n_subsets,
                self.__random_state_seed)

class SubsetRandomRowsEvenDistribution(BaseSubsetIter):
    """Generates subsets where each label appears at about the same frequency

    Parameters
    ----------
    y : np.ndarray
        labels in data
    col_names : list of str
        names of all features in data
    subset_size : int
        number of rows in each subset
    n_subsets : int
        number of subsets to pick
    random_state : int
        random seed

    """
        
    def __init__(self, y, col_names, subset_size, n_subsets=3, 
                 random_state=None):
        super(SubsetRandomRowsEvenDistribution, self).__init__(y, col_names)
        self.__subset_size = subset_size
        self.__n_subsets = n_subsets
        self.__random_state_seed = random_state

    def __iter__(self):
        y = self._y
        subset_size = self.__subset_size
        n_subsets = self.__n_subsets
        count = Counter(y)
        n_categories = len(count)
        proportions = {key: 1.0 / n_categories for key in count}
        n_choices = {key: int(proportions[key] * subset_size) for 
                     key in proportions}
        indices = {key: np.where(y == key)[0] for key in count}
        seed(self.__random_state_seed)
        random_state = getstate()
        for i in xrange(n_subsets):
            setstate(random_state)
            samples = {key: sample(indices[key], n_choices[key]) for key in count}
            random_state = getstate()
            all_indices = np.sort(np.concatenate(samples.values()))
            yield (all_indices, self._col_names, {'sample_num': i})

    def __repr__(self):
        return ('SubsetRandomRowsEvenDistribution('
                'subset_size={}, n_subsets={}, random_state={})').format(
                self.__subset_size,
                self.__n_subsets,
                self.__random_state_seed)

class SubsetSweepNumRows(BaseSubsetIter):
    """Generates subsets with varying number of rows

    Parameters
    ----------
    y : np.ndarray
        labels in data
    col_names : list of str
        names of all features in data
    num_rows : list of int
        number of rows in each subset
    random_state : int
        random seed

    """
        
    def __init__(self, y, col_names, num_rows, random_state=None):
        super(SubsetSweepNumRows, self).__init__(y, col_names)
        self.__num_rows = num_rows
        self.__random_state_seed = random_state

    def __iter__(self):
        y = self._y
        num_rows = self.__num_rows
        seed(self.__random_state_seed)
        random_state = getstate()
        for rows in num_rows:
            setstate(random_state)
            all_indices = np.sort(sample(np.arange(0, y.shape[0]), rows))
            yield (all_indices, self._col_names, {'rows': rows})
            random_state=getstate()

    def __repr__(self):
        return 'SubsetSweepNumRows(num_rows={}, random_state={})'.format(
                self.__num_rows,
                self.__random_state_seed)

class SubsetSweepVaryStratification(BaseSubsetIter):
    """Generates subsets with varying proportion of True and False labels

    Parameters
    ----------
    y : np.ndarray
        labels in data
    col_names : list of str
        names of all features in data
    proportions_positive : list of float
        proportions of positive labels in each subset
    subset_size : int
        number of rows in each subset
    random_state : int
        random seed

    """
        
    def __init__(self, y, col_names, proportions_positive, subset_size, 
                 random_state=None):
        super(SubsetSweepVaryStratification, self).__init__(y, col_names)
        self.__proportions_positive = proportions_positive
        self.__subset_size = subset_size
        self.__random_state_seed = random_state

    def __iter__(self):
        y = self._y
        subset_size = self.__subset_size
        positive_cases = np.where(y)[0]
        negative_cases = np.where(np.logical_not(y))[0]
        seed(self.__random_state_seed)
        random_state = getstate()
        for prop_pos in self.__proportions_positive:
            setstate(random_state)
            positive_sample = sample(positive_cases, int(subset_size * prop_pos))
            negative_sample = sample(negative_cases, int(subset_size * (1 - prop_pos)))
            random_state = getstate()
            # If one of these sets is empty, concatentating them casts to float, so we have
            # to cast it back (hence the astype)
            all_indices = np.sort(np.concatenate([positive_sample, negative_sample])).astype(int)
            yield (all_indices, self._col_names, 'prop_positive={}'.format(prop_pos))

    def __repr__(self):
        return ('SubsetSweepVaryStratification('
                'proportions_positive={}, subset_size={}, '
                'random_state={})').format(
                self.__proportions_positive,
                self.__subset_size,
                self.__random_state_seed)




