
class BaseSubsetIter(object):
    __metaclass__ = abc.ABCMeta

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
    def __iter__(self):
        yield (np.arange(self._y.shape[0]), self._col_names, {})

    def __repr__(self):
        return 'SubsetNoSubset()'

class SubsetRandomRowsActualDistribution(BaseSubsetIter):
        
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


class SubsetSweepExcludeColumns(BaseSubsetIter):
    """
    
    Analyze feature importance when each of a specified set of columns are
    excluded. 
    
    Parameters
    ----------
    M : Numpy structured array
    cols_to_exclude : list of str or None
         List of names of columns to exclude one at a time. If None, tries
         all columns
         
    Returns
    -------
    Numpy Structured array
        First col
            Excluded col name
        Second col
            Accuracy score
        Third col
            Feature importances
    """
    # not providing cv because it's always Kfold
    # returns fitted classifers along a bunch of metadata
    #Why don't we use this as buildinger for slices. AKA the way the cl
    # 
    def __init__(self, M, cols_to_exclude=None):
        raise NotImplementedError

class SubsetSweepLeaveOneColOut(BaseSubsetIter):
    # TODO
    #returns list of list eachone missing a value in order.  
    #needs to be tested
    pass

