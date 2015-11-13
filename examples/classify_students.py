import sys
import os
import cPickle
import re
import numpy as np
from diogenes.grid_search.experiment import Experiment 
from diogenes.grid_search.subset import BaseSubsetIter
from diogenes.grid_search.partition_iterator import SlidingWindowValue
from diogenes.modify import replace_missing_vals
from diogenes.utils import remove_cols
from sklearn.ensemble import RandomForestClassifier
#NEXT REPLACE MISSING VALS
class SubsetSchool(BaseSubsetIter):
    def __init__(self, y, col_names, max_grades):
        super(SubsetSchool, self).__init__(y, col_names)
        self.__max_grades = max_grades
        self.__grades = range(9, 13)

    def __iter__(self):
        # cribbed from Reid
        for max_grade in self.__max_grades:

            # Filter features from higher grade levels.
            higher_grade = False
            higher_grade_regex = '^(?!'
            for grade2 in self.__grades:
                if grade2 > max_grade:
                    higher_grade = True
                    higher_grade_regex += r'grade_' + str(grade2) + '|'
            if higher_grade:
                higher_grade_regex = higher_grade_regex[:-1] # remove last '|'
                higher_grade_regex = higher_grade_regex + ').*'
                #data = data.filter(regex=higher_grade_regex)
                regex = re.compile(higher_grade_regex)
                X_cols = filter(lambda i: regex.search(i), self._col_names)
                yield (np.arange(self._y.shape[0]), X_cols, {'max_grade' : max_grade})

    def __repr__(self):
        return 'SubsetSchool({})'.format(grades)

DATA_PATH = '/home/zar1/hs-scratch/'

fin = open(os.path.join(DATA_PATH, 'data_rec_array.pkl'))
print 'loading data'
M = cPickle.load(fin)
fin.close()
print 'data loaded'

y = M['label']
M = remove_cols(M, ['label', 'student_id', 'index'])

print 'set up data'

M = replace_missing_vals(M, 'constant', np.nan)
print 'imputed'


min_year = min(M['cohort'])

clfs = [{'clf': RandomForestClassifier, 'random_state': [0]}]
csvs = []
train_start = min_year
train_window_size = 2
init_train_window_end = train_start + train_window_size - 1
for max_grade in xrange(9, 12):
    print 'making experiment'
    print max_grade
    test_start = init_train_window_end + (12 - max_grade)
    subsets = [{'subset': SubsetSchool, 'max_grades': [[max_grade]]}]
    cvs = [{'cv': SlidingWindowValue, 'train_start': [train_start], 
            'train_window_size': [1], 'test_start': [test_start], 
            'test_window_size': [1], 'inc_value': [1], 
            'guide_col_name': ['cohort']}]
    exp = Experiment(M, y, clfs=clfs, subsets=subsets, cvs=cvs)
    print 'running'
    exp.run()
    csv_name = '_{}.csv'.format(max_grade)
    print 'making report'
    exp.make_csv(csv_name)
    csvs.append(csv_name)
with open(csvs[0]) as fin:
    header = fin.readline()
with open('report.csv', 'w') as fout:
    fout.write(header)
    for in_csv in csvs:
        with open(in_csv) as fin:
            fin.readline()
            fout.write(fin.read())import sys
import os
import cPickle
import re
import numpy as np
from diogenes.grid_search.experiment import Experiment 
from diogenes.grid_search.subset import BaseSubsetIter
from diogenes.grid_search.partition_iterator import SlidingWindowValue
from diogenes.modify import replace_missing_vals
from diogenes.utils import remove_cols
from sklearn.ensemble import RandomForestClassifier
#NEXT REPLACE MISSING VALS
class SubsetSchool(BaseSubsetIter):
    def __init__(self, y, col_names, max_grades):
        super(SubsetSchool, self).__init__(y, col_names)
        self.__max_grades = max_grades
        self.__grades = range(9, 13)

    def __iter__(self):
        # cribbed from Reid
        for max_grade in self.__max_grades:

            # Filter features from higher grade levels.
            higher_grade = False
            higher_grade_regex = '^(?!'
            for grade2 in self.__grades:
                if grade2 > max_grade:
                    higher_grade = True
                    higher_grade_regex += r'grade_' + str(grade2) + '|'
            if higher_grade:
                higher_grade_regex = higher_grade_regex[:-1] # remove last '|'
                higher_grade_regex = higher_grade_regex + ').*'
                #data = data.filter(regex=higher_grade_regex)
                regex = re.compile(higher_grade_regex)
                X_cols = filter(lambda i: regex.search(i), self._col_names)
                yield (np.arange(self._y.shape[0]), X_cols, {'max_grade' : max_grade})

    def __repr__(self):
        return 'SubsetSchool({})'.format(grades)

DATA_PATH = '/home/zar1/hs-scratch/'

fin = open(os.path.join(DATA_PATH, 'data_rec_array.pkl'))
print 'loading data'
M = cPickle.load(fin)
fin.close()
print 'data loaded'

y = M['label']
M = remove_cols(M, ['label', 'student_id', 'index'])

print 'set up data'

M = replace_missing_vals(M, 'constant', np.nan)
print 'imputed'


min_year = min(M['cohort'])

clfs = [{'clf': RandomForestClassifier, 'random_state': [0]}]
csvs = []
train_start = min_year
train_window_size = 2
init_train_window_end = train_start + train_window_size - 1
for max_grade in xrange(9, 12):
    print 'making experiment'
    print max_grade
    test_start = init_train_window_end + (12 - max_grade)
    subsets = [{'subset': SubsetSchool, 'max_grades': [[max_grade]]}]
    cvs = [{'cv': SlidingWindowValue, 'train_start': [train_start], 
            'train_window_size': [1], 'test_start': [test_start], 
            'test_window_size': [1], 'inc_value': [1], 
            'guide_col_name': ['cohort']}]
    exp = Experiment(M, y, clfs=clfs, subsets=subsets, cvs=cvs)
    print 'running'
    exp.run()
    csv_name = '_{}.csv'.format(max_grade)
    print 'making report'
    exp.make_csv(csv_name)
    csvs.append(csv_name)
with open(csvs[0]) as fin:
    header = fin.readline()
with open('report.csv', 'w') as fout:
    fout.write(header)
    for in_csv in csvs:
        with open(in_csv) as fin:
            fin.readline()
            fout.write(fin.read())
