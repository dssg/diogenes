# Feature Requests From Group Meetings, Week of August 10, 2015

## Australia

* Incorporate SQL queries into perambulation. This would mean that we can make queries and populate table with the grid_search. 

* Descriptive statistics on SQL tables like col_names, num of categories, num of nulls. 

* Plot of feature importance. We want to know how importance scores trail off. And distribution. 

* This group spent the majority of their time figuring out what database schema meant. We probably can't help, but it's good to know that that was the majority of their workload 

## Babies
* Train where x is < than all values in col B. test where col B > x. x is an element of col A

## Cincinnati
* Caching intermediary results. As with Drake, only regenerate files as needed. Have some mechanism to keep track of when we've run what.

* Features like mean value of homes in an area, change of value of homes in this area. Aggregates over a partial set of data in table. average value of home price over last 3 years based on date inspected.

* Choose columns to sum across based on value in a column.

* Distance from this entry to nearest X. For example, distance from this home to nearest abandoned home. This is apparently much easier to do in PostGIS than in Python

## Feeding

* Simple multiprocessing. Thin wrapper around joblib just so people know it's there.

* Deduplicate rows. If rows are identical, remove them.

* ROC curves w/ more than one series. If we have more than one class in our label, we will have more than one series to ROC. We do this in each case by making one class the baseline and comparing it against all other classes

## High School

* The biggest thing is the really confusing cross validation rules that involve leaving out columns. First we pick a max grade, then we have to leave out columns, then the max grade determines how far apart our train and test sets need to be. According to Robin:

```
min_year = train_start = min(available_cohorts)
max_year = test_end = max(available_cohorts)
for max_grade in seq(9, 11){
  for train_end in seq(min_year, max_year - (12 - max_grade)){
    test_start = train_end + (12 - max_grade)
    #...train on cohorts between train_start and train_end
    #...test on cohorts between test_start and test_end
  }
}
# remember, seq is inclusive
```

* pandas.get_dummies

* Transposing 1-to-many relations. 

This circumstance arises when we have a table in a "log" format, where
multiple rows are associated with one identity. For example, say we are 
predicting likelihood that a given student will drop out of school, and we
have a GPA per student per year. So one of the tables we have is formatted like:

| ID  | Grade | GPA |
| ---:| -----:| ---:|
| 1   | 9     | 3.2 |
| 1   | 10    | 3.4 |
| 1   | 11    | 4.0 |
| 2   | 9     | 2.1 |
| 2   | 10    | 2.0 |
| 2   | 11    | 2.3 |
| 2   | 12    | 2.5 |

Then we have another table of features that don't vary by year. For example:

| ID  | Date of Birth | Graduated |
| ---:| -------------:| ---------:|
| 1   | 1988-09-22    | 1         |
| 2   | 1989-08-08    | 0         |

The table that we actually analyze needs one row per student, so we need to
attach a transpose of the former table to the later table, like

| ID  | Date of Birth | Graduated | GPA_9 | GPA_10 | GPA_11 | GPA_12 |
| ---:| -------------:| ---------:| -----:| ------:| ------:| ------:|
| 1   | 1988-09-22    | 1         | 3.2   | 3.4    | 4.0    |        |
| 2   | 1989-08-08    | 0         | 2.1   | 2.0    | 2.3    | 2.5    |

* Removing columns by regex. In the above example, there are circumstances in which we won't to exclude some of the columns (for example, GPA > grade 11). We should have a subsetter that does this. For example, we take a subset that removes anything > grade 11:

| ID  | Date of Birth | Graduated | GPA_9 | GPA_10 | GPA_11 | 
| ---:| -------------:| ---------:| -----:| ------:| ------:| 
| 1   | 1988-09-22    | 1         | 3.2   | 3.4    | 4.0    | 
| 2   | 1989-08-08    | 0         | 2.1   | 2.0    | 2.3    | 

And then another subset that removes anything > grade 10:

| ID  | Date of Birth | Graduated | GPA_9 | GPA_10 | 
| ---:| -------------:| ---------:| -----:| ------:| 
| 1   | 1988-09-22    | 1         | 3.2   | 3.4    | 
| 2   | 1989-08-08    | 0         | 2.1   | 2.0    | 

## Infonavit

* In grid_search: fit train set to a Gaussian, then apply Gaussian to test set. Normalize to train set, then apply to test set. Normalizing across everything would be cheating

* managing shape files. Use them for thresholding to bin GPS data

* We'd rather have a static map than a web page. Just a quick sanity check 

* Sanity check for csv. E.G. Are there the correct amount of delims per row. Except we're not importing any more, so we're not doing this for now.

* Impute based on previous calculations. Treat col w/ missing values as label, train on everything else with RF

* Sanity check to make sure that train/test data includes distribution of labels corresponding to real distribution. E.g. if bimodal

## Labor

* pie charts.

## Police

* Pay attention to feature processors. The different sorts of post-processing one might do on a column.

## Sunlight

### Standardized pdf scraping 

* After some discussion, we've decided that importing data to a structured array in a thorough manner is probably beyond our scope. It would be best to let our clients do it. (probably w/ Pandas). PDF scraping is irrelevant

### Smith-Waterman

* Sunlight developed a faster implementation than is publicly available. This won't go in diogenes, but we should help make sure it gets released.

### JIT

* look at numba. Does this help make things faster if there's no effort involved. Compare to Pypy.

### TF-IDF Score

* A standard tool for topic modeling, clustering, developing feature vectors. Account for its existence. Maybe implement it.

### TIKA

* Import documents into search engine. Not directly relevant, but consider.


## World Bank

* Entity resolution. Different people call different entities the same thing. For example, a series of different nicknames for the same college.

* Figuring out what to name automatically generated columns.

* Feature aggregation into percent “what percent of a suppliers contracts were in Africa before a given contract”. Powers of sets. Things to aggregate over are cross product of a set. See Elissa’s slide

* SQL-esque joins

## Kirsten

* Detect and remove colinear columns

* col renaming. e.g. year to grade level

## Feature generation

* When we generate a feature, keep track of the column name and what it means in metadata that is attached to the structured array. We'll keep metadata in structured arrays by doing a thin subclass like this:

```
>>> class BetterSA(np.ndarray):
...  def __init__(self, *args, **kwargs):
...   super(BetterSA, self).__init__(*args, **kwargs)
...   self.meta = 'metadata'
```
