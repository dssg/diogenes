    ================================================
          ____  _                                 
         / __ \(_)___  ____ ____  ____  ___  _____
        / / / / / __ \/ __ `/ _ \/ __ \/ _ \/ ___/
       / /_/ / / /_/ / /_/ /  __/ / / /  __(__  ) 
      /_____/_/\____/\__, /\___/_/ /_/\___/____/  
                    /____/                        

    ================================================
    searching for an honest classifier
    
#overview
Diogenes is an api utility that provides parralllle implementation of a g

Diogenes is broken into four primary sections and a utils:

1. Read
2. Modify
3. Grid\_Search
4. Display
5. utils

#diogenes.read
##Numpy Structured Arrays

The most common interchange type for Diogenes is the Numpy structured array [http://docs.scipy.org/doc/numpy/user/basics.rec.html]. The numpy 
structured array resembles a SQL or Pandas table. It is a collection of
columns, and each column as a different name and data type. Every time
it makes sense to pass in or return a table-like object, Diogenes prefers
to use structured arrays.

Inside Diogenes, we use "sa" to mean structured array

##User Facing Functions

There are three import functions:

1. open\_csv - takes the path of a csv file and returns an sa
2. open\_csv\_url - takes the url of a csv file and returns an sa
3. connect\_sql - connects to an sql database and returns queries as sas
4. utils.convert\_to\_sa - converts a list of lists or an (unstructured) numpy array to an sa

They accept five data types:

1. csv (1)
2. urls (2)
3. sql (3)
4. list of lists (4)
5. np.arrays (4)
   
##Connection to ArrayEmitter
   This will be filled in at a later point.

#diogenes.modify



##col\_functions
These functions choose or remove columns based on some specification

1. choose\_col\_where() - chooses columns
2. remove\_col\_where(M, arguments) - removes columns

The following are lambdas that can be passed into choose_col_where
or remove_col_where:

    * col\_val\_eq(M, boundary) - picks columns where all values are equal to boudnary
    * col_val\_eq\_any(M, boundary=None) - picks columns where all values are equal to eachother
    * col\_random(M, number\_to\_select) - picks a random subset of columns
    * col\_fewer\_then\_n\_nonzero(M, boundary) - picks columns where all but boundary values are nonzero
    * col\_has\_lt\_threshold\_unique\_values(col, threshold) - picks colums where columns have less than threshold unique values

##row\_functions
These functions choose rows based on some specifications

1. remove\_rows\_where(M, lamd, col\_name, vals) - chooses rows
2. choose\_rows\_where(M, arguments, generated\_name=None) - removes rows
3. where\_all_are\_true - makes an additional binary column that indicates whether conditions are true

The following lambdas can be passed to remove\_rows\_where, choose\_rows\_where, and where\_all\_are\_true

    * row\_is\_outlier(M, col\_name, boundary) - picks rows where value is an outlier
    * row\_val\_eq(M, col\_name, boundary) - picks rows where value is equal to boundary
    * row\_val\_lt(M, col\_name, boundary) - picks rows where value is less than boundary
    * row\_val\_lt\_TIME\_EDITION(M, col\_name, boundary) - picks rows where value is less than boundary in time
    * row\_val\_gt(M, col\_name, boundary) - picks rows where value is greater than boundary
    * row\_val\_between(M, col\_name, boundary) - picks rows where value is between boundary[0] and boundary[1]
    * row\_is\_within\_region(M, col\_names, boundary) picks rows where point signified by two column names is within boundary


##combine

Combine creates a column based on some function of other columns 

1. combine\_cols(M, lambd, col\_names, generated\_name)
   
Takes the following lambdas
   
   * combine\_sum(*args) - creates the sum of columns
   * combine\_mean(*args) - creates the mean of columns

    
##manipulate entries
The standard interface is that the dims of matrix are same, but entries change.

1. label\_encode(M) - converts strings to integers so that each integer uniquely identifies a string
2. replace\_missing\_vals(M, strategy, missing\_val=np.nan, constant=0) - converts missing values to something sensible
3. generate\_bin(col, num\_bins) - creates a feature that sorts another feature into bins
4. normalize(col, mean=None, stddev=None, return\_fit=False) - scale and translate feature
5. distance\_from\_point(lat\_origin, lng\_origin, lat\_col, lng\_col) - generates a column that gives distance of points from given central point


##SA primitives
* stack\_rows - joins the rows from two sas with the same column names and dtypes into one sa. Equivalent to a SQL UNION
* sa\_from\_cols - Converts a list of 1-d numpy arrays to an sa
* append\_cols - Adds columns to an SA
* remove\_cols - removes columns from an SA by name
* join - Merges to sas in a manner resembling SQL join

#diogenes.grid\_search
##Experiment
A class used to organize your grid search of subsampling and models.  The principle goal of Diogenes was to provide an integrated and simple way to search across a large number of unique subsample, classifier and parameters combinations. This is a brute force search of all possible specified configuration.    

* Experiment - The class used to organize a grid search. One initiated, it can be run with the .run function. After being run, an experiment organizes a number of Trials.

* Trial - An object holding all fitted clfs pertaining to a single configuration. Each fitted clf is contained in one Run

* Run - An object holding a single clf. Also has various methods to calculate scores and make graphs. 

##subsetters

grid_search.subset contains a number of classes that can be passed to Experiment. These classes provide various ways to take subsets of rows before fitting them to classifiers

* SubsetNoSubset - takes all rows
* SubsetRandomRowsActualDistribution - takes subsets that reflect proportion of positive to negative labels
* SubsetRandomRowsEvenDistribution - takes subsets that give an even number of positive and negative labels
* SubsetSweepNumRows - takes progressively more rows
* SubsetSweepVaryStratification - takes different percents of positive and negative rows

##partition iterators

grid_search.partition_iterator contains classes that specify different ways to take folds.

* SlidingWindowIdx - varies training and testing sets so that sets slide over array indices
* SlidingWindowValue - varies training and testing sets so that sets slide over column values.

##std\_classifers

grid_search.std_classifiers contain standard sets of classifiers that can be passed into experiment.

#diogenes.display
   diogenes.display provides a number of functions to help visualize trends in data or results. These functions either plot and return figures or provide console output. Additionally, display provides the Report class which can be used to generate pdf reports.


##Report
A class used to organize figures and tables and output pdfs.

##To Console

* pprint\_sa - prints a structured array nicely
* describe\_cols - prints the summary statistics of each column in a structured array
* crosstab - prints the crosstabl of a structured array

##matplotlib
* plot\_simple\_histogram
* plot\_prec\_recall - a nontraditional variation
* plot\_roc - A nontraditional variation
* plot\_box\_plot
* get\_roc\_auc - area under roc curve
* plot\_correlation\_matrix - print the correlation matrix for all columns.
* plot\_correlation\_scatter\_plot - print the correlation scatter plot for all columns.
* plot\_kernel\_density - prints the histogram and the kernel density estimate for a single column.
* plot\_on\_timeline

##metric:
* feature\_pairs\_in\_rf - measures frequency with which features are split consecutively in a random forest
* get\_top\_features - prints top features in a fitted clf.

   
   
##diogenes.utils
* distance - provides distance between two points on earth
* dist\_less\_than - returns true if the given points are closer together than a threshold.
* cast\_np\_sa\_to\_nd - converts an SA to a homogeneous (nonstructured) Numpy array. This is necessary before an SA can be passed to Scikit-Learn functions. Note that every column will be converted to the most permissive time (datetime < int < float < string), so any strings should be converted to integers with
  diogenes.modify.label\_encode before passing them to sklearn. Also, if arbitrary integers don't make sense, modify your sa accordingly before converting it.
 
   
   
##extending diogenes