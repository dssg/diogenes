import numpy as np 
import sqlalchemy as sqla
from diogenes.read import open_csv
from diogenes.read import connect_sql
from uuid import uuid4
from diogenes import utils
import diogenes.grid_search.experiment as exp

from sklearn.ensemble import RandomForestClassifier

class ArrayEmitter(object):
    """
    Array emitter is a tool that accepts tables from either SQL or CSVs in the 
    RG format, then generates Numpy structured arrays in the M format based on 
    selection criteria on those tables.
    
    **RG Tables**

    Tables can be specified from either a CSV file (using the 
    get_rg_from_csv method) or from a SQL query (using the
    get_rg_from_SQL method). Imported tables must adhere to the *RG* format:

    *Table 1--an example RG-format table*

    +------------+------------+----------+-------------+-------+
    | student_id | start_year | end_year |     feature | value |
    +============+============+==========+=============+=======+
    |          0 |       2005 |     2006 |    math_gpa |   2.3 |
    +------------+------------+----------+-------------+-------+
    |          0 |       2005 |     2006 | english_gpa |   4.0 |
    +------------+------------+----------+-------------+-------+
    |          0 |       2005 |     2006 |    absences |     7 |
    +------------+------------+----------+-------------+-------+
    |          0 |       2006 |     2007 |    math_gpa |   2.1 |
    +------------+------------+----------+-------------+-------+
    |          0 |       2006 |     2007 | english_gpa |   3.9 |
    +------------+------------+----------+-------------+-------+
    |          0 |       2006 |     2007 |    absences |     8 |
    +------------+------------+----------+-------------+-------+
    |          1 |       2005 |     2006 |    math_gpa |   3.4 |
    +------------+------------+----------+-------------+-------+
    |          1 |       2005 |     2006 |    absences |     0 |
    +------------+------------+----------+-------------+-------+
    |          1 |       2006 |     2007 |    math_gpa |   3.5 |
    +------------+------------+----------+-------------+-------+
    |          1 |       2007 |     2008 | english_gpa |   2.4 |
    +------------+------------+----------+-------------+-------+
    |          2 |       2004 |     2005 |    math_gpa |   2.4 |
    +------------+------------+----------+-------------+-------+
    |          2 |       2005 |     2006 |    math_gpa |   3.4 |
    +------------+------------+----------+-------------+-------+
    |          2 |       2005 |     2006 |    absences |    14 |
    +------------+------------+----------+-------------+-------+
    |          2 |       2006 |     2007 |    absences |    96 |
    +------------+------------+----------+-------------+-------+

    In an RG-formatted table, there are five columns:
    
    1. The unique identifier of a unit. By "unit," we mean unit in a
       statistical sense, where a population consists of a number of units.
       In Table 1, a unit is a student, and each student is uniquely 
       identified by a value that appears in the student_id column.
       Table 1 defines data for students 0, 1, and 2.
    2. The time at which a certain record begins to be applicable. In Table 1,
       start_year is this start time.
    3. The time at which a certain record ceases to be applicable. In Table 1,
       end_year is this stop time.
    4. The name of a feature applicable to that unit at that time. In Table 1,
       this is "feature" 
    5. The value of the feature for that unit at that time. In Table 1, this is
       Value

    The values in the first column uniquely identify each unit, but there
    can be more than one row in the table per unit. These tables give us
    information in the form of: "For unit u, from time t1 to time t2, feature f 
    had value x"

    In Table 1, the values of the student_id column each correspond to
    one student. Each student may have multiple rows on this table 
    corresponding to multiple features at multiple times. For example, during
    2005-2006, student 0 had a math_gpa of 2.3 and an english_gpa of 4.0.
    During 2006-2007, student 0's math_gpa dropped to 2.1, while his or her
    english_gpa dropped to 3.9.

    If a record does not have a time frame, but can be considered to last
    "forever" (somebody's name, for example) then the start time and end
    time columns can be left NULL. These records will appear in all time
    intervals
    
    If a record only has one time associated
    (for example, the time that a parking tissue was issued) then either 
    start time or stop time can be left NULL, and the other can be filled in.

    **M Tables**

    ArrayEmitter generates M formatted tables based on RG formatted tables. 
    For example, the RG-formatted table Table 1 might result in the following 
    M-formatted table:

    *Table 2*

    +------------+----------+-------------+----------+
    | student_id | math_gpa | english_gpa | absences |
    +============+==========+=============+==========+
    |          0 |      2.2 |        3.95 |        8 |
    +------------+----------+-------------+----------+
    |          1 |     3.45 |         nan |        0 |
    +------------+----------+-------------+----------+
    |          2 |      3.4 |         nan |       96 |
    +------------+----------+-------------+----------+

    In an M-formatted table, each unit has a single row, and each feature has
    its own column. Notice that the student_ids in Table 2 correspond to the
    student_ids in Table 1, and the names of the columns in Table 2 correspond
    to the entries in the "feature" column of Table 1. The process used to 
    determine the values in these columns is elucidated below.

    **Converting an RG-formatted table to an M-formatted table.**

    In order to decide what values appear in our M-formatted table, we:

    1. Optionally select a aggregation methods with set_aggregation and 
       set_default_aggregation
    2. Select a timeframe with emit_M

    When creating the M table, we first take only entries in the RG table
    table that fall within the timeframe specified in emit_M, then we aggregate 
    those entries using the user_specified aggretation method. If an aggreagation 
    method is not specified, ArrayGenerator will take the mean. For example, if
    we have Table 1 stored in table1.csv, and run the following:

    >>> ae = ArrayEmitter()
    >>> ae = ae.get_rg_from_csv('table1.csv')
    >>> ae = ae.set_aggregation('math_gpa', 'AVG')
    >>> ae = ae.set_aggregation('absences', 'MAX')
    >>> ae = ae.set_interval(2005, 2006)
    >>> table2 = ae.emit_M()

    we end up with Table 2

    Notice that math_gpa and english_gpa are the average for 2005 and 2006
    per student, while absences is the max over 2005 and 2006. Also notice
    that english_gpa for student 1 is nan, since the only english_gpa for
    student 1 is from 2007, which is outside of our range. For student 2,
    english_gpa is nan because student 2 has no entries in the table for
    english_gpa.

    **Taking subsets of units**

    In addition to taking subsets of items in RG tables, we might also 
    want to take subsets of units (i.e. rows in M-format tables) according
    to some perameter. For example, we might want to consider only 
    students with a math_gpa at or below 3.4. In order to subset units, we use 
    the select_rows_in_M function. For example:

    >>> ae = ArrayEmitter()
    >>> ae = ae.get_rg_from_csv('table1.csv')
    >>> ae = ae.set_aggregation('math_gpa', 'AVG')
    >>> ae = ae.set_aggregation('absences', 'MAX')
    >>> ae = ae.select_rows_in_M('math_gpa <= 3.4')
    >>> ae = ae.set_interval(2005, 2006)
    >>> table3 = ae.emit_M()

    Gives us 
    
    *Table 3:*

    +------------+----------+-------------+----------+
    | student_id | math_gpa | english_gpa | absences |
    +============+==========+=============+==========+
    |          0 |      2.2 |        3.95 |        8 |
    +------------+----------+-------------+----------+
    |          2 |      3.4 |         nan |       96 |
    +------------+----------+-------------+----------+

    Notice that Table 3 is identical to Table 2, except student 1 has been
    omitted because his/her GPA is higher than 3.4.

    **Note on function call semantics**

    Most methods of ArrayEmitter return new ArrayEmitters rather than 
    modifying the existing ArrayEmitter. 

    Parameters
    ----------
    convert_to_unix_time : boolean
        Iff true, user queries in set_interval will be translated from
        datetimes to unix time (seconds since The Epoch). The user may wish 
        to set this variable if the database stores times in unix time
   
    """

    def __init__(self, convert_to_unix_time=False):
        self.__conn = None
        self.__rg_table_name = None
        self.__selections = []
        self.__aggregations = {}
        self.__default_aggregation = 'AVG'
        self.__col_specs = {}
        self.__convert_to_unix_time = convert_to_unix_time
        self.__start_time = None
        self.__stop_time = None

    def __copy(self):
        cp = ArrayEmitter()
        cp.__conn = self.__conn
        cp.__rg_table_name = self.__rg_table_name
        cp.__selections = list(self.__selections)
        cp.__aggregations = self.__aggregations.copy()
        cp.__default_aggregation = self.__default_aggregation
        cp.__col_specs = self.__col_specs.copy()
        cp.__convert_to_unix_time = self.__convert_to_unix_time 
        cp.__start_time = self.__start_time
        cp.__stop_time = self.__stop_time
        return cp

    def get_rg_from_sql(self, conn_str, table_name, unit_id_col=None, 
                        start_time_col=None, stop_time_col=None, 
                        feature_col=None, val_col=None): 
        """ Gets an RG-formatted matrix from a CSV file
           
        Parameters
        ----------
        conn_str : str
            SQLAlchemy connection string to connect to the database and run
            the query. 

        table_name : str
            The name of the RG-formatted table in the database


        unit_id_col : str or None
            The name of the column containing unique unit IDs. For example,
            in Table 1, this is 'student_id'. If None, ArrayEmitter will
            pick the first otherwise unspecified column

        start_time_col : str or None
            The name of the column containing start time. In Table 1,
            this is 'start_year'. If None, ArrayEmitter will pick the second
            otherwise unspecified column.

        end_time_col : str or None
            The name of the column containing the stop time. In Table 1,
            this is 'end_year'. If None, ArrayEmitter will pick the third
            otherwise unspecified column.

        feature_col : str or None
            The name of the column containing the feature name. In Table 1,
            this is 'feature'. If None, ArrayEmitter will pick the fourth
            otherwise unspecified column.

        val_col : str or None
            The name of the column containing the value for the given
            feature for the given user at the given time. In Table 1,
            this is 'value'. If None, ArrayEmitter will pick the fifth
            otherwise unspecified column.

        Returns
        -------
        ArrayGenerator
            Copy of this ArrayGenerator which has rg_table specified
            
        Examples
        --------
        >>> conn_str = ...
        >>> ae = ArrayEmitter()
        >>> ae = ae.get_rg_from_SQL('SELECT * FROM table_1', 'student_id', 
        ...                         conn_str=conn_str)

        """
        cp = self.__copy()
        cp.__conn = connect_sql(conn_str, allow_pgres_copy_optimization=True)
        cp.__rg_table_name = table_name
        cp.__col_specs['unit_id'] = unit_id_col
        cp.__col_specs['start_time'] = start_time_col
        cp.__col_specs['stop_time'] = stop_time_col
        cp.__col_specs['feature'] = feature_col
        cp.__col_specs['val'] = val_col
        cp.__resolve_cols()
        return cp

    def get_rg_from_csv(self, csv_file_path, parse_datetimes=[], 
                        unit_id_col=None, 
                        start_time_col=None, stop_time_col=None, 
                        feature_col=None, val_col=None):
        """ Get an RG-formatted table from a CSV file.
       
        Parameters
        ----------
        csv_file_path : str
            Path of the csv file to import table from

        parse_datetimes : list of col names
            Columns that should be interpreted as datetimes

        unit_id_col : str or None
            The name of the column containing unique unit IDs. For example,
            in Table 1, this is 'student_id'. If None, ArrayEmitter will
            pick the first otherwise unspecified column

        start_time_col : str or None
            The name of the column containing start time. In Table 1,
            this is 'start_year'. If None, ArrayEmitter will pick the second
            otherwise unspecified column.

        end_time_col : str or None
            The name of the column containing the stop time. In Table 1,
            this is 'end_year'. If None, ArrayEmitter will pick the third
            otherwise unspecified column.

        feature_col : str or None
            The name of the column containing the feature name. In Table 1,
            this is 'feature'. If None, ArrayEmitter will pick the fourth
            otherwise unspecified column.

        val_col : str or None
            The name of the column containing the value for the given
            feature for the given user at the given time. In Table 1,
            this is 'value'. If None, ArrayEmitter will pick the fifth
            otherwise unspecified column.

        Returns
        -------
        ArrayGenerator
            Copy of this ArrayGenerator which has rg_table specified

        Examples
        --------
            
        >>> ae = ArrayEmitter()
        >>> ae = ae.get_rg_from_csv('table_1.csv')             
        """
        # in-memory db
        cp = self.__copy()
        conn = connect_sql('sqlite://')
        cp.__rg_table_name = utils.csv_to_sql(
                conn, 
                csv_file_path, 
                parse_datetimes=parse_datetimes)
        cp.__conn = conn
        cp.__col_specs['unit_id'] = unit_id_col
        cp.__col_specs['start_time'] = start_time_col
        cp.__col_specs['stop_time'] = stop_time_col
        cp.__col_specs['feature'] = feature_col
        cp.__col_specs['val'] = val_col
        cp.__resolve_cols()
        # SQLite doesn't really have datetimes, so we transparently translate
        # to unix times.
        cp.__convert_to_unix_time = True
        return cp

    def set_aggregation(self, feature_name, method):
        """Sets the method used to aggregate across dates in a RG table.

        Parameters
        ----------
        feature_name : str
            Name of feature for which we are aggregating
        method : str 
            Method used to aggregate the feature across year. 
            Can be one of:

                * 'AVG'
                    Mean average

                * 'COUNT'
                    Number of results

                * 'MAX'
                    Largest result

                * 'MIN'
                    Smallest result 

                * 'SUM'
                    Sum of results

            Additionally, method can be any aggregation function supported
            by the database in which the RG table lives.
            
        Returns
        -------
        ArrayGenerator
            Copy of this ArrayGenerator with aggregation set

        Examples
        --------
        >>> ae = ArrayEmitter()
        >>> ... # Populate ag with Table 1 and Table 2
        >>> ae = ae.set_aggregation('math_gpa', 'mean')
        >>> ae = ae.set_aggregation('absences', 'max')
        >>> ae = ae.set_interval(2005, 2006)
        >>> sa = ae.emit_M()

        """
        # TODO make sure method is valid
        cp = self.__copy()
        cp.__aggregations[feature_name] = method
        return cp

    def set_default_aggregation(self, method):
        """Sets the default method used to aggregate across dates

        ArrayEmitter will use the value of set_default_aggregation when
        a method has not been set for a given feature using the
        set_aggregation method.

        When set_default_aggregation has not been called, the default
        aggregation method is 'AVG'

        Parameters
        ----------
        method : str 
            Method used to aggregate features across year. 
            Can be one of:

                * 'AVG'
                    Mean average

                * 'COUNT'
                    Number of results

                * 'MAX'
                    Largest result

                * 'MIN'
                    Smallest result 

                * 'SUM'
                    Sum of results

            Additionally, method can be any aggregation function supported
            by the database in which the RG table lives.

        Returns
        -------
        ArrayGenerator
            Copy of this ArrayGenerator with default aggregation set

        """
        cp = self.__copy()
        cp.__default_aggregation = method
        return cp
    
    def select_rows_in_M(self, where):
        """
        
        Specifies a subset of the units to be returned in the M-table 
        according to some constraint.

        Parameters
        ----------
        where : str
            A statement required to be true about the returned table using
            at least one column name, constant values, parentheses and the 
            operators: =, !=, <, >, <=, >=, AND, OR, NOT, and other things
            that can appear in a SQL WHERE statement

        Returns
        -------
        ArrayGenerator
            A copy of the current ArrayGenerator with the additional where 
            condition added

        Examples
        --------
        >>> ae = ArrayEmitter()
        >>> ... # Populate ag with Table 1 and Table 2
        >>> ae = ae.set_aggregation('math_gpa', 'mean')
        >>> ae = ae.set_aggregation('absences', 'max')
        >>> ae = ae.select_rows_in_M('grad_year == 2007')
        >>> ae = ae.set_interval(2005, 2006)
        >>> sa = ae.emit_M()
        """
        # Note that this copies the original rather than mutating it, so
        # taking a subset does not permanently lose data.

        # We can recycle the mini-language from UPSG Query
        # https://github.com/dssg/UPSG/blob/master/upsg/transform/split.py#L210
        cp = self.__copy()
        cp.__selections.append(where)
        return cp

    def set_interval(self, start_time, stop_time):
        """Sets interval used to create M-formatted table

        Start times and stop times are inclusive

        Parameters
        ----------
        start_time : number or datetime.datetime
            Start time of log tables to include in this sa
        stop_time : number or datetime.datetime
            Stop time of log tables to include in this sa

        Returns
        -------
        ArrayEmitter
            With interval set
        """
        cp = self.__copy()
        cp.__start_time = start_time
        cp.__stop_time = stop_time
        return cp
        
    def __resolve_cols(self):
        col_specs = self.__col_specs
        conn = self.__conn
        table_name = self.__rg_table_name

        # figure out which column is which
        sql_col_name = 'SELECT * FROM {} LIMIT 1;'.format(table_name)
        col_names = conn.execute(sql_col_name).dtype.names
        specified_col_names = [col_name for col_name in 
                               col_specs.itervalues() if col_name
                               is not None]
        unspecified_col_names = [col_name for col_name in col_names if col_name 
                                 not in specified_col_names]
        for spec in ('unit_id', 'start_time', 'stop_time', 'feature', 'val'):
            if col_specs[spec] is None:
                col_specs[spec] = unspecified_col_names.pop(0)

    def get_query(self):
        """Returns SQL query that will be used to create the M-formatted table
        """
        start_time = self.__start_time
        stop_time = self.__stop_time
        if self.__convert_to_unix_time:
            start_time = utils.to_unix_time(start_time)
            stop_time = utils.to_unix_time(stop_time)
        try:
            # If times are numbers, their string representations don't have
            # surrounding single quotes. Otherwise, they do
            float(start_time)
        except ValueError:
            start_time = "'{}'".format(start_time)
        try:
            float(stop_time)
        except ValueError:
            stop_time = "'{}'".format(start_time)

        col_specs = self.__col_specs
        conn = self.__conn
        table_name = self.__rg_table_name

        # get all features
        sql_features = 'SELECT DISTINCT {} FROM {};'.format(
                col_specs['feature'], 
                table_name)
        feat_names = [row[0] for row in conn.execute(sql_features)]

        # figure out aggregations
        aggregations = {feat_name: self.__aggregations[feat_name] if 
                        self.__aggregations.has_key(feat_name) else
                        self.__default_aggregation for feat_name in
                        feat_names}

        # Now we build the complicated sql query
        sql_select_clause = 'SELECT id_tbl.id, {} '.format(
                ', '.join(['{0}_tbl.val AS {0}'.format(feat) for feat in 
                           feat_names]))
        sql_from_clause_top = ('FROM ({}SELECT DISTINCT {} AS id FROM {}) id_tbl '
                               'LEFT JOIN ').format(
                                '(' * len(feat_names),
                                col_specs['unit_id'],
                                table_name)
        sql_from_clause_features = 'LEFT JOIN '.join(
            [("(SELECT {unit_id_col} AS id, {aggr}({val_col}) AS val FROM "
              "{table_name} WHERE "
              "{feature_col} = '{feat_name}' AND "
              "(({start_time_col} >= {start_time} "
              "  AND {start_time_col} <= {stop_time}) "
              " OR {start_time_col} IS NULL) AND "
              "(({stop_time_col} >= {start_time} "
              "  AND {stop_time_col} <= {stop_time}) "
              " OR {stop_time_col} IS NULL) "
              "GROUP BY id) {feat_name}_tbl ON "
              "id_tbl.id = {feat_name}_tbl.id) ").format(
                  unit_id_col=col_specs['unit_id'],
                  aggr=aggregations[feat_name],
                  val_col=col_specs['val'],
                  table_name=table_name,
                  feature_col=col_specs['feature'],
                  start_time_col=col_specs['start_time'],
                  start_time=start_time,
                  stop_time_col=col_specs['stop_time'],
                  stop_time=stop_time,
                  feat_name=feat_name) for feat_name in feat_names])
        # TODO we can probably do something more sophisticated than just 
        # throwing the user's directives in here
        sql_where_clause = ''
        if self.__selections:
            sql_where_clause = "WHERE " + "AND ".join(
                ['({})'.format(sel) for sel in self.__selections]) 
        sql_select = '{} {} {} {}'.format(
            sql_select_clause, 
            sql_from_clause_top,
            sql_from_clause_features, 
            sql_where_clause)
        return sql_select

    def emit_M(self):
        """Creates a structured array in M-format

        Returns
        -------
        np.ndarray
            Numpy structured array constructed using the specified queries and
            subsets
        """
        return self.__conn.execute(self.get_query())

    def subset_over(
            self, 
            label_col,
            interval_train_window_start,
            interval_train_window_size,
            interval_test_window_start,
            interval_test_window_size,
            interval_inc_value,
            interval_expanding=False,
            row_M_col_name=None,
            row_M_train_window_start=None,
            row_M_train_window_size=None,
            row_M_test_window_start=None,
            row_M_test_window_size=None,
            row_M_inc_value=None,
            row_M_expanding=False,
            clfs=[{'clf': RandomForestClassifier}],
            feature_gen_lambda=None):
        """
        Generates ArrayGenerators according to some subsetting directive.

        There are two ways that we determine what the train and test sets are
        for each trial:

        1. The start time/stop time interval. This is the interval used to
           create features in the M-formatted matrix. Setting the start 
           time/stop time of this interval is equalivalent to passing values 
           to set_interval.  variables pertaining to this interval have the 
           interval* prefix.

        2. The rows of the M matrix to select, based on the value of some
           column in the M matrix. Setting the start and end of this interval
           is equivalent to passing values to select_rows_in_M. Values 
           pertaining to this set of rows have the row_M* prefix. Taking
           subsets over rows of M is optional, and it will only occur if
           row_M_col_name is not None

        Parameters
        ----------
        label_col : str
            The name of the column containing labels
        interval_train_window_start : number or datetime
            start of training interval
        interval_train_window_size : number or datetime
            (Initial) size of training interval
        interval_test_window_start : number or datetime
            start of testing interval
        interval_test_window_size : number or datetime
            size of testing interval
        interval_inc_value : datetime, timedelta, or number
            interval to increment train and test interval
        interval_expanding : boolean
            whether or not the training interval is expanding
        row_M_col_name : str or None
            If not None, the name of the feature which will be used to select
            different training and testing sets in addition to the interval

            If None, train and testing sets will use all rows given a 
            particular time interval
        row_M_train_window_start : ? or None
            Start of train window for M rows. If None, uses
            interval_train_window_start
        row_M_train_window_size : ? or None
            (Initial) size of train window for M rows. If None, uses
            interval_train_window_size
        row_M_test_window_start : ? or None
            Start of test window for M rows. If None, uses
            interval_test_window_start
        row_M_train_window_size : ? or None
            size of test window for M rows. If None, uses
            interval_test_window_size
        row_M_inc_value : ? or None
            interval to increment train and test window for M rows. If None,
            uses interval_inc_value
        row_M_expanding : bool
            whether or not the training window for M rows is expanding
        clfs : list of dict
            classifiers and parameters to run with each train/test set. See
            documentation for diogenes.grid_search.experiment.Experiment.
        feature_gen_lambda : (np.ndarray, str, ?, ?, ?, ?) -> np.ndarray or None
            If not None,function to by applied to generated arrays before they 
            are fit to classifiers. Must be a function of signature:

            f(M, test_or_train, interval_start, interval_end, row_M_start,
              row_M_end)

            Where:
            * M is the generated array, 
            * test_or_train is 'test' if this is a test set or 'train' if it's
              a train set
            * interval_start and interval_end define the interval
            * row_M_start and row_M_end define the rows of M that are included

        Returns
        -------
        diogenes.grid_search.experiment.Experiment
            Experiment collecting train/test sets that have been run
        """
        if row_M_train_window_start is None:
            row_M_train_window_start = interval_train_window_start
        if row_M_train_window_size is None:
            row_M_train_window_size = interval_train_window_size
        if row_M_test_window_start is None:
            row_M_test_window_start = interval_test_window_start
        if row_M_test_window_size is None:
            row_M_test_window_size = interval_test_window_size
        if row_M_inc_value is None:
            row_M_inc_value = interval_inc_value

        conn = self.__conn
        col_specs = self.__col_specs
        table_name = self.__rg_table_name

        sql_get_max_interval_end = 'SELECT MAX({}) FROM {}'.format(
               col_specs['stop_time'],
               table_name)
        interval_end = conn.execute(
                sql_get_max_interval_end)[0][0]
        if row_M_col_name is not None:
            sql_get_max_col = ("SELECT MAX({}) FROM {} "
                               "WHERE {} = '{}'").format(
                                   col_specs['val'],
                                   table_name,
                                   col_specs['feature'],
                                   row_M_col_name)
            row_M_end = conn.execute(sql_get_max_col)[0][0]
        else:
            row_M_end = interval_end

        trial_directives = []
        for clf_params in clfs:
            clf = clf_params['clf']
            all_clf_ps = clf_params.copy()
            del all_clf_ps['clf']
            for param_dict in utils.transpose_dict_of_lists(all_clf_ps):
                trial_directives.append((clf, param_dict, []))

        current_interval_train_start = interval_train_window_start
        current_interval_train_end = (interval_train_window_start + 
                                      interval_train_window_size)
        current_interval_test_start = interval_test_window_start
        current_interval_test_end = (interval_test_window_start + 
                                      interval_test_window_size)
        current_row_M_train_start = row_M_train_window_start
        current_row_M_train_end = (row_M_train_window_start + 
                                      row_M_train_window_size)
        current_row_M_test_start = row_M_test_window_start
        current_row_M_test_end = (row_M_test_window_start + 
                                      row_M_test_window_size)
        while (current_interval_test_end <= interval_end and
               current_row_M_test_end <= row_M_end):
            ae_train = self.set_interval(current_interval_train_start,
                                        current_interval_train_end)
            ae_test = self.set_interval(current_interval_test_start,
                                        current_interval_test_end)
            if row_M_col_name is not None:
                ae_train = ae_train.select_rows_in_M(
                        '{col} >= {start} AND {col} <= {stop}'.format(
                            col=row_M_col_name,
                            start=current_row_M_train_start,
                            stop=current_row_M_train_end))
                ae_test = ae_test.select_rows_in_M(
                        '{col} >= {start} AND {col} <= {stop}'.format(
                            col=row_M_col_name,
                            start=current_row_M_test_start,
                            stop=current_row_M_test_end))
            # TODO this should actually run clfs and build an experiment 
            # rather than doing this yield
            data_train = ae_train.emit_M()
            M_train = utils.remove_cols(data_train, label_col)
            y_train = data_train[label_col]
            data_test = ae_test.emit_M()
            M_test = utils.remove_cols(data_test, label_col)
            y_test = data_test[label_col]

            if feature_gen_lambda is not None:
                M_train = feature_gen_lambda(
                        M_train, 
                        'train', 
                        current_interval_train_start,
                        current_interval_train_end,
                        current_row_M_train_start,
                        current_row_M_train_end)
                M_test = feature_gen_lambda(
                        M_test,
                        'test',
                        current_interval_test_start,
                        current_interval_test_end,
                        current_row_M_test_start,
                        current_row_M_test_end)

            col_names = M_train.dtype.names
            M_train_nd = utils.cast_np_sa_to_nd(M_train)
            M_test_nd = utils.cast_np_sa_to_nd(M_test)

            for clf, params, runs in trial_directives:
                clf_inst = clf(**params)
                clf_inst.fit(M_train_nd, y_train)
                runs.append(exp.Run(
                    M_train_nd,
                    y_train,
                    col_names,
                    clf_inst,
                    None,
                    None,
                    col_names,
                    np.arange(len(col_names)),
                    {'train_interval_start': current_interval_train_start,
                     'train_interval_end': current_interval_train_end,
                     'test_interval_start': current_interval_test_start,
                     'test_interval_end': current_interval_test_end},
                    {'train_start': current_row_M_train_start,
                     'train_end': current_row_M_train_end,
                     'test_start': current_row_M_test_start,
                     'test_end': current_row_M_test_end},
                    M_test_nd,
                    y_test))

            if not interval_expanding:
                current_interval_train_start += interval_inc_value
            current_interval_train_end += interval_inc_value
            current_interval_test_start += interval_inc_value
            current_interval_test_end += interval_inc_value
            if not row_M_expanding:
                current_row_M_train_start += row_M_inc_value
            current_row_M_train_end += row_M_inc_value
            current_row_M_test_start += row_M_inc_value
            current_row_M_test_end += row_M_inc_value

        trials = [exp.Trial(
            None,
            None,
            None,
            clf,
            params,
            'Array Emitter',
            {},
            'Array Emitter',
            {},
            [runs]) for clf, params, runs in trial_directives]
        return exp.Experiment(
                None, 
                None, 
                clfs, 
                [{'subset': 'Array Emitter'}],
                [{'cv': 'Array Emitter'}],
                trials)

def _remove_if_present(l, item):
    try:
        l.remove(item)
    except ValueError:
        pass

def M_to_rg(conn_str, from_table, to_table, unit_id_col, 
            start_time_col=None, stop_time_col=None, feature_cols=None):
    conn = connect_sql(conn_str, allow_pgres_copy_optimization=True)
    sql = ('CREATE TABLE IF NOT EXISTS {} '
           '(row_id SERIAL PRIMARY KEY, unit_id INT, start_time TIMESTAMP, '
           ' end_time TIMESTAMP, feat TEXT, val REAL);').format(to_table)
    conn.execute(sql)
    if feature_cols is None:
        sql = 'SELECT * FROM {} LIMIT 1'.format(from_table)
        feature_cols = conn.execute(sql).dtype.names
    feature_cols = list(feature_cols)
    _remove_if_present(feature_cols, unit_id_col)
    _remove_if_present(feature_cols, start_time_col)
    _remove_if_present(feature_cols, stop_time_col)
    sql_numeric = ("INSERT INTO {to_table} (unit_id, start_time, end_time, "
                   "                        feat, val) "
                   "SELECT {unit_id_col}, {start_time_col}, {stop_time_col}, "
                   "        '{{feat_col}}', {{feat_col}} "
                   "FROM {from_table} WHERE {{feat_col}} IS NOT NULL").format(
                       to_table=to_table,
                       unit_id_col=unit_id_col,
                       start_time_col=(start_time_col if start_time_col is not None
                                       else 'NULL'),
                       stop_time_col=(stop_time_col if stop_time_col is not None
                                      else 'NULL'),
                       from_table=from_table)
    sql_boolean = ("INSERT INTO {to_table} (unit_id, start_time, end_time, "
                   "                        feat, val) "
                   "SELECT {unit_id_col}, {start_time_col}, {stop_time_col}, "
                   "        '{{feat_col}}', "
                   "        CASE WHEN {{feat_col}} THEN 1 ELSE 0 END "
                   "FROM {from_table} WHERE {{feat_col}} IS NOT NULL").format(
                       to_table=to_table,
                       unit_id_col=unit_id_col,
                       start_time_col=(start_time_col if start_time_col is not None
                                       else 'NULL'),
                       stop_time_col=(stop_time_col if stop_time_col is not None
                                      else 'NULL'),
                       from_table=from_table)
    sql_char_encode_table_name = ('{to_table}_encode_{from_table}_'
                                  '{{feat_col}}').format(
            from_table=from_table.replace('.', '_'),
            to_table=to_table)
    sql_char = ("DROP TABLE IF EXISTS {encode_table}; "
                "CREATE TABLE {encode_table} AS "
                "   SELECT DISTINCT {{feat_col}} AS val, "
                "       RANK() OVER(ORDER BY {{feat_col}}) AS encoding "
                "   FROM {from_table}; "
                "INSERT INTO {to_table} (unit_id, start_time, end_time, "
                "                        feat, val) "
                "   SELECT {from_table}.{unit_id_col}, "
                "       {start_time_col}, "
                "       {stop_time_col}, "
                "       '{{feat_col}}', "
                "       {encode_table}.encoding "
                "   FROM {from_table} JOIN {encode_table} "
                "   ON {from_table}.{{feat_col}} = {encode_table}.val "
                "   WHERE {from_table}.{{feat_col}} IS NOT NULL ").format(
                       to_table=to_table,
                       unit_id_col=unit_id_col,
                       start_time_col=(from_table + '.' + start_time_col if 
                                       start_time_col is not None
                                       else 'NULL'),
                       stop_time_col=(from_table + '.' + stop_time_col if 
                                      stop_time_col is not None
                                      else 'NULL'),
                       from_table=from_table,
                       encode_table=sql_char_encode_table_name)
    sql_test = ("SELECT {{feat_col}} FROM {from_table} "
                "WHERE {{feat_col}} IS NOT NULL LIMIT 1").format(
                        from_table=from_table)
    for feat_col in feature_cols:
        sql = sql_test.format(feat_col=feat_col)
        type_test = conn.execute(sql)
        type_test_char = type_test.dtype[0].char
        if type_test_char in ('O', 'S'): # string dtype. Must encode
            sql = sql_char.format(feat_col=feat_col)
        elif type_test_char == '?': # boolean. Must translate to int
            sql = sql_boolean.format(feat_col=feat_col)
        else: # numeric dtype. Leave as is
            sql = sql_numeric.format(feat_col=feat_col)
        print sql
        conn.execute(sql.format(feat_col=feat_col))


