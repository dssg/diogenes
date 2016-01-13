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

    +------------+--------------+-----------------+--------------+
    | student_id | math_gpa_AVG | english_gpa_AVG | absences_MAX |
    +============+==============+=================+==============+
    |          0 |          2.2 |            3.95 |            8 |
    +------------+--------------+-----------------+--------------+
    |          1 |         3.45 |             nan |            0 |
    +------------+--------------+-----------------+--------------+
    |          2 |          3.4 |             nan |           96 |
    +------------+--------------+-----------------+--------------+

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
    >>> ae = ae.select_rows_in_M('math_gpa_AVG <= 3.4')
    >>> ae = ae.set_interval(2005, 2006)
    >>> table3 = ae.emit_M()

    Gives us 
    
    *Table 3:*

    +------------+--------------+-----------------+--------------+
    | student_id | math_gpa_AVG | english_gpa_AVG | absences_MAX |
    +============+==============+=================+==============+
    |          0 |          2.2 |            3.95 |            8 |
    +------------+--------------+-----------------+--------------+
    |          2 |          3.4 |             nan |           96 |
    +------------+--------------+-----------------+--------------+

    Notice that Table 3 is identical to Table 2, except student 1 has been
    omitted because his/her GPA is higher than 3.4.

    **Taking labels and features from different time intervals**

    If you need to take labels and the rest of your features from different
    time intervals, set the label column with set_label_feature and set the
    label interval with set_label_interval.

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
        self.__label_feature_name = None
        self.__label_start_time = None
        self.__label_stop_time = None

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
        cp.__label_feature_name = self.__label_feature_name
        cp.__label_start_time = self.__label_start_time
        cp.__label_stop_time = self.__label_stop_time
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

    def set_label_feature(self, feature_name):
        """Sets the feature in the array which will be considered the label

        Returns
        -------
        ArrayGenerator
            Copy of this ArrayGenerator with specified label column

        """
        cp = self.__copy()
        cp.__label_feature_name = feature_name
        return cp

    def set_aggregation(self, feature_name, method):
        """Sets the method or methods used to aggregate across dates in the 
        RG table.

        Parameters
        ----------
        feature_name : str
            Name of feature for which we are aggregating
        method : str or list of strs
            Method or methods used to aggregate the feature across year. 
            If a str, can be one of:

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

            If a list, will create one aggregate column for each method
            in the list, for example: ['AVG', 'MIN', 'MAX']
            
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
        #TODO update docs with multiple aggregations stuff
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

    def set_label_interval(self, start_time, stop_time):
        """Sets interval from which to select labels

        Parameters
        ----------
        start_time : number or datetime.datetime
            Start time of log tables to include in this sa's labels
        stop_time : number or datetime.datetime
            Stop time of log tables to include in this sa's labels

        Returns
        -------
        ArrayEmitter
            With label interval set
        """

        cp = self.__copy()
        cp.__label_start_time = start_time
        cp.__label_stop_time = stop_time
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

    def __clean_time(self, time):
        if self.__convert_to_unix_time:
            time = utils.to_unix_time(time)
        try:
            float(time)
        except (ValueError, TypeError):
            time = "'{}'".format(time)
        return time

    def __feature_subqueries(self, feat_name, start_time, stop_time, label_start_time,
                           label_stop_time):
        aggregations = self.__aggregations
        table_name = self.__rg_table_name
        label_feature_name = self.__label_feature_name
        aggrs = aggregations.get(feat_name, self.__default_aggregation)
        col_specs = self.__col_specs
        if isinstance(aggrs, basestring):
            aggrs = [aggrs]

        feat_table = '{}_tbl'.format(feat_name)
        nicknames = ', '.join(
                ['{feat_name}_{aggr}'.format(
                    feat_name=feat_name,
                    aggr=aggr) for aggr in aggrs])

        select_clause = ',\n'.join(
                ["            {feat_table}.val_{aggr} as {feat_name}_{aggr}".format(
                    feat_table=feat_table,
                    aggr=aggr,
                    feat_name=feat_name) for aggr in aggrs])

        with_clause_0 = ("    {feat_table} AS (\n"
                         "        SELECT\n"
                         "            {unit_id_col} as id,\n").format(
                            unit_id_col=col_specs['unit_id'],
                            feat_table=feat_table)
        with_clause_1 = ',\n'.join(
                ["            {aggr}({val_col}) as val_{aggr}".format(
                    aggr=aggr,
                    val_col=col_specs['val']) for aggr in aggrs]) + '\n'
        with_clause_2 = ("        FROM\n"
                         "            {table_name}\n"
                         "        WHERE\n"
                         "            {feature_col} = '{feat_name}'\n"
                         "            AND\n"
                         "            (\n"
                         "                (\n"
                         "                     {start_time_col} >= {start_time}\n"
                         "                     AND {start_time_col} <= {stop_time}\n"
                         "                )\n"
                         "                OR {start_time_col} IS NULL\n"
                         "            )\n"
                         "            AND\n"
                         "            (\n"
                         "                (\n"
                         "                    {stop_time_col} >= {start_time}\n"
                         "                    AND {stop_time_col} <= {stop_time}\n"
                         "                )\n"
                         "                OR {stop_time_col} IS NULL\n"
                         "            )\n"
                         "        GROUP BY id\n"
                         "    )").format(
                             val_col=col_specs['val'],
                             table_name=table_name,
                             feature_col=col_specs['feature'],
                             start_time_col=col_specs['start_time'],
                             start_time=(label_start_time if 
                                         feat_name == label_feature_name else
                                         start_time),
                             stop_time_col=col_specs['stop_time'],
                             stop_time=(label_stop_time if
                                        feat_name == label_feature_name else
                                        stop_time),
                             feat_name=feat_name)

        return {'table': feat_table,
                'nicknames': nicknames,
                'select': select_clause, 
                'with': '{}{}{}'.format(with_clause_0, with_clause_1, with_clause_2)}
                
    def get_query(self):
        """Returns SQL query that will be used to create the M-formatted table
        """
        start_time = self.__clean_time(self.__start_time)
        stop_time = self.__clean_time(self.__stop_time)

        col_specs = self.__col_specs
        conn = self.__conn
        table_name = self.__rg_table_name

        label_feature_name = self.__label_feature_name
        label_start_time = self.__label_start_time
        label_stop_time = self.__label_stop_time

        if label_feature_name is not None:
            if label_start_time is None:
                label_start_time = start_time
            else:
                label_start_time = self.__clean_time(label_start_time)
            if label_stop_time is None:
                label_stop_time = end_time
            else:
                label_stop_time = self.__clean_time(label_stop_time)


        # get all features
        sql_features = 'SELECT DISTINCT {} FROM {};'.format(
                col_specs['feature'], 
                table_name)
        feat_names = [row[0] for row in conn.execute(sql_features)]

        # get per_feature subqueries
        subqueries = [self.__feature_subqueries(
            feat_name, 
            start_time, 
            stop_time,
            label_start_time,
            label_stop_time) for feat_name in feat_names]

        # build temporary tables
        sql_with_clause_0 = ("WITH\n"
                             "    id_tbl AS (\n"
                             "        SELECT DISTINCT\n" 
                             "            {unit_id_col} AS id\n"
                             "        FROM\n"
                             "            {table_name}\n"
                             "    ),\n\n").format(
                                   unit_id_col=col_specs['unit_id'],
                                   table_name=table_name)
        sql_with_clause_1 = ',\n\n'.join(
                [subquery['with'] for subquery in subqueries])

        # build select out of subtables
        sql_inner_select_clause_0 = (",\n\n"
                                     "    inner_select_tbl AS (\n"
                                     "        SELECT\n"
                                     "            id_tbl.id,\n")
        sql_inner_select_clause_1 = ',\n'.join(
                [subquery['select'] for subquery in subqueries])
                                 
        sql_inner_from_clause_0 = ("\n"
                                   "        FROM\n"
                                   "            id_tbl\n"
                                   "            LEFT JOIN ")
        sql_inner_from_clause_1 = "\n            LEFT JOIN ".join(
            ["{feat_tbl}\n                 ON {feat_tbl}.id = id_tbl.id".format(
                feat_tbl=subquery['table']) for subquery in subqueries])

        sql_select_clause=("\n"
                           "    )\n"
                           "\n"
                           "SELECT\n"
                           "    *\n" 
                           "FROM\n"
                           "    inner_select_tbl")
        # TODO we can probably do something more sophisticated than just 
        # throwing the user's directives in here
        # TODO something with better performance than coalesce
        sql_where_clause = "\nWHERE\n    COALESCE({}) IS NOT NULL".format(
                ', '.join([subquery['nicknames'] for subquery in subqueries]))
        if self.__selections:
            sql_where_clause += " AND\n" + " AND\n".join(
                ['    ({})'.format(sel) for sel in self.__selections]) 

        sql_select = '{}{}{}{}{}{}{}{};'.format(
            sql_with_clause_0,
            sql_with_clause_1,
            sql_inner_select_clause_0,
            sql_inner_select_clause_1,
            sql_inner_from_clause_0,
            sql_inner_from_clause_1,
            sql_select_clause,
            sql_where_clause)
        return sql_select

    def __feature_subqueries_nonlabel(
            self, 
            feat_name):
        aggregations = self.__aggregations
        table_name = self.__rg_table_name
        label_feature_name = self.__label_feature_name
        aggrs = aggregations.get(feat_name, self.__default_aggregation)
        col_specs = self.__col_specs
        if isinstance(aggrs, basestring):
            aggrs = [aggrs]

        feat_table = '{}_tbl'.format(feat_name)
        nicknames = ', '.join(
                ['{feat_name}_{aggr}'.format(
                    feat_name=feat_name,
                    aggr=aggr) for aggr in aggrs])

        select_clause = ',\n'.join(
                ["            {feat_table}.val_{aggr} as {feat_name}_{aggr}".format(
                    feat_table=feat_table,
                    aggr=aggr,
                    feat_name=feat_name) for aggr in aggrs])

        with_clause_0 = ("    {feat_table} AS (\n"
                         "        SELECT\n"
                         "            {table_name}.{unit_id_col} as id,\n").format(
                            unit_id_col=col_specs['unit_id'],
                            feat_table=feat_table)
        with_clause_1 = ',\n'.join(
                ["            {aggr}({table_name}.{val_col}) as val_{aggr}".format(
                    aggr=aggr,
                    val_col=col_specs['val']) for aggr in aggrs]) + '\n'
        with_clause_2 = ("        FROM\n"
                         "            {table_name} JOIN tbl_label\n"
                         "            ON {table_name}.{unit_id_col} = tbl_label.id\n"
                         "        WHERE\n"
                         "            {feature_col} = '{feat_name}'\n"
                         "            AND\n"
                         "            (\n"
                         "                {table_name}.{stop_time_col} < tbl_label.stop_time\n"
                         "                OR {start_time_col} IS NULL\n"
                         "            )\n"
                         "            AND\n"
                         "            (\n"
                         "                {table_name}.{stop_time_col} < tbl_label.stop_time\n"
                         "                OR {stop_time_col} IS NULL\n"
                         "            )\n"
                         "        GROUP BY {table_name}.{unit_id_col}\n"
                         "    )").format(
                             val_col=col_specs['val'],
                             table_name=table_name,
                             feature_col=col_specs['feature'],
                             start_time_col=col_specs['start_time'],
                             stop_time_col=col_specs['stop_time'],
                             feat_name=feat_name,
                             unit_id_col=col_specs['unit_id'])

        return {'table': feat_table,
                'nicknames': nicknames,
                'select': select_clause, 
                'with': '{}{}{}'.format(with_clause_0, with_clause_1, with_clause_2)}

    def __feature_subqueries_label(
            self, 
            start_time,
            stop_time):
        aggregations = self.__aggregations
        table_name = self.__rg_table_name
        feat_name = self.__label_feature_name
        aggrs = aggregations.get(feat_name, self.__default_aggregation)
        col_specs = self.__col_specs
        if isinstance(aggrs, basestring):
            aggrs = [aggrs]

        feat_table = 'label_tbl'
        nicknames = ', '.join(
                ['{feat_name}_{aggr}'.format(
                    feat_name=feat_name,
                    aggr=aggr) for aggr in aggrs])

        select_clause = ',\n'.join(
                ["            {feat_table}.val_{aggr} as {feat_name}_{aggr}".format(
                    feat_table=feat_table,
                    aggr=aggr,
                    feat_name=feat_name) for aggr in aggrs])

        with_clause_0 = ("    {feat_table} AS (\n"
                         "        SELECT\n"
                         "            {unit_id_col} as id,\n").format(
                            unit_id_col=col_specs['unit_id'],
                            feat_table=feat_table)
        with_clause_1 = ("            COALESCE(MAX({stop_time_col}), "
                         "MAX({start_time_col})) AS stop_time\n").format(
                                 start_time_col=col_specs['start_time'],
                                 stop_time_col=col_specs['stop_time'])
        with_clause_2 = ',\n'.join(
                ["            {aggr}({val_col}) as val_{aggr}".format(
                    aggr=aggr,
                    val_col=col_specs['val']) for aggr in aggrs]) + '\n'
        with_clause_3 = ("        FROM\n"
                         "            {table_name}\n"
                         "        WHERE\n"
                         "            {feature_col} = '{feat_name}'\n"
                         "            AND\n"
                         "            (\n"
                         "                (\n"
                         "                     {start_time_col} >= {start_time}\n"
                         "                     AND {start_time_col} <= {stop_time}\n"
                         "                )\n"
                         "                OR {start_time_col} IS NULL\n"
                         "            )\n"
                         "            AND\n"
                         "            (\n"
                         "                (\n"
                         "                    {stop_time_col} >= {start_time}\n"
                         "                    AND {stop_time_col} <= {stop_time}\n"
                         "                )\n"
                         "                OR {stop_time_col} IS NULL\n"
                         "            )\n"
                         "        GROUP BY id\n"
                         "    )").format(
                             val_col=col_specs['val'],
                             table_name=table_name,
                             feature_col=col_specs['feature'],
                             start_time_col=col_specs['start_time'],
                             start_time=start_time,
                             stop_time_col=col_specs['stop_time'],
                             stop_time=stop_time,
                             feat_name=feat_name)

        return {'table': feat_table,
                'nicknames': nicknames,
                'select': select_clause, 
                'with': '{}{}{}{}'.format(with_clause_0, with_clause_1, with_clause_2,
                    with_clause_3)}

    def get_query_with_labels(self):
        """Returns SQL query that will be used to create the M-formatted table
        
        (treating label columns separately)
        """
        col_specs = self.__col_specs
        conn = self.__conn
        table_name = self.__rg_table_name

        label_feature_name = self.__label_feature_name
        label_start_time = self.__label_start_time
        label_stop_time = self.__label_stop_time

        if label_start_time is None:
            label_start_time = start_time
        else:
            label_start_time = self.__clean_time(label_start_time)
        if label_stop_time is None:
            label_stop_time = end_time
        else:
            label_stop_time = self.__clean_time(label_stop_time)


        # get all features
        sql_features = 'SELECT DISTINCT {} FROM {};'.format(
                col_specs['feature'], 
                table_name)
        feat_names = [row[0] for row in conn.execute(sql_features)].remove(
                label_feature_name)

        # get per_feature subqueries
        label_subquery = self.__feature_subqueries_label(
            label_start_time,
            label_stop_time)

        feature_subqueries = [label_subquery] + 
            [self.__feature_subqueries_nonlabel(feat_name)
                for feat_name in feat_names]

        # build temporary tables
        sql_with_clause_0 = ("WITH\n"
                             "    id_tbl AS (\n"
                             "        SELECT DISTINCT\n" 
                             "            {unit_id_col} AS id\n"
                             "        FROM\n"
                             "            {table_name}\n"
                             "    ),\n\n").format(
                                   unit_id_col=col_specs['unit_id'],
                                   table_name=table_name)
        sql_with_clause_1 = ',\n\n'.join(
                [subquery['with'] for subquery in subqueries])

        # build select out of subtables
        sql_inner_select_clause_0 = (",\n\n"
                                     "    inner_select_tbl AS (\n"
                                     "        SELECT\n"
                                     "            id_tbl.id,\n")
        sql_inner_select_clause_1 = ',\n'.join(
                [subquery['select'] for subquery in subqueries])
                                 
        sql_inner_from_clause_0 = ("\n"
                                   "        FROM\n"
                                   "            id_tbl\n"
                                   "            LEFT JOIN ")
        sql_inner_from_clause_1 = "\n            LEFT JOIN ".join(
            ["{feat_tbl}\n                 ON {feat_tbl}.id = id_tbl.id".format(
                feat_tbl=subquery['table']) for subquery in subqueries])

        sql_select_clause=("\n"
                           "    )\n"
                           "\n"
                           "SELECT\n"
                           "    *\n" 
                           "FROM\n"
                           "    inner_select_tbl")
        # TODO we can probably do something more sophisticated than just 
        # throwing the user's directives in here
        # TODO something with better performance than coalesce
        sql_where_clause = "\nWHERE\n    COALESCE({}) IS NOT NULL".format(
                ', '.join([subquery['nicknames'] for subquery in subqueries]))
        if self.__selections:
            sql_where_clause += " AND\n" + " AND\n".join(
                ['    ({})'.format(sel) for sel in self.__selections]) 

        sql_select = '{}{}{}{}{}{}{}{};'.format(
            sql_with_clause_0,
            sql_with_clause_1,
            sql_inner_select_clause_0,
            sql_inner_select_clause_1,
            sql_inner_from_clause_0,
            sql_inner_from_clause_1,
            sql_select_clause,
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
        if self.__label_feature_name:
            query = self.get_query_with_labels()
        else:
            query = self.get_query()
        print query
        return self.__conn.execute(query)

    def subset_over(
            self, 
            label_col,
            interval_train_window_start,
            interval_train_window_end,
            interval_test_window_start,
            interval_test_window_end,
            interval_inc_value,
            label_col_aggr_of_interest='AVG',
            interval_expanding=False,
            label_interval_train_window_start=None,
            label_interval_train_window_end=None,
            label_interval_test_window_start=None,
            label_interval_test_window_end=None,
            label_interval_inc_value=None,
            label_interval_expanding=False,
            row_M_col_name=None,
            row_M_col_aggr_of_interest='AVG',
            row_M_train_window_start=None,
            row_M_train_window_end=None,
            row_M_test_window_start=None,
            row_M_test_window_end=None,
            row_M_inc_value=None,
            row_M_expanding=False,
            clfs=[{'clf': RandomForestClassifier}],
            feature_gen_lambda=None):
        """
        Generates ArrayGenerators according to some subsetting directive.

        There are three ways that we determine what the train and test sets are
        for each trial:

        1. The start time/stop time interval. This is the interval used to
           create features in the M-formatted matrix. Setting the start 
           time/stop time of this interval is equalivalent to passing values 
           to set_interval.  variables pertaining to this interval have the 
           interval* prefix.

        2. The start time/stop time interval for labels. If these values
           are set, then time intervals for the label are different
           than the time intervals for the other features. Variables pertaining
           to this interval have the label_interval* prefix.

        3. The rows of the M matrix to select, based on the value of some
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
        label_col_aggr_of_interest : str
            The type of aggregation which will signify the label
            (for example, use 'AVG' if the label is the 'AVG' of the label
            column in the M-formatted matrix)
        interval_expanding : boolean
            whether or not the training interval is expanding
        label_interval_train_window_start : number or datetime or None
            start of training interval for labels
        label_interval_train_window_size : number or datetime or None
            (Initial) size of training interval for labels
        label_interval_test_window_start : number or datetime or None
            start of testing interval for labels
        label_interval_test_window_size : number or datetime or None
            size of testing interval for labels
        label_interval_inc_value : datetime, timedelta, or number or None
            interval to increment train and test interval for labels
        label_interval_expanding : boolean
            whether or not the training interval for labels is expanding
        row_M_col_name : str or None
            If not None, the name of the feature which will be used to select
            different training and testing sets in addition to the interval

            If None, train and testing sets will use all rows given a 
            particular time interval
        row_M_col_aggr_of_interest : str
            The name of the aggregation used to subset rows of M.
            (For example, use 'AVG' if we want to select rows based on the
            average of the values in the interval)
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
        if row_M_train_window_end is None:
            row_M_train_window_end = interval_train_window_end
        if row_M_test_window_start is None:
            row_M_test_window_start = interval_test_window_start
        if row_M_test_window_end is None:
            row_M_test_window_end = interval_test_window_end
        if row_M_inc_value is None:
            row_M_inc_value = interval_inc_value

        conn = self.__conn
        col_specs = self.__col_specs
        table_name = self.__rg_table_name

        sql_get_max_interval_end = (
            "SELECT MAX(CASE "
            "   WHEN {stop_time_col} > {start_time_col} THEN {stop_time_col} "
            "   ELSE {start_time_col} "
            "END) as max_time FROM {table_name}").format(
               stop_time_col=col_specs['stop_time'],
               start_time_col=col_specs['start_time'],
               table_name=table_name)
        interval_end = conn.execute(
                sql_get_max_interval_end)[0][0]
        # Makesure datetime64s are the same resolution
        if (isinstance(interval_end, np.datetime64) and 
                isinstance(interval_test_window_start, np.datetime64)):
            interval_end = interval_end.astype(interval_test_window_start.dtype)

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

        if label_interval_train_window_start is None:
            label_interval_train_window_start = interval_train_window_start
        if label_interval_train_window_end is None:
            label_interval_train_window_end = interval_train_window_end
        if label_interval_test_window_start is None:
            label_interval_test_window_start = interval_test_window_start
        if label_interval_test_window_end is None:
            label_interval_test_window_end = interval_test_window_end
        if label_interval_inc_value is None:
            label_interval_inc_value = interval_inc_value

        current_interval_train_start = interval_train_window_start
        current_interval_train_end = interval_train_window_end
        current_interval_test_start = interval_test_window_start
        current_interval_test_end = interval_test_window_end
        current_label_interval_train_start = label_interval_train_window_start
        current_label_interval_train_end = label_interval_train_window_end
        current_label_interval_test_start = label_interval_test_window_start
        current_label_interval_test_end = label_interval_test_window_end
        current_row_M_train_start = row_M_train_window_start
        current_row_M_train_end = row_M_train_window_end
        current_row_M_test_start = row_M_test_window_start
        current_row_M_test_end = row_M_test_window_end
        ae = self.set_label_feature(label_col)
        while (current_interval_test_start < interval_end and
               current_row_M_test_start < row_M_end and
               current_label_interval_test_start < interval_end):
            ae_train = ae.set_interval(current_interval_train_start,
                                        current_interval_train_end)
            ae_train = ae_train.set_label_interval(
                    current_label_interval_train_start,
                    current_label_interval_train_end)
            ae_test = ae.set_interval(current_interval_test_start,
                                        current_interval_test_end)
            ae_test = ae_test.set_label_interval(
                    current_label_interval_test_start,
                    current_label_interval_test_end)
            if row_M_col_name is not None:
                ae_train = ae_train.select_rows_in_M(
                        '{col}_{aggr} >= {start} AND {col}_{aggr} <= {stop}'.format(
                            col=row_M_col_name,
                            start=current_row_M_train_start,
                            stop=current_row_M_train_end,
                            aggr=row_M_col_aggr_of_interest))
                ae_test = ae_test.select_rows_in_M(
                        '{col}_{aggr} >= {start} AND {col}_{aggr} <= {stop}'.format(
                            col=row_M_col_name,
                            start=current_row_M_test_start,
                            stop=current_row_M_test_end,
                            aggr=row_M_col_aggr_of_interest))
            # TODO this should actually run clfs and build an experiment 
            # rather than doing this yield
            data_train = ae_train.emit_M()

            # TODO remove label_col_AGGR
            label_plus_aggr = '{}_{}'.format(label_col, label_col_aggr_of_interest)
            M_train = utils.remove_cols(data_train, label_plus_aggr)
            y_train = data_train[label_plus_aggr]
            data_test = ae_test.emit_M()
            M_test = utils.remove_cols(data_test, label_plus_aggr)
            y_test = data_test[label_plus_aggr]

            #import pdb; pdb.set_trace()

            # if column ended up with an Object type, there is no date.
            # remove it
            # TODO something more elegant
            empty_cols_train = [name for name, descr in M_train.dtype.descr
                                if 'O' in descr]
            empty_cols_test = [name for name, descr in M_test.dtype.descr
                                if 'O' in descr]
            empty_cols = list(
                    frozenset(empty_cols_train).union(
                    frozenset(empty_cols_test)))
            M_train = utils.remove_cols(M_train, empty_cols)
            M_test = utils.remove_cols(M_test, empty_cols)

            if feature_gen_lambda is not None:
                M_train, y_train = feature_gen_lambda(
                        M_train, 
                        y_train,
                        'train', 
                        current_interval_train_start,
                        current_interval_train_end,
                        current_label_interval_train_start,
                        current_label_interval_train_end,
                        current_row_M_train_start,
                        current_row_M_train_end)
                M_test, y_test = feature_gen_lambda(
                        M_test,
                        y_test,
                        'test',
                        current_interval_test_start,
                        current_interval_test_end,
                        current_label_interval_test_start,
                        current_label_interval_test_end,
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
                     'test_interval_end': current_interval_test_end,
                     'train_label_interval_start': current_label_interval_train_start,
                     'train_label_interval_end': current_label_interval_train_end,
                     'test_label_interval_start': current_label_interval_test_start,
                     'test_label_interval_end': current_label_interval_test_end},
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
            if not label_interval_expanding:
                current_label_interval_train_start += label_interval_inc_value
            current_label_interval_train_end += label_interval_inc_value
            current_label_interval_test_start += label_interval_inc_value
            current_label_interval_test_end += label_interval_inc_value
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
    """Convert a table in M-format to a table in RG-format

    Data in the M-formatted from_table is appended to the RG-formatted
    to_table. Consequently, the user can run M_to_rg multiple times to
    convert multiple M-formatted tables to the same RG-formatted table

    For character or text columns in from_table, entries will be encoded
    to integers when they are dumped into the RG-formatted table. For each
    of these columns, M_to_rg will create a table with a name of the form:
    [to_table]_encode_[from_table]_[col_name]

    Parameters
    ----------
    conn_str : str
        SQLAlchemy connection string to access database
    from_table : str
        Name of M-formatted table to convert to RG-formatted table
    to_table : str
        Name of RG-formatted table that data will be inserted into. This 
        table must either:
        1. Not yet exist in the database or
        2. Adhere to the schema:
            (row_id SERIAL PRIMARY KEY, unit_id INT, start_time TIMESTAMP, 
             end_time TIMESTAMP, feat TEXT, val REAL);
    unit_id_col : str
        The column in from_table which contains the unique unit ID
    start_time_col : str or None
        The column in from_table which contains the start time. If None,
        start times will be left NULL in the RG-formatted array
    stop_time_col : str or None
        The column in from_table which contains the stop time. If None,
        stop times will be left NULL in the RG-formatted array
    feature_cols : list of str or None
        The columns of from_table which will be inserted into to_table in
        RG-format. If None, will use all columns in from_table except for
        unit_id_col, start_time_col and stop_time_col
    """

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


