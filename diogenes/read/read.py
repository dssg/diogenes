import csv
import urllib2
import os
import cPickle
import subprocess
import sqlalchemy as sqla
from datetime import datetime

import itertools as it
import numpy as np
import pandas as pd

from diogenes.utils import open_csv_as_sa, cast_list_of_list_to_sa

"""

This module provides functions that convert databases in external formats
to Numpy structured arrays.

"""

def open_csv(path, delimiter=',', header=True, col_names=None, parse_datetimes=[]):
    """Creates a structured array from a local .csv file

    Parameters
    ----------
    path : str
        path of the csv file
    delimiter : str
        Character used to delimit csv fields
    header : bool
        If True, assumes the first line of the csv has column names
    col_names : list of str or None
        If header is False, this list will be used for column names
    parse_datetimes : list of col names
        Columns that should be interpreted as datetimes

    Returns
    -------
    numpy.ndarray
        structured array corresponding to the csv

    If header is False and col_names is None, diogenes will assign
    arbitrary column names
    """

    with open(path, 'rU') as fin:
        return open_csv_as_sa(fin, delimiter, header, col_names, parse_datetimes)
    
def open_csv_url(url, delimiter=',', header=True, col_names=None, parse_datetimes=[]):
    """Creates a structured array from a url

    Parameters
    ----------
    url : str
        url of the csv file
    delimiter : str
        Character used to delimit csv fields
    header : bool
        If True, assumes the first line of the csv has column names
    col_names : list of str or None
        If header is False, this list will be used for column names
    parse_datetimes : list of col names
        Columns that should be interpreted as datetimes


    Returns
    -------
    numpy.ndarray
        structured array corresponding to the csv

    If header is False and col_names is None, diogenes will assign
    arbitrary column names
    """
    fin = urllib2.urlopen(url)
    sa = open_csv_as_sa(fin, delimiter, header, col_names, parse_datetimes)
    fin.close()
    return sa

def connect_sql(con_str, allow_caching=False, tmp_dir='.', 
                parse_datetimes=[],
                allow_pgres_copy_optimization=True):
    """Provides an SQLConnection object, which makes structured arrays from SQL

    Parameters
    ----------
    conn_str : str
        SQLAlchemy connection string (http://docs.sqlalchemy.org/en/rel_0_9/core/engines.html)
    allow_caching : bool
        If True, diogenes will cache the results of each query and return
        the cached result if the same query is performed twice. If False,
        each query will be sent to the database
    tmp_dir : str
        If allow_caching is True, the cached results will be stored in 
        tmp_dir. Also where csvs will be stored for postgres servers
    parse_datetimes : list of col names
        Columns that should be interpreted as datetimes

    Returns
    -------
    SQLConnection
        Object that executes SQL queries and returns numpy structured
        arrays

    """
    return SQLConnection(con_str, allow_caching, tmp_dir,
                         parse_datetimes,
                         allow_pgres_copy_optimization)

class SQLError(Exception):
    pass

class SQLConnection(object):
    """Connection to SQL that returns numpy structured arrays
    Intended to vaguely implement DBAPI 2

    Parameters
    ----------
    conn_str : str
        SQLAlchemy connection string (http://docs.sqlalchemy.org/en/rel_0_9/core/engines.html)
    allow_caching : bool
        If True, diogenes will cache the results of each query and return
        the cached result if the same query is performed twice. If False,
        each query will be sent to the database
    tmp_dir : str
        If allow_caching is True, the cached results will be stored in 
        tmp_dir. Also, where csvs will be stored for postgres servers
    parse_datetimes : list of col names
        Columns that should be interpreted as datetimes
    """
    def __init__(self, conn_str, allow_caching=False, tmp_dir='.', 
                 parse_datetimes=[],
                 allow_pgres_copy_optimization=True):
        self.__parse_datetimes = parse_datetimes
        self.psql_optimized = False
        parsed_conn_str = sqla.engine.url.make_url(conn_str)
        exec_fun = self.__execute_sqla
        if (allow_pgres_copy_optimization and 
            parsed_conn_str.drivername == 'postgresql'):
            # try for psql \COPY optimization
            if not subprocess.call(['which', 'psql']):
                # we have psql
                psql_call = ['psql']
                if parsed_conn_str.host:
                    psql_call.append('-h')
                    psql_call.append(parsed_conn_str.host)
                if parsed_conn_str.port:
                    psql_call.append('-p')
                    psql_call.append(str(parsed_conn_str.port))
                if parsed_conn_str.database:
                    psql_call.append('-d')
                    psql_call.append(parsed_conn_str.database)
                if parsed_conn_str.username:
                    psql_call.append('-U')
                    psql_call.append(parsed_conn_str.username)
                if parsed_conn_str.password:
                    os.environ['PGPASSWORD'] = '{}'.format(parsed_conn_str.password)
                psql_call.append('-c')
                self.__psql_call = psql_call
                exec_fun = self.__execute_copy_command
                self.psql_optimized=True
        self.__engine = sqla.create_engine(conn_str)
        self.__tmp_dir = tmp_dir
        if allow_caching:
            self.execute = self.__execute_with_cache(exec_fun)
        else:
            self.execute = self.__execute_no_cache(exec_fun)

    def __execute_copy_command(self, exec_str, repl=None):
        exec_stripped = exec_str.strip()
        if exec_stripped[:6].upper() != 'SELECT':
            # Only do CSV optimization for select statements
            return self.__execute_sqla(exec_str, repl)
        n_semicolons = exec_stripped.count(';')
        if n_semicolons > 1:
            # If it's a multi-statement query, don't optimize
            return self.__execute_sqla(exec_str, repl)
        elif n_semicolons == 1:
            if exec_stripped[-1] != ';':
                # If the semicolon isn't at the end (i.e. we have a 
                # multistatement query) don't optimize
                return self.__execute_sqla(exec_str, repl)
            exec_stripped = exec_stripped[:-1]
        csv_file_name = os.path.join(
                self.__tmp_dir,
                'diogenes_pgres_query_{}.csv'.format(hash(exec_str)))
        command = "\"\\copy ({}) TO '{}' DELIMITER ',' NULL '' CSV HEADER\"".format(
            exec_stripped, 
            csv_file_name)
        psql_call = self.__psql_call + [command]
        #if subprocess.call(psql_call):
        if subprocess.call(' '.join(psql_call), shell=True):
            raise SQLError('Query failed.')
        sa = open_csv(csv_file_name, parse_datetimes=self.__parse_datetimes)
        os.remove(csv_file_name)
        return sa

    def __insert_sanitize(self, item):
        # http://stackoverflow.com/questions/4187185/how-can-i-check-if-my-python-object-is-a-number
        # TODO actually sanitize
        if item is None:
            return 'NULL'
        if isinstance(item, (int, long, float, complex)):
            return item
        return "'{}'".format(item)


    def __execute_sqla(self, exec_str, repl=None):
        if repl is not None:
            exec_str = exec_str.replace('?', '{}').format(
                    *[self.__insert_sanitize(item) for item in repl])
        try:
            df =  pd.read_sql(exec_str, self.__engine, 
                              parse_dates=self.__parse_datetimes)
            if self.__parse_datetimes:
                utils.fix_pandas_datetimes(df, self.__parse_datetimes)
            return df.to_records(index=False)
        except sqla.exc.ResourceClosedError:
            # Query didn't return results
            return None

    def __execute_with_cache(self, exec_fun):
        def fun_with_cache(exec_str, repl=None, invalidate_cache=False):
            pkl_file_name = os.path.join(
                self.__tmp_dir, 
                'diogenes_cache_{}.pkl'.format(hash(exec_str)))
            if os.path.exists(pkl_file_name) and not invalidate_cache:
                with open(pkl_file_name) as fin:
                    return cPickle.load(fin)
            ret = exec_fun(exec_str, repl)
            with open(pkl_file_name, 'w') as fout:
                cPickle.dump(ret, fout)
            return ret
        return fun_with_cache

    def __execute_no_cache(self, exec_fun):
        return lambda exec_str, repl=None, invalidate_cache=False: exec_fun(
                exec_str,
                repl)

    def execute(self, exec_str, invalidate_cache=False):
        """Executes a query

        Parameters
        ----------
        exec_str : str
            SQL query to execute
        invalidate_cache : bool
            If this SQLConnection object was initialized with 
            allow_caching=True, identical queries will always return
            the same result. If invalidate_cache is True, this behavior
            is overriden and the query will be reexecuted.

        Returns
        -------
        numpy.ndarray
            Results of the query in terms of a numpy structured array
        """
        return None



