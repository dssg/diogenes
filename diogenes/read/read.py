import csv
import urllib2
import os
import cPickle
import sqlalchemy as sqla
from datetime import datetime

import itertools as it
import numpy as np

from diogenes.utils import open_csv_as_sa, cast_list_of_list_to_sa

"""

This module provides functions that convert databases in external formats
to Numpy structured arrays.

"""

def open_csv(path, delimiter=',', header=True, col_names=None):
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

    Returns
    -------
    numpy.ndarray
        structured array corresponding to the csv

    If header is False and col_names is None, diogenes will assign
    arbitrary column names
    """

    with open(path, 'rU') as fin:
        return open_csv_as_sa(fin, delimiter, header, col_names)
    
def open_csv_url(url, delimiter=',', header=True, col_names=None):
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

    Returns
    -------
    numpy.ndarray
        structured array corresponding to the csv

    If header is False and col_names is None, diogenes will assign
    arbitrary column names
    """
    fin = urllib2.urlopen(url)
    sa = open_csv_as_sa(fin, delimiter, header, col_names)
    fin.close()
    return sa

def connect_sql(con_str, allow_caching=False, cache_dir='.'):
    """Provides an SQLConnection object, which makes structured arrays from SQL

    Parameters
    ----------
    conn_str : str
        SQLAlchemy connection string (http://docs.sqlalchemy.org/en/rel_0_9/core/engines.html)
    allow_caching : bool
        If True, diogenes will cache the results of each query and return
        the cached result if the same query is performed twice. If False,
        each query will be sent to the database
    cache_dir : str
        If allow_caching is True, the cached results will be stored in 
        cache_dir

    Returns
    -------
    SQLConnection
        Object that executes SQL queries and returns numpy structured
        arrays

    """
    return SQLConnection(con_str, allow_caching, cache_dir)

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
    cache_dir : str
        If allow_caching is True, the cached results will be stored in 
        cache_dir
    """
    def __init__(self, con_str, allow_caching=False, cache_dir='.'):
        self.__engine = sqla.create_engine(con_str)
        self.__cache_dir = cache_dir
        if allow_caching:
            self.execute = self.__execute_with_cache

    def __sql_to_sa(self, exec_str):
        raw_python = self.__engine.execute(exec_str)
        return cast_list_of_list_to_sa(
            raw_python.fetchall(),
            [str(key) for key in raw_python.keys()])

    def __execute_with_cache(self, exec_str, invalidate_cache=False):
        pkl_file_name = os.path.join(
            self.__cache_dir, 
            'diogenes_cache_{}.pkl'.format(hash(exec_str)))
        if os.path.exists(pkl_file_name) and not invalidate_cache:
            with open(pkl_file_name) as fin:
                return cPickle.load(fin)
        ret = self.__sql_to_sa(exec_str)
        with open(pkl_file_name, 'w') as fout:
            cPickle.dump(ret, fout)
        return ret

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
        return self.__sql_to_sa(exec_str)



