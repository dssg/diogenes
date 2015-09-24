import csv
import urllib2
import os
import cPickle
import sqlalchemy as sqla
from datetime import datetime

import itertools as it
import numpy as np

from eights.utils import open_csv_as_sa

def open_csv(path, delimiter=',', header=True, col_names=None):
    with open(path, 'rU') as fin:
        return open_csv_as_sa(fin, delimiter, header, col_names)
    
def open_csv_url(url, delimiter=',', header=True, col_names=None):
    with urllib2.urlopen(url):
        return open_csv_as_sa(fin, delimiter, header, col_names)

def connect_sql(con_str, allow_caching=False, cache_dir='.'):
    return SQLConnection(con_str, allow_caching, cache_dir)

class SQLConnection(object):
    # Intended to vaguely implement DBAPI 2
    # If allow_caching is True, will pickle results in cache_dir and reuse
    # them if it encounters an identical query twice.
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
        return self.__sql_to_sa(exec_str)



