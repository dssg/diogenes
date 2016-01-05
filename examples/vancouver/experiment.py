import numpy as np

from diogenes.array_emitter import ArrayEmitter
from diogenes.modify import replace_missing_vals

from settings import conn_str

def proc_array(M, labels, test_or_train, interval_start, interval_end, 
               label_interval_start, label_interval_end,
               row_M_start, row_M_end):
    nonnan = np.logical_not(np.isnan(labels))
    M = M[nonnan]
    M = replace_missing_vals(M, 'constant', constant=0)
    labels = labels[nonnan]
    return (M, labels)

ae = ArrayEmitter().get_rg_from_sql(
        conn_str, 
        'vancouver._Z_test_rg',
        unit_id_col='unit_id',
        start_time_col='start_time',
        stop_time_col='end_time',
        feature_col='feat',
        val_col='val')

ae = ae.set_aggregation('label', 'MAX')

exp = ae.subset_over(
    label_col='label',
    interval_train_window_start=np.datetime64('2011-01-01'),
    interval_train_window_size=364,
    interval_test_window_start=np.datetime64('2013-01-01'),
    interval_test_window_size=364,
    interval_inc_value=365,
    interval_expanding=True,
    label_interval_train_window_start=np.datetime64('2012-01-01'),
    label_interval_train_window_size=364,
    label_interval_test_window_start=np.datetime64('2013-01-01'),
    label_interval_test_window_size=364,
    label_interval_inc_value=365,
    label_interval_expanding=False,
    feature_gen_lambda=proc_array)
exp.make_csv()
