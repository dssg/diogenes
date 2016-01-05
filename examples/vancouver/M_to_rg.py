from diogenes.array_emitter import M_to_rg
from diogenes.read import connect_sql

from settings import conn_str

to_table = 'vancouver._Z_test_rg'

conn = connect_sql(conn_str)
sql = 'DROP TABLE ' + to_table
conn.execute(sql)

M_to_rg(
        conn_str, 
        'vancouver.test', 
        'vancouver._Z_test_rg', 
        'student_id', 
        start_time_col='sat_date', 
        feature_cols=('sat_score_math', 'sat_score_reading', 'sat_score_writing'))

M_to_rg(
        conn_str, 
        'vancouver.test', 
        'vancouver._Z_test_rg', 
        'student_id', 
        start_time_col='act_date', 
        feature_cols=('act_score_composite', 'act_score_english', 'act_score_math', 'act_score_reading', 'act_score_science'))

M_to_rg(
        conn_str, 
        'vancouver.student', 
        'vancouver._Z_test_rg', 
        'id', 
        feature_cols=('cohort', 'gender', 'race', 'race_ethnicity', 'white', 'asian', 'pacific_islander', 'black', 'hispanic', 'american_indian', 'multi_racial', 'highest_grade_level', 'low_income_ever', 'ell_ever', 'ind_504_ever', 'special_ed_ever'))

M_to_rg(
        conn_str, 
        'vancouver._z_test_labels', 
        'vancouver._Z_test_rg', 
        'student_id', 
        start_time_col='gradtime',
        feature_cols=('label',))
