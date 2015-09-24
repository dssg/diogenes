#Required Import
import numpy as np

from display.display import plot_box_plot



#Investigate
M,labels = rd.open_cvs(file_loc)

#choose to Numpy Structures Arrays

#Descriptive statistics
rd.describe_cols(data)
rd.cross_tabs


rd.plot_correlation_matrix
rd.plot_correlation_scatter_plot
rd.plot_box_plot


# Decontaminate Data
import diogenes.read as dec
replace_with_n_bins
replace_missing_vals

#generate features
def is_this_word_in(a_text, word):
    return word in a_text

M = where_all_are_true(
    M, 
    [(val_eq, 'open_col', NULL),
     (val_eq, 'click', NULL ])
     "n_click,n_open"
     )
          
M = where_all_are_true(
    M, 
    [(is_this_word_in, 'email_text', 'unsubscribe'),
     (val_eq, 'click', NULL ])
     "1_click,n_open"
     )

#Trucate
M1 = remove_rows_=
    M, 
    [(is_this_word_in, 'email_text', 'unsubscribe'),
     (val_eq, 'click', NULL ])
     "1_click,n_open"
     )

#Permabulate/operate

experiment = run_std_classifiers(M1, labesl)

#exp identicle to experiment

exp = Experiment(
    M1, 
    labels, 
    clfs = {AdaBoostClassifier: {'n_estimators': [20,50,100]}, 
           RandomForestClassifier: {'n_estimators': [10,30,50],'max_depth': [None,4,7,15],'n_jobs':[1]}, 
           LogisticRegression:{'C': [1.0,2.0,0.5,0.25],'penalty': ['l1','l2']}, 
           DecisionTreeClassifier: {'max_depth': [None,4,7,15,25]},
           SVC:{'kernel': ['linear','rbf']},
           DummyClassifier:{'strategy': ['stratified','most_frequent','uniform']}
          }
    cvs = {StratifiedKFold:{}}
    )

#display
exp.report()


M = add_these(
   [[(val_eq, 'open_col', NULL),(val_eq, 'click', NULL ]),"n_click,n_open"],
   [[(val_eq, 'open_col', NULL),(val_eq, 'click', NULL ]),"n_click,n_open"]
   )






    




     
