import numpy as np
import sklearn.datasets

from diogenes.read import (cast_np_nd_to_sa, describe_cols,)
from diogenes.display import (plot_correlation_scatter_plot,
                               plot_correlation_matrix, 
                               plot_kernel_density,
                               plot_box_plot)

#import numpy array
M = sklearn.datasets.load_iris().data
labels = sklearn.datasets.load_iris().target

M = cast_np_nd_to_sa(M)


#M is multi class, we want to remove those rows.
keep_index = np.where(labels!=2)

labels = labels[keep_index]
M = M[keep_index]




if False:
    for x in describe_cols(M):
        print x

if False:
   plot_correlation_scatter_plot(M) 
   plot_correlation_matrix(M)
   plot_kernel_density(M['f0']) #no designation of col name
   plot_box_plot(M['f0']) #no designation of col name


if False:
    from diogenes.generate import val_between, choose_rows_where, append_cols  #val_btwn, where
    #generate a composite rule
    M = choose_rows_where(M, 
                          [{'func': val_between, 
                            'col_name': 'f0', 
                            'vals': (3.5, 5.0)},
                           {'func': val_between, 
                            'col_name': 'f1', 
                            'vals': (2.7, 3.1)}
                           ], 
                           'a new col_name')

    #new eval function
    def rounds_to_val(M, col_name, boundary):
        return (np.round(M[col_name]) == boundary)
    
    M = choose_rows_where(M,
                          [{'func': rounds_to_val, 
                            'col_name': 'f0', 
                            'vals': 5}],
                            'new_col')
    
    from  diogenes.modify import (fewer_then_n_nonzero_in_col, 
                                 remove_rows_where,
                                 remove_cols,
                                 val_eq)
    #remove Useless row
    M = fewer_then_n_nonzero_in_col(M,1)
    M = append_cols(M, labels, 'labels')
    M = remove_rows_where(M, val_eq, 'labels', 2)
    labels=M['labels']
    M = remove_cols(M, 'labels')


from diogenes.operate import run_std_classifiers, run_alt_classifiers #run_alt_classifiers not working yet
exp = run_std_classifiers(M,labels)
exp.make_csv()
import pdb; pdb.set_trace()


####################Communicate#######################



#Pretend .1 is wrong so set all values of .1 in M[3] as .2
# make a new column where its a test if col,val, (3,.2), (2,1.4) is true.


import pdb; pdb.set_trace()

#from decontaminate import remove_null, remove_999, case_fix, 
#from generate import donut
#from aggregate import append_on_right, append_on_bottom

#from operate import run_list, fiveFunctions
#from display import graph_all, results_invtestiage

#investiage
#M_orginal = csv_open(file_loc, file_descpiption)  # this is our original files
#results = diogenes.read.describe_all(M_orginal)
#results_invtestiage(results)

#decontaminate
#aggregate
#generate
#M = np.array([]) #this is the master Matrix we train on.
#labels = np.array([]) # this is tells us

#models = [] #list of functions

#operate

#display


#func_list = [sklearn.randomforest,sklearn.gaussian, ]


#If main:
#run on single csv
