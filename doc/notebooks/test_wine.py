import numpy as np
import sklearn.datasets

from diogenes.read import open_csv_url
from diogenes.display import (plot_correlation_scatter_plot,
                               plot_correlation_matrix, 
                               plot_kernel_density,
                               plot_box_plot)

from diogenes.grid_search import Experiment 
from diogenes.grid_search import DBG_std_clfs as std_clfs
from diogenes.utils import remove_cols


data = open_csv_url(
            'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv',  
            delimiter=';')
y = data['quality']
M = remove_cols(data, 'quality')

y = y < np.average(y)


if False:
    for x in describe_cols(M):
        print x

if False:
   plot_correlation_scatter_plot(M) 
   plot_correlation_matrix(M)
   plot_kernel_density(M['f0']) #no designation of col name
   plot_box_plot(M['f0']) #no designation of col name

exp = Experiment(M, y, clfs=std_clfs)
exp.make_csv()

