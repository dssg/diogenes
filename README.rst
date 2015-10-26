::

    ================================================
          ____  _                                 
         / __ \(_)___  ____ ____  ____  ___  _____
        / / / / / __ \/ __ `/ _ \/ __ \/ _ \/ ___/
       / /_/ / / /_/ / /_/ /  __/ / / /  __(__  ) 
      /_____/_/\____/\__, /\___/_/ /_/\___/____/  
                    /____/                        

    ================================================


------------
Introduction
------------

Diogenes is a a Python library and workflow templet for machine learning.
Principally it wraps sklearn providing enhanced functionality and simplified 
interface of often used workflows. 

------------
Installation
------------

`pip install git+git://github.com/dssg/diogenes.git`

Required Packages
=================

Python packages
---------------
- `Python 2.7 <https://www.python.org/>`_
- `Numpy 1.10.1 <http://www.numpy.org/>`_
- `scikit-learn <http://scikit-learn.org/stable/>`_
- `pandas <http://pandas.pydata.org/>`_
- `SQLAlchemy <http://www.sqlalchemy.org/>`_
- `pdfkit <https://github.com/pdfkit/pdfkit>`_
- `plotlib <http://matplotlib.org/>`_

Other packages
--------------

- `wkhtmltopdf <http://wkhtmltopdf.org/>`_

-------
Example
-------
::

    import diogenes
    import numpy as np
    # Get data from Wine Quality data set
    data = diogenes.read.open_csv_url(
        'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv',
        delimiter=';')
    # Take labels from the quality column
    labels = data['quality']
    # Make this a binary classification problem
    labels = labels < np.average(labels)
    # Remove labels from data to make feature set
    M = diogenes.modify.remove_cols(data, 'quality')
    # Print statistics of features
    diogenes.display.pprint_sa(diogenes.display.describe_cols(M))
    # Plot correlation between features
    diogenes.display.plot_correlation_matrix(M)
    # Set up grid search experiment using different classifiers
    exp = diogenes.grid_search.experiment.Experiment(
        M, 
        labels, 
        clfs=diogenes.grid_search.standard_clfs.std_clfs)
    # Make a report for the experiment to find best-performing classifiers
    exp.make_report()


----------
Next Steps
----------

Check out the `documentation <http://dssg.github.io/diogenes>`_

----
Misc
----
my_* are included in the .gitignore.  We recommend a standard such as my_experiment, my_storage for local folders.


