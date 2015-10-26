# Diogenes User Guide

Diogenes is a set of tools designed to make common machine learning tasks
easier. Diogenes is divided into several parts:

* :mod:`diogenes.read` provides tools for reading data from external sources
  into Diogenes' preferred Numpy 
  `structured array <http://docs.scipy.org/doc/numpy/user/basics.rec.html>`_
  format.
* :mod:`diogenes.display` provides tools for summarizing data and the 
  performance of trained classifiers.
* :mod:`diogenes.modify` provides tools for manipulating arrays and
  generating features.
* :mod:`diogenes.grid_search` provides tools for finding the best classifier,
  testing classifier sensitivity to data sets, and cross-validating 
  classifier performance.
* :mod:`diogenes.array_emitter` provides tools for processing and iterating
  over "RG-formatted" (transposed) arrays.
* :mod:`diogenes.utils` provides miscilaneous utilities--mostly for processing
  structured arrays.

