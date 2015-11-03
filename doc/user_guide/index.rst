==========
User Guide
==========

Diogenes is a set of tools designed to make common machine learning tasks
easier. Diogenes is divided into several parts:

* :doc:`read` provides tools for reading data from external 
  sources into Diogenes' preferred Numpy 
  `structured array <http://docs.scipy.org/doc/numpy/user/basics.rec.html>`_
  format.
* :doc:`display` provides tools for summarizing data and the 
  performance of trained classifiers.
* :doc:`modify` provides tools for manipulating arrays and
  generating features.
* :doc:`grid_search` provides tools for finding the best classifier,
  testing classifier sensitivity to data sets, and cross-validating 
  classifier performance.
* :doc:`array_emitter` provides tools for processing and iterating
  over "RG-formatted" (transposed) arrays.
* :doc:`utils` provides miscilaneous utilities--mostly for processing
  structured arrays.

Contents:

.. toctree::
   :maxdepth: 2

   read.rst
   display.rst
   modify.rst
   grid_search.rst
   array_emitter.rst
   utils.rst
