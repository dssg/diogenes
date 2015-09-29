****************************************
Using Eights for Feature Generation
****************************************
Quick test::

   import numpy as np
   
   import diogenes.read
   import diogenes.generate
   
   M = [[1,2,3], [2,3,4], [3,4,5]]
   col_names = ['heigh','weight', 'age']
   lables= [0,0,1]
   
   # Eights uses Structured arrays, which allow for different data types in different columns
   M = diogenes.read.convert_list_of_list_to_sa(np.array(M), c_name=col_names)
   #By convention M is the our matrix on which our ML algo will run
   
   #This is a sample lambada statment, to show how easy it is to craft your own.  
   #the signitutre(M, col_name, boundary) is standardized.  
   def test_equality(M, col_name, boundary):
       return M[col_name] == boundary

   #This generates a new frow where the values are all true
   M_new = diogenes.generate.choose_rows_where(
                  M,
                  [test_equality, test_equality, test_equality], 
                  ['height','weight', 'age'], 
                  [1,2,3], 
                  ('new_column_name',)
                  )
   # Read top to bottom:
   # If test_equality in column 'height' == 1 AND
   # If test_equality in column 'weight' == 2 AND
   # If test_equality in column 'age' == 3 
   # return true


import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sklearn.datasets
import diogenes as e


diab = sklearn.datasets.load_diabetes()

data = diab.data
target = diab.target

M= e.rd.convert_list_of_list_to_sa(data)

#quick sanity check
#e.com.plot_simple_histogram(target)   
# i want to bin all values above 210
y = (target >=205)

# twist this into a classification problem

e.rd.simple_CV(data,target, RandomForestClassifier) 























