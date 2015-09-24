omg there are so many

http://patorjk.com/software/taag/#p=display&v=3&f=3D-ASCII&t=diogenes                                              
                                                                   
                                                                   
       __.....__     .--.            .                             
   .-''         '.   |__|  .--./)  .'|                             
  /     .-''"'-.  `. .--. /.''\\  <  |             .|              
 /     /________\   \|  || |  | |  | |           .' |_             
 |                  ||  | \`-' /   | | .'''-.  .'     |       _    
 \    .-------------'|  | /("'`    | |/.'''. \'--.  .-'     .' |   
  \    '-.____...---.|  | \ '---.  |  /    | |   |  |      .   | / 
   `.             .' |__|  /'""'.\ | |     | |   |  |    .'.'| |// 
     `''-...... -'        ||     ||| |     | |   |  '.'.'.'.-'  /  
                          \'. __// | '.    | '.  |   / .'   \_.'   
                           `'---'  '---'   '---' `'-''              
                           
      _       _     _       
     (_)     | |   | |      
  ___ _  __ _| |__ | |_ ___ 
 / _ \ |/ _` | '_ \| __/ __|
|  __/ | (_| | | | | |_\__ \
 \___|_|\__, |_| |_|\__|___/
         __/ |              
        |___/               
        
 _______ _________ _______          _________ _______ 
(  ____ \\__   __/(  ____ \|\     /|\__   __/(  ____ \
| (    \/   ) (   | (    \/| )   ( |   ) (   | (    \/
| (__       | |   | |      | (___) |   | |   | (_____ 
|  __)      | |   | | ____ |  ___  |   | |   (_____  )
| (         | |   | | \_  )| (   ) |   | |         ) |
| (____/\___) (___| (___) || )   ( |   | |   /\____) |
(_______/\_______/(_______)|/     \|   )_(   \_______)
                                                      
                                                      
                                                      
By default we use pass structured arrays from within functions. However to accommodate both list of list and list of nd.arrays, we cast structured arrays as: 
name, M = sa_to_nd(M)

For our the rapid analysis of different CLF's in SKLEARN we use dictionaries of dictionaries of Lists. Where lists are the slice indices, the outermost dictionary is the test, the inner dictionary is the run.  For instance, Test["sweepParamtersize"]['one'] == nd.array([1,2,3])

M is a structured array
Y is the target in SKLEARN parlance. It is the known labels.



supported plots:      (ROC, PER_RECALL, ACC, N_TOP_FEAT, AUC)
supported clfs:       (RF, SVM, DESC_TREE, ADA_BOOST)
supported subsetting: (LEAVE_ONE_COL_OUT, SWEEP_TRAINING_SIZE, )
supported cv:         (K_FOLD, STRAT_ACTUAL_K_FOLD, STRAT_EVEN_K_FOLD)         


plots = ['roc', 'acc']
clfs =  [('random forest', ['RF PARMS'] ),
         ('svm',           ['SVM PARMS'] )]
         
subsets = [('leave one out col',   ['PARMS'] ), 
           ('sweep training size', ['PARMS'] )]

cv =     [('cv', ['parms']),
          ('stratified cv', ['parms'])]
          
runOne = Experiment(plots, clfs, subsets, cv)

exp = Experiment(
      [RF: {'depth': [10, 100],
            'n_trees': [40, 50]},
       SVM: {'param_1': [1, 2],
             'param_2': ['a', 'b']}],
      [LEAVE_ONE_COL_OUT: {'col_names': ['f0', 'f1', 'f2', 'f3']},
       SWEEP_TRAINING_SIZE: {'sizes': (10, 20, 40)}
      ],
      [STRAT_ACTUAL_K_FOLD : {'y': y}])


M is our matrix to train our ML algo on, its always a structured array (or an nd.array, or a list of one dimension arrays).
Labels are the supervised learnings gold standard labels.  It is the TRUTH. Aways a one dim numpy.array
If we use a collection of columns, that are not the full matrix it is cols.
col is the one dimeion equiv of cols, it has the same type as labels.
When we pass in a classifier, its an sklearn base estimator class as opposed to instance
When we are passing a cross validation type, its a sklearn partition iterator.
When we pass a subset method it is an iterator for which each iteration returns a set of indices. 
Parameters for sklearn etc is always pass dictionaries of string to something.  
There will be other stuff, but the above carved in rice paper.  






 