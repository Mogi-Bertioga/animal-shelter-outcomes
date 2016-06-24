'''
USP - ICMC
Kaggle: Shelter animal outcomes
Authors: Thomio Watanabe
 - separate dataset
 - train machine learning methods
 - evaluate result

http://scikit-learn.org/stable/modules/cross_validation.html
http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html

'''

import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.externals import joblib
from sklearn.metrics import log_loss


input_table = pd.read_csv('../output/filtered_dataset.csv')


# One model for each output class
# Output classes
adoption = input_table['Adoption']
died = input_table['Died']
euthanasia = input_table['Euthanasia']
return_to_owner = input_table['Return_to_owner']
transfer = input_table['Transfer']

target_table = pd.concat( [adoption, died, euthanasia, return_to_owner, transfer], axis = 1 )

input_table.drop('Adoption', axis = 1, inplace = True)
input_table.drop('Died', axis = 1, inplace = True)
input_table.drop('Euthanasia', axis = 1, inplace = True)
input_table.drop('Return_to_owner', axis = 1, inplace = True)
input_table.drop('Transfer', axis = 1, inplace = True)


def train_svm( input_table, target_table, target ):
    # Separate train and test samples
    # Use cross validation
    number_samples = len(input_table) - 1
    kfold = KFold( number_samples , n_folds = 10 )
    for train_index, test_index in kfold:
#        print 'train: ', train_index, 'test: ',test_index
        train_X, train_Y = input_table.iloc[train_index], target_table[ target ].iloc[train_index]
        test_X, test_Y   = input_table.iloc[test_index], target_table[ target ].iloc[test_index]
        clf = LinearSVC()
        clf.fit( train_X, train_Y )
#        clf = clf.fit(train_X, train_Y)
        scores = cross_val_score(clf, test_X, test_Y)
    print target, ' log loss: ', log_loss( target_table[target], clf.predict(input_table) )
    print target, ' model score: ', scores.mean()
    # Save svm forest model to file
    output_file = './svm/' + target + '.pkl'
    joblib.dump( clf, output_file )
    return scores.mean()


#  LinearSVM
#    Adoption  log loss:  7.99613114602
#    Adoption  model score:  0.742505164508
#    Return_to_owner  log loss:  5.90916545787
#    Return_to_owner  model score:  0.809131683039
#    Transfer  log loss:  7.09802397171
#    Transfer  model score:  0.780310386428
#    Died  log loss:  5.84419089639
#    Died  model score:  0.697577276525
#    Euthanasia  log loss:  7.8480969867
#    Euthanasia  model score:  0.720967741935



train_svm( input_table, target_table, 'Adoption' )
#train_svm( input_table, target_table, 'Died' )
#train_svm( input_table, target_table, 'Euthanasia' )
train_svm( input_table, target_table, 'Return_to_owner' )
train_svm( input_table, target_table, 'Transfer' )


# Reduce training set to died and euthanasia attributes
died_index = np.where( died == 1 ) # 197 samples
input_aux = input_table
input_samples = input_aux.iloc[died_index[0]]
input_aux = input_aux.drop( input_aux.index[ died_index[0] ] )
aux_array = input_aux.iloc[ range( 2 * len(died_index[0]) ) ]
input_samples = pd.concat( [input_samples, aux_array] )

target_aux = target_table
output_samples = target_aux.iloc[died_index[0]]
target_aux = target_aux.drop( target_aux.index[ died_index[0] ] )
aux_array = target_aux.iloc[ range( 2 * len(died_index[0]) ) ]
output_samples = pd.concat( [output_samples, aux_array] )

# shuffle rows
arr = np.arange( len(input_samples) )
np.random.shuffle(arr)

train_svm( input_samples.iloc[arr], output_samples.iloc[arr], 'Died' )


euthanasia_index = np.where( euthanasia == 1 ) #  samples
input_aux = input_table
input_samples = input_aux.iloc[euthanasia_index[0]]
input_aux = input_aux.drop( input_aux.index[ euthanasia_index[0] ] )
aux_array = input_aux.iloc[ range( 2 * len(euthanasia_index[0]) ) ]
input_samples = pd.concat( [input_samples, aux_array] )

target_aux = target_table
output_samples = target_aux.iloc[euthanasia_index[0]]
target_aux = target_aux.drop( target_aux.index[ euthanasia_index[0] ] )
aux_array = target_aux.iloc[ range( 2 * len(euthanasia_index[0]) ) ]
output_samples = pd.concat( [output_samples, aux_array] )

# shuffle rows
arr = np.arange( len(input_samples) )
np.random.shuffle(arr)

train_svm( input_samples.iloc[arr], output_samples.iloc[arr], 'Euthanasia' )



