'''
USP - ICMC
Kaggle: Shelter animal outcomes
Authors: Thomio Watanabe
 - separate dataset
 - train machine learning methods
 - evaluate result

http://scikit-learn.org/stable/modules/cross_validation.html
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

'''

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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


def train_randomforest( input_table, target_table, target, n_trees ):
    # Separate train and test samples
    # Use cross validation
    number_samples = len(input_table) - 1
    kfold = KFold( number_samples , n_folds = 10 )
    for train_index, test_index in kfold:
#        print 'train: ', train_index, 'test: ',test_index
        train_X, train_Y = input_table.iloc[train_index], target_table[ target ].iloc[train_index]
        test_X, test_Y   = input_table.iloc[test_index], target_table[ target ].iloc[test_index]
        clf = RandomForestClassifier(n_estimators = n_trees, n_jobs = 3, random_state = 5)
        clf = clf.fit(train_X, train_Y)
        scores = cross_val_score(clf, test_X, test_Y)
    print target, 'log loss: ', log_loss( target_table[target], clf.predict(input_table) )
    print target, ' model score: ', scores.mean()
    # Save random forest model to file
    output_file = './randomforest/' + target + '.pkl'
    joblib.dump( clf, output_file )
    return scores.mean()


# Random Forest
#    Adoption log loss:  1.07769347703
#    Adoption  model score:  0.753733561747
#    Return_to_owner log loss:  0.8153735878
#    Return_to_owner  model score:  0.812496584656
#    Transfer log loss:  0.973023132411
#    Transfer  model score:  0.788548825752
#    Died log loss:  0.935069406938
#    Died  model score:  0.678070175439
#    Euthanasia log loss:  0.829235184674
#    Euthanasia  model score:  0.74893851668


train_randomforest( input_table, target_table, 'Adoption', 100 )
#train_randomforest( input_table, target_table, 'Died', 10)
#train_randomforest( input_table, target_table, 'Euthanasia', 10 )
train_randomforest( input_table, target_table, 'Return_to_owner', 100 )
train_randomforest( input_table, target_table, 'Transfer', 100 )


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

train_randomforest( input_samples.iloc[arr], output_samples.iloc[arr], 'Died', 50 )


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

train_randomforest( input_samples.iloc[arr], output_samples.iloc[arr], 'Euthanasia', 50 )

