'''
USP - ICMC
Kaggle: Shelter animal outcomes
Authors: Thomio Watanabe
 - separate dataset
 - train machine learning methods
 - evaluate result
'''

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.externals import joblib


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


died_samples = np.where( died == 1) # 197 samples
euthanasia_samples = np.where( euthanasia == 1) # 1555 samples


def train_randomforest( input_table, target_table, target ):
    # Separate train and test samples
    # Use cross validation
    number_samples = len(input_table) - 1
    kfold = KFold( number_samples , n_folds = 10 )
    for train_index, test_index in kfold:
        print 'train: ', train_index, 'test: ',test_index
        train_X, train_Y = input_table.iloc[train_index], target_table[ target ].iloc[train_index]
        test_X, test_Y   = input_table.iloc[test_index], target_table[ target ].iloc[test_index]
        clf = RandomForestClassifier(n_estimators = 20)
        clf = clf.fit(train_X, train_Y)
        scores = cross_val_score(clf, test_X, test_Y)
    print target, ' model score: ', scores.mean()
    # Save random forest model to file
    output_file = './randomforest/' + target + '.pkl'
    joblib.dump( clf, output_file )
    return scores.mean()


# Create model and evaluate

# n_estimators = 10
# adoption =
# died =
# euthanasia =
# return_to_owner =

# n_estimators = 20
# adoption = 0.746622495423
# died = 0.992889773433
# euthanasia = 0.928144112788
# return_to_owner = 0.811752985536
# transfer = 0.773205210658

train_randomforest( input_table, target_table, 'Adoption' )
#train_randomforest( input_table, target_table, 'Died' )
#train_randomforest( input_table, target_table, 'Euthanasia' )
train_randomforest( input_table, target_table, 'Return_to_owner' )
train_randomforest( input_table, target_table, 'Transfer' )


# Reduce training set to died and euthanasia attributes
died_target = input_table.drop( input_table.index[ died_samples[0] ] )
euthanasia_target = input_table.drop( input_table.index[ euthanasia_samples[0] ] )

train_randomforest( input_table.iloc[died_samples[0]], died_target[0:len(died_samples[0])], 'Died' )
train_randomforest( input_table.iloc[euthanasia_samples[0]], euthanasia_target[0:len(euthanasia_samples[0])], 'Euthanasia' )


