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


df = pd.read_csv('../output/filtered_dataset.csv')


# One model for each output class
# Output classes
adoption = df['Adoption']
died = df['Died']
euthanasia = df['Euthanasia']
return_to_owner = df['Return_to_owner']


df.drop('Adoption', axis = 1, inplace = True)
df.drop('Died', axis = 1, inplace = True)
df.drop('Euthanasia', axis = 1, inplace = True)
df.drop('Return_to_owner', axis = 1, inplace = True)


# input attributes
input_table = df


# Separate train and test samples
# Use cross validation
number_samples = len(input_table) - 1
kfold = KFold( number_samples , n_folds = 10 )


# Create model and evaluate

# n_estimators = 10
# adoption = 0.818
# died = 0.992
# euthanasia = 0.932
# return_to_owner = 0.809

# n_estimators = 20
# adoption = 0.8195
# died = 0.9928
# euthanasia = 0.9337
# return_to_owner = 0.8147

for train_index, test_index in kfold:
    print 'train: ', train_index, 'test: ',test_index
    train_X, train_Y = input_table.iloc[train_index], adoption.iloc[train_index]
    test_X, test_Y   = input_table.iloc[test_index], adoption.iloc[test_index]
    clf = RandomForestClassifier(n_estimators = 20)
    clf = clf.fit(train_X, train_Y)
    scores = cross_val_score(clf, test_X, test_Y)

print 'Adoption model score: ', scores.mean()



for train_index, test_index in kfold:
    print 'train: ', train_index, 'test: ',test_index
    train_X, train_Y = input_table.iloc[train_index], died.iloc[train_index]
    test_X, test_Y   = input_table.iloc[test_index], died.iloc[test_index]
    clf = RandomForestClassifier(n_estimators = 20)
    clf = clf.fit(train_X, train_Y)
    scores = cross_val_score(clf, test_X, test_Y)

print 'died model score: ', scores.mean()


for train_index, test_index in kfold:
    print 'train: ', train_index, 'test: ',test_index
    train_X, train_Y = input_table.iloc[train_index], euthanasia.iloc[train_index]
    test_X, test_Y   = input_table.iloc[test_index], euthanasia.iloc[test_index]
    clf = RandomForestClassifier(n_estimators = 20)
    clf = clf.fit(train_X, train_Y)
    scores = cross_val_score(clf, test_X, test_Y)

print 'euthanasia model score: ', scores.mean()


for train_index, test_index in kfold:
    print 'train: ', train_index, 'test: ',test_index
    train_X, train_Y = input_table.iloc[train_index], return_to_owner.iloc[train_index]
    test_X, test_Y   = input_table.iloc[test_index], return_to_owner.iloc[test_index]
    clf = RandomForestClassifier(n_estimators = 20)
    clf = clf.fit(train_X, train_Y)
    scores = cross_val_score(clf, test_X, test_Y)

print 'return_to_owner model score: ', scores.mean()



