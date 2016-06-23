# -*- coding: utf-8 -*-

import sklearn
from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt
import pandas
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from lib import missmap
from lib import shuffle
import re
import sys

class animal_shelter:

    def __init__(self):
        self.flag_save_files = True
        self.flag_restore_session = True

    def parse_age(self, row):
        age = row
        try:
            result = re.search(r'([0-9]+)[^a-z]*([a-z]+)', age)
        except:
            return row
        if result:
            quantity = float(result.group(1))
            time_unit = result.group(2)
            if time_unit == 'week' or time_unit == 'weeks':
                quantity = quantity * 7
            elif time_unit == 'month' or time_unit == 'months':
                quantity = quantity * 30
            elif time_unit == 'year' or time_unit == 'years':
                quantity = quantity * 365
            row = quantity
        return row

    # N = number of unique values
    def keep_most_popular_values_for_feature(self, feature_name, N=30):
        total = self.df[feature_name].shape[0]
        counts = self.df[feature_name].value_counts()
        sum = 0
        different_values = 0
        outros_list = list()
        for value, count in counts.iteritems():
            outros = False
            #print value, count
            sum += count
            different_values += 1
            #if sum > 0.9*total:
            #    outros = True
            if different_values >= N:
                outros = True
            if outros:
                outros_list.append(value)
        #print '---------------------'
        #print 'Total\tSum\t%\tUnique'
        #print '%d\t%d\t%.1f\t%d' % (total, sum, 100.0*float(sum)/float(total), different_values)
        #print '---------------------'
        self.df[feature_name].replace(outros_list, 'Outros', inplace=True)

    def shuffle(self):
        self.df = shuffle.shuffle(self.df)

    def fix_missing(self):
        self.df = self.df.fillna('NA')

    def convert_age_to_numerical(self):
        self.df['AgeuponOutcome'] = self.df['AgeuponOutcome'].apply(self.parse_age)
        self.df['AgeuponOutcome'].fillna(self.df['AgeuponOutcome'].mean(), inplace=True)
        #print self.df['AgeuponOutcome'].as_matrix().dtype
        #print preprocessing.scale(self.df['AgeuponOutcome'].as_matrix())
        self.df['AgeuponOutcome'] = preprocessing.scale(self.df['AgeuponOutcome'].as_matrix())
        self.save_to_file(self.df, 'convert_age_to_numerical_df.csv')

    def categorical_to_numerical_one_hot(self):
        print 'Converting categorical to numerical...'
        vect = DictVectorizer(sparse = False)
        self.vect = vect
        input_dict = self.input_df.T.to_dict().values()
        self.input_df = vect.fit_transform(input_dict)
        #print self.input_df
        print 'Done.'
        self.save_to_file(self.input_df, 'categorial_to_numerical_one_hot_input_df.csv')

    def categorical_to_numerical_unique(self):
        print 'Converting categorical to numerical...'
        self.input_df = self.input_df.apply(LabelEncoder().fit_transform)
        print 'Done.'
        self.save_to_file(self.input_df, 'categorial_to_numerical_unique.csv')

    def target_to_numerical(self):
        print 'Converting target to numerical...'
        self.target_df = self.target_df.apply(LabelEncoder().fit_transform)
        print 'Done.'
        self.save_to_file(self.target_df, 'target_to_numerical_target_df.csv')

    def load_train_dataset(self):
        self.df = pandas.read_csv('../data/train.csv.gz', compression='gzip')

    def select_attributes(self):
        self.input_variables = [5, 6, 7, 8, 9]
        self.output_variable = [3]
        self.input_df = self.df[self.input_variables]
        self.target_df = self.df[self.output_variable]
        self.save_to_file(self.input_df, 'select_attributes_input_df.csv')

    def split_dataset(self, ts=0, te=1000, vs=1001, ve=2001):
        self.train_start_index = ts
        self.train_end_index = te
        self.validation_start_index = vs
        self.validation_end_index = ve
        self.train_input = self.input_df[self.train_start_index:self.train_end_index][:]
        self.save_to_file(self.train_input, 'split_dataset_train_input.csv')
        self.train_target = self.target_df[self.train_start_index:self.train_end_index][:]
        self.save_to_file(self.train_target, 'split_dataset_train_target.csv')
        self.validation_input = self.input_df[self.validation_start_index:self.validation_end_index][:]
        self.save_to_file(self.validation_input, 'split_dataset_validation_input.csv')
        self.validation_target = self.target_df[self.validation_start_index:self.validation_end_index][:]
        self.save_to_file(self.validation_target, 'split_dataset_validation_target.csv')

    def restore_session(self):
        self.train_input = self.file_read('split_dataset_train_input.csv')
        self.train_target = self.file_read('split_dataset_train_target.csv')
        self.validation_input = self.file_read('split_dataset_validation_input.csv')
        self.validation_target = self.file_read('split_dataset_validation_target.csv')

    def data_summary(self):
        print 'Data shape: ', self.dataset.shape
        print self.dataset.describe()

    def learn_svm(self):
        print 'Learning SVM...'
        self.clf_svm = svm.SVC()
        self.clf_svm.fit(self.train_input, self.train_target['OutcomeType'])
        print 'Done.'

    def predict_svm(self):
        print 'Predicting SVM...'
        #print 'Predicting rows %d through %d...' % (self.validation_start_index, self.validation_end_index)
        y_pred = self.clf_svm.predict(self.validation_input)
        y_pred = y_pred.reshape(y_pred.shape[0], 1)
        total = self.validation_input.shape[0]
        print  y_pred.shape
        print self.validation_target.shape
        correct = (self.validation_target == y_pred).sum()
        accuracy = (float(correct)/float(total))*100.0
        print "Number of mislabeled points out of a total %d points: %d" % (total, total-correct)
        print "Accuracy: %.1f%%" % (accuracy)
        print 'Done.'
        
    def missing_values(self):
        missmap.missmap(self.dataset)
        plt.savefig('output/missing.png', bbox_inches='tight')

    def print_scikit_version(self):
        print 'scikit-learn version: ', sklearn.__version__

    def save_to_file(self, df, filename):
        if self.flag_save_files:
            filename = '../output/' + filename
            print 'Saving to file %s...' % (filename)
            df = pandas.DataFrame(df)
            df.to_csv(filename, index=False)
            print 'Done.'

    def file_read(self, filename):
        filename = '../output/' + filename
        print 'Reading file %s...' % (filename)
        df = pandas.read_csv(filename)
        print 'Done.'
        return df

# Init script
app = animal_shelter()

if not app.flag_restore_session:

    # Step 1: Load train dataset (read CSV)
    app.load_train_dataset()

    # Step: Shuffle?
    app.shuffle()

    # Keep most popular breeds
    app.keep_most_popular_values_for_feature('Breed', 5)

    # Keep most popular names
    app.keep_most_popular_values_for_feature('Name', 5)

    # Keep most popular colors
    app.keep_most_popular_values_for_feature('Color', 5)

    # Step 2: Parse age and normalize from 0 to 1, NA's as 0.5
    app.convert_age_to_numerical()

    # Step: Replace missing values
    # Média? Moda?
    app.fix_missing()

    # Step 4: Select attributes
    # Remover AnimalID e OutcomeSubType
    app.select_attributes()

    # Step: Convert target variable to numerical
    app.target_to_numerical()

    # Step 5: Convert categorical variables to numeric
    #app.categorical_to_numerical_one_hot()
    app.categorical_to_numerical_unique()

    app.split_dataset(ts=0, te=1000, vs=2000,ve=3000)

else:
    app.restore_session()

# Step 7: Validation
#    - X-Validation
#        - Leave one out vs. k-fold
#        - Treinamento:
#            - Random Forest 
#        - Testing:
#            - Apply Model

# Step 8: Validate performance (accuracy, etc)

# Quick splitting dataset in train and validation subsets

# Step: Learn Naive Bayes
app.learn_svm()
app.predict_svm()
