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
        pass

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

    def shuffle(self):
        self.df = shuffle.shuffle(self.df)

    def fix_missing(self):
        self.df = self.df.fillna('NA')

    def convert_age_to_numerical(self):
        self.df['AgeuponOutcome'] = self.df['AgeuponOutcome'].apply(self.parse_age)
        self.save_to_file(self.df, 'convert_age_to_numerical_df.csv')

    def categorical_to_numerical_one_hot(self):
        print 'Converting categorical to numerical...'
        vect = DictVectorizer(sparse = False)
        self.vect = vect
        input_dict = self.input_df.T.to_dict().values()
        self.input_df = vect.fit_transform(input_dict)
        print 'Done.'
        self.save_to_file(self.input_df, 'categorial_to_numerical_one_hot_input_df.csv')

    def categorical_to_numerical_unique(self):
        print 'Converting categorical to numerical...'
        self.input_df = self.input_df.apply(LabelEncoder().fit_transform)
        print 'Done.'
        self.save_to_file(self.input_df, 'categorial_to_numerical_unique.csv')

    def target_to_numerical(self):
        output = self.dataset[:][self.output_variable]
        le = LabelEncoder()
        le.fit(output.values.flatten())
        self.le = le
        output = le.transform(output.values.flatten())
        print 'Done.'

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
        self.dataset = self.df[self.train_start_index:self.train_end_index][:]
        self.validation = self.df[self.validation_start_index:self.validation_end_index][:]

    def data_summary(self):
        print 'Data shape: ', self.dataset.shape
        print self.dataset.describe()

    def learn_naive_bayes(self):
        nb = MultinomialNB()
        self.clf = nb.fit(input, output)
        #print self.clf

    def predict_naive_bayes(self):
        print 'Predicting rows %d through %d...' % (self.validation_start_index, self.validation_end_index)
        y_pred = self.clf.predict(self.input_validation, self.output_validation)
        total = input_validation.shape[0]
        correct = (output_validation == y_pred).sum()
        accuracy = (float(correct)/float(total))*100.0
        print "Number of mislabeled points out of a total %d points: %d" % (total, total-correct)
        print "Accuracy: %.1f%%" % (accuracy)
        
    def missing_values(self):
        missmap.missmap(self.dataset)
        plt.savefig('output/missing.png', bbox_inches='tight')

    def print_scikit_version(self):
        print 'scikit-learn version: ', sklearn.__version__

    def save_to_file(self, df, filename):
        filename = '../output/' + filename
        print 'Saving to file %s...' % (filename)
        df = pandas.DataFrame(df)
        df.to_csv(filename, index=False)
        print 'Done.'

# Init script
app = animal_shelter()

# Step 1: Load train dataset (read CSV)
app.load_train_dataset()

# Step 2: Parse age
app.convert_age_to_numerical()

# Step 3: Shuffle?

# Step 4: Select attributes
# Remover AnimalID e OutcomeSubType
app.select_attributes()

# Step 5: Convert categorical variables to numeric
#app.categorical_to_numerical_one_hot()
app.categorical_to_numerical_unique()

# Step 6: Replace missing values
# Média? Moda?
app.fix_missing()

# Step 7: Validation
#    - X-Validation
#        - Leave one out vs. k-fold
#        - Treinamento:
#            - Random Forest 
#        - Testing:
#            - Apply Model

# Step 8: Validate performance (accuracy, etc)
