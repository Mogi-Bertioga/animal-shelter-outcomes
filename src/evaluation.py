import sklearn
from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt
import pandas
import numpy as np
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
#from sklearn.naive_bayes import BernoulliNB

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
        result = re.search(r'([0-9]+)[^a-z]*([a-z]+)', age)
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
        else:
            row = 99999
        return row

    def load_train_dataset(self):
        self.df = pandas.read_csv('../data/train.csv.gz', compression='gzip')
        self.df = shuffle.shuffle(self.df)
        self.df = self.df.fillna('NA')
        self.input_variables = [1, 5, 6, 7, 8, 9]
        self.output_variable = [3]
        self.df['AgeuponOutcome'] = self.df['AgeuponOutcome'].apply(self.parse_age)

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
        print 'Learning from rows %d through %d...' % (self.train_start_index, self.train_end_index)
        vect = DictVectorizer(sparse = False)
        self.vect = vect
        input = self.dataset[:][self.input_variables]
        input_dict = input.T.to_dict().values()
        input = vect.fit_transform(input_dict)
        print input.shape
        print input
        print pandas.DataFrame(input).describe()
        sys.exit()
        output = self.dataset[:][self.output_variable]
        le = LabelEncoder()
        le.fit(output.values.flatten())
        self.le = le
        output = le.transform(output.values.flatten())
        nb = MultinomialNB()
        self.clf = nb.fit(input, output)
        #print self.clf

    def predict_naive_bayes(self):
        print 'Predicting rows %d through %d...' % (self.validation_start_index, self.validation_end_index)
        le = self.le
        vect = self.vect
        input_validation = self.validation[:][self.input_variables]
        input_validation_dict = input_validation.T.to_dict().values()
        input_validation = vect.transform(input_validation_dict)
        output_validation = self.validation[:][self.output_variable]
        output_validation = le.transform(output_validation.values.flatten())
        y_pred = self.clf.predict(input_validation)
        total = input_validation.shape[0]
        correct = (output_validation == y_pred).sum()
        #print output_validation
        #print y_pred
        accuracy = (float(correct)/float(total))*100.0
        print "Number of mislabeled points out of a total %d points: %d" % (total, total-correct)
        print "Accuracy: %.1f%%" % (accuracy)
        
    def missing_values(self):
        missmap.missmap(self.dataset)
        plt.savefig('output/missing.png', bbox_inches='tight')

    def print_scikit_version(self):
        print 'scikit-learn version: ', sklearn.__version__

app = animal_shelter()
app.load_train_dataset()
app.split_dataset(0, 7000)
app.learn_naive_bayes()

for i in range(0,5):
    length = 1000
    start = 10000 + i*(length)
    app.split_dataset(0, 3000, start, start+length)
    app.predict_naive_bayes()

#app.data_summary()
#app.missing_values()
