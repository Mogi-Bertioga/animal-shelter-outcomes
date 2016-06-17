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
from sklearn.naive_bayes import BernoulliNB
import random
import lib.missmap

class animal_shelter:

    def __init__(self):
        pass

    # Source: http://stackoverflow.com/a/25319311/1501575
    def shuffle(self, df):
        index = list(df.index)
        random.shuffle(index)
        df = df.ix[index]
        df.reset_index()
        return df

    def load_train_dataset(self):
        self.df = pandas.read_csv('data/train.csv.gz', compression='gzip')
        self.df = self.shuffle(self.df)

    def split_dataset(self, ts=0, te=1000, vs=1001, ve=2001):
        self.train_start_index = ts
        self.train_end_index = te
        self.validation_start_index = vs
        self.validation_end_index = ve
        self.dataset = self.df[self.train_start_index:self.train_end_index][:]
        self.dataset = self.dataset.fillna('NA')
        self.input_variables = [1, 5, 6, 7, 8, 9]
        self.output_variable = [3]
        self.validation = self.df[self.validation_start_index:self.validation_end_index][:]
        self.validation = self.validation.fillna('NA')

    def data_summary(self):
        print 'Data shape: ', self.dataset.shape
        print self.dataset.describe()

    def learn_naive_bayes(self):
        vect = DictVectorizer(sparse = False)
        self.vect = vect
        input = self.dataset[:][self.input_variables]
        input_dict = input.T.to_dict().values()
        input = vect.fit_transform(input_dict)
        output = self.dataset[:][self.output_variable]
        le = LabelEncoder()
        le.fit(output.values.flatten())
        self.le = le
        output = le.transform(output.values.flatten())
        nb = BernoulliNB()
        self.clf = nb.fit(input, output)

    def predict_naive_bayes(self):
        le = self.le
        vect = self.vect
        input_validation = self.validation[:][self.input_variables]
        input_validation_dict = input_validation.T.to_dict().values()
        input_validation = vect.transform(input_validation_dict)
        output_validation = self.validation[:][self.output_variable]
        output_validation = le.transform(output_validation.values.flatten())
        y_pred = self.clf.predict(input_validation)
        total = input_validation.shape[0]
        errors = (output_validation != y_pred).sum()
        accuracy = (1.0 - float(errors)/float(total))*100.0
        print "Number of mislabeled points out of a total %d points: %d" % (total, errors)
        print "Accuracy: %.1f%%" % (accuracy)
        
    def missing_values(self):
        lib.missmap.missmap(self.dataset)
        plt.savefig('output/missing.png', bbox_inches='tight')

    def print_scikit_version(self):
        print 'scikit-learn version: ', sklearn.__version__

app = animal_shelter()
app.load_train_dataset()
app.split_dataset(0, 3000)
app.learn_naive_bayes()

for i in range(0,5):
    length = 1000
    start = 4000 + i*(length)
    app.split_dataset(0, 3000, start, start+length)
    app.predict_naive_bayes()

#app.data_summary()
#app.missing_values()
