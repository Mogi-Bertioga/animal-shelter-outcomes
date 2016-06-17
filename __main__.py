import sklearn
from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt
import pandas
from sklearn.naive_bayes import GaussianNB
import lib.missmap

class animal_shelter:

    def __init__(self):
        pass

    def load_train_dataset(self):
        self.dataset = pandas.read_csv('data/train.csv.gz', compression='gzip')

    def data_summary(self):
        print 'Data shape: ', self.dataset.shape
        print self.dataset.describe()

    def learn_naive_bayes(self):
        gnb = GaussianNB()
        y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
        print("Number of mislabeled points out of a total %d points : %d" % (iris.data.shape[0],(iris.target != y_pred).sum()))
        
    def missing_values(self):
        lib.missmap.missmap(self.dataset)
        plt.savefig('output/missing.png', bbox_inches='tight')

    def print_scikit_version(self):
        print 'scikit-learn version: ', sklearn.__version__

app = animal_shelter()
app.load_train_dataset()
app.data_summary()
#app.missing_values()
