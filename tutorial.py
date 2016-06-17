import sklearn
from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt

class learning_digits:

    def __init__(self):
        self.digits = datasets.load_digits()

    def print_summary(self):
        print self.digits.data
        print self.digits.target
        print self.digits.data.shape

    def learn(self):
        self.clf = svm.SVC(gamma=0.001, C=100.0)
        self.clf.fit(self.digits.data[:-1], self.digits.target[:-1]) 

    def predict(self, img_index):
        prediction, = self.clf.predict(self.digits.data[img_index])
        print 'Image ID: %d; Prediction: %d' % (img_index, prediction)
        figdata = self.digits.data[img_index].reshape(8, 8)
        plt.imshow(figdata, cmap=plt.cm.gray_r, interpolation='none')
        plt.show()

    def print_scikit_version(self):
        print 'scikit-learn version: ', sklearn.__version__

ld = learning_digits()
ld.print_scikit_version()
ld.learn()
ld.predict(200)
