import pandas as p
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class NaiveBayes:
    def __init__(self):
        self.pys = np.zeros(24)
        self.pxs = np.zeros((24,10000))

    def train(self,X,Y):
        (nrow, ncol) = X.shape
        totaly = 0
        print ('calculating Y..')
        i = 0
        for row in Y:
            i += 1
            totaly += row
        print ('Finished calculating Y')
        self.pys = totaly/nrow
        print ('calculating X..')
        for i in range(0,nrow):
            if (i % 10000 == 0):
                print ("At row %d" % i)
            curx = X[i,:]
            cury = Y[i,:]
            for j in range(0,24):
                self.pxs[j,:] += (cury[j] * curx)
        print ('Finished calculating X')
        for row in self.pxs:
            row = row/(row.sum())
        

    def predict(self,X):
        (nrow, ncol) = X.shape
        Y = np.zeros((nrow, 24))
        for i in range(0,nrow):
            if (i % 10000 == 0):
                print ("At row %d" % i)
            for j in range(0,24):
                temp = (self.pxs[j,:]).reshape(ncol,1)
                prod = X[j,:] * temp
                Y[i,j] = self.pys[j] * prod
        for i in range(0,nrow):
            Y[i,0:5] = Y[i,0:5]/((Y[i,0:5]).sum())
            Y[i,5:9] = Y[i,5:9]/((Y[i,5:9]).sum())
        return Y

paths = ['new_train.csv', 'new_val.csv']
t = p.read_csv(paths[0])
t2 = p.read_csv(paths[1])

tfidf = TfidfVectorizer(max_features=10000, strip_accents='unicode', analyzer='word')
tfidf.fit(t['tweet'])
X = tfidf.transform(t['tweet'])
test = tfidf.transform(t2['tweet'])
Y = np.array(t.ix[:,4:])
testY = np.array(t2.ix[:,4:])

nb = NaiveBayes()
print ("Training..")
nb.train(X,Y)
print ("Done Training")

print 'Train error: {0}'.format(np.sqrt(np.sum(np.array(np.array(nb.predict(X))-Y)**2)/ (X.shape[0]*24.0)))
print 'Validation error: {0}'.format(np.sqrt(np.sum(np.array(np.array(nb.predict(test))-testY)**2)/ (X.shape[0]*24.0)))

