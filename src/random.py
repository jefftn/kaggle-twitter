import pandas as p
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

paths = ['new_train.csv', 'new_val.csv']
t = p.read_csv(paths[0])
t2 = p.read_csv(paths[1])

tfidf = TfidfVectorizer(max_features=10000, strip_accents='unicode', analyzer='word')
tfidf.fit(t['tweet'])
X = tfidf.transform(t['tweet'])
test = tfidf.transform(t2['tweet'])
Y = np.array(t.ix[:,4:])
testY = np.array(t2.ix[:,4:])


def randomp(row, col):
    a = np.random.uniform(0, 1, size=(row * col))
    return a.reshape(row,col)

(row,col) = X.shape
predictY = randomp(row,col)

(row,col) = test.shape
predictTestY = randomp(row,col)
    
    

print 'Train error: {0}'.format(np.sqrt(np.sum(np.array(np.array(predictY)-Y)**2)/ (X.shape[0]*24.0)))
print 'Validation error: {0}'.format(np.sqrt(np.sum(np.array(np.array(predictTestY)-testY)**2)/ (X.shape[0]*24.0)))
print ''


#col = '%f,'*23 + '%f'
#np.savetxt('predictY.csv', predictY,col, delimiter=',')
#np.savetxt('predictTestY.csv', predictTestY,col, delimiter=',')


print 'Train error: {0}'.format(np.sqrt(np.sum(np.array(np.array(predictY)-Y)**2)/ (X.shape[0]*24.0)))
print 'Validation error: {0}'.format(np.sqrt(np.sum(np.array(np.array(predictTestY)-testY)**2)/ (X.shape[0]*24.0)))

