import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB 
import time

num_test = 10000
data = pd.read_csv('data/train.csv')

for i in range(0, 5):
	l = len(data['tweet'])
	l5 = l/5
	test = data[l5 * i: l5 * (i+1)]
	testlab = np.array(test.ix[:,4:])
	train = data[:l5*i]
	trainlab = np.array(train.ix[:,4:])
	train2 = data[l5*(i+1):]
	trainlab2 = np.array(train2.ix[:,4:])
	train = np.array(train["tweet"])
	train2 = np.array(train2["tweet"])
	trainfin = np.concatenate((train, train2))
	trainlabfin = np.concatenate((trainlab, trainlab2))
	clf = MultinomialNB()
	clf.fit(trainfin, trainlabfin)
	pred = clf.predict(test) 
	s_sum = pred[:, 0] + pred[:, 1] + pred[:, 2] + pred[:, 3] + pred[:, 4] 
	w_sum = pred[:, 5] + pred[:, 6] + pred[:, 7] + pred[:, 8] 
	for i in range(0,5):
		pred[:, i] = pred[:,i]/s_sum
	for i in range(5,9):
		pred[:, i] = pred[:, i]/w_sum
	print(format(np.sqrt(np.sum(np.array(np.array(pred-testlab)**2)/ (testf.shape[0]*24.0))))) 

