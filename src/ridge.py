import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
import time

num_test = 10000
data = pd.read_csv('data/train.csv')

tfidf = TfidfVectorizer(max_features=10000, strip_accents='unicode', 
	analyzer='word')

#cross for local optima -do not use
# totalpred = [] 
# for k in range(4, 28):
# 	print("k = " + str(k))
# 	train = data[num_test:]
# 	trainlab = np.array(train.ix[:,k])
# 	tfidf.fit(train["tweet"])
# 	trainf = tfidf.transform(train["tweet"])
# 	test = data[:num_test]
# 	testlab = np.array(test.ix[:,k])
# 	testf = tfidf.transform(test["tweet"])
# 	alphas = [8, 1, 3, 2, 2, 4, 3, 5, 2, 1, 2, 2, 2, 0.5, 1, 3, 1, 5, 2, 1, 2, 4, 0.5, 1]
# 	clf = Ridge(alpha=alphas[k-4])
# 	clf.fit(trainf, trainlab)
# 	pred = clf.predict(testf) 
# 	for i in range(0,len(pred)):
# 		if pred[i] < 0:
# 			pred[i] = 0
# 		elif pred[i] > 1:
# 			pred[i] = 1  	
# 	totalpred.append(pred.tolist()) 
# totaltest = data[:num_test]
# totaltestlab = totaltest.ix[:,4:]
# print(format(np.sqrt(np.sum(np.array(np.array(np.transpose(totalpred)-totaltestlab)**2)/ (testf.shape[0]*24.0))))) 

#test for best alpha for each label 
# for k in range(4, 28):
# 	print("k = " + str(k))
# 	train = data[num_test:]
# 	trainlab = np.array(train.ix[:,k])
# 	tfidf.fit(train["tweet"])
# 	trainf = tfidf.transform(train["tweet"])

# 	test = data[:num_test]
# 	testlab = np.array(test.ix[:,k])
# 	testf = tfidf.transform(test["tweet"])

# 	alphas = [.1, .2, .5, .7, 1, 2, 3, 4, 5, 6,7, 8,9, 10]
# 	errors = []
# 	for a in alphas: 
# 		clf = Ridge(alpha=a)
# 		clf.fit(trainf, trainlab)
# 		pred = clf.predict(testf) 
# 		for i in range(0,len(pred)):
# 			if pred[i] < 0:
# 				pred[i] = 0
# 			elif pred[i] > 1:
# 				pred[i] = 1  	
# 		errors.append(np.sqrt(np.sum(np.array(np.array(pred-testlab)**2)/ (testf.shape[0]))))
# 	print(alphas[errors.index(min(errors))])

#test for best global alpha
# alphas = [.1, .2, .5, .7, 2, 5, 7, 10]
# for a in alphas: 
# 	clf = Ridge(alpha=a)
# 	clf.fit(trainf, trainlab)
# 	pred = clf.predict(testf) 
# 	for i in range(0,len(pred)):
# 		for j in range(0, len(pred[i])):
# 			if pred[i][j] < 0:
# 				 pred[i][j] = 0
# 			elif pred[i][j] > 1:
# 				pred[i][j] = 1 
# 	print(format(np.sqrt(np.sum(np.array(np.array(pred-testlab)**2)/ (testf.shape[0]*24.0))))) 

#k cross for best global alpha 
# for i in range(0, 5):
# 	l = len(data['tweet'])
# 	l5 = l/5
# 	test = data[l5 * i: l5 * (i+1)]
# 	testlab = np.array(test.ix[:,4:])
# 	train = data[:l5*i]
# 	trainlab = np.array(train.ix[:,4:])
# 	train2 = data[l5*(i+1):]
# 	trainlab2 = np.array(train2.ix[:,4:])
# 	train = np.array(train["tweet"])
# 	train2 = np.array(train2["tweet"])
# 	trainfin = np.concatenate((train, train2))
# 	trainlabfin = np.concatenate((trainlab, trainlab2))
# 	tfidf.fit(trainfin)
# 	trainf = tfidf.transform(trainfin)
# 	testf = tfidf.transform(test["tweet"])
# 	clf = Ridge(alpha = 2)
# 	clf.fit(trainf, trainlabfin)
# 	pred = clf.predict(testf)
# 	for i in range(0,len(pred)):
# 		for j in range(0, len(pred[i])):
# 			if pred[i][j] < 0:
# 				 pred[i][j] = 0
# 			elif pred[i][j] > 1:
# 				pred[i][j] = 1 
# 	print(format(np.sqrt(np.sum(np.array(np.array(pred-testlab)**2)/ (testf.shape[0]*24.0))))) 

#k cross for optimal local alphas
# for i in range(0, 5):
# 	l = len(data['tweet'])
# 	l5 = l/5
# 	totalpred = [] 
# 	for k in range(4, 28):
# 		test = data[l5 * i: l5 * (i+1)]
# 		testlab = np.array(test.ix[:,k])
# 		train = data[:l5*i]
# 		trainlab = np.array(train.ix[:,k])
# 		train2 = data[l5*(i+1):]
# 		trainlab2 = np.array(train2.ix[:,k])
# 		train = np.array(train["tweet"])
# 		train2 = np.array(train2["tweet"])
# 		trainfin = np.concatenate((train, train2))
# 		trainlabfin = np.concatenate((trainlab, trainlab2))
# 		tfidf.fit(trainfin)
# 		trainf = tfidf.transform(trainfin)
# 		testf = tfidf.transform(test["tweet"])
# 		alphas = [8, 1, 3, 2, 2, 4, 3, 5, 2, 1, 2, 2, 2, 0.5, 1, 3, 1, 5, 2, 1, 2, 4, 0.5, 1]
# 		clf = Ridge(alpha=alphas[k-4])
# 		clf.fit(trainf, trainlabfin)
# 		pred = clf.predict(testf)
# 		for j in range(0,len(pred)):
# 			if pred[j] < 0:
# 				pred[j] = 0
# 			elif pred[j] > 1:
# 				pred[j] = 1  	
# 		totalpred.append(pred.tolist()) 
# 	totaltest = data[l5 * i: l5 * (i+1)]
# 	totaltestlab = totaltest.ix[:,4:]
# 	print(format(np.sqrt(np.sum(np.array(np.array(np.transpose(totalpred)-totaltestlab)**2)/ (testf.shape[0]*24.0))))) 


#test for best 2 step alpha
# train = data[num_test:]
# trainlab = np.array(train.ix[:,4:])
# tfidf.fit(train["tweet"])
# trainf = tfidf.transform(train["tweet"])
# test = data[:num_test]
# testlab = np.array(test.ix[:,4:])
# testf = tfidf.transform(test["tweet"])
# alphas = [0.1, 0.2, 0.5, 0.7, 1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]
# #set up ridge two step
# clf = Ridge(alpha=2)
# clf.fit(trainf, trainlab)
# predTrain= clf.predict(trainf) 
# for i in range(0,len(predTrain)):
# 	for j in range(0, len(predTrain[i])):
# 		if predTrain[i][j] < 0:
# 			 predTrain[i][j] = 0
# 		elif predTrain[i][j] > 1:
# 			predTrain[i][j] = 1
# for a in alphas: 
# 	clf2 = Ridge(alpha = a)
# 	clf2.fit(predTrain, trainlab)
# 	#test testdata 
# 	pred = clf.predict(testf)
# 	for i in range(0,len(pred)):
# 		for j in range(0, len(pred[i])):
# 			if pred[i][j] < 0:
# 				 pred[i][j] = 0
# 			elif pred[i][j] > 1:
# 				pred[i][j] = 1
# 	pred = clf2.predict(pred)
# 	for i in range(0,len(pred)):
# 		for j in range(0, len(pred[i])):
# 			if pred[i][j] < 0:
# 				 pred[i][j] = 0
# 			elif pred[i][j] > 1:
# 				pred[i][j] = 1
# 	print(str(a) + ": " + format(np.sqrt(np.sum(np.array(np.array(pred-testlab)**2)/ (testf.shape[0]*24.0))))) 

#cross for two step prediction 
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
	tfidf.fit(trainfin)
	trainf = tfidf.transform(trainfin)
	testf = tfidf.transform(test["tweet"])
	clf = Ridge(alpha = 2)
	clf.fit(trainf, trainlabfin)
	pred = clf.predict(trainf)
	for i in range(0,len(pred)):
		for j in range(0, len(pred[i])):
			if pred[i][j] < 0:
				 pred[i][j] = 0
			elif pred[i][j] > 1:
				pred[i][j] = 1 
	clf2 = Ridge(alpha = 10)
	clf2.fit(pred, trainlabfin)
	pred = clf.predict(testf)
	for i in range(0,len(pred)):
		for j in range(0, len(pred[i])):
			if pred[i][j] < 0:
				 pred[i][j] = 0
			elif pred[i][j] > 1:
				pred[i][j] = 1 
	pred = clf2.predict(pred)
	for i in range(0,len(pred)):
		for j in range(0, len(pred[i])):
			if pred[i][j] < 0:
				 pred[i][j] = 0
			elif pred[i][j] > 1:
				pred[i][j] = 1 
	print(format(np.sqrt(np.sum(np.array(np.array(pred-testlab)**2)/ (testf.shape[0]*24.0)))))