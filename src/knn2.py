import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import neighbors
import time
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer

thres = 0.006
num_test = 10000
#without stemming and with stop words Pred error: 0.202183161285
#with stemming and stop words 0.201094689893
#without stemming and without stop words Pred error: 0.19875830122
#with stemming and without stop words Pred error: 0.198967769784
# with stemming and stemmed custom stop words Pred error: 0.265828643485
class MyTokenizer(object):
	def __init__(self):
		self.ps = PorterStemmer()
	def __call__(self, tweet):
		r = RegexpTokenizer(r'\w+')
		return r.tokenize(tweet)
		#return [self.ps.stem(t) for t in r.tokenize(tweet)]

data = pd.read_csv('data/train.csv')
test = data[:num_test]
train = data[num_test:]

print train['tweet'][num_test]
print len(train['tweet'])
word_freqs = {}
total_words = 0
tokenizer = RegexpTokenizer(r'\w+')
for i in range(num_test, len(train['tweet'])):
        word_list = tokenizer.tokenize((train['tweet'][i]).lower())
        for word in word_list:
                total_words += 1
                if not word in word_freqs:
                        word_freqs[word] = 1
                else:
                        word_freqs[word] = word_freqs[word] + 1

print "total num of words: "+str(total_words)
print word_freqs['the']

stop_words = []
for key in word_freqs:
        if word_freqs[key]/(total_words*1.0) > thres:
                stop_words.append(key)

print stop_words

ps = PorterStemmer()
r = RegexpTokenizer(r'\w+')
stop_words = [ps.stem(t) for t in stop_words]
print stop_words

tfidf = TfidfVectorizer(max_features=1750, strip_accents='unicode', 
	 tokenizer = MyTokenizer(), analyzer='word')


tfidf.fit(train['tweet'])
trainf = tfidf.transform(train['tweet'])
testf = tfidf.transform(test['tweet'])
trainlab = np.array(train.ix[:,4:])

#0.20027248411 with n_neighbors = 4
#0.19905705874 with n_neighbors = 1
#0.199360243758 with n_neighbors = 2
def knn_pseudo_rel(k):
	knn = neighbors.KNeighborsRegressor(n_neighbors=k)
	knn.fit(trainf,trainlab)
	n = 10
	k = 2
	nearest = []
	newtest = []
	for i in range(0,n):
		nearest.extend(knn.kneighbors(testf[(i*1000):((i+1)*(1000))], n_neighbors=k, return_distance=False))
	for i in range(0, n*1000):
		newtest.append(test['tweet'][i])
		for j in range(0, k):
			newtest[i] = newtest[i] + " " + train['tweet'][nearest[i][j] + 10000]
	newtestf = tfidf.transform(newtest)
	pred = []
	for i in range(0,n):
		pred.extend(knn.predict(newtestf[(i*1000):((i+1)*(1000))]))
		print(i) 
	#RMSE:
	testlab = np.array(test.ix[:,4:])
	err = format(np.sqrt(np.sum(np.array(np.array(pred-testlab)**2)/ (testf.shape[0]*24.0))))
	print err

def knn(k):
	knn = neighbors.KNeighborsRegressor(n_neighbors=k)
	knn.fit(trainf,trainlab)
	n = 10
	pred = []
	for i in range(0,n):
		pred.extend(knn.predict(testf[(i*1000):((i+1)*(1000))]))
		print(i) 
	#RMSE:
	testlab = np.array(test.ix[:,4:])
	err = format(np.sqrt(np.sum(np.array(np.array(pred-testlab)**2)/ (testf.shape[0]*24.0))))
	print err

def num_feat_select(n,k):
	tfidf = TfidfVectorizer(max_features=n, strip_accents='unicode', 
		 tokenizer = MyTokenizer(), analyzer='word')
	tfidf.fit(train['tweet'])
	trainf = tfidf.transform(train['tweet'])
	testf = tfidf.transform(test['tweet'])
	trainlab = np.array(train.ix[:,4:])
	knn = neighbors.KNeighborsRegressor(n_neighbors=k)
	knn.fit(trainf,trainlab)
	print 'here'
	tim = time.time();

	n = 10
	pred = []
	for i in range(0,n):
		pred.extend(knn.predict(testf[(i*1000):((i+1)*(1000))]))
		print(i)
	print "time: " + str(time.time() - tim) 

	#RMSE:
	testlab = np.array(test.ix[:,4:])
	err = format(np.sqrt(np.sum(np.array(np.array(pred-testlab)**2)/ (testf.shape[0]*24.0))))
	return err        


# k = [25,28,30,32,34,36,38,40,42,44,46]
# all_errors = []
# for i in k:
#         err = knn(i)
#         print err
#         all_errors.append((i,err))

#print knn(30)

#all_err = []
#for n in range(10000,110000,10000):
#        print 'testing ' + str(n)
#        err = num_feat_select(n,30)
#        all_err.append((n,err))
#print all_err

#knn_pseudo_rel(30)
knn_pseudo_rel(30)









