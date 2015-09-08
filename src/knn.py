import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import neighbors
import time
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from sklearn.linear_model import Ridge
import numpy as np
import cogent.maths.stats.test as stats

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
# for i in range(num_test, len(train['tweet'])):
#         word_list = tokenizer.tokenize((train['tweet'][i]).lower())
#         for word in word_list:
#                 total_words += 1
#                 if not word in word_freqs:
#                         word_freqs[word] = 1
#                 else:
#                         word_freqs[word] = word_freqs[word] + 1

# print "total num of words: "+str(total_words)
# print word_freqs['the']

# stop_words = []
# for key in word_freqs:
#         if word_freqs[key]/(total_words*1.0) > thres:
#                 stop_words.append(key)

#print stop_words

ps = PorterStemmer()
r = RegexpTokenizer(r'\w+')
#stop_words = [ps.stem(t) for t in stop_words]
#print stop_words

#tfidf = TfidfVectorizer(max_features=1750, strip_accents='unicode', 
#	 tokenizer = MyTokenizer(), analyzer='word')

tfidf = TfidfVectorizer(max_features=1750, strip_accents='unicode', 
	 tokenizer = MyTokenizer(), stop_words = 'english', analyzer='word')

#print(tfidf.get_stop_words())
tfidf.fit(train['tweet'])
trainf = tfidf.transform(train['tweet'])
testf = tfidf.transform(test['tweet'])
trainlab = np.array(train.ix[:,4:])


def five_fold():
	print 'starting 5 fold'
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
		knn = neighbors.KNeighborsRegressor(n_neighbors=30)
		knn.fit(trainf,trainlabfin)
		# clf = Ridge(alpha = 5)
		# clf.fit(trainf, trainlabfin)
		n = 15
		pred = []
		for i in range(0,n):
			pred.extend(knn.predict(testf[(i*1000):((i+1)*(1000))]))
			print(i)
		pred.extend(knn.predict(testf[15000:15589]))
		print(format(np.sqrt(np.sum(np.array(np.array(pred-testlab)**2)/ (testf.shape[0]*24.0)))))
five_fold()

def knn(k):
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

print 'knn:'
print knn(30)

def knn_twice(k):
	knn1 = neighbors.KNeighborsRegressor(n_neighbors=k)
	knn1.fit(trainf,trainlab)
	print 'here'
	tim = time.time();

	n = len(train)/1000
	pred1 = []
	for i in range(0,n):
		pred1.extend(knn1.predict(trainf[(i*1000):((i+1)*(1000))]))
		print(i)
	pred1.extend(knn1.predict(trainf[67000:67946]))
	print "time: " + str(time.time() - tim)
	#knn = neighbors.KNeighborsRegressor(n_neighbors=k)
	#knn.fit(pred1,trainlab)
	ridge = Ridge(alpha=1.0)
	ridge.fit(pred1, trainlab)

	n = 10
	pred2 = []
	for i in range(0,n):
		pred2.extend(knn1.predict(testf[(i*1000):((i+1)*(1000))].toarray()))
		print(i)	

	n = 10
	pred = []
	for i in range(0,n):
		pred.extend(ridge.predict(pred2[(i*1000):((i+1)*(1000))]))
		print(i)	

	#RMSE:
	testlab = np.array(test.ix[:,4:])
	err = format(np.sqrt(np.sum(np.array(np.array(pred-testlab)**2)/ (testf.shape[0]*24.0))))
	return err

knn_twice(30)

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
	print err        


# k = [25,28,30,32,34,36,38,40,42,44,46]
# all_errors = []
# for i in k:
#         err = knn(i)
#         print err
#         all_errors.append((i,err))

#print 'knn twice: ' + str(knn_twice(30))
#print knn(30)
#print all_err

#all_err = []
#for n in range(500,2000,250):
 #       print 'testing ' + str(n)
  #      err = num_feat_select(n,30)
   #     all_err.append((n,err))
    #    print (n,err)
        
def k_fold(n, k):
	#tfidf = TfidfVectorizer(max_features=n, strip_accents='unicode', 
	#	 tokenizer = MyTokenizer(), analyzer='word')
	#tfidf.fit(train['tweet'])
	#trainf = tfidf.transform(train['tweet'])
	#testf = tfidf.transform(test['tweet'])
	#trainlab = np.array(train.ix[:,4:])

	num_folds = 5
	for i in range(1, num_folds):
		knn = neighbors.KNeighborsRegressor(n_neighbors=k)
		num_test = 50000
		train = data[:num_test]
		fold_size = len(train)/num_folds

		#first_part = train[0: fold_size*i]
		#first_part.append(train[fold_size*(i+1):])
		train2 = train[0: fold_size*i].join(train[fold_size*(i+1):])
		
		print train
		print train2
		#tfidf = TfidfVectorizer(max_features=n, strip_accents='unicode', 
		 #tokenizer = MyTokenizer(), analyzer='word')
		tfidf = TfidfVectorizer(max_features=n, strip_accents='unicode', 
                        analyzer='word')
		tfidf.fit(train2['tweet'])

		trainf2 = tfidf.transform(train2['tweet'])
		#trainf2 = trainf[0: fold_size*i]+trainf[fold_size*(i+1):]
		trainlab2 = np.array(train2.ix[:,4:])

		test2 = train[fold_size*i: fold_size*(i+1)]
		testf2 = tfidf.transform(test2['tweet'])
		testlab2 = np.array(test2.ix[:,4:])		

		knn.fit(trainf2,trainlab2)
		print 'here'
		tim = time.time();

		#pred = knn.predict(testf2)
		n = 10
		pred = []
		# for i in range(0,fold_size):
		# 		pred.extend(knn.predict(testf2[(i*1000):((i+1)*(1000))]))
		# 		print(i)
		for i in range(0,fold_size):
				pred.extend(knn.predict(testf2[i:i+1]))
				#print(i)
		print "time: " + str(time.time() - tim)  

		
		#RMSE:
		err = format(np.sqrt(np.sum(np.array(np.array(pred-testlab2)**2)/ (testf2.shape[0]*24.0))))
		print err        

# print 'k-fold:'
# k_fold(1750, 30)

all_err = []
def grid_search(n_arr,k_arr):
        for n in n_arr:
                for k in k_arr:
                        print 'testing n = ' + str(n) + ' k = ' + str(k) 
                        err = num_feat_select(n,k)
                        all_err.append((n,k,err))
                        print (n,k,err)

n_arr = [500, 1000,1250, 1500,1750,2000,2500,5000,7500,10000]
k_arr = [20,25,30,40,50,60,80,100,150,200,500,750,1000]

#grid_search(n_arr,k_arr)

def t_test(nums1, nums2, col_num):
        if len(nums1) != len(nums2):
                print 'fail'
                return
        w1 = []
        w2 = []
        for i in range(0,len(nums1)):
                w1.append(nums1[i][col_num])
                w2.append(nums2[i][col_num])
                
        t, prob = stats.t_two_sample(w1,w2)
        print 'results:'
        print t, prob
        return t, prob


        










        











