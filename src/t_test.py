import pandas as p
import nltk
from nltk.tokenize import *
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
# two sample t-tests
import numpy as np
import cogent.maths.stats.test as stats

# Testing the library
def oneTrial(n,f=np.random.normal):
    N = 500      # num of individual tests
    SZ = n*2*N    # need this many nums
    draw = f(loc=50,scale=3,size=SZ)
    counter = 0
    for i in range(0,SZ,n*2):
        nums1 = draw[i:i+n]
        nums2 = draw[i+n:i+2*n]
        t, prob = stats.t_two_sample(nums1,nums2)
        if prob < 0.05:  counter += 1
    return 1.0*counter / N

n = 3             # sample size
L = list()
#for i in range(100):
    #if i and not i % 10:  print 'i =', i
    #L.append(oneTrial(n))

#print 'avg %3.4f, std %3.4f' % (np.mean(L), np.std(L)),
#print 'min %3.4f, max %3.4f' % (min(L), max(L))

# f=np.random.normal
# draw = f(loc=50,scale=3,size=2000)
# nums1 = draw[0:1000]
# nums2 = draw[1000:2000]
# t, prob = stats.t_two_sample(nums1,nums2)
# print t
# print prob

# nums1 = [[.5,.3],[.2,.3]]
# nums2 = [[.5,.3],[.2,.3]]
# nums1 = [.4,.2]
# nums2 = [.2,.3]
# t, prob = stats.t_two_sample(nums1,nums2)
# print t
# print prob

nums1 = [0.1893, 0.1896, 0.1893, 0.1894, 0.1898]
nums1 = [0.1880, 0.1882, 0.1890, 0.1884, 0.1882]
nums1 = [0.1576, 0.1587, 0.1584, 0.1576, 0.1577]
nums2 = [0.1574, 0.1586, 0.1581, 0.1574, 0.1576]

def t_test(nums1, nums2):
    t, prob = stats.t_two_sample(nums1,nums2)
    print t
    print prob   

t_test(nums1, nums2)













