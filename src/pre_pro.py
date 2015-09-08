import pandas as p
import nltk
from nltk.tokenize import *
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
lmz = WordNetLemmatizer()
paths = ['data/train.csv','data/train.csv']
t = p.read_csv(paths[0])
#print t



#t['insertfieldhere'] is list
#lists are t['id'],t['tweet'],t['state'],t['location'] blah blah


#pre-processing
stops = set(stopwords.words("english"))
tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+$')
for i in range(0, len(t['tweet'])):
	tokenized_word = regexp_tokenize(t['tweet'][i],r'\w+|^#*\w+|\d+\.\d+')
	filtered = [w for w in tokenized_word if not w in stops]
	final = []
	for word in filtered:
		final.append(lmz.lemmatize(word).lower())
	t['tweet'][i] = final


print t['tweet']
