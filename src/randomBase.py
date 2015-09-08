import pandas as pd
import numpy as np

data = pd.read_csv('data/train.csv')
labels = data.ix[:, 4:]
l = len(data["tweet"])
names = ['s1', 's2', 's3', 's4', 's5', 'w1', 'w2', 'w3', 'w4', 
		 'k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', 'k9',
		 'k10', 'k11', 'k12', 'k13', 'k14', 'k15']

for i in range(0,5):
	d = {}
	for n in names:
		d[n] = np.random.random(l)
	pred = pd.DataFrame(d) 

	s_sum = pred['s1'] + pred['s2'] + pred['s3'] + pred['s4'] + pred['s5']
	w_sum = pred['w1'] + pred['w2'] + pred['w3'] + pred['w4']

	for n in ['s1', 's2', 's3', 's4', 's5']:
		pred[n] = pred[n]/s_sum
	for n in ['w1', 'w2', 'w3', 'w4']:
		pred[n] = pred[n]/w_sum  

	print(format(np.sqrt(np.sum(np.array(np.array(pred-labels)**2)/ (labels.shape[0]*24.0)))))