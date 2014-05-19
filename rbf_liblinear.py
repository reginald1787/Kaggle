
"""
Use randomforest, SVM, regression for classification


"""

import re
import string
import numpy as np
from sklearn.ensemble import RandomForestClassifier as randomforest
#import scipy.sparse as sparse
from sklearn.feature_extraction import DictVectorizer as dict2mat
from liblinearutil import *
from liblinear import *

def rdfreadvw(filename='mytrain.vw'):
	# n=100
	# i=0
	X=[]
	Y = []
	idY = []
	# feature={}
	# feature_value=1
	with open('../../allstate/%s'%filename,'r') as trainvw:
	
		for line in trainvw:
			a = line.split()
			Y.append(int(a[0]))
			idY.append(int(re.findall('\d+',a[1])[0]))
			newdict = {}
			for word in a[4:]:
				[key,value] = string.split(word,':')
				# if key not in feature.keys():
				# 	feature[key]=feature_value
				# 	feature_value+=1
				newdict[key] = float(value)
			X.append(newdict)
			# i+=1
			# if i>=n:
			# 	break
	trainvw.close()
	
	v = dict2mat(sparse=True)
	data = v.fit_transform(X)
	
	return data,Y,idY

def readvw(filename):
	X = []
	Y = []
	idY = []
	feature={}
	feature_value=1
	with open('../../allstate/%s'%filename,'r') as trainvw:
	
		for line in trainvw:
			a = line.split()
			Y.append(int(a[0]))
			#idY.append(int(a[1]))
			idY.append(int(re.findall('\d+',a[1])[0]))
			newdict = {}
			for word in a[4:]:
				[key,value] = string.split(word,':')
				if key not in feature.keys():
					feature[key]=feature_value
					feature_value+=1

				newdict[feature[key]] = float(value)
			X.append(newdict)
			# i+=1
			# if i>=n:
			# 	break
	#print trainvw.readlines(1)
	#print trainvw.readlines(2)
	trainvw.close()
	#print feature
	#print feature_value
	return X,Y,idY

def liblinear():
	x,y,yid = readvw()
	test,testy,testid = readvw('test.vw')

	#print x[0]
	#print x[1]
	prob = problem(y,x)
	param = parameter('-s 2 -c 5')
	m = train(prob, param)
	#m = load_model('shop.model')
	#p_label, p_acc, p_val = predict(y, x, m, '-b 1')
	#print type(m)
	
	p_label, p_acc, p_val = predict([], test, m)

	#print p_val[:20]
	with open('pred3.txt','w') as f:
		for f1,f2 in zip(p_label,testid):
			print >> f, f1, f2
	
	p_label, p_acc, p_val = predict(y, x, m)
	ACC, MSE, SCC = evaluations(y, p_label)
	#print p_acc,p_val
	save_model('shop3.model', m)

def rdf():
	x,y,yid = rdfreadvw()
	test,testy,testid = rdfreadvw('mytest.vw')
	#print len(testy),len(testid)
	train = x.toarray()
	test = test.toarray()
	rf = randomforest()
	rf.fit(train,y)
	print rf.score(train,y)
	pred = rf.predict(test)
	# #predprob = rf.predict_proba(test)


	with open('rfpred2.txt','w') as f:
		for f1,f2 in zip(pred,testid):
			print >> f, f1, f2

	# with open('rfpredprob.txt','w') as f:
	# for f1,f2 in zip(predprob,testid):
	# 	print >> f, f1, f2

	# np.savetxt('rfpred.txt',pred)
	# np.savetxt('rfpredprob.txt',predprob)



if __name__ == '__main__':
	
	rdf()
	
	
	
