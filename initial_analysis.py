#read_data

import re
import pydot
import pylab as pl
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB
from sklearn.externals.six import StringIO
#from liblinearutil import *

# Warning: this is a large data set. The decompressed files require about 22GB of space.

# This data captures the process of offering incentives (a.k.a. coupons) to a large number of customers and forecasting those who will become loyal to the product. 
# Let's say 100 customers are offered a discount to purchase two bottles of water. Of the 100 customers, 60 choose to redeem the offer. 
# These 60 customers are the focus of this competition. You are asked to predict which of the 60 will return (during or after the promotional period) to purchase the same item again.

# To create this prediction, you are given a minimum of a year of shopping history prior to each customer's incentive, 
# as well as the purchase histories of many other shoppers (some of whom will have received the same offer). 
# The transaction history contains all items purchased, not just items related to the offer. Only one offer per customer is included in the data. 
# The training set is comprised of offers issued before 2013-05-01. The test set is offers issued on or after 2013-05-01.

# Files

# You are provided four relational files:

# transactions.csv - contains transaction history for all customers for a period of at least 1 year prior to their offered incentive
# trainHistory.csv - contains the incentive offered to each customer and information about the behavioral response to the offer
# testHistory.csv - contains the incentive offered to each customer but does not include their response (you are predicting the repeater column for each id in this file)
# offers.csv - contains information about the offers
# Fields

# All of the fields are anonymized and categorized to protect customer and sales information. 
# The specific meanings of the fields will not be provided (so don't bother asking). 
# Part of the challenge of this competition is learning the taxonomy of items in a data-driven way.

# history
# id - A unique id representing a customer
# chain - An integer representing a store chain
# offer - An id representing a certain offer
# market - An id representing a geographical region
# repeattrips - The number of times the customer made a repeat purchase
# repeater - A boolean, equal to repeattrips > 0
# offerdate - The date a customer received the offer

# transactions
# id - see above
# chain - see above
# dept - An aggregate grouping of the Category (e.g. water)
# category - The product category (e.g. sparkling water)
# company - An id of the company that sells the item
# brand - An id of the brand to which the item belongs
# date - The date of purchase
# productsize - The amount of the product purchase (e.g. 16 oz of water)
# productmeasure - The units of the product purchase (e.g. ounces)
# purchasequantity - The number of units purchased
# purchaseamount - The dollar amount of the purchase

# offers
# offer - see above
# category - see above
# quantity - The number of units one must purchase to get the discount
# company - see above
# offervalue - The dollar value of the offer
# brand - see above

# The transactions file can be joined to the history file by (id,chain). The history file can be joined to the offers file by (offer). 
# The transactions file can be joined to the offers file by (category, brand, company). A negative value in productquantity and purchaseamount indicates a return.


def classifer(X,Y):
	
	
	h = .02  # step size in the mesh
	# we create an instance of SVM and fit out data. We do not scale our
	# data since we want to plot the support vectors
	C = 1.0  # SVM regularization parameter
	# svc = svm.SVC(kernel='linear', C=C).fit(X, Y)
	# rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, Y)
	# poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, Y)
	lin_svc = svm.LinearSVC(C=C).fit(X, Y)

	# create a mesh to plot in
	x_min, x_max = X[:, 0].min()+1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min()+1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
					 np.arange(y_min, y_max, h))

	# title for the plots
	titles = ['SVC with linear kernel',
		  'SVC with RBF kernel',
		  'SVC with polynomial (degree 3) kernel',
		  'LinearSVC (linear kernel)']


	#for i, clf in enumerate((svc, rbf_svc, poly_svc, lin_svc)):
		# Plot the decision boundary. For that, we will assign a color to each
		# point in the mesh [x_min, m_max]x[y_min, y_max].
	clf = lin_svc
	#pl.subplot(2, 2, i + 1)
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

	# Put the result into a color plot
	Z = Z.reshape(xx.shape)
	pl.contourf(xx, yy, Z, cmap=pl.cm.Paired)
	pl.axis('off')

	# Plot also the training points
	pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)

	#pl.title(titles[i])

	pl.show()



def singlefactor(data,fac='market',result='repeattrips'):
	fact = data[[fac,result]]
	x = fact.sort(fac)[fac]
	plt.hist(x,90)
	plt.title('%s factor'%fac)
	plt.show()

def twofactor(data,fac1='chain',fac2='offer',result='repeattrips'):
		
	x = data[fac1].values
	y = data[fac2].values
	z = data[result].values
	z_min = min(z)
	z_max = max(z)
	#volume = (15 * price_data.volume[:-2] / price_data.volume[0])**2
	#close = 0.003 * price_data.close[:-2] / 0.003 * price_data.open[:-2]		
	fig,ax = plt.subplots()
	#drawer = ax.scatter(x,y,c=z,alpha=0.5)
	drawer = ax.scatter(x,y,c=z,alpha=0.5)
	ax.set_xlabel('%s'%fac1, fontsize=20)
	ax.set_ylabel('%s'%fac2, fontsize=20)
	ax.set_title('%s'%result)
	#ax.grid(True)
	cbar = fig.colorbar(drawer,ticks=[z_min,z_max])
	cbar.ax.set_yticklabels(['< %d'%z_min, '> %d'%z_max])
	fig.tight_layout()

	plt.show()

def liblinear(traindata,trainlabel,testdata):
	lin_svc = svm.LinearSVC()
	lin_svc.fit(traindata,trainlabel)
	predict = lin_svc.predict(testdata)
	np.savetxt('predict.csv',predict,delimiter=",")
	# print predict.shape
	# print predict

def naivebayes(traindata,trainlabel,testdata):
	BNB = BernoulliNB()
	BNB.fit(traindata,trainlabel)
	predict = BNB.predict(testdata)
	np.savetxt('NBpredict2.csv',predict,delimiter=",")
	#print BNB.score(testdata,predict)


def feature(train,test):

	# categorical features:

	chain = ['chain'+str(item) for item in  set(list(train['chain'].values)+list(test['chain'].values))]
	market = ['market'+str(item) for item in set(list(train['market'].values)+list(test['market'].values))]
	offer = ['offer'+str(item) for item in set(list(train['offer'].values)+list(test['offer'].values))]
	category = ['category'+str(item) for item in set(list(train['category'].values)+list(test['category'].values))]
	company = ['company'+str(item) for item in set(list(train['company'].values)+list(test['company'].values))]
	brand = ['brand'+str(item) for item in set(list(train['brand'].values)+list(test['brand'].values))]


	feature_key = chain+market+offer+category+company+brand #+offerdate
	feature_dict = dict(zip(feature_key,range(len(feature_key))))
	#print sorted(feature_dict.values()), len(feature_key)
	#print feature_dict.keys()

	# numerical features:



	#tansformation

	#table = train[['chain','market','offer','offerdate']].values
	traintable = train[['chain','market','offer','category','company','brand']].values
	testtable = test[['chain','market','offer','category','company','brand']].values

	traindata = featuredata(traintable,feature_key,feature_dict)
	testdata = featuredata(testtable,feature_key,feature_dict)

	return traindata,testdata


def featuredata(table,feature_key,feature_dict):
	nrow,ncol = table.shape

	data = sparse.dok_matrix((nrow,len(feature_key)))

	for i in range(nrow):
		#for j,name in zip(range(ncol),['chain','market','offer','offerdate']):
		for j,name in zip(range(ncol),['chain','market','offer','category','company','brand']):
			#print j,name
			data[i,feature_dict[name+str(table[i,j])]]=1

			#print feature_dict[name+str(table[i,j])]
	#print data

	#label = (train['repeater']=='t').astype(int)
	return data

def score(predict,label):
	n = len(predict)
	if n == len(label):
		#nonzero = np.subtract(predict,label)
		predict-=label
		#correct = nonzero[nonzero==0]
		correct = predict[predict==0]
		
		return float(len(correct))/n
	else:
		print 'error in length'
		return 0

def contigency_test():
	train = pd.read_csv('trainHistory.csv')
	table = pd.DataFrame({})


if __name__ == '__main__':

	
	# #liblinear(data,label,testdata)
	naivebayes(data,label,testdata)

	
