# -*- coding: UTF-8 -*-

"""
Original written by Triskelion, posted by BreakfastPirate.

Modifications:

Keep old features include "has_bought_brand, has_bought_company, has_bought_category,
has_bought_brand_category, has_bought_brand_company, has_bought_brand_company_category" 
within 30, 60, 90 days.

Add new features include "max_chain_visited_portion, max_market_visited_portion" within 30, 60, 90 days;
use "repeatertime" to duplicate samples as to enchance loyal customes' weights.

"""

from datetime import datetime, date
from collections import defaultdict
import numpy as np

loc_offers = "offers.csv"
loc_transactions = "transactions.csv"
loc_train = "trainHistory.csv"
loc_test = "testHistory.csv"

# will be created
loc_reduced = "reduced.csv" 
loc_out_train = "mytrain.vw"
loc_out_test = "mytest.vw"

###

def reduce_data(loc_offers, loc_transactions, loc_reduced):
  start = datetime.now()
  #get all categories and comps on offer in a dict
  offers_cat = {}
  offers_co = {}
  for e, line in enumerate( open(loc_offers) ):
    offers_cat[ line.split(",")[1] ] = 1
    offers_co[ line.split(",")[3] ] = 1
  #open output file
  with open(loc_reduced, "wb") as outfile:
    #go through transactions file and reduce
    reduced = 0
    for e, line in enumerate( open(loc_transactions) ):
      if e == 0:
        outfile.write( line ) #print header
      else:
        #only write when if category in offers dict
        if line.split(",")[3] in offers_cat or line.split(",")[4] in offers_co:
          outfile.write( line )
          reduced += 1
      #progress
      if e % 5000000 == 0:
        print e, reduced, datetime.now() - start
  print e, reduced, datetime.now() - start

#reduce_data(loc_offers, loc_transactions, loc_reduced)


def diff_days(s1,s2):
	date_format = "%Y-%m-%d"
	a = datetime.strptime(s1, date_format)
	b = datetime.strptime(s2, date_format)
	delta = b - a
	return delta.days

def generate_features(loc_train, loc_test, loc_transactions, loc_out_train, loc_out_test):
	#keep a dictionary with the offerdata
	offers = {}
	for e, line in enumerate( open(loc_offers) ):
		row = line.strip().split(",")
		offers[ row[0] ] = row

	#keep two dictionaries with the shopper id's from test and train
	train_ids = {}
	test_ids = {}
	for e, line in enumerate( open(loc_train) ):
		if e > 0:
			row = line.strip().split(",")
			train_ids[row[0]] = row
	for e, line in enumerate( open(loc_test) ):
		if e > 0:
			row = line.strip().split(",")
			test_ids[row[0]] = row
	#open two output files
	with open(loc_out_train, "wb") as out_train, open(loc_out_test, "wb") as out_test:
		#iterate through reduced dataset 
		last_id = 0
		features = defaultdict(float)
		#chain = {}
		for e, line in enumerate( open(loc_transactions) ):
			if e > 0: #skip header
				#poor man's csv reader
				row = line.strip().split(",")
				#write away the features when we get to a new shopper id
				if last_id != row[0] and e != 1:
					
					#generate negative features
					if "has_bought_company" not in features:
						features['never_bought_company'] = 1
					
					if "has_bought_category" not in features:
						features['never_bought_category'] = 1
						
					if "has_bought_brand" not in features:
						features['never_bought_brand'] = 1
						
					if "has_bought_brand" in features and "has_bought_category" in features and "has_bought_company" in features:
						features['has_bought_brand_company_category'] = 1
					
					if "has_bought_brand" in features and "has_bought_category" in features:
						features['has_bought_brand_category'] = 1
					
					if "has_bought_brand" in features and "has_bought_company" in features:
						features['has_bought_brand_company'] = 1
						
					if "chain_visited" not in features:
						features['never_visited_chain'] = 1

					#features['max_chain_visited_portion'] = np.array(chain.values())/float(sum(chain.values()))).max()

					outline = ""
					test = False
					for k, v in features.items():
						
						if k == "label" and v == 0.5:
							#test
							outline = "1 '" + last_id + " |f" + outline
							test = True
						elif k == "label":
							outline = str(v) + " '" + last_id + " |f" + outline
						else:
							outline += " " + k+":"+str(v) 
					outline += "\n"
					if test:
						out_test.write( outline )
					else:
						out_train.write( outline )
					#print "Writing features or storing them in an array"
					#reset features
					#reset chain
					features = defaultdict(float)
					#chain = {}
				#generate features from transaction record
				#check if we have a test sample or train sample
				if row[0] in train_ids or row[0] in test_ids:
					#generate label and history
					if row[0] in train_ids:
						history = train_ids[row[0]]
						if train_ids[row[0]][5] == "t":
							features['label'] = 1
						else:
							features['label'] = 0
					else:
						history = test_ids[row[0]]
						features['label'] = 0.5
						
					#print "label", label 
					#print "trainhistory", train_ids[row[0]]
					#print "transaction", row
					#print "offers", offers[ train_ids[row[0]][2] ]
					#print


					# features: offer_value, offer_quantity, total_spend are temporarilly disabled 
					# features: chain_visited is added

					#chain = {}
					#market = {}

					# if row[1] in chain.keys():
					# 	chain[row[1]]+=1
					# else:
					# 	chain[row[1]]=1

					if history[1] == row[1]:
						features['chain_visited'] += 1.0
						date_diff_days = diff_days(row[6],history[-1])
						if date_diff_days < 30:
							features['chain_visited_30'] += 1.0
							
						if date_diff_days < 60:
							features['chain_visited_60'] += 1.0
							
						if date_diff_days < 90:
							features['chain_visited_90'] += 1.0
							
						if date_diff_days < 180:
							features['chain_visited_180'] += 1.0
							

					#features['offer_value'] = offers[ history[2] ][4]
					#features['offer_quantity'] = offers[ history[2] ][2]
					offervalue = offers[ history[2] ][4]
					
					#features['total_spend'] += float( row[10] )

					# drop quantity and amount feature
					
					if offers[ history[2] ][3] == row[4]:
						features['has_bought_company'] += 1.0
						#features['has_bought_company_q'] += float( row[9] )
						#features['has_bought_company_a'] += float( row[10] )
						date_diff_days = diff_days(row[6],history[-1])
						if date_diff_days < 30:
							features['has_bought_company_30'] += 1.0
							#features['has_bought_company_q_30'] += float( row[9] )
							#features['has_bought_company_a_30'] += float( row[10] )
						if date_diff_days < 60:
							features['has_bought_company_60'] += 1.0
							#features['has_bought_company_q_60'] += float( row[9] )
							#features['has_bought_company_a_60'] += float( row[10] )
						if date_diff_days < 90:
							features['has_bought_company_90'] += 1.0
							#features['has_bought_company_q_90'] += float( row[9] )
							#features['has_bought_company_a_90'] += float( row[10] )
						if date_diff_days < 180:
							features['has_bought_company_180'] += 1.0
							#features['has_bought_company_q_180'] += float( row[9] )
							#features['has_bought_company_a_180'] += float( row[10] )
					
					if offers[ history[2] ][1] == row[3]:	
						features['has_bought_category'] += 1.0
						#features['has_bought_category_q'] += float( row[9] )
						#features['has_bought_category_a'] += float( row[10] )
						date_diff_days = diff_days(row[6],history[-1])
						if date_diff_days < 30:
							features['has_bought_category_30'] += 1.0
							#features['has_bought_category_q_30'] += float( row[9] )
							#features['has_bought_category_a_30'] += float( row[10] )
						if date_diff_days < 60:
							features['has_bought_category_60'] += 1.0
							#features['has_bought_category_q_60'] += float( row[9] )
							#features['has_bought_category_a_60'] += float( row[10] )
						if date_diff_days < 90:
							features['has_bought_category_90'] += 1.0
							#features['has_bought_category_q_90'] += float( row[9] )
							#features['has_bought_category_a_90'] += float( row[10] )						
						if date_diff_days < 180:
							features['has_bought_category_180'] += 1.0
							#features['has_bought_category_q_180'] += float( row[9] )
							#features['has_bought_category_a_180'] += float( row[10] )		

					if offers[ history[2] ][5] == row[5]:
						features['has_bought_brand'] += 1.0
						#features['has_bought_brand_q'] += float( row[9] )
						#features['has_bought_brand_a'] += float( row[10] )
						date_diff_days = diff_days(row[6],history[-1])
						if date_diff_days < 30:
							features['has_bought_brand_30'] += 1.0
							#features['has_bought_brand_q_30'] += float( row[9] )
							#features['has_bought_brand_a_30'] += float( row[10] )
						if date_diff_days < 60:
							features['has_bought_brand_60'] += 1.0
							#features['has_bought_brand_q_60'] += float( row[9] )
							#features['has_bought_brand_a_60'] += float( row[10] )
						if date_diff_days < 90:
							features['has_bought_brand_90'] += 1.0
							#features['has_bought_brand_q_90'] += float( row[9] )
							#features['has_bought_brand_a_90'] += float( row[10] )						
						if date_diff_days < 180:
							features['has_bought_brand_180'] += 1.0
							#features['has_bought_brand_q_180'] += float( row[9] )
							#features['has_bought_brand_a_180'] += float( row[10] )	

					# add geography features: max_chain_visited_portion, max_market visited_portion


				last_id = row[0]
				if e % 100000 == 0:
					print e
					
#generate_features(loc_train, loc_test, loc_transactions, loc_out_train, loc_out_test)


if __name__ == '__main__':
	#reduce_data(loc_offers, loc_transactions, loc_reduced)
	generate_features(loc_train, loc_test, loc_reduced, loc_out_train, loc_out_test)

	
	
