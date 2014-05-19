"Generate a Kaggle submission file from VW predictions"

loc_preds = "preds2.txt"
#loc_preds = "/home/lzhong/pf/kaggle/liblinear-1.94/python/rfpred2.txt"
loc_submission = "7submission.csv"
loc_test = "testHistory.csv"

def generate_submission(loc_preds, loc_test, loc_submission):
	preds = {}
	for e, line in enumerate( open(loc_preds) ):
		row = line.strip().split(" ")
		preds[ row[1] ] = row[0]
		
	
	with open(loc_submission, "wb") as outfile:
		for e, line in enumerate( open(loc_test) ):
			if e == 0:
				outfile.write( "id,repeatProbability\n" )
			else:
				row = line.strip().split(",")
				if row[0] not in preds:
					outfile.write(row[0]+",0\n")
				else:
					outfile.write(row[0]+","+preds[row[0]]+"\n")
					
#generate_submission(loc_preds, loc_test, loc_submission)

if __name__ == '__main__':
	generate_submission(loc_preds, loc_test, loc_submission)
