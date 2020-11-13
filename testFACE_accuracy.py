import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle as pk


def main():
	recourse_dict_path = './tmp/Face_recourse_points.pk'
	clf_path = './tmp/LR_classifier_face.pk'
	FEATURE_SET = ['x1', 'x2']

	recourse_dict = pk.load(open(recourse_dict_path, 'rb'))
	clf = pk.load(open(clf_path, 'rb'))

	negatives = 0
	flipped = 0
	for sample_id in recourse_dict:
		sample = recourse_dict[sample_id]
		name = sample['name']
		factX = sample['factual_instance'][FEATURE_SET]
		coutfactX = sample['counterfactual_target'][FEATURE_SET]
		cost = sample['cost']
		
		if (clf.predict([factX])[0] == 0):
			negatives += 1
			if (clf.predict([coutfactX])[0] == 1):
				flipped += 1

	print("flip accuracy: ", (1.0*flipped)/negatives, " for", len(recourse_dict), " points")


if __name__=="__main__":
	main()
