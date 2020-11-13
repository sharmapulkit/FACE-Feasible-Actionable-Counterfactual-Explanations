import sys
import numpy as np
import pandas as pd
import pickle as pk
from sklearn.linear_model import LogisticRegression

import utils
import distribution
from FACE import *

### KDE kernel
class Kernel_obj:
	def __init__(self, distrib):
		self._distribution = distrib

	def kernelKDE(self, xi, xj):
		"""
		KDE kernel value
		"""
		mean = 0.5*(xi + xj)
		dist = np.linalg.norm(xi - xj, 2)
		density_at_mean = self._distribution.pdf(mean)
		return density_at_mean*dist


class distance_obj:
	def __init__(self):
		return

	def computeDistance(self, xi, xj):
		dist = np.linalg.norm(xi - xj, 2)
		return dist

def main():
	# dataPath = "./data/synthetic_one_hot"
	# datasetName = 'synthetic_lin'
	# FEATURE_COLUMNS = ['x1', 'x2', 'x3']

	dataPath = "./data/synthetic_face_dataset.pk"
	datasetName = 'paper_synthetic'
	FEATURE_COLUMNS = ['x1', 'x2']
	TARGET_COLUMNS = ['y']

	tp = 0.6 # Prediction threshold
	td = 0.001 # density threshold
	epsilon = 0.05 # margin for creating connections in graphs

	# data_scf_obj = utils.load_data(dataPath)
	# data = data_scf_obj.data_frame_kurz.iloc[:500]
	data = pk.load(open(dataPath, 'rb'))
	X = data[FEATURE_COLUMNS]
	y = data[TARGET_COLUMNS]

	### Train a logistic regression model
	clf = LogisticRegression(random_state=utils.random_seed)
	clf.fit(X, y)
	print("Training accuracy:", clf.score(X, y))

	### Get the negatively classified points
	negative_points = utils.get_negatively_classified(data, clf, FEATURE_COLUMNS)
	print("# negative points:", len(negative_points))

	### Initialize FACE object
	distrib = distribution.distribution(data)
	kernel = Kernel_obj(distrib)
	dist_obj = distance_obj()
	face = FACE(data, distrib, dist_obj, kernel, FEATURE_COLUMNS, TARGET_COLUMNS, epsilon, clf)
	feasibility_constraints = utils.getFeasibilityConstraints(FEATURE_COLUMNS, dataset_name=datasetName)
	face.make_graph(feasibility_constraints, epsilon)

	recourse_points = {}
	for n_id, n in enumerate(negative_points):
		# if (n_id > 2):
		# 	break
		print("Computing recourse for: {}/{}".format(n_id, len(negative_points)))
		recourse_point, cost, recourse_path = face.compute_recourse(n, tp, td)

		recourse_points[n_id] = {}
		recourse_points[n_id]['name'] = n
		recourse_points[n_id]['factual_instance'] = negative_points[n]
		recourse_points[n_id]['counterfactual_target'] = recourse_point
		recourse_points[n_id]['cost'] = cost
		recourse_points[n_id]['path'] = recourse_path

	# print(recourse_points)
	pk.dump(clf, open("./tmp/LR_classifier_face.pk", 'wb'))
	pk.dump(recourse_points, open("./tmp/Face_recourse_points.pk", 'wb'))
	return recourse_points

def unit_test():
	dataPath = "./data/synthetic_face_dataset.pk"
	FEATURE_COLUMNS = ['x1', 'x2']
	TARGET_COLUMNS = ['y']
	epsilon = 0.05 # margin for creating connections in graphs
	data = pk.load(open(dataPath, 'rb'))
	clf = LogisticRegression(random_state=utils.random_seed)
	distrib = distribution.distribution(data)
	kernel = Kernel_obj(distrib)
	dist_obj = distance_obj()
	face = FACE(data, distrib, dist_obj, kernel, FEATURE_COLUMNS, TARGET_COLUMNS, epsilon, clf)
	face.unit_test_djikstra()

if __name__=="__main__":
	# main()
	unit_test()

