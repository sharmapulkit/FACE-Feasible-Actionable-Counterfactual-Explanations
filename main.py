#!/bin/sh

import sys
import numpy as np
import pandas as pd
import pickle as pk
import random

from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from scipy.special import gamma as gamma_function
from sklearn import preprocessing

import utils
import distribution
from FACE import *
from kernel import *
from dataLoader import *

EPSILON = 1e-8

class distance_obj:
	def __init__(self):
		return

	def computeDistance(self, xi, xj):
		dist = np.linalg.norm(xi - xj, 2)
		return dist

def plot_recourse(data, face_recourse, plot_idx=0):
	all_pos = data[data.y == 1]
	all_pos_x1 = all_pos.x1.values
	all_pos_x2 = all_pos.x2.values

	all_neg = data[data.y == 0]
	all_neg_x1 = all_neg.x1.values
	all_neg_x2 = all_neg.x2.values

	plt.plot(all_pos_x1, all_pos_x2, '*')
	plt.plot(all_neg_x1, all_neg_x2, '*')
	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.title('FACE Synthetic data set')

	assert(plot_idx < len(data))

	plot_pt = face_recourse[plot_idx]['factual_instance']
	plot_cfpt = face_recourse[plot_idx]['counterfactual_target']
	points_x1 = []
	points_x2 = []
	points_x1 = [data.iloc[x]['x1'] for x in face_recourse[plot_idx]['path']]
	points_x2 = [data.iloc[x]['x2'] for x in face_recourse[plot_idx]['path']]
	plt.plot(points_x1, points_x2, color='green')
	plt.plot(plot_pt['x1'], plot_pt['x2'], 'o',color='red')
	plt.plot(plot_cfpt['x1'], plot_cfpt['x2'], 'o', color='red')
	plt.savefig('./tmp/recourse_path_{}.jpg'.format(plot_idx))

def data_statistics(data, distance, datasetName):
	print("Computing Data Statistics ...")

	dists = []
	N = len(data)
	for i in range(0, N):
		for j in range(i+1, N):
			xi = data.iloc[i]
			xj = data.iloc[j]
			dist = distance.computeDistance(xi, xj)
			dists.append(dist)

	plt.hist(dists, bins='auto')
	plt.title("Distribution of distances in the graph for given dataset")
	plt.xlabel("distance")
	plt.ylabel("number of points")
	plt.savefig("data_statistics_{}.png".format(datasetName))
	
	## Calculate important statistics
	mean = np.mean(dists)
	var = np.var(dists)
	p10 = np.percentile(dists, 10)
	p25 = np.percentile(dists, 25)
	p50 = np.percentile(dists, 50)
	p75 = np.percentile(dists, 75)
	min_ = np.min(dists)
	max_ = np.max(dists)

	print("Statistics:\n mean: {}\n var: {}\n p10: {}\n p25: {}\n p50: {}\n p75: {}\n min: {}\n max: {}".format(mean, var, p10, p25, p50, p75, min_, max_))

def main_synthetic_face(epsilon=0.2, tp=0.6, td=0.001):
	"""
	tp = 0.6 # Prediction threshold
	td = 0.001 # density threshold
	epsilon = 0.18 # margin for creating connections in graphs
	"""
	# dataPath = "./data/synthetic_one_hot"
	# datasetName = 'synthetic_lin'
	# FEATURE_COLUMNS = ['x1', 'x2', 'x3']
	
	datasetName = 'synthetic_one_hot'
	data, FEATURE_COLUMNS, TARGET_COLUMNS = load_synthetic_one_hot()
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
	kernel = Kernel_obj(distrib, Num_points=len(data), knnK=5)
	kernel.fitKernel(X)

	dist_obj = distance_obj()
	face = FACE(data, distrib, dist_obj, kernel, FEATURE_COLUMNS, TARGET_COLUMNS, epsilon, clf)
	feasibility_constraints = utils.getFeasibilityConstraints(FEATURE_COLUMNS, dataset_name=datasetName)
	face.make_graph(feasibility_constraints, epsilon)

	recourse_points = {}
	path_lengths = []
	for n_id, n in enumerate(negative_points):
		# if (n_id > 250):
		# 	break
		print("Computing recourse for: {}/{}".format(n_id, len(negative_points)))
		recourse_point, cost, recourse_path = face.compute_recourse(n, tp, td)

		recourse_points[n_id] = {}
		recourse_points[n_id]['name'] = n
		recourse_points[n_id]['factual_instance'] = negative_points[n]
		recourse_points[n_id]['counterfactual_target'] = recourse_point
		recourse_points[n_id]['cost'] = cost
		recourse_points[n_id]['path'] = recourse_path
		if (recourse_path is not None):
			path_lengths.append(len(recourse_path))

	# print(recourse_points)
	pk.dump(clf, open("./tmp/LR_classifier_face_{}_KDEkernel_eps{}_td{}.pk".format(datasetName, epsilon, td), 'wb'))
	pk.dump(recourse_points, open("./tmp/Face_recourse_points_KDEkernel_{}_eps{}_td{}.pk".format(datasetName, epsilon, td), 'wb'))
	print("Mean Path length:", np.mean(path_lengths))
	print("Median Path Length:", np.median(path_lengths))

	### Plot recourse for 10th data point
	# plot_recourse(data, recourse_points, 10)
	return recourse_points, np.median(path_lengths)

def cross_validate_path_length(datasetName='synthetic_face'):
	loop_name = 'tp'
	path_lengths = {}
	path_scores = {}
	if (loop_name == 'td'):
		td_range = np.arange(0.15, 0.42, 0.05)
		for _td in td_range:
			print("td:", _td)
			r_points, m_path_len = main(epsilon=0.2, td=_td, datasetName=datasetName)
			path_lengths[_td] = np.mean(m_path_len)
			path_scores[_td] = np.mean(m_path_len)*len(m_path_len)/len(r_points)
	if (loop_name == 'tp'):
		tp_range = np.arange(0.55, 0.76, 0.05)
		for _tp in tp_range:
			print("tp:", _tp)
			r_points, m_path_len = main(epsilon=0.2, td=_tp, datasetName=datasetName)
			path_lengths[_tp] = np.mean(m_path_len)
			path_scores[_tp] = np.mean(m_path_len)*len(m_path_len)/len(r_points)
	if (loop_name == 'eps'):
		# eps_range = [0.18, 0.22, 0.26, 0.30, 0.34, 0.38]
		eps_range = [0.4, 0.45, 0.5]
		for eps in eps_range:
			print("eps:", eps)
			r_points, m_path_len = main(epsilon=eps, datasetName=datasetName)
			path_lengths[eps] = np.mean(m_path_len)
			path_scores[eps] = np.mean(m_path_len)*len(m_path_len)/len(r_points)

	
	pk.dump(path_scores, open("tmp/path_scores_variation_{}_{}.pk".format(loop_name, datasetName), "wb"))
	print("Path lengths:", path_lengths)
	print("PathLength*fracRecourseProvided:", path_scores)
	lists = path_scores.items()
	x, y = zip(*lists)
	plt.plot(x, y)
	plt.xlabel(loop_name)
	plt.ylabel("Recourse Path Scores")
	plt.title("PathLength*fracRecourseProvided variation with {}".format(loop_name))
	plt.savefig("Path_Score_variation_{}_{}.jpg".format(loop_name, datasetName))

def main(epsilon=0.2, tp=0.6, td=0.001, datasetName='german_credit', expIter='0.0'):
	"""
	tp = 0.6 # Prediction threshold
	td = 0.0001 # density threshold
	epsilon = 0.3 # margin for creating connections in graphs
	# dataPath = "./data/synthetic_one_hot"
	# datasetName = 'synthetic_lin'
	# FEATURE_COLUMNS = ['x1', 'x2', 'x3']
	"""

	data, FEATURE_COLUMNS, TARGET_COLUMNS = load_dataset(datasetName=datasetName)
	TEST_SIZE = 0.3

	X = data[FEATURE_COLUMNS]
	y = data[TARGET_COLUMNS]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=utils.random_seed, shuffle=True)
	data_train = pd.concat([y_train, X_train], axis=1)
	data_test = pd.concat([y_test, X_test], axis=1)
	print("Data train:", data_train.shape)
	print("Data Train columns:", data_train.columns)

	### Train a logistic regression model
	clf = LogisticRegression(random_state=utils.random_seed)
	clf.fit(X_train, y_train)
	print("Training accuracy:", clf.score(X_train, y_train))
	print("Testing accuracy:", clf.score(X_test, y_test))

	### Get the negatively classified points
	negative_points = utils.get_negatively_classified(data_test, clf, FEATURE_COLUMNS)
	print("# negative points:", len(negative_points))

	### Initialize FACE object
	dist_obj = distance_obj()
	# data_statistics(X, dist_obj, datasetName=datasetName) ## Get data statistics
	# distrib = distribution.distribution(data)
	distrib = distribution.kernel_distribution(data)
	# distrib = distribution.synthetic_distribution_face(data)
	# print("PDF:", distrib.pdf(data.iloc[0][FEATURE_COLUMNS], data.iloc[0][TARGET_COLUMNS]))
	kernel = Kernel_obj(distrib, Num_points=len(data))	
	kernel.fitKernel(X)
	distrib.setKernel(kernel)

	face = FACE(data, distrib, dist_obj, kernel, FEATURE_COLUMNS, TARGET_COLUMNS, epsilon, clf)
	feasibility_constraints = utils.getFeasibilityConstraints(FEATURE_COLUMNS, dataset_name=datasetName)
	face.make_graph(feasibility_constraints, epsilon)

	recourse_points = {}
	path_lengths = []
	for n_id, n in enumerate(negative_points):
		print("Computing recourse for: {}/{}".format(n_id, len(negative_points)))
		recourse_point, cost, recourse_path = face.compute_recourse(n, tp, td)

		recourse_points[n_id] = {}
		recourse_points[n_id]['name'] = n
		recourse_points[n_id]['factual_instance'] = negative_points[n]
		recourse_points[n_id]['counterfactual_target'] = recourse_point
		recourse_points[n_id]['cost'] = cost
		recourse_points[n_id]['path'] = recourse_path
		if (recourse_path is not None):
			path_lengths.append(len(recourse_path))

	# print(recourse_points)
	pk.dump(clf, open("./tmp/LR_classifier_face_data{}_eps{}_tp{}_td{}_expIter{}.pk".format(datasetName, epsilon, tp, td, expIter), 'wb'))
	pk.dump(recourse_points, open("./tmp/Face_recourse_points_{}_eps{}_tp{}_td{}_expIter{}.pk".format(datasetName, epsilon, tp, td, expIter), 'wb'))

	print("Mean Path length:", np.mean(path_lengths))
	print("Recourse Found:", len(path_lengths)/len(negative_points))
	print("Median Path Length:", np.median(path_lengths))

	return recourse_points, path_lengths

def main_train_test(epsilon=0.2, tp=0.6, td=0.001, datasetName='german_credit', expIter='0.0'):
	"""
	tp = 0.6 # Prediction threshold
	td = 0.0001 # density threshold
	epsilon = 0.3 # margin for creating connections in graphs
	# dataPath = "./data/synthetic_one_hot"
	# datasetName = 'synthetic_lin'
	# FEATURE_COLUMNS = ['x1', 'x2', 'x3']
	"""

	data, data_recourse_test, FEATURE_COLUMNS, TARGET_COLUMNS = load_german_synthetic_sampled_dataset()
	TEST_SIZE = 0.8

	X = data[FEATURE_COLUMNS]
	y = data[TARGET_COLUMNS]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=utils.random_seed, shuffle=True)
	data_train = pd.concat([y_train, X_train], axis=1)
	data_test = pd.concat([y_test, X_test], axis=1)
	print("Data train:", data_train.shape)
	print("Data Train columns:", data_train.columns)

	### Train a logistic regression model
	clf = LogisticRegression(random_state=utils.random_seed)
	clf.fit(X_train, y_train)
	print("Training accuracy:", clf.score(X_train, y_train))
	print("Testing accuracy:", clf.score(X_test, y_test))

	### Get the negatively classified points
	negative_points = utils.get_negatively_classified(data_recourse_test, clf, FEATURE_COLUMNS)
	print("# negative points:", len(negative_points) , "/", len(data_recourse_test))

	### Initialize FACE object
	dist_obj = distance_obj()
	# data_statistics(X, dist_obj, datasetName=datasetName) ## Get data statistics
	# distrib = distribution.distribution(data)
	distrib = distribution.kernel_distribution(data)
	# distrib = distribution.synthetic_distribution_face(data)
	# print("PDF:", distrib.pdf(data.iloc[0][FEATURE_COLUMNS], data.iloc[0][TARGET_COLUMNS]))
	kernel = Kernel_obj(distrib, Num_points=len(data))
	kernel.fitKernel(X)
	distrib.setKernel(kernel)

	face = FACE(data, distrib, dist_obj, kernel, FEATURE_COLUMNS, TARGET_COLUMNS, epsilon, clf)
	feasibility_constraints = utils.getFeasibilityConstraints(FEATURE_COLUMNS, dataset_name=datasetName)
	face.make_graph(feasibility_constraints, epsilon)

	recourse_points = {}
	path_lengths = []
	for n_id, n in enumerate(negative_points):
		print("Computing recourse for: {}/{}".format(n_id, len(negative_points)))
		recourse_point, cost, recourse_path = face.compute_recourse(n, tp, td)

		recourse_points[n_id] = {}
		recourse_points[n_id]['name'] = n
		recourse_points[n_id]['factual_instance'] = negative_points[n]
		recourse_points[n_id]['counterfactual_target'] = recourse_point
		recourse_points[n_id]['cost'] = cost
		recourse_points[n_id]['path'] = recourse_path
		if (recourse_path is not None):
			path_lengths.append(len(recourse_path))

	# print(recourse_points)
	pk.dump(clf, open("./tmp/LR_classifier_face_data{}_eps{}_tp{}_td{}_expIter{}.pk".format(datasetName, epsilon, tp, td, expIter), 'wb'))
	pk.dump(recourse_points, open("./tmp/Face_recourse_points_{}_eps{}_tp{}_td{}_expIter{}.pk".format(datasetName, epsilon, tp, td, expIter), 'wb'))

	print("Mean Path length:", np.mean(path_lengths))
	print("Recourse Found:", len(path_lengths)/len(negative_points))
	print("Median Path Length:", np.median(path_lengths))

	return recourse_points, path_lengths

def unit_test():
	dataPath = "./data/synthetic_face_dataset.pk"
	Num_Features = 2
	FEATURE_COLUMNS = [('x' + str(i+1)) for i in range(0, len(Num_Features))]
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
	# main_train_test(epsilon=0.35, td=0.001, tp=0.55, datasetName='synthetic_german_one_hot_sampled', expIter='4.1.3')
	main_synthetic_face(epsilon=0.2, datasetName='synthetic_face')
	# cross_validate_path_length(datasetName='synthetic_german_one_hot')
	# cross_validate_path_length(datasetName='synthetic_face')
	# unit_test()

