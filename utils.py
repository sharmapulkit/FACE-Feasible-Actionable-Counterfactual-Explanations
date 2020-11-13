import sys
import numpy as np
import pandas as pd
import pickle as pk

sys.path.append('../recourse')

import Feasibility

random_seed = 482

### KDE kernel
class kernel:
	def __init__(self):
		pass

	def kernelKDE(xi, xj, density):
		"""
		KDE kernel
		"""
		mean = 0.5*(xi + xj)
		dist = np.linalg.norm(xi - xj, 2)
		density_at_mean = density(mean)
		return density_at_mean*dist

def make_graph(X, density, distance, kernel, feasibility_constraints, epsilon):
	N = len(X)
	G = np.zeros((N, N)) ### Adjacency Graph
	for i in range(0, N):
		for j in range(i+1, N):
			xi = X.iloc[i]
			xj = X.iloc[j]
			if not ( (distance.computeDistance(xi, xj) > epsilon) and (feasibility_constraints.check_constraints(xi, xj) is False) ):
				wij = distance.computeDistance(xi, xj)*kernel.kernelKDE(xi, xj)
				G[i][j] = wij # Add edge to the graph
				G[j][i] = wij # Add edge to the graph
	return G

def make_graph_adjList(X, density, distance, kernel, feasibility_constraints, epsilon):
	N = len(X)
	G = {}
	for i in range(0, N):
		for j in range(i+1, N):
			xi = X.iloc[i]
			xj = X.iloc[j]
			if not ( (distance.computeDistance(xi, xj) > epsilon) and (feasibility_constraints.check_constraints(xi, xj) is False) ):
				wij = distance.computeDistance(xi, xj)*kernel.kernelKDE(xi, xj)
				if (i in G):
					G[i][j] = wij # Add edge to the graph
				else:
					G[i] = {j: wij}

				if (j in G):
					G[j][i] = wij # Add edge to the graph
				else:
					G[j] = {i: wij}
	return G

def load_data(dataPath):
	with open(dataPath, 'rb') as f:
		data = pk.load(f)
	return data

def get_negatively_classified(data, clf, FEATURE_COLUMNS):
	"""
	Get a dict of negatively classified data points
	data: pandas df
	"""
	negatives = {} # Id -> data point (y, X)
	for row_id, row in data.iterrows():
		if (clf.predict([row[FEATURE_COLUMNS]]) == 0):
			negatives[row_id] = row

	return negatives


def density_function(xi, X):
	### Uniform density
	return 1/len(X)

def getFeasibilityConstraints(FEATURE_COLUMNS, dataset_name):
	if (dataset_name == 'synthetic_lin'):
		## Initialize feasibility constraints
		feasibility_consts = Feasibility.feasibility_consts(FEATURE_COLUMNS)

		feasibility_consts.feasibility_set['x1'].mutability = False
		feasibility_consts.feasibility_set['x2'].step_direction = 1

		return feasibility_consts
	else:
		print("Unknown dataset. Initializing with no constraints")
		feasibility_consts = Feasibility.feasibility_consts(FEATURE_COLUMNS)

		return feasibility_consts


