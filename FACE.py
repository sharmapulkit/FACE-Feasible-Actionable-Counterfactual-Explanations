import numpy as np
import pandas as pd

import utils

random_seed = 482
np.random.seed(random_seed)

class FACE:
	def __init__(self, data, density_method, distance_obj, kernel_obj, feature_columns, target_column, epsilon, clf=None):
		self._data = data
		self._density = density_method
		self._distance = distance_obj
		self._epsilon = epsilon
		self._clf = clf
		self._kernel_obj = kernel_obj
		self.set_feature_names(feature_columns, target_column)

		self._Graph = None

	def set_feature_names(self, features, target):
		self.FEATURE_COLUMNS = features
		self.TARGET_COLUMN = target

	def getDistance(self, xi, xj):
		self._distance.computeDistance(xi, xj)

	def train_classifier(self):	
		if (self._clf is None):
			self._clf = LogisticRegression(random_state=random_seed)

		X = self._data[self.FEATURE_COLUMNS]
		y = self._data[self.TARGET_COLUMN]
		self._clf.fit(X, y)
		print("Trained classifier")


	# def shortestPath(self, source, target):
	# 	"""
	# 	Djikstra to find the shortest path in Graph
	# 	Returns shortest_path, path_cost
	# 	"""
	# 	path = [source]
	# 	path_cost = 0
	# 	current = source
	# 	## while current is not target
	# 	while (current is not target):
	# 		## update distances of all neighbors
	# 		distances = {}
	# 		for elem_id, elem in enumerate(self._Graph[current]):
	# 			if (elem > 0):
	# 				distances[elem_id] = elem

	# 		minimum_cost = float('inf')
	# 		closest = -1
	# 		## Pick the closest neighbor
	# 		for key in distances:
	# 			if (distances[key] < minimum_cost):
	# 				minimum_cost = distances[key]
	# 				closest = key

	# 		if (closest == -1):
	# 			return []

	# 		path.append(closest)
	# 		path_cost += minimum_cost
	# 		current = closest

	# 	return path, path_cost

	def shortestPath(self, source, target):
		"""
		Djikstra to find the shortest path in Graph
		source: source_id
		target: target_id
		Returns shortest_path, path_cost
		"""
		path = [source]
		path_cost = 0
		current = source
		visited = []
		## while current is not target
		while (current is not target):
			## update distances of all neighbors
			visited.append(current)
			distances = self._Graph[current]
			minimum_cost = float('inf')
			closest = -1

			## Pick the closest neighbor
			for key in distances:
				if ((distances[key] < minimum_cost) and not (key in visited)):
					minimum_cost = distances[key]
					closest = key

			if (closest == -1):
				return [], -1

			path.append(closest)
			path_cost += minimum_cost
			current = closest
			# print("Current:", current)

		return path, path_cost

	def make_graph(self, feasibilitySet, epsilon):
		"""
		Make the graph using given data points
		"""
		X = self._data[self.FEATURE_COLUMNS]
		# self._Graph = utils.make_graph(X, self._density, self._distance, self._kernel_obj, feasibilitySet, epsilon) ## Adjacency matrix representation of graph
		self._Graph = utils.make_graph_adjList(X, self._density, self._distance, self._kernel_obj, feasibilitySet, epsilon) ## Adjacency matrix representation of graph

	def get_candidates(self, tp, td):
		"""
		Returns a dictionary of idices of candidate points for recourse
		"""
		candidates = {}
		for x_id, x in self._data[self.FEATURE_COLUMNS].iterrows():
			# print(self._clf.predict_log_proba([x])[0][1])
			if (self._clf.predict_log_proba([x])[0][1] > np.log(tp) and self._density.pdf(x) > td):
				candidates[x_id] = x

		return candidates

	def compute_recourse(self, source, tp, td):
		"""	
		source: Source point id
		X: dataset - pd dataframe
		density: probability density function
		distance: distance function between 2 points
		tp: threshold for classifier prediction probability
		td: threshold for pdf
		epislon: constant for distance threshold
		c: constrains for feasibility
		clf: classifier sklearn object
		"""		
		assert (self._Graph is not None)

		I = self.get_candidates(tp, td) ### Indices of candidates
		
		min_path_cost = float('inf')
		min_target = -1
		min_path = None
		for candidate_id in I:
			candidate = I[candidate_id]
			closest_target_path, path_cost = self.shortestPath(source, candidate_id)
			if (path_cost == -1):
				continue
			if (path_cost < min_path_cost):
				min_target = candidate
				min_path_cost = path_cost
				min_path = closest_target_path

		return min_target, min_path_cost, min_path

	#### Unit tests
	def unit_test_djikstra(self):
		G = {0: {1: 3, 2:4}, 1: {0: 3, 2: 4}, 2: {0:4, 1: 4, 3: 3}, 3: {2: 3, 4: 4, 5: 1}, 4:{3: 4, 5: 1}, 5:{3:1, 4:1}}
		self._Graph = G
		path, path_cost = self.shortestPath(0, 5)
		print("Path:", path, " Cost:", path_cost)
		assert(path == [0, 1, 2, 3, 5])
		# assert(path_cost == 7)
		print("unit test djikstra passed...")

