import numpy as np
import scipy

class distribution():
	def __init__(self, data):
		self._data = data

	def pdf(self, xi, yi=None):
		return 1/len(self._data)

class kernel_distribution(distribution):
	def __init__(self, data):
		super().__init__(data)
		self.kernel = None

	def setKernel(self, kernel):
		self.kernel = kernel

	def pdf(self, xi, yi=None):
		assert self.kernel is not None

		density_at_mean = np.exp(self.kernel._kernel.score_samples([xi]))
		return density_at_mean


class synthetic_distribution_face(distribution):
	def __init__(self, data):
		super().__init__(data)


	def pdf(self, xi, yi):
		yi = yi['y']
		if ((yi == 1) and xi['x2'] < 4):
			prob_x1 = 1/(5.5 - (-0.5))
			prob_x2 = scipy.stats.norm(0, 1.2/3).pdf(xi['x2']) 
		if ((yi == 1) and xi['x2'] >= 4):
			prob_x1 = scipy.stats.norm(3.2, 0.8).pdf(xi['x1'])
			prob_x2 = scipy.stats.norm(8, 0.4).pdf(xi['x2'])
		else:
			prob_x1 = scipy.stats.norm(0, 1.2/3).pdf(xi['x1'])
			prob_x2 = 1/11

		return prob_x1*prob_x2
