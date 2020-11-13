import numpy as np

class distribution():
	def __init__(self, data):
		self._data = data

	def pdf(self, xi):
		return 1/len(self._data)
