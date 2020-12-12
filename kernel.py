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

import utils
import distribution
from FACE import *

EPSILON = 1e-8

### KDE kernel
class Kernel_obj:
	def __init__(self, distrib, bandwidth=0.5, Num_points=1, knnK=1):
		self._distribution = distrib
		self._kernel = KernelDensity(bandwidth, kernel='tophat')
		self.Num_points = Num_points
		self.knnK = knnK
		self.volume = None

	def fitKernel(self, X):
		print("Fitting kernel...")
		self._kernel.fit(X)

	def kernelUniform(self, xi, xj):
		"""
		KDE kernel value
		"""
		mean = 0.5*(xi + xj)
		dist = np.linalg.norm(xi - xj, 2)
		density_at_mean = self._distribution.pdf(mean)
		return density_at_mean*dist

	def kernelKDE(self, xi, xj):
		"""
		KDE kernel values
		"""
		mean = 0.5*(xi + xj)
		dist = np.linalg.norm(xi - xj, 2)
		density_at_mean = np.exp(self._kernel.score_samples([mean]))
		return (1/(density_at_mean + EPSILON))*dist
		# return density_at_mean*dist

	def kernelKNN(self, xi, xj):
		"""
		kNN kernel
		"""
		dim = len(xi)
		if self.volume is None:
			self.volume = np.pi**(dim//2) / gamma_function(dim//2 + 1)
		dist = np.linalg.norm(xi - xj, 2)
		density_at_mean = self.knnK/(self.Num_points*self.volume)*dist
		return density_at_mean


class distance_obj:
	def __init__(self):
		return

	def computeDistance(self, xi, xj):
		dist = np.linalg.norm(xi - xj, 2)
		return dist
