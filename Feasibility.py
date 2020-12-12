import numpy as np
import pandas as pd

class constraintProps:
	def __init__(self, mutable=True, step_direction=0):
		self._mutable = mutable
		self._step_direction = step_direction

	@property
	def mutable(self):
		return self._mutable

	@mutable.setter
	def mutable(self, v):
		self._mutable = v

	@property
	def step_direction(self):
		return self._step_direction

	@step_direction.setter
	def step_direction(self, v):
		self._step_direction = v

	def set_mutability(self, m):
		self._mutable = m

	def set_direction(self, s):
		self._step_direction = s


class feasibility_consts:
	def __init__(self, FEATURE_COLUMNS):
		self._FEATURE_COLUMNS = FEATURE_COLUMNS
		self._feasibility_set = {}
		# for feat in FEATURE_COLUMNS:
		# 	self._feasibility_set[feat] = constraintProps()

	@property
	def feasibility_set(self):
		return self._feasibility_set

	def set_constraint(self, feat, mutability=True, step_direction=0):
		self._feasibility_set[feat] = constraintProps(mutability, step_direction)

	def check_constraints(self, source, dest):
		if (len(self._feasibility_set) == 0):
			return True

		delta = dest - source
		for feat in self._FEATURE_COLUMNS:
			if feat in self._feasibility_set:
				if ((delta[feat] != 0) and (self._feasibility_set[feat]._mutable is False)):
					return False
				if (delta[feat]*self._feasibility_set[feat]._step_direction < 0):
					return False

		return	True


