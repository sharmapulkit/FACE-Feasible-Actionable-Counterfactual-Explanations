import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde as kde
from scipy.interpolate import interp1d
from scipy.stats import percentileofscore
from dataLoader import *

class q_cost():
    def __init__(self, data):
        self._data = data
        self._data_values = data.values
        self.step_size = 0.01 #'relative'
        # self.generate_grid()
        return

    def generate_grid(self):
        """Generate grid of feasible values"""

        # end points
        self.lb = self._data.min(axis=0)
        self.ub = self._data.max(axis=0)
        step = self.step_size

        # if self._variable_type == int:
        start = np.floor(self.lb)
        stop = np.ceil(self.ub)

        # if self.step_type == 'relative':
        step = np.multiply(step, stop - start)
        print(np.arange(start.values, stop.values + step.values, step.values))

        # if self._variable_type == int:
        #     step = np.ceil(step)

        # generate grid
        # try:
        grid = np.arange(start, stop + step, step)
        # except Exception:
        #     ipsh()

        # cast grid
        # if self._variable_type == int:
        #     grid = grid.astype('int')

        self._grid = grid

    def interpolator(self, x, left_buffer = 1e-6, right_buffer = 1e-6):

        # check buffers
        left_buffer = float(left_buffer)
        right_buffer = float(right_buffer)
        assert 0.0 <= left_buffer < 1.0
        assert 0.0 <= right_buffer < 1.0
        assert left_buffer + right_buffer < 1.0

        # build kde estimator using observed values
        kde_estimator = kde(x)

        # build the CDF over the grid
        pdf = kde_estimator(self._grid)
        cdf_raw = np.cumsum(pdf)
        total = cdf_raw[-1] + left_buffer + right_buffer
        cdf = (left_buffer + cdf_raw) / total
        self._interpolator = interp1d(x = self._grid, y = cdf, copy = False, fill_value = (left_buffer, 1.0 - right_buffer), bounds_error = False, assume_sorted = True)

    def cost(self, x_source, x_dest):

        x_source = x_source.values
        x_dest = x_dest.values
        log_score = 0.0
        for i in range(0, 7):
            percentile_source = percentileofscore(self._data_values[:, i], x_source[i])/100
            percentile_dest = percentileofscore(self._data_values[:, i], x_dest[i])/100
            log_score += np.log((1 - percentile_dest)/(1 - percentile_source))

        return log_score

if __name__=="__main__":
    data, data_recourse_test, FEATURE_COLUMNS, TARGET_COLUMNS = load_german_synthetic_sampled_dataset()
    TEST_SIZE = 0.8

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMNS]

    cc = q_cost(data[FEATURE_COLUMNS])
    score = cc.cost(data[FEATURE_COLUMNS].iloc[0], data[FEATURE_COLUMNS].iloc[1])
    print(score)


