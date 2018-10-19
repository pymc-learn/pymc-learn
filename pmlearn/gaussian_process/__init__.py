# -*- coding: utf-8 -*-

# Author: Daniel Emaasit <daniel.emaasit@gmail.com>
#         (mostly translation, see implementation details)
# License: BSD 3 clause

"""
The :mod:`pmlearn.gaussian_process` module implements Gaussian Process
based regression and classification.
"""

from .gpr import GaussianProcessRegressor
from .gpr import StudentsTProcessRegressor
from .gpr import SparseGaussianProcessRegressor
# from .gpc import GaussianProcessClassifier
# from . import kernels


__all__ = ['GaussianProcessRegressor',
           'StudentsTProcessRegressor',
           'SparseGaussianProcessRegressor']