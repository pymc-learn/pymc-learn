# -*- coding: utf-8 -*-

# Author: Daniel Emaasit <daniel.emaasit@gmail.com>
#         (mostly translation, see implementation details)
# License: BSD 3 clause

"""
The :mod:`pmlearn.mixture` module implements mixture models.
"""

from .gaussian_mixture import GaussianMixture
from .dirichlet_process import DirichletProcessMixture


__all__ = ['GaussianMixture',
           'DirichletProcessMixture']