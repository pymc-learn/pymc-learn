# -*- coding: utf-8 -*-

# Author: Daniel Emaasit <daniel.emaasit@gmail.com>
#         (mostly translation, see implementation details)
# License: BSD 3 clause

"""
The :mod:`pmlearn.linear_model` module implements generalized linear models.
It includes Bayesian Regression, Bayesian Logistic Regression, Hierarchical
Logistic Regression.
"""

from .base import LinearRegression
from .logistic import (HierarchicalLogisticRegression,
                       LogisticRegression)


__all__ = ['LinearRegression',
           'HierarchicalLogisticRegression',
           'LogisticRegression']
