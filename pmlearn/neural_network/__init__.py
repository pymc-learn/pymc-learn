# -*- coding: utf-8 -*-

# Author: Daniel Emaasit <daniel.emaasit@gmail.com>
#         (mostly translation, see implementation details)
# License: BSD 3 clause

"""
The :mod:`pmlearn.neural_network` module includes models based on neural
networks.
"""

from .multilayer_perceptron import MLPClassifier

__all__ = ['MLPClassifier']
