# -*- coding: utf-8 -*-

# Author: Daniel Emaasit <daniel.emaasit@gmail.com>
#         (mostly translation, see implementation details)
# License: BSD 3 clause

"""
The :mod:`pmlearn.naive_bayes` module implements Naive Bayes algorithms. These
are supervised learning methods based on applying Bayes' theorem with strong
(naive) feature independence assumptions.
"""

from .naive_bayes import GaussianNB


__all__ = ['GaussianNB']
