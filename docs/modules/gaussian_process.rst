

.. _gaussian_process:

==================
Gaussian Processes
==================

.. currentmodule:: sklearn.gaussian_process

**Gaussian Processes (GP)** are a generic supervised learning method designed
to solve *regression* and *probabilistic classification* problems.

.. _gpr:

Gaussian Process Regression (GPR)
=================================

.. currentmodule:: sklearn.gaussian_process

The :class:`GaussianProcessRegressor` implements Gaussian processes (GP) for
regression purposes. For this, the prior of the GP needs to be specified. The
prior mean is assumed to be constant and zero (for ``normalize_y=False``) or the
training data's mean (for ``normalize_y=True``). The prior's
covariance is specified by a passing a :ref:`kernel <gp_kernels>` object.

.. _gp_kernels:

Kernels for Gaussian Processes
==============================
.. currentmodule:: sklearn.gaussian_process.kernels

Kernels (also called "covariance functions" in the context of GPs) are a crucial
ingredient of GPs which determine the shape of prior and posterior of the GP.
They encode the assumptions on the function being learned by defining the "similarity"
of two data points combined with the assumption that similar data points should
have similar target values. Two categories of kernels can be distinguished:
stationary kernels depend only on the distance of two data points and not on their
absolute values :math:`k(x_i, x_j)= k(d(x_i, x_j))` and are thus invariant to
translations in the input space, while non-stationary kernels
depend also on the specific values of the data points. Stationary kernels can further
be subdivided into isotropic and anisotropic kernels, where isotropic kernels are
also invariant to rotations in the input space. For more details, we refer to
Chapter 4 of [RW2006]_.


References
----------

.. [RW2006] Carl Eduard Rasmussen and Christopher K.I. Williams, "Gaussian Processes for Machine Learning", MIT Press 2006, Link to an official complete PDF version of the book `here <http://www.gaussianprocess.org/gpml/chapters/RW.pdf>`_ .

.. currentmodule:: sklearn.gaussian_process
