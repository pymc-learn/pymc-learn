.. _mixture:

.. _gmm:

=======================
Gaussian mixture models
=======================

.. currentmodule:: pmlearn.mixture

``pmlearn.mixture`` is a package which enables one to learn
Gaussian Mixture Models.

A Gaussian mixture model is a probabilistic model that assumes all the
data points are generated from a mixture of a finite number of
Gaussian distributions with unknown parameters.

pymc-learn implements different classes to estimate Gaussian
mixture models, that correspond to different estimation strategies,
detailed below.

Gaussian Mixture
================

A :meth:`GaussianMixture.fit` method is provided that learns a Gaussian
Mixture Model from train data. Given test data, it can assign to each
sample the Gaussian it mostly probably belong to using
the :meth:`GaussianMixture.predict` method.

..
    Alternatively, the probability of each
    sample belonging to the various Gaussians may be retrieved using the
    :meth:`GaussianMixture.predict_proba` method.

.. _dirichlet_process:

The Dirichlet Process
======================

Here we describe variational inference algorithms on Dirichlet process
mixture. The Dirichlet process is a prior probability distribution on
*clusterings with an infinite, unbounded, number of partitions*.
Variational techniques let us incorporate this prior structure on
Gaussian mixture models at almost no penalty in inference time, comparing
with a finite Gaussian mixture model.