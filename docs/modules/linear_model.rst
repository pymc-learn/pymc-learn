.. _linear_model:

=========================
Generalized Linear Models
=========================

.. currentmodule:: pmlearn.linear_model

The following are a set of methods intended for regression in which
the target value is expected to be a linear combination of the input
variables. In mathematical notion, if :math:`\hat{y}` is the predicted
value.

.. math::    \hat{y}(\beta, x) = \beta_0 + \beta_1 x_1 + ... + \beta_p x_p

Where :math:`\beta = (\beta_1,
..., \beta_p)` are the coefficients and :math:`\beta_0` is the y-intercept.

To perform classification with generalized linear models, see
:ref:`bayesian_logistic_regression`.

.. _bayesian_linear_regression:

Bayesian Linear Regression
==========================

To obtain a fully probabilistic model, the output :math:`y` is assumed
to be Gaussian distributed around :math:`X w`:

.. math::  p(y|X,w,\alpha) = \mathcal{N}(y|X w,\alpha)

Alpha is again treated as a random variable that is to be estimated from the
data.

.. topic:: References

 * A good introduction to Bayesian methods is given in C. Bishop: Pattern
   Recognition and Machine learning

 * Original Algorithm is detailed in the  book `Bayesian learning for neural
   networks` by Radford M. Neal

.. _bayesian_logistic_regression:

Bayesian Logistic regression
==============================

Bayesian Logistic regression, despite its name, is a linear model for
classification rather than regression. Logistic regression is also
known in the literature as logit regression, maximum-entropy classification (MaxEnt)
or the log-linear classifier. In this model, the probabilities describing the
possible outcomes of a single trial are modeled
using a `logistic function <https://en.wikipedia.org/wiki/Logistic_function>`_.

The implementation of logistic regression in pymc-learn can be accessed from
class :class:`LogisticRegression`.