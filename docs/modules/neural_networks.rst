.. _neural_networks_supervised:

==================================
Neural network models (supervised)
==================================

.. currentmodule:: pmlearn.neural_network


.. NOTE::

    Unlike scikit-learn, this implementation of neural networks in pymc-learn is
    intended for large-scale applications. Pymc-learn relies on Theano for GPU
    support.

    scikit-learn offers no GPU support.

.. _multilayer_perceptron:

Multi-layer Perceptron
======================

**Multi-layer Perceptron (MLP)** is a supervised learning algorithm that learns
a function :math:`f(\cdot): R^m \rightarrow R^o` by training on a dataset,
where :math:`m` is the number of dimensions for input and :math:`o` is the
number of dimensions for output. Given a set of features :math:`X = {x_1, x_2, ..., x_m}`
and a target :math:`y`, it can learn a non-linear function approximator for either
classification or regression.


Classification
==============

Class :class:`MLPClassifier` implements a multi-layer perceptron (MLP) algorithm
that trains using `Backpropagation <http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm>`_.

MLP trains on two arrays: array X of size (n_samples, n_features), which holds
the training samples represented as floating point feature vectors; and array
y of size (n_samples,), which holds the target values (class labels) for the
training samples::

    >>> from pmlearn.neural_network import MLPClassifier
    >>> X = [[0., 0.], [1., 1.]]
    >>> y = [0, 1]
    >>> clf = MLPClassifier()
    ...
    >>> clf.fit(X, y)                         # doctest: +NORMALIZE_WHITESPACE
    >>> clf.predict([[2., 2.], [-1., -2.]])
    array([1, 0])



.. topic:: References:

    * `"Learning representations by back-propagating errors."
      <http://www.iro.umontreal.ca/~pift6266/A06/refs/backprop_old.pdf>`_
      Rumelhart, David E., Geoffrey E. Hinton, and Ronald J. Williams.

    * `"Stochastic Gradient Descent" <http://leon.bottou.org/projects/sgd>`_ L. Bottou - Website, 2010.

    * `"Backpropagation" <http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm>`_
      Andrew Ng, Jiquan Ngiam, Chuan Yu Foo, Yifan Mai, Caroline Suen - Website, 2011.

    * `"Efficient BackProp" <http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf>`_
      Y. LeCun, L. Bottou, G. Orr, K. Müller - In Neural Networks: Tricks
      of the Trade 1998.

    *  `"Adam: A method for stochastic optimization."
       <http://arxiv.org/pdf/1412.6980v8.pdf>`_
       Kingma, Diederik, and Jimmy Ba. arXiv preprint arXiv:1412.6980 (2014).
