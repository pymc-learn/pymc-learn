.. _naive_bayes:

===========
Naive Bayes
===========

.. currentmodule:: pmlearn.naive_bayes


Naive Bayes methods are a set of supervised learning algorithms
based on applying Bayes' theorem with the "naive" assumption of
conditional independence between every pair of features given the
value of the class variable. Bayes' theorem states the following
relationship, given class variable :math:`y` and dependent feature
vector :math:`x_1` through :math:`x_n`, :

.. math::

   P(y \mid x_1, \dots, x_n) = \frac{P(y) P(x_1, \dots x_n \mid y)}
                                    {P(x_1, \dots, x_n)}

Using the naive conditional independence assumption that

.. math::

   P(x_i | y, x_1, \dots, x_{i-1}, x_{i+1}, \dots, x_n) = P(x_i | y),

for all :math:`i`, this relationship is simplified to

.. math::

   P(y \mid x_1, \dots, x_n) = \frac{P(y) \prod_{i=1}^{n} P(x_i \mid y)}
                                    {P(x_1, \dots, x_n)}

Since :math:`P(x_1, \dots, x_n)` is constant given the input,
we can use the following classification rule:

.. math::

   P(y \mid x_1, \dots, x_n) \propto P(y) \prod_{i=1}^{n} P(x_i \mid y)

   \Downarrow

   \hat{y} = \arg\max_y P(y) \prod_{i=1}^{n} P(x_i \mid y),

and we can use Maximum A Posteriori (MAP) estimation to estimate
:math:`P(y)` and :math:`P(x_i \mid y)`;
the former is then the relative frequency of class :math:`y`
in the training set.

The different naive Bayes classifiers differ mainly by the assumptions they
make regarding the distribution of :math:`P(x_i \mid y)`.

In spite of their apparently over-simplified assumptions, naive Bayes
classifiers have worked quite well in many real-world situations, famously
document classification and spam filtering. They require a small amount
of training data to estimate the necessary parameters. (For theoretical
reasons why naive Bayes works well, and on which types of data it does, see
the references below.)

Naive Bayes learners and classifiers can be extremely fast compared to more
sophisticated methods.
The decoupling of the class conditional feature distributions means that each
distribution can be independently estimated as a one dimensional distribution.
This in turn helps to alleviate problems stemming from the curse of
dimensionality.

On the flip side, although naive Bayes is known as a decent classifier,
it is known to be a bad estimator, so the probability outputs from
``predict_proba`` are not to be taken too seriously.

.. topic:: References:

 * H. Zhang (2004). `The optimality of Naive Bayes.
   <http://www.cs.unb.ca/~hzhang/publications/FLAIRS04ZhangH.pdf>`_
   Proc. FLAIRS.

.. _gaussian_naive_bayes:

Gaussian Naive Bayes
--------------------

:class:`GaussianNB` implements the Gaussian Naive Bayes algorithm for
classification. The likelihood of the features is assumed to be Gaussian:

.. math::

   P(x_i \mid y) &= \frac{1}{\sqrt{2\pi\sigma^2_y}} \exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma^2_y}\right)

The parameters :math:`\sigma_y` and :math:`\mu_y`
are estimated using maximum likelihood.

    >>> from sklearn import datasets
    >>> iris = datasets.load_iris()
    >>> from pmlearn.naive_bayes import GaussianNB
    >>> gnb = GaussianNB()
    >>> y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
    >>> print("Number of mislabeled points out of a total %d points : %d"
    ...       % (iris.data.shape[0],(iris.target != y_pred).sum()))
    Number of mislabeled points out of a total 150 points : 6

