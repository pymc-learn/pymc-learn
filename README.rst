pymc-learn: Practical Probabilistic Machine Learning in Python
===============================================================

.. image:: https://github.com/pymc-learn/pymc-learn/blob/master/docs/logos/pymc-learn-logo.jpg?raw=true
    :width: 350px
    :alt: Pymc-Learn logo
    :align: center

|Travis| |Coverage| |Docs| |License| |Pypi| |Binder|

**Contents:**

    #. `Github repo`_
    #. `What is pymc-learn?`_
    #. `Quick Install`_
    #. `Quick Start`_
    #. `Index`_


.. _Github repo: https://github.com/pymc-learn/pymc-learn

----

What is pymc-learn?
------------------------

*pymc-learn is a library for practical probabilistic
machine learning in Python*.

It provides a variety of state-of-the art probabilistic models for supervised
and unsupervised machine learning. **It is inspired by**
`scikit-learn <http://scikit-learn.org>`_ **and focuses on bringing probabilistic
machine learning to non-specialists**. It uses a syntax that mimics scikit-learn.
Emphasis is put on ease of use, productivity, flexibility, performance,
documentation, and an API consistent with scikit-learn. It depends on scikit-learn
and `PyMC3 <https://docs.pymc.io/>`_ and is distributed under the new BSD-3 license,
encouraging its use in both academia and industry.

Users can now have calibrated quantities of uncertainty in their models
using powerful inference algorithms -- such as MCMC or Variational inference --
provided by `PyMC3 <https://docs.pymc.io/>`_.
See :doc:`why` for a more detailed description of why ``pymc-learn`` was
created.

.. NOTE::
   ``pymc-learn`` leverages and extends the Base template provided by the
   PyMC3 Models project: https://github.com/parsing-science/pymc3_models

----

Familiar user interface
-----------------------
``pymc-learn`` mimics scikit-learn. You don't have to completely rewrite
your scikit-learn ML code.

.. code-block:: python

    from sklearn.linear_model \                         from pmlearn.linear_model \
      import LinearRegression                             import LinearRegression
    lr = LinearRegression()                             lr = LinearRegression()
    lr.fit(X, y)                                        lr.fit(X, y)

The difference between the two models is that ``pymc-learn`` estimates model
parameters using Bayesian inference algorithms such as MCMC or variational
inference. This produces calibrated quantities of uncertainty for model
parameters and predictions.

----

Quick Install
-----------------

You can install ``pymc-learn`` from PyPi using pip as follows:

.. code-block:: bash

   pip install pymc-learn


Or from source as follows:

.. code-block:: bash

   pip install git+https://github.com/pymc-learn/pymc-learn


.. CAUTION::
   ``pymc-learn`` is under heavy development.

Dependencies
................

``pymc-learn`` is tested on Python 2.7, 3.5 & 3.6 and depends on Theano,
PyMC3, Scikit-learn, NumPy, SciPy, and Matplotlib (see ``requirements.txt``
for version information).

----


Quick Start
------------------

.. code-block:: python

    # For regression using Bayesian Nonparametrics
    >>> from sklearn.datasets import make_friedman2
    >>> from pmlearn.gaussian_process import GaussianProcessRegressor
    >>> from pmlearn.gaussian_process.kernels import DotProduct, WhiteKernel
    >>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    >>> kernel = DotProduct() + WhiteKernel()
    >>> gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)
    >>> gpr.score(X, y)
    0.3680...
    >>> gpr.predict(X[:2,:], return_std=True)
    (array([653.0..., 592.1...]), array([316.6..., 316.6...]))

----

Scales to Big Data & Complex Models
-----------------------------------

Recent research has led to the development of variational inference algorithms
that are fast and almost as flexible as MCMC. For instance Automatic
Differentation Variational Inference (ADVI) is illustrated in the code below.

.. code-block:: python

    from pmlearn.neural_network import MLPClassifier
    model = MLPClassifier()
    model.fit(X_train, y_train, inference_type="advi")


Instead of drawing samples from the posterior, these algorithms fit
a distribution (e.g. normal) to the posterior turning a sampling problem into
an optimization problem. ADVI is provided PyMC3.

----

Citing pymc-learn
------------------

To cite ``pymc-learn`` in publications, please use the following::

   Emaasit, Daniel (2018). Pymc-learn: Practical probabilistic machine
   learning in Python. arXiv preprint arXiv:1810.xxxxx.

Or using BibTex as follows:

.. code-block:: latex

    @article{emaasit2018pymc,
      title={Pymc-learn: Practical probabilistic machine learning in {P}ython},
      author={Emaasit, Daniel and others},
      journal={arXiv preprint arXiv:1810.xxxxx},
      year={2018}
    }

If you want to cite ``pymc-learn`` for its API, you may also want to consider
this reference::

   Carlson, Nicole (2018). Custom PyMC3 models built on top of the scikit-learn
   API. https://github.com/parsing-science/pymc3_models

Or using BibTex as follows:

.. code-block:: latex

    @article{Pymc3_models,
      title={pymc3_models: Custom PyMC3 models built on top of the scikit-learn API,
      author={Carlson, Nicole},
      journal={},
      url={https://github.com/parsing-science/pymc3_models}
      year={2018}
    }

License
..............

`New BSD-3 license <https://github.com/pymc-learn/pymc-learn/blob/master/LICENSE>`__

----

Index
-----

**Getting Started**

* :doc:`install`
* :doc:`support`
* :doc:`why`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   install.rst
   support.rst
   why.rst

----

**User Guide**

The main documentation. This contains an in-depth description of all models
and how to apply them.

* :doc:`user_guide`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: User Guide

   user_guide.rst

----

**Examples**

Pymc-learn provides probabilistic models for machine learning,
in a familiar scikit-learn syntax.

* :doc:`regression`
* :doc:`classification`
* :doc:`mixture`
* :doc:`neural_networks`
* :doc:`api`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Examples

   regression.rst
   classification.rst
   mixture.rst
   neural_networks.rst

----

**API Reference**

``pymc-learn`` leverages and extends the Base template provided by the PyMC3
Models project: https://github.com/parsing-science/pymc3_models.

* :doc:`api`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: API Reference

   api.rst

----

**Help & reference**

* :doc:`develop`
* :doc:`support`
* :doc:`changelog`
* :doc:`cite`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Help & reference

   develop.rst
   support.rst
   changelog.rst
   cite.rst

.. |Binder| image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/pymc-learn/pymc-learn/master?filepath=%2Fdocs%2Fnotebooks?urlpath=lab

.. |Travis| image:: https://travis-ci.com/pymc-learn/pymc-learn.svg?branch=master
   :target: https://travis-ci.com/pymc-learn/pymc-learn

.. |Coverage| image:: https://coveralls.io/repos/github/pymc-learn/pymc-learn/badge.svg?branch=master
   :target: https://coveralls.io/github/pymc-learn/pymc-learn?branch=master

.. |Python27| image:: https://img.shields.io/badge/python-2.7-blue.svg
   :target: https://badge.fury.io/py/pymc-learn

.. |Python36| image:: https://img.shields.io/badge/python-3.6-blue.svg
   :target: https://badge.fury.io/py/pymc-learn

.. |Docs| image:: https://readthedocs.org/projects/pymc-learn/badge/?version=latest
   :target: https://pymc-learn.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. |License| image:: https://img.shields.io/badge/license-BSD-blue.svg
   :alt: Hex.pm
   :target: https://github.com/pymc-learn/pymc-learn/blob/master/LICENSE

.. |Pypi| image:: https://badge.fury.io/py/pymc-learn.svg
   :target: https://badge.fury.io/py/pymc-learn
