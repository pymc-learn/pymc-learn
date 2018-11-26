pymc-learn: Practical Probabilistic Machine Learning in Python
===============================================================

.. image:: https://github.com/pymc-learn/pymc-learn/blob/master/docs/logos/pymc-learn-logo.jpg?raw=true
    :width: 350px
    :alt: Pymc-Learn logo
    :align: center

|status| |Travis| |Coverage| |Docs| |License| |Pypi| |Binder|

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


Transitioning from PyMC3 to PyMC4
..................................

.. raw:: html

    <embed>
        <blockquote class="twitter-tweet" data-lang="en"><p lang="en" dir="ltr">.<a href="https://twitter.com/pymc_learn?ref_src=twsrc%5Etfw">@pymc_learn</a> has been following closely the development of <a href="https://twitter.com/hashtag/PyMC4?src=hash&amp;ref_src=twsrc%5Etfw">#PyMC4</a> with the aim of switching its backend from <a href="https://twitter.com/hashtag/PyMC3?src=hash&amp;ref_src=twsrc%5Etfw">#PyMC3</a> to PyMC4 as the latter grows to maturity. Core devs are invited. Here&#39;s the tentative roadmap for PyMC4: <a href="https://t.co/Kwjkykqzup">https://t.co/Kwjkykqzup</a> cc <a href="https://twitter.com/pymc_devs?ref_src=twsrc%5Etfw">@pymc_devs</a> <a href="https://t.co/Ze0tyPsIGH">https://t.co/Ze0tyPsIGH</a></p>&mdash; pymc-learn (@pymc_learn) <a href="https://twitter.com/pymc_learn/status/1059474316801249280?ref_src=twsrc%5Etfw">November 5, 2018</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
    </embed>

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

``pymc-learn`` requires a working Python interpreter (2.7 or 3.5+).
It is recommend installing Python and key numerical libraries using the `Anaconda Distribution <https://www.anaconda.com/download/>`_,
which has one-click installers available on all major platforms.

Assuming a standard Python environment is installed on your machine
(including pip), ``pymc-learn`` itself can be installed in one line using pip:

You can install ``pymc-learn`` from PyPi using pip as follows:

.. code-block:: bash

   pip install pymc-learn


Or from source as follows:

.. code-block:: bash

   pip install git+https://github.com/pymc-learn/pymc-learn


.. CAUTION::
   ``pymc-learn`` is under heavy development.

   It is recommended installing ``pymc-learn`` in a Conda environment because it
   provides `Math Kernel Library <https://anaconda.org/anaconda/mkl-service>`_ (MKL)
   routines to accelerate math functions. If you are having trouble, try using
   a distribution of Python that includes these packages like
   `Anaconda <https://www.anaconda.com/download/>`_.



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
   learning in Python. arXiv preprint arXiv:1811.00542.

Or using BibTex as follows:

.. code-block:: latex

    @article{emaasit2018pymc,
      title={Pymc-learn: Practical probabilistic machine learning in {P}ython},
      author={Emaasit, Daniel and others},
      journal={arXiv preprint arXiv:1811.00542},
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

.. |Binder| image:: https://img.shields.io/badge/try-online-579ACA.svg?logo=data:image/png;base64,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
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

.. |status| image:: https://img.shields.io/badge/Status-Beta-blue.svg