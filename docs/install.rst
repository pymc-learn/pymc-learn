Install pymc-learn
===================

``pymc-learn`` requires a working Python interpreter (2.7 or 3.3+).
It is recommend installing Python and key numerical libraries using the `Anaconda Distribution <https://www.continuum.io/downloads>`_,
which has one-click installers available on all major platforms.

Assuming a standard Python environment is installed on your machine (including pip), ``pymc-learn`` itself can be installed in one line using pip:

.. code-block:: python

    pip install git+https://github.com/pymc-learn/pymc-learn

This also installs required dependencies including Theano.
For alternative Theano installations (e.g., gpu), please see the
instructions on the main `Theano webpage <http://deeplearning.net/software/theano/>`_ and `TensorFlow webpage <https://www.tensorflow.org/>`_, respectively.
