Install pymc-learn
===================

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


This also installs required dependencies including Theano.
For alternative Theano installations (e.g., gpu), please see the
instructions on the main `Theano webpage <http://deeplearning.net/software/theano/>`_.

Transitioning from PyMC3 to PyMC4
..................................

.. raw:: html

    <embed>
        <blockquote class="twitter-tweet" data-lang="en"><p lang="en" dir="ltr">.<a href="https://twitter.com/pymc_learn?ref_src=twsrc%5Etfw">@pymc_learn</a> has been following closely the development of <a href="https://twitter.com/hashtag/PyMC4?src=hash&amp;ref_src=twsrc%5Etfw">#PyMC4</a> with the aim of switching its backend from <a href="https://twitter.com/hashtag/PyMC3?src=hash&amp;ref_src=twsrc%5Etfw">#PyMC3</a> to PyMC4 as the latter grows to maturity. Core devs are invited. Here&#39;s the tentative roadmap for PyMC4: <a href="https://t.co/Kwjkykqzup">https://t.co/Kwjkykqzup</a> cc <a href="https://twitter.com/pymc_devs?ref_src=twsrc%5Etfw">@pymc_devs</a> <a href="https://t.co/Ze0tyPsIGH">https://t.co/Ze0tyPsIGH</a></p>&mdash; pymc-learn (@pymc_learn) <a href="https://twitter.com/pymc_learn/status/1059474316801249280?ref_src=twsrc%5Etfw">November 5, 2018</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
    </embed>