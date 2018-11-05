Why pymc-learn?
===============

There are several probabilistic machine learning frameworks available today.
Why use ``pymc-learn`` rather than any other? Here are some of the reasons why
you may be compelled to use ``pymc-learn``.

----

.. toctree::
   :maxdepth: 1


pymc-learn prioritizes user experience
---------------------------------------

- *Familiarity*: ``pymc-learn`` mimics the syntax of `scikit-learn <https://scikit-learn.org>`_ -- a popular Python library for machine learning -- which has a consistent & simple API, and is very user friendly.

- *Ease of use*: This makes ``pymc-learn`` easy to learn and use for first-time users.

- *Productivity*: For scikit-learn users, you don't have to completely rewrite your code. Your code looks almost the same. You are more productive, allowing you to try more ideas faster.

.. code-block:: python

    from sklearn.linear_model \                         from pmlearn.linear_model \
      import LinearRegression                             import LinearRegression
    lr = LinearRegression()                             lr = LinearRegression()
    lr.fit(X, y)                                        lr.fit(X, y)

- *Flexibility*: This ease of use does not come at the cost of reduced flexibility. Given that ``pymc-learn`` integrates with `PyMC4 <https://docs.pymc.io>`_, it enables you to implement anything you could have built in the base language.

- *Performance*. The primary inference algorithm is gradient-based automatic differentiation variational inference (ADVI) (Kucukelbir et al., 2017), which estimates a divergence measure between approximate and true posterior distributions. Pymc-learn scales to complex, high-dimensional models thanks to GPU-accelerated tensor math and reverse-mode automatic differentiation via TensorFlow (TensorFlow Development Team, 2016), and it scales to large datasets thanks to estimates computed over mini-batches of data in ADVI.

----


Why do we need pymc-learn?
--------------------------
Currently, there is a growing need for principled machine learning approaches by
non-specialists in many fields including the pure sciences (e.g. biology, physics,
chemistry), the applied sciences (e.g. political science, biostatistics),
engineering (e.g. transportation, mechanical), medicine (e.g. medical imaging),
the arts (e.g visual art), and software industries.

This has lead to increased adoption of probabilistic modeling. This trend is
attributed in part to three major factors:

(1) the need for transparent models with calibrated quantities of uncertainty, i.e. "models should know when they don't know",

(2) the ever-increasing number of promising results achieved on a variety of fundamental problems in AI (Ghahramani, 2015), and

(3) the emergency of probabilistic programming languages (PPLs) that provide a fexible framework to build richly structured probabilistic models that incorporate domain knowledge.

However, usage of PPLs requires a specialized understanding of probability
theory, probabilistic graphical modeling, and probabilistic inference. Some PPLs
also require a good command of software coding. These requirements make it
difficult for non-specialists to adopt and apply probabilistic machine learning
to their domain problems.

``Pymc-learn`` seeks to address these challenges by providing state-of-the art
implementations of several popular probabilistic machine learning models.
**It is inspired by scikit-learn** (Pedregosa et al., 2011) **and focuses on
bringing probabilistic machine learning to non-specialists**. It puts emphasis
on:

(1) ease of use,

(2) productivity,

(3) fexibility,

(4) performance,

(5) documentation, and

(6) an API consistent with scikit-learn.

The underlying probabilistic models are built using pymc4 (Salvatier et al., 2019).


Python is the lingua franca of Data Science
--------------------------------------------

Python has become the dominant language for both data science, and
general programming:

.. image:: https://zgab33vy595fw5zq-zippykid.netdna-ssl.com/wp-content/uploads/2017/09/growth_major_languages-1-1024x878.png
   :alt: Growth of major programming languages
   :width: 75%

This popularity is driven both by computational libraries like Numpy, Pandas, and
Scikit-Learn and by a wealth of libraries for visualization, interactive
notebooks, collaboration, and so forth.

.. image:: https://zgab33vy595fw5zq-zippykid.netdna-ssl.com/wp-content/uploads/2017/09/related_tags_over_time-1-1024x1024.png
   :alt: Stack overflow traffic to various packages
   :width: 75%

*Image credit to Stack Overflow blogposts*
`#1 <https://stackoverflow.blog/2017/09/06/incredible-growth-python>`_
*and*
`#2 <https://stackoverflow.blog/2017/09/14/python-growing-quickly/>`_

----

Why scikit-learn and PyMC4
---------------------------
PyMC4 is a Python package for probabilistic machine learning that enables users
to build bespoke models for their specific problems using a probabilistic
modeling framework.

*scikit-learn* which has become the standard
library for machine learning provides a simple API that makes it very easy for
users to train, score, save and load models in production.

The ``pymc-learn`` project developed probabilistic versions of the estimators in *scikit-learn*.
This provides users with probabilistic models in a simple workflow that mimics
the scikit-learn API.

----


Quantification of uncertainty
------------------------------

Today, many data-driven solutions are seeing a heavy use of machine
learning for understanding phenomena and predictions. For instance, in cyber security,
this may include monitoring streams
of network data and predicting unusual events that deviate from the norm.
For example, an employee downloading large volumes of intellectual property
(IP) on a weekend. **Immediately**, we are faced with our first challenge,
that is, we are dealing with quantities (unusual volume & unusual period)
whose values are uncertain. To be more concrete, we start off very uncertain
whether this download event is unusually large and then slowly get more and
more certain as we uncover more clues such as the period of the week,
performance reviews for the employee, or did they visit WikiLeaks?, etc.

In fact, the need to deal with uncertainty arises throughout our increasingly
data-driven world. Whether it is Uber autonomous vehicles dealing with
predicting pedestrians on roadways or Amazon's logistics apparatus that has to
optimize its supply chain system. All these applications have to handle and
manipulate uncertainty. Consequently, we need a principled framework for
quantifying uncertainty which will allow us to create applications and build
solutions in ways that can represent and process uncertain values.

Fortunately, there is a simple framework for manipulating uncertain
quantities which uses probability to quantify the degree of uncertainty.
To quote Prof. Zhoubin Ghahramani, Uber's Chief Scientist and Professor of AI
at University of Cambridge:

    Just as Calculus is the fundamental mathematical principle for calculating
    rates of change, Probability is the fundamental mathematical principle for
    quantifying uncertainty.

The probabilistic approach to machine learning is an exciting area of research
that is currently receiving a lot of attention in many conferences and Journals
such as `NIPS <https://nips.cc>`_, `UAI <http://www.auai.org>`_,
`AISTATS <https://www.aistats.org>`_, `JML <http://www.jmlr.org/>`_,
`IEEE PAMI <https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=34>`_, etc.

----

References
------------

1. Ghahramani, Z. (2015). Probabilistic machine learning and artificial intelligence. Nature, 521(7553), 452.

2. Bishop, C. M. (2013). Model-based machine learning. Phil. Trans. R. Soc. A, 371(1984), 20120222.

3. Murphy, K. P. (2012). Machine learning: a probabilistic perspective. MIT Press.

4. Barber, D. (2012). Bayesian reasoning and machine learning. Cambridge University Press.

5. PyMC4.

6. Alp Kucukelbir, Dustin Tran, Rajesh Ranganath, Andrew Gelman, and David M Blei. Automatic differentiation variational inference. The Journal of Machine Learning Research, 18(1):430{474, 2017.

7. Fabian Pedregosa, Gael Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, et al. Scikit-learn: Machine learning in python. Journal of machine learning research, 12(Oct): 2825-2830, 2011.

8. TensorFlow