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

- ``pymc-learn`` mimics the syntax of `scikit-learn <https://scikit-learn.org>`_ -- a popular Python library for machine learning -- which has a consistent & simple API, and is very user friendly.

- This makes ``pymc-learn`` easy to learn and use for first-time users.

- For scikit-learn users, you don't have to completely rewrite your code. Your code looks almost the same. You are more productive, allowing you to try more ideas faster.

.. code-block:: python

    from sklearn.linear_model \                         from pmlearn.linear_model \
      import LinearRegression                             import LinearRegression
    lr = LinearRegression()                             lr = LinearRegression()
    lr.fit(X, y)                                        lr.fit(X, y)

- This ease of use does not come at the cost of reduced flexibility: because ``pymc-learn`` integrates with `PyMC3 <https://docs.pymc.io>`_, it enables you to implement anything you could have built in the base language.


----



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

Why scikit-learn and PyMC3
---------------------------
PyMC3 is a Python package for probabilistic machine learning that enables users
to build bespoke models for their specific problems using a probabilistic
modeling framework. However, PyMC3 lacks the steps between creating a model and
reusing it with new data in production. The missing steps include: scoring a
model, saving a model for later use, and loading the model in production
systems.

In contrast, *scikit-learn* which has become the standard
library for machine learning provides a simple API that makes it very easy for
users to train, score, save and load models in production. However,
*scikit-learn* may not have the model for a user's specific problem.
These limitations have led to the development of the open
source *pymc3-models* library which provides a template to build bespoke
PyMC3 models on top of the *scikit-learn* API and reuse them in
production. This enables users to easily and quickly train, score, save and
load their bespoke models just like in *scikit-learn*.

The ``pymc-learn`` project adopted and extended the template in *pymc3-models*
to develop probabilistic versions of the estimators in *scikit-learn*.
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
...........

1. Ghahramani, Z. (2015). Probabilistic machine learning and artificial intelligence. Nature, 521(7553), 452.

2. Bishop, C. M. (2013). Model-based machine learning. Phil. Trans. R. Soc. A, 371(1984), 20120222.

3. Murphy, K. P. (2012). Machine learning: a probabilistic perspective. MIT Press.

4. Barber, D. (2012). Bayesian reasoning and machine learning. Cambridge University Press.

5. Salvatier, J., Wiecki, T. V., & Fonnesbeck, C. (2016). Probabilistic programming in Python using PyMC3. PeerJ Computer Science, 2, e55.