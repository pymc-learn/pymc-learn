"""
Naive Bayes models.
"""

# Authors: Rémi Louf <remilouf@gmail.com>, <remi@sounds.am>
#          Daniel Emaasit <daniel.emaasit@gmail.com>
#
# License: BSD 3 clause

import functools as ft

import numpy as np
import pymc3 as pm
import scipy.stats
import theano

from ..exceptions import PymcLearnError
from ..base import BayesianModel, BayesianClassifierMixin


class GaussianNBClassifierMixin(BayesianClassifierMixin):
    """Mixin class for naive Bayes classifiers

    """
    def fit(self, X, y, inference_type='advi', minibatch_size=None,
            inference_args=None):
        """
        Train the Naive Bayes model.

        Parameters
        ----------
        X : numpy array, shape [num_training_samples, num_pred].
           Contains the data points.

        y : numpy array, shape [num_training_samples,].
           Contains the category of the data points.

        inference_type : string, specifies which inference method to call.
            Default is 'advi'. Currently, only 'advi' and 'nuts'
            are implemented.

        minibatch_size : int, number of samples to include in each minibatch
            for ADVI. Defaults to None so minibatch is not run by default.

        inference_args : dict, arguments to be passed to the inference methods.
            Check the PyMC3 documentation.

        Returns
        -------
        The current instance of the GaussianNB class.
        """
        self.num_training_samples, self.num_pred = X.shape
        self.num_cats = len(np.unique(y))
        self.inference_type = inference_type

        if not inference_args:
            inference_args = self._set_default_inference_args()

        if not self.cached_model:
            self.cached_model = self.create_model()

        if minibatch_size:
            with self.cached_model:
                minibatches = {
                    self.shared_vars['model_input']: pm.Minibatch(
                        X, batch_size=minibatch_size),
                    self.shared_vars['model_output']: pm.Minibatch(
                        y, batch_size=minibatch_size),
                }

                inference_args['more_replacements'] = minibatches
        else:
            self._set_shared_vars({'model_input': X, 'model_output': y})

        self._inference(inference_type, inference_args)

        return self

    def predict_proba(self, X):
        """
        Predicts the probabilities that data points belong to each category.

        Given a new data point :math:`\\vec{x}`, we want to estimate the
        probability that it belongs to a category :math:`c`.
        Following the notations in [1], the probability
        reads:

        .. math::

            P(y=c|\\vec{x}, \\mathcal{D}) = P(y=c|\\mathcal{D}) \\prod_{j=1}^{n_{dims}} \\
                                            P(x_j|y=c, \\mathcal{D})

        We previously used the data :math:`\\mathcal{D}` to estimate the
        distribution of the parameters :math:`\\vec{\\mu}`, :math:`\\vec{\\pi}`
        and :math:`\\vec{\\sigma}`. To compute the above probability, we need
        to integrate over the values of these parameters:

        .. math::

            P(y=c|\\vec{x}, \\mathcal{D}) = \\left[\\int Cat(y=c|\\vec{\\pi})P(\\vec{\\pi}|\\
                                            \\mathcal{D})\\mathrm{d}\\vec{\\pi}\\right]
                                            \\int P(\\vec{x}|\\vec{\\mu}, \\vec{\\sigma})\\
                                            P(\\vec{\\mu}|\\mathcal{D})\\
                                            P(\\vec{\\sigma}|\\mathcal{D})\\
                                            \\mathrm{d}\\vec{\\mu}\\mathrm{d}\\vec{\\sigma}

        Parameters
        ----------
        X : numpy array, shape [num_training_samples, num_pred].
           Contains the points for which we want to predict the class

        Returns
        -------
        A numpy array of shape [num_training_samples, num_cats] that contains
        the probabilities that each sample belong to each category.

        References
        ----------
        .. [1] Murphy, K. P. (2012). Machine learning: a probabilistic
        perspective.
        """

        if self.trace is None:
            raise PymcLearnError("Run fit() on the model before predict()")

        posterior_prediction = np.array([])
        for x in X:
            prob_per_sample = scipy.stats.norm(self.trace['mu'],
                                               self.trace['sigma']).pdf(x)
            prob_per_feature = [
                    np.sum(prob_per_sample[:, :, i], axis=0)/len(
                        self.trace['mu']) for i in range(self.num_pred)
            ]
            prob_per_class = self.normalize(ft.reduce(
                lambda x, y: x*y, prob_per_feature))
            if len(posterior_prediction) == 0:
                posterior_prediction = prob_per_class
            else:
                posterior_prediction = np.vstack((posterior_prediction,
                                                  prob_per_class))

        return posterior_prediction

    def predict(self, X):
        """
        Classify new data with a trained Naive Bayes model. The output is the
        point estimate of the posterior predictive distribution that
        corresponds to the one-hot loss function.

        Parameters
        ----------
        X : numpy array, shape [num_training_samples, num_pred].
           Contains the data to classify.

        Returns
        -------
        A numpy array of shape [num_training_samples,] that contains the
        predicted class to which the data points belong.
        """
        proba = self.predict_proba(X)
        predictions = np.argmax(proba, axis=1)
        return predictions

    def normalize(self, array):
        """
        Normalize values in the array to get probabilities.
        Parameters
        ----------
        array : numpy array of shape [1,]
        Returns
        -------
        A normalized array
        """
        return array / np.sum(array)


class GaussianNB(BayesianModel, GaussianNBClassifierMixin):
    """Gaussian Naive Bayes (GaussianNB) classification built using PyMC3.

    The Gaussian Naive Bayes algorithm assumes that the random variables
    that describe each class and each feature are independent and distributed
    according to Normal distributions.

    Example
    -------
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> Y = np.array([1, 1, 1, 2, 2, 2])
    >>> from pmlearn.naive_bayes import GaussianNB
    >>> clf = GaussianNB()
    >>> clf.fit(X, Y)
    GaussianNB(priors=None, var_smoothing=1e-09)
    >>> print(clf.predict([[-0.8, -1]]))
    [1]
    >>> clf_pf = GaussianNB()
    >>> clf_pf.partial_fit(X, Y, np.unique(Y))
    GaussianNB(priors=None, var_smoothing=1e-09)
    >>> print(clf_pf.predict([[-0.8, -1]]))
    [1]

    See the documentation of the `create_model` method for details on the model
    itself.
    """

    def __init__(self):
        super(GaussianNB, self).__init__()

    def create_model(self):
        """
        Creates and returns the PyMC3 model.

        We note :math:`x_{jc}` the value of the j-th element of the data
        vector :math:`x` conditioned on x belonging to the class :math:`c`.
        The Gaussian Naive Bayes algorithm models :math:`x_{jc}` as:

        .. math::

            x_{jc} \\sim Normal(\\mu_{jc}, \\sigma_{jc})

        While the probability that :math:`x` belongs to the
        class :math:`c` is given by the categorical distribution:

        .. math::

            P(y=c|x_i) = Cat(\\pi_1, \dots, \\pi_C)

        where :math:`\pi_i` is the probability that a vector belongs
        to category :math:`i`.

        We assume that the :math:`\pi_i` follow a Dirichlet distribution:

        .. math::

            \\pi \\sim Dirichlet(\\alpha)

        with hyperparameter :math:`\\alpha = [1, .., 1]`. The :math:`\\mu_{jc}`
        are sampled from a Normal distribution centred on :math:`0` with
        variance :math:`100`, and the :math:`\\sigma_{jc}` are sampled from a
        HalfNormal distribuion of variance :math:`100`:

        .. math::

            \\mu_{jc} \\sim Normal(0, 100)

            \\sigma_{jc} \\sim HalfNormal(100)

        Note that the Gaussian Naive Bayes model is equivalent to a Gaussian
        mixture with a diagonal covariance [1].

        Returns
        -------
        A PyMC3 model

        References
        ----------
        .. [1] Murphy, K. P. (2012). Machine learning: a probabilistic
        perspective.
        """

        # The data
        X = theano.shared(np.zeros((self.num_training_samples, self.num_pred)))
        y = theano.shared(np.zeros(self.num_training_samples, dtype=int))

        self.shared_vars = {
            'model_input': X,
            'model_output': y
        }

        model = pm.Model()
        with model:
            # Priors
            alpha = np.ones(self.num_cats)
            pi = pm.Dirichlet('pi', alpha, shape=self.num_cats)
            mu = pm.Normal('mu', mu=0, sd=100, shape=(self.num_cats,
                                                      self.num_pred))
            sigma = pm.HalfNormal('sigma', 100, shape=(self.num_cats,
                                                       self.num_pred))

            # Assign classes to data points
            z = pm.Categorical('z', pi, shape=self.num_training_samples,
                               observed=y)

            # The components are independent and normally distributed
            xi = pm.Normal('xi', mu=mu[z], sd=sigma[z], observed=X)

        return model

    def save(self, file_prefix):
        params = {
            'inference_type': self.inference_type,
            'num_cats': self.num_cats,
            'num_pred': self.num_pred,
            'num_training_samples': self.num_training_samples
        }
        super(GaussianNB, self).save(file_prefix, params)

    def load(self, file_profile):
        params = super(GaussianNB, self).load(file_profile,
                                              load_custom_params=True)

        self.inference_type = params['inference_type']
        self.num_cats = params['num_cats']
        self.num_pred = params['num_pred']
        self.num_training_samples = params['num_training_samples']
