"""Base classes for all Bayesian models."""

# Authors: Daniel Emaasit <daniel.emaasit@gmail.com>
#          Nicole Carlson <nicole@parsingscience.com>
# License: BSD 3 clause

import joblib
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin, ClassifierMixin, DensityMixin

from .exceptions import PymcLearnError


class BayesianModel(BaseEstimator):
    """Base class for all Bayesian models in pymc-learn

    Notes
    -----
    All Bayesian models should specify all the parameters that can be set at
    the class level in their ``__init__`` as explicit keyword arguments
    (no ``*args`` or **kwargs``).
    """
    def __init__(self):
        self.cached_model = None
        self.default_advi_sample_draws = 10000
        self.inference_type = None
        self.num_pred = None
        self.shared_vars = None
        self.summary = None
        self.trace = None
        self.num_training_samples = None

    def create_model(self):
        """Create model
        """
        raise NotImplementedError

    def _set_shared_vars(self, shared_vars):
        """
        Sets theano shared variables for the PyMC3 model.
        """
        for key in shared_vars.keys():
            self.shared_vars[key].set_value(shared_vars[key])

    def _inference(self, inference_type='advi', inference_args=None):
        """
        Calls internal methods for two types of inferences. Raises an error
        if the inference_type is not supported.

        Parameters
        ==========
        inference_type : string, specifies which inference method to call.
           Defaults to 'advi'. Currently, only 'advi' and 'nuts' are supported

        inference_args : dict, arguments to be passed to the inference methods.
           Check the PyMC3 docs to see what is permitted. Defaults to None.
        """
        if inference_type == 'advi':
            self._advi_inference(inference_args)
        elif inference_type == 'nuts':
            self._nuts_inference(inference_args)
        else:
            raise PymcLearnError('{} is not a supported type'
                                 ' of inference'.format(inference_type))

    def _advi_inference(self, inference_args):
        """
        Runs variational ADVI and then samples from those results.

        Parameters
        ----------
        inference_args : dict, arguments to be passed to the PyMC3 fit method.
           See PyMC3 doc for permissible values.
        """
        with self.cached_model:
            inference = pm.ADVI()
            approx = pm.fit(method=inference, **inference_args)

        self.approx = approx
        self.trace = approx.sample(draws=self.default_advi_sample_draws)
        self.summary = pm.summary(self.trace)
        self.advi_hist = inference.hist

    def _nuts_inference(self, inference_args):
        """
        Runs NUTS inference.

        Parameters
        ----------
        inference_args : dict, arguments passed to the PyMC3 sample method.
           See PyMC3 doc for permissible values.
        """
        with self.cached_model:
            step = pm.NUTS()
            nuts_trace = pm.sample(step=step, **inference_args)

        self.trace = nuts_trace
        self.summary = pm.summary(self.trace)

    def _set_default_inference_args(self):
        """
        Set default values for inference arguments if none are provided,
        dependent on inference type.

        ADVI
        -----
        callbacks : list containing a parameter stopping check.

        n : number of iterations for ADVI fit, defaults to 200000

        NUTS
        -----
        draws : the number of samples to draw, defaults to 2000
        """
        if self.inference_type == 'advi':
            inference_args = {
                'n': 200000,
                'callbacks': [pm.callbacks.CheckParametersConvergence()]
            }
        elif self.inference_type == 'nuts':
            inference_args = {
                'draws': 2000
            }
        else:
            inference_args = None

        return inference_args

    def save(self, file_prefix, custom_params=None):
        """
        Saves the trace and custom params to files with the given file_prefix.

        Parameters
        ----------
        file_prefix : str, path and prefix used to identify where to save the
        trace for this model.
            Ex: given file_prefix = "path/to/file/"
            This will attempt to save to "path/to/file/trace.pickle"

        custom_params : Dictionary of custom parameters to save.
           Defaults to None
        """
        fileObject = open(file_prefix + 'trace.pickle', 'wb')
        joblib.dump(self.trace, fileObject)
        fileObject.close()

        if custom_params:
            fileObject = open(file_prefix + 'params.pickle', 'wb')
            joblib.dump(custom_params, fileObject)
            fileObject.close()

    def load(self, file_prefix, load_custom_params=False):
        """
        Loads a saved version of the trace, and custom param files with the
        given file_prefix.

        Parameters
        ----------
        file_prefix : str, path and prefix used to identify where to load the
        saved trace for this model.
            Ex: given file_prefix = "path/to/file/"
            This will attempt to load "path/to/file/trace.pickle"

        load_custom_params : Boolean flag to indicate whether custom parameters
        should be loaded. Defaults to False.

        Returns
        ----------
        custom_params : Dictionary of custom parameters
        """
        self.trace = joblib.load(file_prefix + 'trace.pickle')

        custom_params = None
        if load_custom_params:
            custom_params = joblib.load(file_prefix + 'params.pickle')

        return custom_params

    def plot_elbo(self):
        """
        Plot the ELBO values after running ADVI minibatch.
        """
        if self.inference_type != 'advi':
            raise PymcLearnError(
                'This method should only be called after calling fit with '
                'ADVI minibatch.'
            )

        sns.set_style("white")
        plt.plot(-self.advi_hist)
        plt.ylabel('ELBO')
        plt.xlabel('iteration')
        sns.despine()


class BayesianRegressorMixin(RegressorMixin):
    """Mixin for regression models in pmlearn

    """
    def fit(self, X, y, inference_type='advi', minibatch_size=None,
            inference_args=None):
        """
        Train the Linear Regression model

        Parameters
        ----------
        X : numpy array, shape [n_samples, n_features]

        y : numpy array, shape [n_samples, ]

        inference_type : string, specifies which inference method to call.
           Defaults to 'advi'. Currently, only 'advi' and 'nuts' are supported

        minibatch_size : number of samples to include in each minibatch for
           ADVI, defaults to None, so minibatch is not run by default

        inference_args : dict, arguments to be passed to the inference methods.
           Check the PyMC3 docs for permissable values. If no arguments are
           specified, default values will be set.
        """
        self.num_training_samples, self.num_pred = X.shape

        self.inference_type = inference_type

        if y.ndim != 1:
            y = np.squeeze(y)

        if not inference_args:
            inference_args = self._set_default_inference_args()

        if self.cached_model is None:
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

    def predict(self, X, return_std=False):
        """
        Predicts values of new data with a trained Linear Regression model

        Parameters
        ----------
        X : numpy array, shape [n_samples, n_features]

        return_std : Boolean flag
           Boolean flag of whether to return standard deviations with mean
           values. Defaults to False.
        """

        if self.trace is None:
            raise PymcLearnError('Run fit on the model before predict.')

        num_samples = X.shape[0]

        if self.cached_model is None:
            self.cached_model = self.create_model()

        self._set_shared_vars({'model_input': X,
                               'model_output': np.zeros(num_samples)})

        ppc = pm.sample_ppc(self.trace, model=self.cached_model, samples=2000)

        if return_std:
            return ppc['y'].mean(axis=0), ppc['y'].std(axis=0)
        else:
            return ppc['y'].mean(axis=0)


class BayesianClassifierMixin(ClassifierMixin):
    """Mixin for regression models in pmlearn

    """
    def fit(self, X, y, inference_type='advi', minibatch_size=None,
            inference_args=None):
        """ Train the Multilayer perceptron model

        Parameters
        ----------
        X : numpy array, shape [n_samples, n_features]

        y : numpy array, shape [n_samples, ]

        inference_type : string, specifies which inference method to call.
        Defaults to 'advi'. Currently, only 'advi' and 'nuts' are supported

        minibatch_size : number of samples to include in each minibatch
        for ADVI, defaults to None, so minibatch is not run by default

        inference_args : dict, arguments to be passed to the inference methods.
        Check the PyMC3 docs for permissable values. If no arguments are
        specified, default values will be set.
        """
        self.num_training_samples, self.num_pred = X.shape

        self.inference_type = inference_type

        if y.ndim != 1:
            y = np.squeeze(y)

        if not inference_args:
            inference_args = self._set_default_inference_args()

        if self.cached_model is None:
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

    def predict_proba(self, X, return_std=False):
        """ Perform Prediction

        Predicts values of new data with a trained Gaussian Process
        Regression model

        Parameters
        ----------
        X : numpy array, shape [n_samples, n_features]

        return_std : Boolean
            Whether to return standard deviations with mean values.
            Defaults to False.
        """

        if self.trace is None:
            raise PymcLearnError('Run fit on the model before predict.')

        num_samples = X.shape[0]

        if self.cached_model is None:
            self.cached_model = self.create_model()

        self._set_shared_vars({'model_input': X,
                               'model_output': np.zeros(num_samples)})

        ppc = pm.sample_ppc(self.trace, model=self.cached_model, samples=2000)

        if return_std:
            return ppc['y'].mean(axis=0), ppc['y'].std(axis=0)
        else:
            return ppc['y'].mean(axis=0)

    def predict(self, X):
        """
        Predicts labels of new data with a trained model

        Parameters
        ----------
        X : numpy array, shape [n_samples, n_features]

        """
        ppc_mean = self.predict_proba(X)

        pred = ppc_mean > 0.5

        return pred


class BayesianDensityMixin(DensityMixin):
    """Mixin for regression models in pmlearn

    """
    def fit(self, X, num_components, inference_type='advi',
            minibatch_size=None, inference_args=None):
        """
        Train the Gaussian Mixture Model model

        Parameters
        ----------
        X : numpy array, shape [n_samples, n_features]

        n_truncate : numpy array, shape [n_samples, ]

        inference_type : string, specifies which inference method to call.
        Defaults to 'advi'. Currently, only 'advi' and 'nuts' are supported

        minibatch_size : number of samples to include in each minibatch for
        ADVI,
        defaults to None, so minibatch is not run by default

        inference_args : dict, arguments to be passed to the inference methods.
        Check the PyMC3 docs for permissable values. If no arguments are
        specified,
        default values will be set.
        """
        self.num_components = num_components
        self.num_training_samples, self.num_pred = X.shape

        self.inference_type = inference_type

        # if y.ndim != 1:
        #     y = np.squeeze(y)

        if not inference_args:
            inference_args = self._set_default_inference_args()

        if self.cached_model is None:
            self.cached_model = self.create_model()

        if minibatch_size:
            with self.cached_model:
                minibatches = {
                    self.shared_vars['model_input']: pm.Minibatch(
                        X, batch_size=minibatch_size)
                    # ,
                    # self.shared_vars['model_output']: pm.Minibatch(
                    # y, batch_size=minibatch_size),
                    # self.shared_vars['model_components']: pm.Minibatch(
                    # components, batch_size=minibatch_size)
                }

                inference_args['more_replacements'] = minibatches
        else:
            self._set_shared_vars({'model_input': X})

        self._inference(inference_type, inference_args)

        return self

    def predict_proba(self, X, return_std=False):
        """
        Predicts probabilities of new data with a trained GaussianMixture Model

        Parameters
        ----------
        X : numpy array, shape [n_samples, n_features]

        cats : numpy array, shape [n_samples, ]

        return_std : Boolean flag of whether to return standard deviations with
        mean probabilities. Defaults to False.
        """

        if self.trace is None:
            raise PymcLearnError('Run fit on the model before predict.')

        # num_samples = X.shape[0]

        if self.cached_model is None:
            self.cached_model = self.create_model()

        self._set_shared_vars({'model_input': X})
        K = self.num_components

        with self.cached_model:
            pi = pm.Dirichlet("probability",
                              a=np.array([1.0, 1.0, 1.0]),
                              shape=K)
            _vars = [pi]

            ppc = pm.sample_ppc(self.trace,
                                # model=self.cached_model,
                                vars=_vars,
                                samples=2000,
                                size=len(X))

        if return_std:
            return ppc['probability'].mean(axis=0), \
                   ppc['probability'].std(axis=0)
        else:
            return ppc['probability'].mean(axis=0)

    def predict(self, X):
        """
        Predicts labels of new data with a trained model

        Parameters
        ----------
        X : numpy array, shape [n_samples, n_features]

        cats : numpy array, shape [n_samples, ]
        """
        ppc_mean = self.predict_proba(X)

        # pred = ppc_mean > 0.5
        #
        # return pred
        return ppc_mean

    def score(self, X, y, cats):
        """
        Scores new data with a trained model.

        Parameters
        ----------
        X : numpy array, shape [n_samples, n_features]

        y : numpy array, shape [n_samples, ]

        cats : numpy array, shape [n_samples, ]
        """
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X, cats))
