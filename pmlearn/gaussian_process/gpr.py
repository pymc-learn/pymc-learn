"""Gaussian process regression. """

# Authors: Daniel Emaasit <daniel.emaasit@gmail.com>
#
# License: BSD 3 clause

import numpy as np
import pymc3 as pm
import theano

from ..exceptions import PymcLearnError
from ..base import BayesianModel, BayesianRegressorMixin

from .kernels import RBF


class GaussianProcessRegressorMixin(BayesianRegressorMixin):
    """Mixin class for Gaussian Process Regression

    """
    def predict(self, X, return_std=False):
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

        with self.cached_model:
            f_pred = self.gp.conditional("f_pred", X)
            self.ppc = pm.sample_ppc(self.trace,
                                     vars=[f_pred],
                                     samples=2000)

        if return_std:
            return self.ppc['f_pred'].mean(axis=0), \
                   self.ppc['f_pred'].std(axis=0)
        else:
            return self.ppc['f_pred'].mean(axis=0)


class GaussianProcessRegressor(BayesianModel,
                               GaussianProcessRegressorMixin):
    """ Gaussian Process Regression built using PyMC3.

    Fit a Gaussian process model and estimate model parameters using
    MCMC algorithms or Variational Inference algorithms

    Parameters
    ----------
    prior_mean : mean object
        The mean specifying the mean function of the GP. If None is passed,
        the mean "pm.gp.mean.Zero()" is used as default.

    kernel : covariance function (kernel)
        The function specifying the covariance of the GP. If None is passed,
        the kernel "RBF()" is used as default.

    Examples
    --------
    >>> from sklearn.datasets import make_friedman2
    >>> from pmlearn.gaussian_process import GaussianProcessRegressor
    >>> from pmlearn.gaussian_process.kernels import DotProduct, WhiteKernel
    >>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    >>> kernel = DotProduct() + WhiteKernel()
    >>> gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)
    >>> gpr.score(X, y) # doctest: +ELLIPSIS
    0.3680...
    >>> gpr.predict(X[:2,:], return_std=True) # doctest: +ELLIPSIS
    (array([653.0..., 592.1...]), array([316.6..., 316.6...]))


    Reference
    ----------
    Rasmussen and Williams (2006). Gaussian Processes for Machine Learning.
    """

    def __init__(self, prior_mean=None, kernel=None):
        self.ppc = None
        self.gp = None
        self.num_training_samples = None
        self.num_pred = None
        self.prior_mean = prior_mean
        self.kernel = kernel

        super(GaussianProcessRegressor, self).__init__()

    def create_model(self):
        """ Creates and returns the PyMC3 model.

        Note: The size of the shared variables must match the size of the
        training data. Otherwise, setting the shared variables later will
        raise an error. See http://docs.pymc.io/advanced_theano.html

        Returns
        ----------
        model: the PyMC3 model.
        """
        model_input = theano.shared(np.zeros([self.num_training_samples,
                                              self.num_pred]))

        model_output = theano.shared(np.zeros(self.num_training_samples))

        self.shared_vars = {
            'model_input': model_input,
            'model_output': model_output,
        }

        model = pm.Model()

        with model:
            length_scale = pm.Gamma('length_scale', alpha=2, beta=1,
                                    shape=(1, self.num_pred))
            signal_variance = pm.HalfCauchy('signal_variance', beta=5,
                                            shape=1)
            noise_variance = pm.HalfCauchy('noise_variance', beta=5,
                                           shape=1)

            if self.kernel is None:
                cov_function = signal_variance ** 2 * RBF(
                    input_dim=self.num_pred,
                    ls=length_scale)
            else:
                cov_function = self.kernel

            if self.prior_mean is None:
                mean_function = pm.gp.mean.Zero()
            else:
                mean_function = self.prior_mean

            self.gp = pm.gp.Latent(mean_func=mean_function,
                                   cov_func=cov_function)

            f = self.gp.prior('f', X=model_input.get_value())

            y = pm.Normal('y', mu=f, sd=noise_variance, observed=model_output)

        return model

    def save(self, file_prefix):
        params = {
            'inference_type': self.inference_type,
            'num_pred': self.num_pred,
            'num_training_samples': self.num_training_samples
        }

        super(GaussianProcessRegressor, self).save(file_prefix, params)

    def load(self, file_prefix):
        params = super(GaussianProcessRegressor, self).load(
            file_prefix, load_custom_params=True)

        self.inference_type = params['inference_type']
        self.num_pred = params['num_pred']
        self.num_training_samples = params['num_training_samples']


class StudentsTProcessRegressor(GaussianProcessRegressor):
    """ StudentsT Process Regression built using PyMC3.

    Fit a StudentsT process model and estimate model parameters using
    MCMC algorithms or Variational Inference algorithms

    Parameters
    ----------
    prior_mean : mean object
        The mean specifying the mean function of the StudentsT process.
        If None is passed, the mean "pm.gp.mean.Zero()" is used as default.

    Examples
    --------
    >>> from sklearn.datasets import make_friedman2
    >>> from pmlearn.gaussian_process import StudentsTProcessRegressor
    >>> from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
    >>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    >>> kernel = DotProduct() + WhiteKernel()
    >>> spr = StudentsTProcessRegressor(kernel=kernel).fit(X, y)
    >>> spr.score(X, y) # doctest: +ELLIPSIS
    0.3680...
    >>> spr.predict(X[:2,:], return_std=True) # doctest: +ELLIPSIS
    (array([653.0..., 592.1...]), array([316.6..., 316.6...]))


    Reference
    ----------
    Rasmussen and Williams (2006). Gaussian Processes for Machine Learning.
    """

    def __init__(self, prior_mean=0.0):
        super(StudentsTProcessRegressor, self).__init__(prior_mean=prior_mean)

    def create_model(self):
        """ Creates and returns the PyMC3 model.

        Note: The size of the shared variables must match the size of the
        training data. Otherwise, setting the shared variables later will raise
        an error. See http://docs.pymc.io/advanced_theano.html

        Returns
        ----------
        model : the PyMC3 model
        """
        model_input = theano.shared(np.zeros([self.num_training_samples,
                                              self.num_pred]))

        model_output = theano.shared(np.zeros(self.num_training_samples))

        self.shared_vars = {
            'model_input': model_input,
            'model_output': model_output,
        }

        self.gp = None
        model = pm.Model()

        with model:
            length_scale = pm.Gamma('length_scale', alpha=2, beta=0.5,
                                    shape=(1, self.num_pred))
            signal_variance = pm.HalfCauchy('signal_variance', beta=2,
                                            shape=1)
            noise_variance = pm.HalfCauchy('noise_variance', beta=2,
                                           shape=1)
            degrees_of_freedom = pm.Gamma('degrees_of_freedom', alpha=2,
                                          beta=0.1, shape=1)

            # cov_function = signal_variance**2 * pm.gp.cov.ExpQuad(
            # 1, length_scale)
            cov_function = signal_variance ** 2 * pm.gp.cov.Matern52(
                1, length_scale)

            # mean_function = pm.gp.mean.Zero()
            mean_function = pm.gp.mean.Constant(self.prior_mean)

            self.gp = pm.gp.Latent(mean_func=mean_function,
                                   cov_func=cov_function)

            f = self.gp.prior('f', X=model_input.get_value())

            y = pm.StudentT('y', mu=f, lam=1 / signal_variance,
                            nu=degrees_of_freedom, observed=model_output)

        return model

    def save(self, file_prefix):
        params = {
            'inference_type': self.inference_type,
            'num_pred': self.num_pred,
            'num_training_samples': self.num_training_samples
        }

        super(StudentsTProcessRegressor, self).save(file_prefix, params)

    def load(self, file_prefix):
        params = super(StudentsTProcessRegressor, self).load(
            file_prefix, load_custom_params=True)

        self.inference_type = params['inference_type']
        self.num_pred = params['num_pred']
        self.num_training_samples = params['num_training_samples']


class SparseGaussianProcessRegressor(GaussianProcessRegressor):
    """ Sparse Gaussian Process Regression built using PyMC3.

    Fit a Sparse Gaussian process model and estimate model parameters using
    MCMC algorithms or Variational Inference algorithms

    Parameters
    ----------
    prior_mean : mean object
        The mean specifying the mean function of the GP. If None is passed,
        the mean "pm.gp.mean.Zero()" is used as default.

    Examples
    --------
    >>> from sklearn.datasets import make_friedman2
    >>> from pmlearn.gaussian_process import SparseGaussianProcessRegressor
    >>> from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
    >>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    >>> kernel = DotProduct() + WhiteKernel()
    >>> sgpr = SparseGaussianProcessRegressor(kernel=kernel).fit(X, y)
    >>> sgpr.score(X, y) # doctest: +ELLIPSIS
    0.3680...
    >>> sgpr.predict(X[:2,:], return_std=True) # doctest: +ELLIPSIS
    (array([653.0..., 592.1...]), array([316.6..., 316.6...]))


    Reference
    ----------
    Rasmussen and Williams (2006). Gaussian Processes for Machine Learning.
    """

    def __init__(self, prior_mean=0.0):
        super(SparseGaussianProcessRegressor, self).__init__(
            prior_mean=prior_mean)

    def create_model(self):
        """ Creates and returns the PyMC3 model.

        Note: The size of the shared variables must match the size of the
        training data. Otherwise, setting the shared variables later will
        raise an error. See http://docs.pymc.io/advanced_theano.html

        Returns
        ----------
        model : the PyMC3 model
        """
        model_input = theano.shared(np.zeros([self.num_training_samples,
                                              self.num_pred]))

        model_output = theano.shared(np.zeros(self.num_training_samples))

        self.shared_vars = {
            'model_input': model_input,
            'model_output': model_output,
        }

        self.gp = None
        model = pm.Model()

        with model:
            length_scale = pm.Gamma('length_scale', alpha=2, beta=1,
                                    shape=(1, self.num_pred))
            signal_variance = pm.HalfCauchy('signal_variance', beta=5,
                                            shape=1)
            noise_variance = pm.HalfCauchy('noise_variance', beta=5,
                                           shape=1)

            # cov_function = signal_variance**2 * pm.gp.cov.ExpQuad(
            # 1, length_scale)
            cov_function = signal_variance ** 2 * pm.gp.cov.Matern52(
                1, length_scale)

            # mean_function = pm.gp.mean.Zero()
            mean_function = pm.gp.mean.Constant(self.prior_mean)

            self.gp = pm.gp.MarginalSparse(mean_func=mean_function,
                                           cov_func=cov_function,
                                           approx="FITC")

            # initialize 20 inducing points with K-means
            # gp.util
            Xu = pm.gp.util.kmeans_inducing_points(20,
                                                   X=model_input.get_value())

            y = self.gp.marginal_likelihood('y',
                                            X=model_input.get_value(),
                                            Xu=Xu,
                                            y=model_output.get_value(),
                                            sigma=noise_variance)

        return model

    def save(self, file_prefix):
        params = {
            'inference_type': self.inference_type,
            'num_pred': self.num_pred,
            'num_training_samples': self.num_training_samples
        }

        super(SparseGaussianProcessRegressor, self).save(file_prefix, params)

    def load(self, file_prefix):
        params = super(SparseGaussianProcessRegressor, self).load(
            file_prefix, load_custom_params=True)

        self.inference_type = params['inference_type']
        self.num_pred = params['num_pred']
        self.num_training_samples = params['num_training_samples']