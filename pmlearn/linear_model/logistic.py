"""
Logistic regression models.
"""

# Authors: Nicole Carlson <parsing-science@gmail.com>
#
# License: BSD 3 clause

import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt

from .base import BayesianModel
from .base import BayesianLinearClassifierMixin


class LogisticRegression(BayesianModel, BayesianLinearClassifierMixin):
    """Bayesian Logistic Regression built using PyMC3

    """
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.num_cats = None

    def create_model(self):
        """
        Creates and returns the PyMC3 model.

        Note: The size of the shared variables must match the size of the
        training data. Otherwise, setting the shared variables later will
        raise an error. See http://docs.pymc.io/advanced_theano.html

        Returns
        ----------
        the PyMC3 model
        """
        model_input = theano.shared(
            np.zeros([self.num_training_samples, self.num_pred]))

        model_output = theano.shared(
            np.zeros(self.num_training_samples, dtype='int'))

        model_cats = theano.shared(
            np.zeros(self.num_training_samples, dtype='int'))

        self.shared_vars = {
            'model_input': model_input,
            'model_output': model_output,
            'model_cats': model_cats
        }

        model = pm.Model()

        with model:
            alpha = pm.Normal('alpha', mu=0, sd=100,
                              shape=(self.num_cats,))
            betas = pm.Normal('beta', mu=0, sd=100,
                              shape=(self.num_cats, self.num_pred))

            c = model_cats

            linear_function = alpha[c] + tt.sum(betas[c] * model_input, 1)

            p = pm.invlogit(linear_function)

            y = pm.Bernoulli('y', p, observed=model_output)

        return model

    def save(self, file_prefix):
        params = {
            'inference_type': self.inference_type,
            'num_cats': self.num_cats,
            'num_pred': self.num_pred,
            'num_training_samples': self.num_training_samples
        }

        super(LogisticRegression, self).save(file_prefix, params)

    def load(self, file_prefix):
        params = super(LogisticRegression, self).load(
            file_prefix, load_custom_params=True)

        self.inference_type = params['inference_type']
        self.num_cats = params['num_cats']
        self.num_pred = params['num_pred']
        self.num_training_samples = params['num_training_samples']


class HierarchicalLogisticRegression(BayesianModel,
                                     BayesianLinearClassifierMixin):
    """
    Custom Hierachical Logistic Regression built using PyMC3.
    """

    def __init__(self):
        super(HierarchicalLogisticRegression, self).__init__()
        self.num_cats = None

    def create_model(self):
        """
        Creates and returns the PyMC3 model.

        Note: The size of the shared variables must match the size of the
        training data. Otherwise, setting the shared variables later will
        raise an error. See http://docs.pymc.io/advanced_theano.html

        Returns
        ----------
        the PyMC3 model
        """
        model_input = theano.shared(
            np.zeros([self.num_training_samples, self.num_pred]))

        model_output = theano.shared(
            np.zeros(self.num_training_samples, dtype='int'))

        model_cats = theano.shared(
            np.zeros(self.num_training_samples, dtype='int'))

        self.shared_vars = {
            'model_input': model_input,
            'model_output': model_output,
            'model_cats': model_cats
        }

        model = pm.Model()

        with model:
            mu_alpha = pm.Normal('mu_alpha', mu=0, sd=100)
            sigma_alpha = pm.HalfNormal('sigma_alpha', sd=100)

            mu_beta = pm.Normal('mu_beta', mu=0, sd=100)
            sigma_beta = pm.HalfNormal('sigma_beta', sd=100)

            alpha = pm.Normal('alpha', mu=mu_alpha, sd=sigma_alpha,
                              shape=(self.num_cats,))
            betas = pm.Normal('beta', mu=mu_beta, sd=sigma_beta,
                              shape=(self.num_cats, self.num_pred))

            c = model_cats

            linear_function = alpha[c] + tt.sum(betas[c] * model_input, 1)

            p = pm.invlogit(linear_function)

            y = pm.Bernoulli('y', p, observed=model_output)

        return model

    def save(self, file_prefix):
        params = {
            'inference_type': self.inference_type,
            'num_cats': self.num_cats,
            'num_pred': self.num_pred,
            'num_training_samples': self.num_training_samples
        }

        super(HierarchicalLogisticRegression, self).save(file_prefix, params)

    def load(self, file_prefix):
        params = super(HierarchicalLogisticRegression,
                       self).load(file_prefix, load_custom_params=True)

        self.inference_type = params['inference_type']
        self.num_cats = params['num_cats']
        self.num_pred = params['num_pred']
        self.num_training_samples = params['num_training_samples']