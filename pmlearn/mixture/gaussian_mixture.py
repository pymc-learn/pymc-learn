"""Gaussian Mixture Model. """

# Authors: Daniel Emaasit <daniel.emaasit@gmail.com>
#
# License: BSD 3 clause

import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt

from ..base import BayesianModel, BayesianDensityMixin


class GaussianMixture(BayesianModel, BayesianDensityMixin):
    """
    Custom Gaussian Mixture Model built using PyMC3.
    """

    def __init__(self):
        super(GaussianMixture, self).__init__()
        self.num_components = None

    def create_model(self):
        """
        Creates and returns the PyMC3 model.

        Note: The size of the shared variables must match the size of the
        training data. Otherwise, setting the shared variables later will raise
        an error. See http://docs.pymc.io/advanced_theano.html

        Returns
        ----------
        the PyMC3 model
        """
        model_input = theano.shared(np.zeros([self.num_training_samples,
                                              self.num_pred]))

        # model_output = theano.shared(np.zeros(self.num_training_samples))

        model_components = theano.shared(np.zeros(self.num_training_samples,
                                                  dtype='int'))

        self.shared_vars = {
            'model_input': model_input
            # ,
            # 'model_output': model_output,
            # 'model_components': model_components
        }

        model = pm.Model()

        with model:

            K = self.num_components
            D = self.num_pred

            pi = pm.Dirichlet("pi", a=np.ones(K) / K, shape=K)
            means = tt.stack([pm.Uniform('cluster_center_{}'.format(k),
                                         lower=0.,
                                         upper=10.,
                                         shape=D) for k in range(K)])
            lower = tt.stack([pm.LKJCholeskyCov(
                'cluster_variance_{}'.format(k),
                n=D,
                eta=2.,
                sd_dist=pm.HalfNormal.dist(sd=1.)) for k in range(K)])
            chol = tt.stack([pm.expand_packed_triangular(
                D, lower[k]) for k in range(K)])

            component_dists = [pm.MvNormal.dist(mu=means[k],
                                                chol=chol[k],
                                                shape=D) for k in range(K)]

            X = pm.Mixture("X", w=pi, comp_dists=component_dists,
                           observed=model_input)

        return model

    def save(self, file_prefix):
        params = {
            'inference_type': self.inference_type,
            # 'num_cats': self.num_cats,
            'num_pred': self.num_pred,
            'num_training_samples': self.num_training_samples
        }

        super(GaussianMixture, self).save(file_prefix, params)

    def load(self, file_prefix):
        params = super(GaussianMixture, self).load(
            file_prefix, load_custom_params=True)

        self.inference_type = params['inference_type']
        # self.num_cats = params['num_cats']
        self.num_pred = params['num_pred']
        self.num_training_samples = params['num_training_samples']
