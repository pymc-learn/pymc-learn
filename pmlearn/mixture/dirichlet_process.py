"""Dirichlet Process Mixture Model. """

# Authors: Daniel Emaasit <daniel.emaasit@gmail.com>
#
# License: BSD 3 clause

import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt

from ..exceptions import PymcLearnError
from ..base import BayesianModel, BayesianDensityMixin
from .util import logp_gmix


class DirichletProcessMixture(BayesianModel, BayesianDensityMixin):
    """
    Custom Dirichlet Process Mixture Model built using PyMC3.
    """

    def __init__(self):
        super(DirichletProcessMixture, self).__init__()
        self.num_truncate = None

    def create_model(self):
        """
        Creates and returns the PyMC3 model.

        Note: The size of the shared variables must match the size of the
        training data. Otherwise, setting the shared variables later will raise
        an error. See http://docs.pymc.io/advanced_theano.html

        The DensityDist class is used as the likelihood term. The second
        argument, logp_gmix(mus, pi, np.eye(D)), is a python function which
        recieves observations (denoted by 'value') and returns the tensor
        representation of the log-likelihood.

        Returns
        ----------
        the PyMC3 model
        """
        model_input = theano.shared(np.zeros([self.num_training_samples,
                                              self.num_pred]))

        # model_output = theano.shared(np.zeros(self.num_training_samples))

        # model_truncate = theano.shared(np.zeros(self.num_training_samples,
        #                                     dtype='int'))

        self.shared_vars = {
            'model_input': model_input
            # ,
            # 'model_output': model_output,
            # 'model_truncate': model_truncate
        }

        # Log likelihood of normal distribution
        # def logp_normal(mu, tau, value):
        #     # log probability of individual samples
        #     k = tau.shape[0]
        #
        #     def delta(mu):
        #         return value - mu
        #     # delta = lambda mu: value - mu
        #     return (-1 / 2.) * (k * T.log(2 * np.pi) + T.log(1./det(tau)) +
        #                          (delta(mu).dot(tau) * delta(
        # mu)).sum(axis=1))

        # Log likelihood of Gaussian mixture distribution
        # def logp_gmix(mus, pi, tau):
        #     def logp_(value):
        #         logps = [T.log(pi[i]) + logp_normal(mu, tau, value)
        #                  for i, mu in enumerate(mus)]
        #
        #         return T.sum(
        # logsumexp(T.stacklists(logps)[:, :self.num_training_samples],
        # axis=0))
        #
        #     return logp_

        def stick_breaking(v):
            portion_remaining = tt.concatenate(
                [[1], tt.extra_ops.cumprod(1 - v)[:-1]])
            return v * portion_remaining

        model = pm.Model()

        with model:

            K = self.num_truncate
            D = self.num_pred

            alpha = pm.Gamma('alpha', 1.0, 1.0)
            v = pm.Beta('v', 1, alpha, shape=K)
            pi_ = stick_breaking(v)
            pi = pm.Deterministic('pi', pi_/pi_.sum())

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

            component_dists = [pm.MvNormal(
                'component_dist_%d' % k,
                mu=means[k],
                chol=chol[k],
                shape=D) for k in range(K)]

            # rand = [pm.MvNormal(
            # 'rand_{}'.format(k),
            # mu=means[k], chol=Chol[k], shape=D) for k in range(K)]
            rand = pm.Normal.dist(0, 1).random

            X = pm.DensityDist(
                'X',
                logp_gmix(
                    mus=component_dists, pi=pi,
                    tau=np.eye(D),
                    num_training_samples=model_input.get_value().shape[0]),
                observed=model_input, random=rand)

        return model

    def predict_proba(self, X, return_std=False):
        """
        Predicts probabilities of new data with a trained Dirichlet Process
        Mixture Model

        Parameters
        ----------
        X : numpy array, shape [n_samples, n_features]

        cats : numpy array, shape [n_samples, ]

        return_std : Boolean flag
           Boolean flag of whether to return standard deviations with mean
           probabilities. Defaults to False.
        """

        if self.trace is None:
            raise PymcLearnError('Run fit on the model before predict.')

        # num_samples = X.shape[0]

        if self.cached_model is None:
            self.cached_model = self.create_model()

        self._set_shared_vars({'model_input': X})
        _vars = self.cached_model.free_RVs[8:11]

        ppc = pm.sample_ppc(self.trace,
                            model=self.cached_model,
                            vars=_vars,
                            samples=2000,
                            size=len(X))
        return(ppc)

        # if return_std:
        #     return ppc['X'].mean(axis=0), ppc['X'].std(axis=0)
        # else:
        #     return ppc['X'].mean(axis=0)

    def save(self, file_prefix):
        params = {
            'inference_type': self.inference_type,
            # 'num_cats': self.num_cats,
            'num_pred': self.num_pred,
            'num_training_samples': self.num_training_samples
        }

        super(DirichletProcessMixture, self).save(file_prefix, params)

    def load(self, file_prefix):
        params = super(DirichletProcessMixture, self).load(
            file_prefix, load_custom_params=True)

        self.inference_type = params['inference_type']
        # self.num_cats = params['num_cats']
        self.num_pred = params['num_pred']
        self.num_training_samples = params['num_training_samples']
