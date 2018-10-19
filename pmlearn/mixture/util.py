import numpy as np
# import pymc3 as pm
from pymc3.math import logsumexp
# import theano
import theano.tensor as T
from theano.tensor.nlinalg import det


class logp_gmix(object):
    def __init__(self, mus, pi, tau, num_training_samples):
        self.mus = mus
        self.pi = pi
        self.tau = tau
        self.num_training_samples = num_training_samples

    def __call__(self, value):

        def logp_normal(mu, tau, value):
            # log probability of individual samples
            k = tau.shape[0]

            def delta(mu):
                return value - mu
            # delta = lambda mu: value - mu
            return (-1 / 2.) * (k * T.log(2 * np.pi) + T.log(1. / det(tau)) +
                                (delta(mu).dot(tau) * delta(mu)).sum(axis=1))

        logps = [T.log(self.pi[i]) + logp_normal(mu, self.tau, value)
                 for i, mu in enumerate(self.mus)]

        return T.sum(logsumexp(T.stacklists(logps)[:, :self.num_training_samples],
                               axis=0))

        # Log likelihood of normal distribution
        # def logp_normal(mu, tau, value):
        #     # log probability of individual samples
        #     k = tau.shape[0]
        #
        #     def delta(mu):
        #         return value - mu
        #     # delta = lambda mu: value - mu
        #     return (-1 / 2.) * (k * T.log(2 * np.pi) + T.log(1./det(tau)) +
        #                          (delta(mu).dot(tau) * delta(mu)).sum(axis=1))

        # Log likelihood of Gaussian mixture distribution
        # def logp_gmix(mus, pi, tau):
        #     def logp_(value):
        #         logps = [T.log(pi[i]) + logp_normal(mu, tau, value)
        #                  for i, mu in enumerate(mus)]
        #
        #         return T.sum(logsumexp(T.stacklists(logps)[:, :self.num_training_samples], axis=0))
        #
        #     return logp_
