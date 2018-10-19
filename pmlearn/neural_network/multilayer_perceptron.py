"""
Multilayer perceptron
"""

# Authors: Daniel Emaasit <daniel.emaasit@gmail.com>
#
# License: BSD 3 clause

import numpy as np
import pymc3 as pm
import theano

from ..base import BayesianModel, BayesianClassifierMixin

floatX = theano.config.floatX


class MLPClassifier(BayesianModel, BayesianClassifierMixin):
    """ Multilayer perceptron classification built using PyMC3.

    Fit a Multilayer perceptron classification model and estimate
    model parameters using
    MCMC algorithms or Variational Inference algorithms

    Parameters
    ----------


    Examples
    --------


    Reference
    ----------
    http://twiecki.github.io/blog/2016/06/01/bayesian-deep-learning/
    """
    def __init__(self, n_hidden=5):
        self.n_hidden = n_hidden
        self.num_training_samples = None
        self.num_pred = None
        self.total_size = None

        super(MLPClassifier, self).__init__()

    def create_model(self):
        """

        Returns
        -------

        """
        model_input = theano.shared(np.zeros([self.num_training_samples,
                                              self.num_pred]))

        model_output = theano.shared(np.zeros(self.num_training_samples))

        self.shared_vars = {
            'model_input': model_input,
            'model_output': model_output,
        }

        self.total_size = len(model_output.get_value())

        # Initialize random weights between each layer
        init_1 = np.random.randn(self.num_pred, self.n_hidden).astype(floatX)
        init_2 = np.random.randn(self.n_hidden, self.n_hidden).astype(floatX)
        init_out = np.random.randn(self.n_hidden).astype(floatX)

        model = pm.Model()

        with model:
            # Weights from input to hidden layer
            weights_in_1 = pm.Normal('w_in_1', 0, sd=1,
                                     shape=(self.num_pred, self.n_hidden),
                                     testval=init_1)

            # Weights from 1st to 2nd layer
            weights_1_2 = pm.Normal('w_1_2', 0, sd=1,
                                    shape=(self.n_hidden, self.n_hidden),
                                    testval=init_2)

            # Weights from hidden layer to output
            weights_2_out = pm.Normal('w_2_out', 0, sd=1,
                                      shape=(self.n_hidden,),
                                      testval=init_out)

            # Build neural-network using tanh activation function
            act_1 = pm.math.tanh(pm.math.dot(model_input, weights_in_1))
            act_2 = pm.math.tanh(pm.math.dot(act_1, weights_1_2))
            act_out = pm.math.sigmoid(pm.math.dot(act_2, weights_2_out))

            # Binary classification -> Bernoulli likelihood
            y = pm.Bernoulli('y',
                             act_out,
                             observed=model_output,
                             total_size=self.total_size)
        return model

    def save(self, file_prefix):
        params = {
            'inference_type': self.inference_type,
            'num_pred': self.num_pred,
            'num_training_samples': self.num_training_samples
        }

        super(MLPClassifier, self).save(file_prefix, params)

    def load(self, file_prefix):
        params = super(MLPClassifier, self).load(file_prefix,
                                                 load_custom_params=True)

        self.inference_type = params['inference_type']
        self.num_pred = params['num_pred']
        self.num_training_samples = params['num_training_samples']
