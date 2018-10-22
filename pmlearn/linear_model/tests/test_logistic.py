"""Testing for Logistic regression """

# Authors: Daniel Emaasit <daniel.emaasit@gmail.com>
#
# License: BSD 3 clause

import pytest
import numpy.testing as npt
import pandas.testing as pdt
import shutil
import tempfile

import numpy as np
from pymc3 import summary
from sklearn.model_selection import train_test_split

from pmlearn.exceptions import NotFittedError
from pmlearn.linear_model import HierarchicalLogisticRegression


class TestHierarchicalLogisticRegression(object):
    def setup_method(self):
        def numpy_invlogit(x):
            return 1 / (1 + np.exp(-x))

        self.num_cats = 3
        self.num_pred = 1
        self.num_samples_per_cat = 1000

        self.alphas = np.random.randn(self.num_cats)
        self.betas = np.random.randn(self.num_cats, self.num_pred)
        # TODO: make this more efficient; right now, it's very explicit
        # so I understand it.
        x_a = np.random.randn(self.num_samples_per_cat, self.num_pred)
        y_a = np.random.binomial(1,
                                 numpy_invlogit(self.alphas[0] +
                                                np.sum(self.betas[0] * x_a, 1)
                                                ))
        x_b = np.random.randn(self.num_samples_per_cat, self.num_pred)
        y_b = np.random.binomial(1,
                                 numpy_invlogit(self.alphas[1] +
                                                np.sum(self.betas[1] * x_b, 1)
                                                ))
        x_c = np.random.randn(self.num_samples_per_cat, self.num_pred)
        y_c = np.random.binomial(1,
                                 numpy_invlogit(self.alphas[2] +
                                                np.sum(self.betas[2] * x_c, 1)
                                                ))

        X = np.concatenate([x_a, x_b, x_c])
        y = np.concatenate([y_a, y_b, y_c])
        cats = np.concatenate([
            np.zeros(self.num_samples_per_cat, dtype=np.int),
            np.ones(self.num_samples_per_cat, dtype=np.int),
            2*np.ones(self.num_samples_per_cat, dtype=np.int)
        ])

        self.X_train, self.X_test, self.cat_train, self.cat_test, \
        self.y_train, self.y_test = train_test_split(
            X, cats, y, test_size=0.4
        )

        self.advi_hlr = HierarchicalLogisticRegression()

        self.test_dir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.test_dir)


class TestHierarchicalLogisticRegressionFit(TestHierarchicalLogisticRegression):
    def test_advi_fit_returns_correct_model(self):
        # Note: print is here so PyMC3 output won't overwrite the test name
        print('')
        self.advi_hlr.fit(self.X_train, self.y_train, self.cat_train,
                          minibatch_size=500, inference_args={"n": 50000})

        npt.assert_equal(self.num_cats, self.advi_hlr.num_cats)
        npt.assert_equal(self.num_pred, self.advi_hlr.num_pred)

        #TODO: Figure out best way to test
        #np.testing.assert_almost_equal(self.alphas,
        # self.advi_hlr.trace['alphas'].mean(), decimal=1)
        #np.testing.assert_almost_equal(self.betas,
        # self.advi_hlr.trace['betas'].mean(), decimal=1)

        # For now, just check that the estimated parameters
        # have the correct signs
        npt.assert_equal(
            np.sign(self.alphas),
            np.sign(self.advi_hlr.trace['alpha'].mean(axis=0))
        )
        npt.assert_equal(
            np.sign(self.betas),
            np.sign(self.advi_hlr.trace['beta'].mean(axis=0))
        )


class TestHierarchicalLogisticRegressionPredictProba(
    TestHierarchicalLogisticRegression):

    def test_predict_proba_returns_probabilities(self):
        print('')
        self.advi_hlr.fit(self.X_train, self.y_train, self.cat_train,
                          minibatch_size=500, inference_args={"n": 50000})
        probs = self.advi_hlr.predict_proba(self.X_test, self.cat_test)
        npt.assert_equal(probs.shape, self.y_test.shape)

    def test_predict_proba_returns_probabilities_and_std(self):
        print('')
        self.advi_hlr.fit(self.X_train, self.y_train, self.cat_train,
                          minibatch_size=500, inference_args={"n": 50000})
        probs, stds = self.advi_hlr.predict_proba(self.X_test, self.cat_test,
                                                  return_std=True)
        npt.assert_equal(probs.shape, self.y_test.shape)
        npt.assert_equal(stds.shape, self.y_test.shape)

    def test_predict_proba_raises_error_if_not_fit(self):
        with pytest.raises(NotFittedError):
            advi_hlr = HierarchicalLogisticRegression()
            advi_hlr.predict_proba(self.X_train, self.cat_train)


class TestHierarchicalLogisticRegressionPredict(
    TestHierarchicalLogisticRegression):

    def test_predict_returns_predictions(self):
        print('')
        self.advi_hlr.fit(self.X_train, self.y_train, self.cat_train,
                          minibatch_size=500, inference_args={"n": 50000})
        preds = self.advi_hlr.predict(self.X_test, self.cat_test)
        npt.assert_equal(preds.shape, self.y_test.shape)


class TestHierarchicalLogisticRegressionScore(
    TestHierarchicalLogisticRegression):

    def test_score_scores(self):
        print('')
        self.advi_hlr.fit(self.X_train, self.y_train, self.cat_train,
                          minibatch_size=500, inference_args={"n": 50000})
        score = self.advi_hlr.score(self.X_test, self.y_test, self.cat_test)
        naive_score = np.mean(self.y_test)
        npt.assert_array_less(naive_score, score)


class TestHierarchicalLogisticRegressionSaveandLoad(
    TestHierarchicalLogisticRegression):

    def test_save_and_load_work_correctly(self):
        print('')
        self.advi_hlr.fit(self.X_train, self.y_train, self.cat_train,
                          minibatch_size=500, inference_args={"n": 50000})
        probs1 = self.advi_hlr.predict_proba(self.X_test, self.cat_test)
        self.advi_hlr.save(self.test_dir)

        hlr2 = HierarchicalLogisticRegression()

        hlr2.load(self.test_dir)

        npt.assert_equal(self.advi_hlr.num_cats, hlr2.num_cats)
        npt.assert_equal(self.advi_hlr.num_pred, hlr2.num_pred)
        npt.assert_equal(self.advi_hlr.num_training_samples,
                         hlr2.num_training_samples)
        pdt.assert_frame_equal(summary(self.advi_hlr.trace),
                               summary(hlr2.trace))

        probs2 = hlr2.predict_proba(self.X_test, self.cat_test)

        npt.assert_almost_equal(probs2, probs1, decimal=1)
