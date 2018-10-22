"""Testing for Linear regression """

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
from sklearn.linear_model import LinearRegression as skLinearRegression
from sklearn.model_selection import train_test_split

from pmlearn.exceptions import NotFittedError
from pmlearn.linear_model import LinearRegression


class TestLinearRegression(object):

    def setup_method(self):
        self.num_pred = 1
        self.alpha = 2
        self.betas = 3
        self.s = 1

        X = np.random.randn(1000, 1)
        noise = self.s * np.random.randn(1000, 1)
        y = self.betas * X + self.alpha + noise
        y = np.squeeze(y)

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.4)

        self.advi_lr = LinearRegression()
        self.nuts_lr = LinearRegression()

        self.test_dir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.test_dir)


class TestLinearRegressionFit(TestLinearRegression):
    def test_advi_fit_returns_correct_model(self):
        # Note: print is here so PyMC3 output won't overwrite the test name
        print('')
        self.advi_lr.fit(self.X_train, self.y_train)

        npt.assert_equal(self.num_pred, self.advi_lr.num_pred)

        npt.assert_almost_equal(self.alpha,
                                self.advi_lr.summary['mean']['alpha__0'],
                                decimal=1)
        npt.assert_almost_equal(self.betas,
                                self.advi_lr.summary['mean']['betas__0_0'],
                                decimal=1)
        npt.assert_almost_equal(self.s,
                                self.advi_lr.summary['mean']['s'],
                                decimal=1)

    def test_nuts_fit_returns_correct_model(self):
        # Note: print is here so PyMC3 output won't overwrite the test name
        print('')
        self.nuts_lr.fit(self.X_train, self.y_train, inference_type='nuts',
                         inference_args={'draws': 2000})

        npt.assert_equal(self.num_pred, self.nuts_lr.num_pred)

        npt.assert_almost_equal(self.alpha,
                                self.nuts_lr.summary['mean']['alpha__0'],
                                decimal=1)
        npt.assert_almost_equal(self.betas,
                                self.nuts_lr.summary['mean']['betas__0_0'],
                                decimal=1)
        npt.assert_almost_equal(self.s,
                                self.nuts_lr.summary['mean']['s'],
                                decimal=1)


class TestLinearRegressionPredict(TestLinearRegression):
    def test_predict_returns_predictions(self):
        print('')
        self.advi_lr.fit(self.X_train, self.y_train)
        preds = self.advi_lr.predict(self.X_test)
        npt.assert_equal(preds.shape, self.y_test.shape)

    def test_predict_returns_mean_predictions_and_std(self):
        print('')
        self.advi_lr.fit(self.X_train, self.y_train)
        preds, stds = self.advi_lr.predict(self.X_test, return_std=True)
        npt.assert_equal(preds.shape, self.y_test.shape)
        npt.assert_equal(stds.shape, self.y_test.shape)

    def test_predict_raises_error_if_not_fit(self):
        with pytest.raises(NotFittedError):
            lr = LinearRegression()
            lr.predict(self.X_train)


class TestLinearRegressionScore(TestLinearRegression):
    def test_score_matches_sklearn_performance(self):
        print('')
        sklr = skLinearRegression()
        sklr.fit(self.X_train, self.y_train)
        sklr_score = sklr.score(self.X_test, self.y_test)

        self.advi_lr.fit(self.X_train, self.y_train)
        score = self.advi_lr.score(self.X_test, self.y_test)
        npt.assert_almost_equal(sklr_score, score, decimal=1)


class TestLinearRegressionSaveandLoad(TestLinearRegression):
    def test_save_and_load_work_correctly(self):
        print('')
        self.advi_lr.fit(self.X_train, self.y_train)
        score1 = self.advi_lr.score(self.X_test, self.y_test)
        self.advi_lr.save(self.test_dir)

        lr2 = LinearRegression()
        lr2.load(self.test_dir)

        npt.assert_equal(self.advi_lr.inference_type, lr2.inference_type)
        npt.assert_equal(self.advi_lr.num_pred, lr2.num_pred)
        npt.assert_equal(self.advi_lr.num_training_samples,
                         lr2.num_training_samples)
        pdt.assert_frame_equal(summary(self.advi_lr.trace), summary(lr2.trace))

        score2 = lr2.score(self.X_test, self.y_test)
        npt.assert_almost_equal(score1, score2, decimal=1)
