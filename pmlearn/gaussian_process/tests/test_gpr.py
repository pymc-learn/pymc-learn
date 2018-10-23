"""Testing for Gaussian process regression """

# Authors: Daniel Emaasit <daniel.emaasit@gmail.com>
#
# License: BSD 3 clause

import pytest
import numpy.testing as npt
import pandas.testing as pdt
import shutil
import tempfile

import numpy as np
import pymc3 as pm
from pymc3 import summary
from sklearn.gaussian_process import \
    GaussianProcessRegressor as skGaussianProcessRegressor
from sklearn.model_selection import train_test_split


from pmlearn.exceptions import NotFittedError
from pmlearn.gaussian_process import (GaussianProcessRegressor,
                                      StudentsTProcessRegressor,
                                      SparseGaussianProcessRegressor)


class TestGaussianProcessRegressor(object):

    def setup_method(self):
        self.num_pred = 1
        self.num_training_samples = 300

        self.length_scale = 1.0
        self.signal_variance = 0.1
        self.noise_variance = 0.1

        X = np.linspace(start=0, stop=10,
                        num=self.num_training_samples)[:, None]

        cov_func = self.signal_variance ** 2 * pm.gp.cov.ExpQuad(
            1, self.length_scale)
        mean_func = pm.gp.mean.Zero()

        f_true = np.random.multivariate_normal(
            mean_func(X).eval(),
            cov_func(X).eval() + 1e-8 * np.eye(self.num_training_samples),
            1).flatten()
        y = f_true + \
            self.noise_variance * np.random.randn(self.num_training_samples)

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.3)

        self.advi_gpr = GaussianProcessRegressor()
        
        self.test_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Tear down
        """
        shutil.rmtree(self.test_dir)


class TestGaussianProcessRegressorFit(TestGaussianProcessRegressor):
    def test_advi_fit_returns_correct_model(self):
        # This print statement ensures PyMC3 output won't overwrite
        # the test name
        print('')
        self.advi_gpr.fit(self.X_train, self.y_train,
                          inference_args={"n": 25000})

        npt.assert_equal(self.num_pred, self.advi_gpr.num_pred)
        npt.assert_almost_equal(
            self.signal_variance,
            self.advi_gpr.summary['mean']['signal_variance__0'],
            0)
        npt.assert_almost_equal(
            self.length_scale,
            self.advi_gpr.summary['mean']['length_scale__0_0'],
            0)
        npt.assert_almost_equal(
            self.noise_variance,
            self.advi_gpr.summary['mean']['noise_variance__0'],
            0)


class TestGaussianProcessRegressorPredict(TestGaussianProcessRegressor):
    def test_predict_returns_predictions(self):
        print('')
        self.advi_gpr.fit(self.X_train, self.y_train,
                          inference_args={"n": 25000})
        preds = self.advi_gpr.predict(self.X_test)
        npt.assert_equal(self.y_test.shape, preds.shape)

    def test_predict_returns_mean_predictions_and_std(self):
        print('')
        self.advi_gpr.fit(self.X_train, self.y_train,
                          inference_args={"n": 25000})
        preds, stds = self.advi_gpr.predict(self.X_test, return_std=True)
        npt.assert_equal(self.y_test.shape, preds.shape)
        npt.assert_equal(self.y_test.shape, stds.shape)

    def test_predict_raises_error_if_not_fit(self):
        print('')
        with pytest.raises(NotFittedError):
            advi_gpr = GaussianProcessRegressor()
            advi_gpr.predict(self.X_train)


class TestGaussianProcessRegressorScore(TestGaussianProcessRegressor):
    def test_score_matches_sklearn_performance(self):
        print('')
        sk_gpr = skGaussianProcessRegressor()
        sk_gpr.fit(self.X_train, self.y_train)
        sk_gpr_score = sk_gpr.score(self.X_test, self.y_test)

        self.advi_gpr.fit(self.X_train, self.y_train,
                          inference_args={"n": 25000})
        advi_gpr_score = self.advi_gpr.score(self.X_test, self.y_test)

        npt.assert_almost_equal(sk_gpr_score, advi_gpr_score, 1)


class TestGaussianProcessRegressorSaveAndLoad(TestGaussianProcessRegressor):
    def test_save_and_load_work_correctly(self):
        print('')
        self.advi_gpr.fit(self.X_train, self.y_train,
                          inference_args={"n": 25000})
        score1 = self.advi_gpr.score(self.X_test, self.y_test)
        self.advi_gpr.save(self.test_dir)

        gpr2 = GaussianProcessRegressor()
        gpr2.load(self.test_dir)

        npt.assert_equal(self.advi_gpr.inference_type, gpr2.inference_type)
        npt.assert_equal(self.advi_gpr.num_pred, gpr2.num_pred)
        npt.assert_equal(self.advi_gpr.num_training_samples,
                         gpr2.num_training_samples)
        pdt.assert_frame_equal(summary(self.advi_gpr.trace),
                               summary(gpr2.trace))

        score2 = gpr2.score(self.X_test, self.y_test)
        npt.assert_almost_equal(score1, score2, 0)


class TestStudentsTProcessRegressor(object):

    def setup_method(self):
        self.num_pred = 1
        self.num_training_samples = 500

        self.length_scale = 1.0
        self.signal_variance = 0.1
        self.noise_variance = 0.1
        self.degrees_of_freedom = 1.0

        X = np.linspace(start=0, stop=10,
                        num=self.num_training_samples)[:, None]

        cov_func = self.signal_variance ** 2 * pm.gp.cov.ExpQuad(
            1, self.length_scale)
        mean_func = pm.gp.mean.Zero()

        f_true = np.random.multivariate_normal(
            mean_func(X).eval(),
            cov_func(X).eval() + 1e-8 * np.eye(self.num_training_samples),
            1).flatten()
        y = f_true + \
            self.noise_variance * \
            np.random.standard_t(self.degrees_of_freedom,
                                 size=self.num_training_samples)

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.3)

        self.advi_stpr = StudentsTProcessRegressor()

        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)


class TestStudentsTProcessRegressorFit(TestStudentsTProcessRegressor):
    def test_advi_fit_returns_correct_model(self):
        # This print statement ensures PyMC3 output won't overwrite
        # the test name
        print('')
        self.advi_stpr.fit(self.X_train, self.y_train,
                           inference_args={"n": 25000})

        npt.assert_equal(self.num_pred, self.advi_stpr.num_pred)
        npt.assert_almost_equal(
            self.signal_variance,
            self.advi_stpr.summary['mean']['signal_variance__0'],
            0)
        npt.assert_almost_equal(
            self.length_scale,
            self.advi_stpr.summary['mean']['length_scale__0_0'],
            0)
        npt.assert_almost_equal(
            self.noise_variance,
            self.advi_stpr.summary['mean']['noise_variance__0'],
            0)


class TestStudentsTProcessRegressorPredict(TestStudentsTProcessRegressor):
    def test_predict_returns_predictions(self):
        print('')
        self.advi_stpr.fit(self.X_train, self.y_train,
                           inference_args={"n": 25000})
        preds = self.advi_stpr.predict(self.X_test)
        npt.assert_equal(self.y_test.shape, preds.shape)

    def test_predict_returns_mean_predictions_and_std(self):
        print('')
        self.advi_stpr.fit(self.X_train, self.y_train,
                           inference_args={"n": 25000})
        preds, stds = self.advi_stpr.predict(self.X_test, return_std=True)
        npt.assert_equal(self.y_test.shape, preds.shape)
        npt.assert_equal(self.y_test.shape, stds.shape)

    def test_predict_raises_error_if_not_fit(self):
        print('')
        with pytest.raises(NotFittedError):
            advi_stpr = StudentsTProcessRegressor()
            advi_stpr.predict(self.X_train)


class TestStudentsTProcessRegressorScore(TestStudentsTProcessRegressor):
    def test_score_matches_sklearn_performance(self):
        print('')
        sk_gpr = skGaussianProcessRegressor()
        sk_gpr.fit(self.X_train, self.y_train)
        sk_gpr_score = sk_gpr.score(self.X_test, self.y_test)

        self.advi_stpr.fit(self.X_train, self.y_train,
                           inference_args={"n": 25000})
        advi_stpr_score = self.advi_stpr.score(self.X_test, self.y_test)

        npt.assert_almost_equal(sk_gpr_score, advi_stpr_score, 0)


class TestStudentsTProcessRegressorSaveAndLoad(TestStudentsTProcessRegressor):
    def test_save_and_load_work_correctly(self):
        print('')
        self.advi_stpr.fit(self.X_train, self.y_train,
                           inference_args={"n": 25000})
        score1 = self.advi_stpr.score(self.X_test, self.y_test)
        self.advi_stpr.save(self.test_dir)

        stpr2 = StudentsTProcessRegressor()
        stpr2.load(self.test_dir)

        npt.assert_equal(self.advi_stpr.inference_type, stpr2.inference_type)
        npt.assert_equal(self.advi_stpr.num_pred, stpr2.num_pred)
        npt.assert_equal(self.advi_stpr.num_training_samples,
                         stpr2.num_training_samples)
        pdt.assert_frame_equal(summary(self.advi_stpr.trace),
                               summary(stpr2.trace))

        score2 = stpr2.score(self.X_test, self.y_test)
        npt.assert_almost_equal(score1, score2, 0)


class TestSparseGaussianProcessRegressor(object):

    def setup_method(self):
        self.num_pred = 1
        self.num_training_samples = 1000

        self.length_scale = 1.0
        self.signal_variance = 0.1
        self.noise_variance = 0.1

        X = np.linspace(start=0, stop=10,
                        num=self.num_training_samples)[:, None]

        cov_func = self.signal_variance ** 2 * pm.gp.cov.ExpQuad(
            1, self.length_scale)
        mean_func = pm.gp.mean.Zero()

        f_true = np.random.multivariate_normal(
            mean_func(X).eval(),
            cov_func(X).eval() + 1e-8 * np.eye(self.num_training_samples),
            1).flatten()
        y = f_true + \
            self.noise_variance * np.random.randn(self.num_training_samples)

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.3)

        self.advi_sgpr = SparseGaussianProcessRegressor()

        self.test_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Tear down
        """
        shutil.rmtree(self.test_dir)


class TestSparseGaussianProcessRegressorFit(TestSparseGaussianProcessRegressor):
    def test_advi_fit_returns_correct_model(self):
        # This print statement ensures PyMC3 output won't overwrite
        # the test name
        print('')
        self.advi_sgpr.fit(self.X_train, self.y_train)

        npt.assert_equal(self.num_pred, self.advi_sgpr.num_pred)
        npt.assert_almost_equal(
            self.signal_variance,
            self.advi_sgpr.summary['mean']['signal_variance__0'],
            0)
        npt.assert_almost_equal(
            self.length_scale,
            self.advi_sgpr.summary['mean']['length_scale__0_0'],
            0)
        npt.assert_almost_equal(
            self.noise_variance,
            self.advi_sgpr.summary['mean']['noise_variance__0'],
            0)


class TestSparseGaussianProcessRegressorPredict(
    TestSparseGaussianProcessRegressor):

    def test_predict_returns_predictions(self):
        print('')
        self.advi_sgpr.fit(self.X_train, self.y_train,
                           inference_args={"n": 25000})
        preds = self.advi_sgpr.predict(self.X_test)
        npt.assert_equal(self.y_test.shape, preds.shape)

    def test_predict_returns_mean_predictions_and_std(self):
        print('')
        self.advi_sgpr.fit(self.X_train, self.y_train,
                           inference_args={"n": 25000})
        preds, stds = self.advi_sgpr.predict(self.X_test, return_std=True)
        npt.assert_equal(self.y_test.shape, preds.shape)
        npt.assert_equal(self.y_test.shape, stds.shape)

    def test_predict_raises_error_if_not_fit(self):
        print('')
        with pytest.raises(NotFittedError):
            advi_sgpr = SparseGaussianProcessRegressor()
            advi_sgpr.predict(self.X_train)


class TestSparseGaussianProcessRegressorScore(
    TestSparseGaussianProcessRegressor):

    def test_score_matches_sklearn_performance(self):
        print('')
        sk_gpr = skGaussianProcessRegressor()
        sk_gpr.fit(self.X_train, self.y_train)
        sk_gpr_score = sk_gpr.score(self.X_test, self.y_test)

        self.advi_sgpr.fit(self.X_train, self.y_train)
        advi_sgpr_score = self.advi_sgpr.score(self.X_test, self.y_test)

        npt.assert_almost_equal(sk_gpr_score, advi_sgpr_score, 0)


class TestSparseGaussianProcessRegressorSaveAndLoad(
    TestSparseGaussianProcessRegressor):

    def test_save_and_load_work_correctly(self):
        print('')
        self.advi_sgpr.fit(self.X_train, self.y_train)
        score1 = self.advi_sgpr.score(self.X_test, self.y_test)
        self.advi_sgpr.save(self.test_dir)

        sgpr2 = SparseGaussianProcessRegressor()
        sgpr2.load(self.test_dir)

        npt.assert_equal(self.advi_sgpr.inference_type, sgpr2.inference_type)
        npt.assert_equal(self.advi_sgpr.num_pred, sgpr2.num_pred)
        npt.assert_equal(self.advi_sgpr.num_training_samples,
                         sgpr2.num_training_samples)
        pdt.assert_frame_equal(summary(self.advi_sgpr.trace),
                               summary(sgpr2.trace))

        score2 = sgpr2.score(self.X_test, self.y_test)
        npt.assert_almost_equal(score1, score2, 0)
