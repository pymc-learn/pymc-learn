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
from pymc3 import summary
from sklearn.gaussian_process import \
    GaussianProcessRegressor as skGaussianProcessRegressor
from sklearn.model_selection import train_test_split


from pmlearn.exceptions import NotFittedError
from pmlearn.gaussian_process import (GaussianProcessRegressor,
                                      SparseGaussianProcessRegressor,
                                      StudentsTProcessRegressor)


class TestGaussianProcessRegressor(object):
    """
    Compare the logp of GPR models in pmlearn to sklearn
    """

    def setup_method(self):
        """Setup the data for testing
        """
        self.num_pred = 1
        self.num_training_samples = 20
        self.length_scale = 0.1
        self.signal_variance = 0.01
        self.noise_variance = 0.01
        self.X = np.random.randn(self.num_training_samples, self.num_pred)
        self.y = np.random.randn(self.num_training_samples) * \
                 self.noise_variance
        self.Xnew = np.random.randn(50, self.num_pred)
        self.pnew = np.random.randn(50) * self.noise_variance
        with pm.Model() as model:
            cov_func = self.signal_variance**2 * \
                       pm.gp.cov.ExpQuad(self.num_pred, self.length_scale)
            gp = pm.gp.Latent(cov_func=cov_func)
            f = gp.prior("f", self.X, reparameterize=False)
            p = gp.conditional("p", self.Xnew)

        self.latent_logp = model.logp({"f": self.y, "p": self.pnew})
        self.plogp = p.logp({"f": self.y, "p": self.pnew})

        self.test_gpr = GaussianProcessRegressor(kernel=cov_func)
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
        self.test_gpr.fit(self.X, self.y)

        npt.assert_equal(self.num_pred, self.test_gpr.num_pred)
        npt.assert_almost_equal(self.signal_variance,
                               int(self.test_GPR.summary['mean']['signal_variance__0']),
                               0)
        self.assertAlmostEqual(self.length_scale,
                               int(self.test_GPR.summary['mean']['length_scale__0_0']),
                               0)
        self.assertAlmostEqual(self.noise_variance,
                               int(self.test_GPR.summary['mean']['noise_variance__0']),
                               0)

    # def test_nuts_fit_returns_correct_model(self):
    #     # This print statement ensures PyMC3 output won't overwrite the test name
    #     print('')
    #     self.test_nuts_GPR.fit(self.X_train, self.y_train, inference_type='nuts')
    #
    #     self.assertEqual(self.num_pred, self.test_nuts_GPR.num_pred)
    #     self.assertAlmostEqual(self.signal_variance,
    #                            int(self.test_nuts_GPR.summary['mean']['signal_variance__0']),
    #                            0)
    #     self.assertAlmostEqual(self.length_scale,
    #                            int(self.test_nuts_GPR.summary['mean']['length_scale__0_0']),
    #                            0)
    #     self.assertAlmostEqual(self.noise_variance,
    #                            int(self.test_nuts_GPR.summary['mean']['noise_variance__0']),
    #                            0)


# class GaussianProcessRegressorPredictTestCase(GaussianProcessRegressorTestCase):
#     def test_predict_returns_predictions(self):
#         print('')
#         self.test_GPR.fit(self.X_train, self.y_train)
#         preds = self.test_GPR.predict(self.X_test)
#         self.assertEqual(self.y_test.shape, preds.shape)
#
#     def test_predict_returns_mean_predictions_and_std(self):
#         print('')
#         self.test_GPR.fit(self.X_train, self.y_train)
#         preds, stds = self.test_GPR.predict(self.X_test, return_std=True)
#         self.assertEqual(self.y_test.shape, preds.shape)
#         self.assertEqual(self.y_test.shape, stds.shape)
#
#     def test_predict_raises_error_if_not_fit(self):
#         print('')
#         with self.assertRaises(NotFittedError) as no_fit_error:
#             test_GPR = GaussianProcessRegressor()
#             test_GPR.predict(self.X_train)
#
#         expected = 'Run fit on the model before predict.'
#         self.assertEqual(str(no_fit_error.exception), expected)


# class GaussianProcessRegressorScoreTestCase(GaussianProcessRegressorTestCase):
#     def test_score_matches_sklearn_performance(self):
#         print('')
#         skGPR = skGaussianProcessRegressor()
#         skGPR.fit(self.X_train, self.y_train)
#         skGPR_score = skGPR.score(self.X_test, self.y_test)
#
#         self.test_GPR.fit(self.X_train, self.y_train)
#         test_GPR_score = self.test_GPR.score(self.X_test, self.y_test)
#
#         self.assertAlmostEqual(skGPR_score, test_GPR_score, 1)
#
#
# class GaussianProcessRegressorSaveAndLoadTestCase(GaussianProcessRegressorTestCase):
#     def test_save_and_load_work_correctly(self):
#         print('')
#         self.test_GPR.fit(self.X_train, self.y_train)
#         score1 = self.test_GPR.score(self.X_test, self.y_test)
#         self.test_GPR.save(self.test_dir)
#
#         GPR2 = GaussianProcessRegressor()
#         GPR2.load(self.test_dir)
#
#         self.assertEqual(self.test_GPR.inference_type, GPR2.inference_type)
#         self.assertEqual(self.test_GPR.num_pred, GPR2.num_pred)
#         self.assertEqual(self.test_GPR.num_training_samples, GPR2.num_training_samples)
#         pd.testing.assert_frame_equal(summary(self.test_GPR.trace),
#                                       summary(GPR2.trace))
#
#         score2 = GPR2.score(self.X_test, self.y_test)
#         self.assertAlmostEqual(score1, score2, 1)
#
#
# class StudentsTProcessRegressorTestCase(unittest.TestCase):
#
#     def setUp(self):
#         self.num_training_samples = 150
#         self.num_pred = 1
#
#         self.length_scale = 2.0
#         self.noise_variance = 1.0
#         self.signal_variance = 1.0
#         self.degrees_of_freedom = 3.0
#
#         X = np.linspace(start=0, stop=10, num=self.num_training_samples)[:, None]
#         cov_func = self.signal_variance**2 * pm.gp.cov.ExpQuad(self.num_pred,
#                                                                self.length_scale)
#
#         mean_func = pm.gp.mean.Zero()
#         f_ = np.random.multivariate_normal(mean_func(X).eval(),
#                                            cov_func(X).eval() + 1e-8 * np.eye(self.num_training_samples),
#                                            self.num_pred
#                                            ).flatten()
#
#         y = f_ + self.noise_variance * np.random.standard_t(self.degrees_of_freedom,
#                                                             size=self.num_training_samples)
#         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
#             X, y, test_size=0.3
#         )
#
#         self.test_STPR = StudentsTProcessRegressor()
#         # self.test_nuts_STPR = StudentsTProcessRegressor()
#         self.test_dir = tempfile.mkdtemp()
#
#     def tearDown(self):
#         shutil.rmtree(self.test_dir)
#
#
# class StudentsTProcessRegressorFitTestCase(StudentsTProcessRegressorTestCase):
#     def test_advi_fit_returns_correct_model(self):
#         # This print statement ensures PyMC3 output won't overwrite the test name
#         print('')
#         self.test_STPR.fit(self.X_train, self.y_train)
#
#         self.assertEqual(self.num_pred, self.test_STPR.num_pred)
#         self.assertAlmostEqual(self.signal_variance,
#                                int(self.test_STPR.summary['mean']['signal_variance__0']),
#                                0)
#         self.assertAlmostEqual(self.length_scale,
#                                int(self.test_STPR.summary['mean']['length_scale__0_0']),
#                                0)
#         self.assertAlmostEqual(self.noise_variance,
#                                int(self.test_STPR.summary['mean']['noise_variance__0']),
#                                0)

    # def test_nuts_fit_returns_correct_model(self):
    #     # This print statement ensures PyMC3 output won't overwrite the test name
    #     print('')
    #     self.test_nuts_STPR.fit(self.X_train, self.y_train, inference_type='nuts')
    #
    #     self.assertEqual(self.num_pred, self.test_nuts_STPR.num_pred)
    #     self.assertAlmostEqual(self.signal_variance,
    #                            int(self.test_nuts_STPR.summary['mean']['signal_variance__0']),
    #                            0)
    #     self.assertAlmostEqual(self.length_scale,
    #                            int(self.test_nuts_STPR.summary['mean']['length_scale__0_0']),
    #                            0)
    #     self.assertAlmostEqual(self.noise_variance,
    #                            int(self.test_nuts_STPR.summary['mean']['noise_variance__0']),
    #                            0)


# class StudentsTProcessRegressorPredictTestCase(StudentsTProcessRegressorTestCase):
#     def test_predict_returns_predictions(self):
#         print('')
#         self.test_STPR.fit(self.X_train, self.y_train)
#         preds = self.test_STPR.predict(self.X_test)
#         self.assertEqual(self.y_test.shape, preds.shape)
#
#     def test_predict_returns_mean_predictions_and_std(self):
#         print('')
#         self.test_STPR.fit(self.X_train, self.y_train)
#         preds, stds = self.test_STPR.predict(self.X_test, return_std=True)
#         self.assertEqual(self.y_test.shape, preds.shape)
#         self.assertEqual(self.y_test.shape, stds.shape)
#
#     def test_predict_raises_error_if_not_fit(self):
#         print('')
#         with self.assertRaises(NotFittedError) as no_fit_error:
#             test_STPR = StudentsTProcessRegressor()
#             test_STPR.predict(self.X_train)
#
#         expected = 'Run fit on the model before predict.'
#         self.assertEqual(str(no_fit_error.exception), expected)
#
#
# class StudentsTProcessRegressorScoreTestCase(StudentsTProcessRegressorTestCase):
#     def test_score_matches_sklearn_performance(self):
#         print('')
#         skGPR = skGaussianProcessRegressor()
#         skGPR.fit(self.X_train, self.y_train)
#         skGPR_score = skGPR.score(self.X_test, self.y_test)
#
#         self.test_STPR.fit(self.X_train, self.y_train)
#         test_STPR_score = self.test_STPR.score(self.X_test, self.y_test)
#
#         self.assertAlmostEqual(skGPR_score, test_STPR_score, 1)
#
#
# class StudentsTProcessRegressorSaveAndLoadTestCase(StudentsTProcessRegressorTestCase):
#     def test_save_and_load_work_correctly(self):
#         print('')
#         self.test_STPR.fit(self.X_train, self.y_train)
#         score1 = self.test_STPR.score(self.X_test, self.y_test)
#         self.test_STPR.save(self.test_dir)
#
#         STPR2 = StudentsTProcessRegressor()
#         STPR2.load(self.test_dir)
#
#         self.assertEqual(self.test_STPR.inference_type, STPR2.inference_type)
#         self.assertEqual(self.test_STPR.num_pred, STPR2.num_pred)
#         self.assertEqual(self.test_STPR.num_training_samples, STPR2.num_training_samples)
#         pd.testing.assert_frame_equal(summary(self.test_STPR.trace),
#                                       summary(STPR2.trace))
#
#         score2 = STPR2.score(self.X_test, self.y_test)
#         self.assertAlmostEqual(score1, score2, 1)
#
#
# class SparseGaussianProcessRegressorTestCase(unittest.TestCase):
#
#     def setUp(self):
#         self.num_training_samples = 150
#         self.num_pred = 1
#
#         self.length_scale = 1.0
#         self.noise_variance = 2.0
#         self.signal_variance = 3.0
#
#         X = np.linspace(start=0, stop=10, num=self.num_training_samples)[:, None]
#         cov_func = self.signal_variance**2 * pm.gp.cov.ExpQuad(self.num_pred,
#                                                                self.length_scale)
#
#         mean_func = pm.gp.mean.Zero()
#         f_ = np.random.multivariate_normal(mean_func(X).eval(),
#                                            cov_func(X).eval() + 1e-8 * np.eye(self.num_training_samples),
#                                            self.num_pred
#                                            ).flatten()
#
#         y = f_ + self.noise_variance * np.random.randn(self.num_training_samples)
#         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
#             X, y, test_size=0.3
#         )
#
#         self.test_SGPR = SparseGaussianProcessRegressor()
#         # self.test_nuts_SGPR = SparseGaussianProcessRegressor()
#         self.test_dir = tempfile.mkdtemp()
#
#     def tearDown(self):
#         shutil.rmtree(self.test_dir)
#
#
# class SparseGaussianProcessRegressorFitTestCase(SparseGaussianProcessRegressorTestCase):
#     def test_advi_fit_returns_correct_model(self):
#         # This print statement ensures PyMC3 output won't overwrite the test name
#         print('')
#         self.test_SGPR.fit(self.X_train, self.y_train)
#
#         self.assertEqual(self.num_pred, self.test_SGPR.num_pred)
#         self.assertAlmostEqual(self.signal_variance,
#                                int(self.test_SGPR.summary['mean']['signal_variance__0']),
#                                0)
#         self.assertAlmostEqual(self.length_scale,
#                                int(self.test_SGPR.summary['mean']['length_scale__0_0']),
#                                0)
#         self.assertAlmostEqual(self.noise_variance,
#                                int(self.test_SGPR.summary['mean']['noise_variance__0']),
#                                0)

    # def test_nuts_fit_returns_correct_model(self):
    #     # This print statement ensures PyMC3 output won't overwrite the test name
    #     print('')
    #     self.test_nuts_SGPR.fit(self.X_train, self.y_train, inference_type='nuts')
    #
    #     self.assertEqual(self.num_pred, self.test_nuts_SGPR.num_pred)
    #     self.assertAlmostEqual(self.signal_variance,
    #                            int(self.test_nuts_SGPR.summary['mean']['signal_variance__0']),
    #                            0)
    #     self.assertAlmostEqual(self.length_scale,
    #                            int(self.test_nuts_SGPR.summary['mean']['length_scale__0_0']),
    #                            0)
    #     self.assertAlmostEqual(self.noise_variance,
    #                            int(self.test_nuts_SGPR.summary['mean']['noise_variance__0']),
    #                            0)


# class SparseGaussianProcessRegressorPredictTestCase(SparseGaussianProcessRegressorTestCase):
#     def test_predict_returns_predictions(self):
#         print('')
#         self.test_SGPR.fit(self.X_train, self.y_train)
#         preds = self.test_SGPR.predict(self.X_test)
#         self.assertEqual(self.y_test.shape, preds.shape)
#
#     def test_predict_returns_mean_predictions_and_std(self):
#         print('')
#         self.test_SGPR.fit(self.X_train, self.y_train)
#         preds, stds = self.test_SGPR.predict(self.X_test, return_std=True)
#         self.assertEqual(self.y_test.shape, preds.shape)
#         self.assertEqual(self.y_test.shape, stds.shape)
#
#     def test_predict_raises_error_if_not_fit(self):
#         print('')
#         with self.assertRaises(NotFittedError) as no_fit_error:
#             test_SGPR = SparseGaussianProcessRegressor()
#             test_SGPR.predict(self.X_train)
#
#         expected = 'Run fit on the model before predict.'
#         self.assertEqual(str(no_fit_error.exception), expected)
#
#
# class SparseGaussianProcessRegressorScoreTestCase(SparseGaussianProcessRegressorTestCase):
#     def test_score_matches_sklearn_performance(self):
#         print('')
#         skGPR = skGaussianProcessRegressor()
#         skGPR.fit(self.X_train, self.y_train)
#         skGPR_score = skGPR.score(self.X_test, self.y_test)
#
#         self.test_SGPR.fit(self.X_train, self.y_train)
#         test_SGPR_score = self.test_SGPR.score(self.X_test, self.y_test)
#
#         self.assertAlmostEqual(skGPR_score, test_SGPR_score, 1)
#
#
# class SparseGaussianProcessRegressorSaveAndLoadTestCase(SparseGaussianProcessRegressorTestCase):
#     def test_save_and_load_work_correctly(self):
#         print('')
#         self.test_SGPR.fit(self.X_train, self.y_train)
#         score1 = self.test_SGPR.score(self.X_test, self.y_test)
#         self.test_SGPR.save(self.test_dir)
#
#         SGPR2 = SparseGaussianProcessRegressor()
#         SGPR2.load(self.test_dir)
#
#         self.assertEqual(self.test_SGPR.inference_type, SGPR2.inference_type)
#         self.assertEqual(self.test_SGPR.num_pred, SGPR2.num_pred)
#         self.assertEqual(self.test_SGPR.num_training_samples, SGPR2.num_training_samples)
#         pd.testing.assert_frame_equal(summary(self.test_SGPR.trace),
#                                       summary(SGPR2.trace))
#
#         score2 = SGPR2.score(self.X_test, self.y_test)
#         self.assertAlmostEqual(score1, score2, 1)