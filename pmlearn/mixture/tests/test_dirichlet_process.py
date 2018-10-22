import unittest
import shutil
import tempfile

import numpy as np
# import pandas as pd
# import pymc3 as pm
# from pymc3 import summary
# from sklearn.mixture import BayesianGaussianMixture as skBayesianGaussianMixture
from sklearn.model_selection import train_test_split

from pmlearn.exceptions import NotFittedError
from pmlearn.mixture import DirichletProcessMixture


class DirichletProcessMixtureTestCase(unittest.TestCase):

    def setUp(self):
        self.num_truncate = 3
        self.num_components = 3
        self.num_pred = 1
        self.num_training_samples = 100

        self.pi = np.array([0.35, 0.4, 0.25])
        self.means = np.array([0, 5, 10])
        self.sigmas = np.array([0.5, 0.5, 1.0])

        self.components = np.random.randint(0,
                                            self.num_components,
                                            self.num_training_samples)

        X = np.random.normal(loc=self.means[self.components],
                             scale=self.sigmas[self.components])
        X.shape = (self.num_training_samples, 1)

        self.X_train, self.X_test = train_test_split(X, test_size=0.3)

        self.test_DPMM = DirichletProcessMixture()
        self.test_nuts_DPMM = DirichletProcessMixture()
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)


# class DirichletProcessMixtureFitTestCase(DirichletProcessMixtureTestCase):
#     def test_advi_fit_returns_correct_model(self):
#         # This print statement ensures PyMC3 output won't overwrite the test name
#         print('')
#         self.test_DPMM.fit(self.X_train)
#
#         self.assertEqual(self.num_pred, self.test_DPMM.num_pred)
#         self.assertEqual(self.num_components, self.test_DPMM.num_components)
#         self.assertEqual(self.num_truncate, self.test_DPMM.num_truncate)
#
#         self.assertAlmostEqual(self.pi[0],
#                                self.test_DPMM.summary['mean']['pi__0'],
#                                0)
#         self.assertAlmostEqual(self.pi[1],
#                                self.test_DPMM.summary['mean']['pi__1'],
#                                0)
#         self.assertAlmostEqual(self.pi[2],
#                                self.test_DPMM.summary['mean']['pi__2'],
#                                0)
#
#         self.assertAlmostEqual(
#             self.means[0],
#             self.test_DPMM.summary['mean']['cluster_center_0__0'],
#             0)
#         self.assertAlmostEqual(
#             self.means[1],
#             self.test_DPMM.summary['mean']['cluster_center_1__0'],
#             0)
#         self.assertAlmostEqual(
#             self.means[2],
#             self.test_DPMM.summary['mean']['cluster_center_2__0'],
#             0)
#
#         self.assertAlmostEqual(
#             self.sigmas[0],
#             self.test_DPMM.summary['mean']['cluster_variance_0__0'],
#             0)
#         self.assertAlmostEqual(
#             self.sigmas[1],
#             self.test_DPMM.summary['mean']['cluster_variance_1__0'],
#             0)
#         self.assertAlmostEqual(
#             self.sigmas[2],
#             self.test_DPMM.summary['mean']['cluster_variance_2__0'],
#             0)
#
#     def test_nuts_fit_returns_correct_model(self):
#         # This print statement ensures PyMC3 output won't overwrite the test name
#         print('')
#         self.test_nuts_DPMM.fit(self.X_train,
#                                 inference_type='nuts',
#                                 inference_args={'draws': 1000,
#                                                 'chains': 2})
#
#         self.assertEqual(self.num_pred, self.test_nuts_DPMM.num_pred)
#         self.assertEqual(self.num_components, self.test_nuts_DPMM.num_components)
#         self.assertEqual(self.num_components, self.test_nuts_DPMM.num_truncate)
#
#         self.assertAlmostEqual(self.pi[0],
#                                self.test_nuts_DPMM.summary['mean']['pi__0'],
#                                0)
#         self.assertAlmostEqual(self.pi[1],
#                                self.test_nuts_DPMM.summary['mean']['pi__1'],
#                                0)
#         self.assertAlmostEqual(self.pi[2],
#                                self.test_nuts_DPMM.summary['mean']['pi__2'],
#                                0)
#
#         self.assertAlmostEqual(
#             self.means[0],
#             self.test_nuts_DPMM.summary['mean']['cluster_center_0__0'],
#             0)
#         self.assertAlmostEqual(
#             self.means[1],
#             self.test_nuts_DPMM.summary['mean']['cluster_center_1__0'],
#             0)
#         self.assertAlmostEqual(
#             self.means[2],
#             self.test_nuts_DPMM.summary['mean']['cluster_center_2__0'],
#             0)
#
#         self.assertAlmostEqual(
#             self.sigmas[0],
#             self.test_nuts_DPMM.summary['mean']['cluster_variance_0__0'],
#             0)
#         self.assertAlmostEqual(
#             self.sigmas[1],
#             self.test_nuts_DPMM.summary['mean']['cluster_variance_1__0'],
#             0)
#         self.assertAlmostEqual(
#             self.sigmas[2],
#             self.test_nuts_DPMM.summary['mean']['cluster_variance_2__0'],
#             0)
#
#
class DirichletProcessMixturePredictTestCase(DirichletProcessMixtureTestCase):
    # def test_predict_returns_predictions(self):
    #     print('')
    #     self.test_DPMM.fit(self.X_train, self.y_train)
    #     preds = self.test_DPMM.predict(self.X_test)
    #     self.assertEqual(self.y_test.shape, preds.shape)

    # def test_predict_returns_mean_predictions_and_std(self):
    #     print('')
    #     self.test_DPMM.fit(self.X_train, self.y_train)
    #     preds, stds = self.test_DPMM.predict(self.X_test, return_std=True)
    #     self.assertEqual(self.y_test.shape, preds.shape)
    #     self.assertEqual(self.y_test.shape, stds.shape)

    def test_predict_raises_error_if_not_fit(self):
        print('')
        with self.assertRaises(NotFittedError) as no_fit_error:
            test_DPMM = DirichletProcessMixture()
            test_DPMM.predict(self.X_train)

        expected = 'Run fit on the model before predict.'
        self.assertEqual(str(no_fit_error.exception), expected)


# class DirichletProcessMixtureScoreTestCase(DirichletProcessMixtureTestCase):
#     def test_score_matches_sklearn_performance(self):
#         print('')
#         skDPMM = skBayesianGaussianMixture(n_components=3)
#         skDPMM.fit(self.X_train)
#         skDPMM_score = skDPMM.score(self.X_test)
#
#         self.test_DPMM.fit(self.X_train)
#         test_DPMM_score = self.test_DPMM.score(self.X_test)
#
#         self.assertAlmostEqual(skDPMM_score, test_DPMM_score, 0)
#
#
# class DirichletProcessMixtureSaveAndLoadTestCase(DirichletProcessMixtureTestCase):
#     def test_save_and_load_work_correctly(self):
#         print('')
#         self.test_DPMM.fit(self.X_train)
#         score1 = self.test_DPMM.score(self.X_test)
#         self.test_DPMM.save(self.test_dir)
#
#         DPMM2 = DirichletProcessMixture()
#         DPMM2.load(self.test_dir)
#
#         self.assertEqual(self.test_DPMM.inference_type, DPMM2.inference_type)
#         self.assertEqual(self.test_DPMM.num_pred, DPMM2.num_pred)
#         self.assertEqual(self.test_DPMM.num_training_samples,
#                          DPMM2.num_training_samples)
#         self.assertEqual(self.test_DPMM.num_truncate, DPMM2.num_truncate)
#
#         pd.testing.assert_frame_equal(summary(self.test_DPMM.trace),
#                                       summary(DPMM2.trace))
#
#         score2 = DPMM2.score(self.X_test)
#         self.assertAlmostEqual(score1, score2, 0)
