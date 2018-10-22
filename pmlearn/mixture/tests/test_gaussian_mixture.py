import unittest
import shutil
import tempfile

import numpy as np
# import pandas as pd
# import pymc3 as pm
# from pymc3 import summary
# from sklearn.mixture import GaussianMixture as skGaussianMixture
from sklearn.model_selection import train_test_split

from pmlearn.exceptions import NotFittedError
from pmlearn.mixture import GaussianMixture


class GaussianMixtureTestCase(unittest.TestCase):

    def setUp(self):
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

        self.test_GMM = GaussianMixture()
        self.test_nuts_GMM = GaussianMixture()
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)


# class GaussianMixtureFitTestCase(GaussianMixtureTestCase):
#     def test_advi_fit_returns_correct_model(self):
#         # This print statement ensures PyMC3 output won't overwrite the test name
#         print('')
#         self.test_GMM.fit(self.X_train)
#
#         self.assertEqual(self.num_pred, self.test_GMM.num_pred)
#         self.assertEqual(self.num_components, self.test_GMM.num_components)
#
#         self.assertAlmostEqual(self.pi[0],
#                                self.test_GMM.summary['mean']['pi__0'],
#                                0)
#         self.assertAlmostEqual(self.pi[1],
#                                self.test_GMM.summary['mean']['pi__1'],
#                                0)
#         self.assertAlmostEqual(self.pi[2],
#                                self.test_GMM.summary['mean']['pi__2'],
#                                0)
#
#         self.assertAlmostEqual(
#             self.means[0],
#             self.test_GMM.summary['mean']['cluster_center_0__0'],
#             0)
#         self.assertAlmostEqual(
#             self.means[1],
#             self.test_GMM.summary['mean']['cluster_center_1__0'],
#             0)
#         self.assertAlmostEqual(
#             self.means[2],
#             self.test_GMM.summary['mean']['cluster_center_2__0'],
#             0)
#
#         self.assertAlmostEqual(
#             self.sigmas[0],
#             self.test_GMM.summary['mean']['cluster_variance_0__0'],
#             0)
#         self.assertAlmostEqual(
#             self.sigmas[1],
#             self.test_GMM.summary['mean']['cluster_variance_1__0'],
#             0)
#         self.assertAlmostEqual(
#             self.sigmas[2],
#             self.test_GMM.summary['mean']['cluster_variance_2__0'],
#             0)
#
#     def test_nuts_fit_returns_correct_model(self):
#         # This print statement ensures PyMC3 output won't overwrite the test name
#         print('')
#         self.test_nuts_GMM.fit(self.X_train,
#                                inference_type='nuts')
#
#         self.assertEqual(self.num_pred, self.test_nuts_GMM.num_pred)
#         self.assertEqual(self.num_components, self.test_nuts_GMM.num_components)
#
#         self.assertAlmostEqual(self.pi[0],
#                                self.test_nuts_GMM.summary['mean']['pi__0'],
#                                0)
#         self.assertAlmostEqual(self.pi[1],
#                                self.test_nuts_GMM.summary['mean']['pi__1'],
#                                0)
#         self.assertAlmostEqual(self.pi[2],
#                                self.test_nuts_GMM.summary['mean']['pi__2'],
#                                0)
#
#         self.assertAlmostEqual(
#             self.means[0],
#             self.test_nuts_GMM.summary['mean']['cluster_center_0__0'],
#             0)
#         self.assertAlmostEqual(
#             self.means[1],
#             self.test_nuts_GMM.summary['mean']['cluster_center_1__0'],
#             0)
#         self.assertAlmostEqual(
#             self.means[2],
#             self.test_nuts_GMM.summary['mean']['cluster_center_2__0'],
#             0)
#
#         self.assertAlmostEqual(
#             self.sigmas[0],
#             self.test_nuts_GMM.summary['mean']['cluster_variance_0__0'],
#             0)
#         self.assertAlmostEqual(
#             self.sigmas[1],
#             self.test_nuts_GMM.summary['mean']['cluster_variance_1__0'],
#             0)
#         self.assertAlmostEqual(
#             self.sigmas[2],
#             self.test_nuts_GMM.summary['mean']['cluster_variance_2__0'],
#             0)
#
#
class GaussianMixturePredictTestCase(GaussianMixtureTestCase):
    # def test_predict_returns_predictions(self):
    #     print('')
    #     self.test_GMM.fit(self.X_train, self.y_train)
    #     preds = self.test_GMM.predict(self.X_test)
    #     self.assertEqual(self.y_test.shape, preds.shape)

    # def test_predict_returns_mean_predictions_and_std(self):
    #     print('')
    #     self.test_GMM.fit(self.X_train, self.y_train)
    #     preds, stds = self.test_GMM.predict(self.X_test, return_std=True)
    #     self.assertEqual(self.y_test.shape, preds.shape)
    #     self.assertEqual(self.y_test.shape, stds.shape)

    def test_predict_raises_error_if_not_fit(self):
        print('')
        with self.assertRaises(NotFittedError) as no_fit_error:
            test_GMM = GaussianMixture()
            test_GMM.predict(self.X_train)

        expected = 'Run fit on the model before predict.'
        self.assertEqual(str(no_fit_error.exception), expected)


# class GaussianMixtureScoreTestCase(GaussianMixtureTestCase):
#     def test_score_matches_sklearn_performance(self):
#         print('')
#         skGMM = skGaussianMixture(n_components=3)
#         skGMM.fit(self.X_train)
#         skGMM_score = skGMM.score(self.X_test)
#
#         self.test_GMM.fit(self.X_train)
#         test_GMM_score = self.test_GMM.score(self.X_test)
#
#         self.assertAlmostEqual(skGMM_score, test_GMM_score, 0)
#
#
# class GaussianMixtureSaveAndLoadTestCase(GaussianMixtureTestCase):
#     def test_save_and_load_work_correctly(self):
#         print('')
#         self.test_GMM.fit(self.X_train)
#         score1 = self.test_GMM.score(self.X_test)
#         self.test_GMM.save(self.test_dir)
#
#         GMM2 = GaussianMixture()
#         GMM2.load(self.test_dir)
#
#         self.assertEqual(self.test_GMM.inference_type, GMM2.inference_type)
#         self.assertEqual(self.test_GMM.num_pred, GMM2.num_pred)
#         self.assertEqual(self.test_GMM.num_training_samples,
#                          GMM2.num_training_samples)
#         pd.testing.assert_frame_equal(summary(self.test_GMM.trace),
#                                       summary(GMM2.trace))
#
#         score2 = GMM2.score(self.X_test)
#         self.assertAlmostEqual(score1, score2, 0)
