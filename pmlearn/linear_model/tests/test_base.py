import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
from pymc3 import summary
from sklearn.linear_model import LinearRegression as skLinearRegression
from sklearn.model_selection import train_test_split

from pmlearn.exceptions import PymcLearnError
from pmlearn.linear_model import LinearRegression


class LinearRegressionTestCase(unittest.TestCase):
    def setUp(self):
        self.num_pred = 1
        self.alpha = 3
        self.betas = 4
        self.s = 2

        X = np.random.randn(1000, 1)
        noise = self.s * np.random.randn(1000, 1)
        Y = self.betas * X + self.alpha + noise
        Y = np.squeeze(Y)

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, Y, test_size=0.4
        )

        self.test_LR = LinearRegression()
        self.nuts_LR = LinearRegression()

        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)


class LinearRegressionFitTestCase(LinearRegressionTestCase):
    def test_advi_fit_returns_correct_model(self):
        # Note: print is here so PyMC3 output won't overwrite the test name
        print('')
        self.test_LR.fit(self.X_train, self.Y_train)

        self.assertEqual(self.num_pred, self.test_LR.num_pred)

        np.testing.assert_almost_equal(self.alpha, self.test_LR.summary['mean']['alpha__0'], decimal=1)
        np.testing.assert_almost_equal(self.betas, self.test_LR.summary['mean']['betas__0_0'], decimal=1)
        np.testing.assert_almost_equal(self.s, self.test_LR.summary['mean']['s'], decimal=1)

    def test_nuts_fit_returns_correct_model(self):
        # Note: print is here so PyMC3 output won't overwrite the test name
        print('')
        self.nuts_LR.fit(self.X_train, self.Y_train, inference_type='nuts', inference_args={'draws': 2000})

        self.assertEqual(self.num_pred, self.nuts_LR.num_pred)

        np.testing.assert_almost_equal(self.alpha, self.nuts_LR.summary['mean']['alpha__0'], decimal=1)
        np.testing.assert_almost_equal(self.betas, self.nuts_LR.summary['mean']['betas__0_0'], decimal=1)
        np.testing.assert_almost_equal(self.s, self.nuts_LR.summary['mean']['s'], decimal=1)


class LinearRegressionPredictTestCase(LinearRegressionTestCase):
    def test_predict_returns_predictions(self):
        print('')
        self.test_LR.fit(self.X_train, self.Y_train)
        preds = self.test_LR.predict(self.X_test)
        self.assertEqual(preds.shape, self.Y_test.shape)

    def test_predict_returns_mean_predictions_and_std(self):
        print('')
        self.test_LR.fit(self.X_train, self.Y_train)
        preds, stds = self.test_LR.predict(self.X_test, return_std=True)
        self.assertEqual(preds.shape, self.Y_test.shape)
        self.assertEqual(stds.shape, self.Y_test.shape)

    def test_predict_raises_error_if_not_fit(self):
        with self.assertRaises(PymcLearnError) as no_fit_error:
            test_LR = LinearRegression()
            test_LR.predict(self.X_train)

        expected = 'Run fit on the model before predict.'
        self.assertEqual(str(no_fit_error.exception), expected)


class LinearRegressionScoreTestCase(LinearRegressionTestCase):
    def test_score_matches_sklearn_performance(self):
        print('')
        skLR = skLinearRegression()
        skLR.fit(self.X_train, self.Y_train)
        skLR_score = skLR.score(self.X_test, self.Y_test)

        self.test_LR.fit(self.X_train, self.Y_train)
        score = self.test_LR.score(self.X_test, self.Y_test)
        np.testing.assert_almost_equal(skLR_score, score, decimal=1)


class LinearRegressionSaveandLoadTestCase(LinearRegressionTestCase):
    def test_save_and_load_work_correctly(self):
        print('')
        self.test_LR.fit(self.X_train, self.Y_train)
        score1 = self.test_LR.score(self.X_test, self.Y_test)
        self.test_LR.save(self.test_dir)

        LR2 = LinearRegression()

        LR2.load(self.test_dir)

        self.assertEqual(self.test_LR.inference_type, LR2.inference_type)
        self.assertEqual(self.test_LR.num_pred, LR2.num_pred)
        self.assertEqual(self.test_LR.num_training_samples, LR2.num_training_samples)
        pd.testing.assert_frame_equal(summary(self.test_LR.trace), summary(LR2.trace))

        score2 = LR2.score(self.X_test, self.Y_test)

        np.testing.assert_almost_equal(score1, score2, decimal=1)
