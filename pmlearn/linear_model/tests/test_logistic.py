import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
from pymc3 import summary
from sklearn.model_selection import train_test_split

from pmlearn.exceptions import NotFittedError
from pmlearn.linear_model import HierarchicalLogisticRegression


class HierarchicalLogisticRegressionTestCase(unittest.TestCase):
    def setUp(self):
        def numpy_invlogit(x):
            return 1 / (1 + np.exp(-x))

        self.num_cats = 3
        self.num_pred = 1
        self.num_samples_per_cat = 100000

        self.alphas = np.random.randn(self.num_cats)
        self.betas = np.random.randn(self.num_cats, self.num_pred)
        #TODO: make this more efficient; right now, it's very explicit so I understand it.
        x_a = np.random.randn(self.num_samples_per_cat, self.num_pred)
        y_a = np.random.binomial(1, numpy_invlogit(self.alphas[0] + np.sum(self.betas[0] * x_a, 1)))
        x_b = np.random.randn(self.num_samples_per_cat, self.num_pred)
        y_b = np.random.binomial(1, numpy_invlogit(self.alphas[1] + np.sum(self.betas[1] * x_b, 1)))
        x_c = np.random.randn(self.num_samples_per_cat, self.num_pred)
        y_c = np.random.binomial(1, numpy_invlogit(self.alphas[2] + np.sum(self.betas[2] * x_c, 1)))

        X = np.concatenate([x_a, x_b, x_c])
        Y = np.concatenate([y_a, y_b, y_c])
        cats = np.concatenate([
            np.zeros(self.num_samples_per_cat, dtype=np.int),
            np.ones(self.num_samples_per_cat, dtype=np.int),
            2*np.ones(self.num_samples_per_cat, dtype=np.int)
        ])

        self.X_train, self.X_test, self.cat_train, self.cat_test, self.Y_train, self.Y_test = train_test_split(
            X, cats, Y, test_size=0.4
        )

        self.test_HLR = HierarchicalLogisticRegression()

        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)


class HierarchicalLogisticRegressionFitTestCase(HierarchicalLogisticRegressionTestCase):
    def test_fit_returns_correct_model(self):
        # Note: print is here so PyMC3 output won't overwrite the test name
        print('')
        self.test_HLR.fit(self.X_train, self.Y_train, self.cat_train, minibatch_size=2000)

        self.assertEqual(self.num_cats, self.test_HLR.num_cats)
        self.assertEqual(self.num_pred, self.test_HLR.num_pred)

        #TODO: Figure out best way to test
        #np.testing.assert_almost_equal(self.alphas, self.test_HLR.trace['alphas'].mean(), decimal=1)
        #np.testing.assert_almost_equal(self.betas, self.test_HLR.trace['betas'].mean(), decimal=1)

        # For now, just check that the estimated parameters have the correct signs
        np.testing.assert_equal(
            np.sign(self.alphas),
            np.sign(self.test_HLR.trace['alpha'].mean(axis=0))
        )
        np.testing.assert_equal(
            np.sign(self.betas),
            np.sign(self.test_HLR.trace['beta'].mean(axis=0))
        )


class HierarchicalLogisticRegressionPredictProbaTestCase(HierarchicalLogisticRegressionTestCase):
    def test_predict_proba_returns_probabilities(self):
        print('')
        self.test_HLR.fit(self.X_train, self.Y_train, self.cat_train, minibatch_size=2000)
        probs = self.test_HLR.predict_proba(self.X_test, self.cat_test)
        self.assertEqual(probs.shape, self.Y_test.shape)

    def test_predict_proba_returns_probabilities_and_std(self):
        print('')
        self.test_HLR.fit(self.X_train, self.Y_train, self.cat_train, minibatch_size=2000)
        probs, stds = self.test_HLR.predict_proba(self.X_test, self.cat_test, return_std=True)
        self.assertEqual(probs.shape, self.Y_test.shape)
        self.assertEqual(stds.shape, self.Y_test.shape)

    def test_predict_proba_raises_error_if_not_fit(self):
        with self.assertRaises(NotFittedError) as no_fit_error:
            test_HLR = HierarchicalLogisticRegression()
            test_HLR.predict_proba(self.X_train, self.cat_train)

        expected = 'Run fit on the model before predict.'
        self.assertEqual(str(no_fit_error.exception), expected)


class HierarchicalLogisticRegressionPredictTestCase(HierarchicalLogisticRegressionTestCase):
    def test_predict_returns_predictions(self):
        print('')
        self.test_HLR.fit(self.X_train, self.Y_train, self.cat_train, minibatch_size=2000)
        preds = self.test_HLR.predict(self.X_test, self.cat_test)
        self.assertEqual(preds.shape, self.Y_test.shape)


class HierarchicalLogisticRegressionScoreTestCase(HierarchicalLogisticRegressionTestCase):
    def test_score_scores(self):
        print('')
        self.test_HLR.fit(self.X_train, self.Y_train, self.cat_train, minibatch_size=2000)
        score = self.test_HLR.score(self.X_test, self.Y_test, self.cat_test)
        naive_score = np.mean(self.Y_test)
        self.assertGreaterEqual(score, naive_score)


class HierarchicalLogisticRegressionSaveandLoadTestCase(HierarchicalLogisticRegressionTestCase):
    def test_save_and_load_work_correctly(self):
        print('')
        self.test_HLR.fit(self.X_train, self.Y_train, self.cat_train, minibatch_size=2000)
        probs1 = self.test_HLR.predict_proba(self.X_test, self.cat_test)
        self.test_HLR.save(self.test_dir)

        HLR2 = HierarchicalLogisticRegression()

        HLR2.load(self.test_dir)

        self.assertEqual(self.test_HLR.num_cats, HLR2.num_cats)
        self.assertEqual(self.test_HLR.num_pred, HLR2.num_pred)
        self.assertEqual(self.test_HLR.num_training_samples, HLR2.num_training_samples)
        pd.testing.assert_frame_equal(summary(self.test_HLR.trace), summary(HLR2.trace))

        probs2 = HLR2.predict_proba(self.X_test, self.cat_test)

        np.testing.assert_almost_equal(probs2, probs1, decimal=1)
