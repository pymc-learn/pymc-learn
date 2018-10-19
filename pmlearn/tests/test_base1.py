import unittest

from pmlearn.base import BayesianModel


class BayesianModelTestCase(unittest.TestCase):
    def test_create_model_raises_not_implemented_error(self):
        with self.assertRaises(NotImplementedError):
            BM = BayesianModel()
            BM.create_model()

    def test_fit_raises_not_implemented_error(self):
        with self.assertRaises(NotImplementedError):
            BM = BayesianModel()
            BM.fit()

    def test_predict_raises_not_implemented_error(self):
        with self.assertRaises(NotImplementedError):
            BM = BayesianModel()
            BM.predict()

    def test_score_raises_not_implemented_error(self):
        with self.assertRaises(NotImplementedError):
            BM = BayesianModel()
            BM.score()
