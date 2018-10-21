"""Unit tests for :mod:`pmlearn.base`. """

# Authors: Daniel Emaasit <daniel.emaasit@gmail.com>
#
# License: BSD 3 clause

import pytest

from ..base import BayesianModel


class TestBayesianModel(object):
    """Test class for base bayesian model

    """
    def test_create_model_raises_not_implemented_error(self):
        """Assert that NotImplementedError is raised

        """
        with pytest.raises(NotImplementedError):
            bm = BayesianModel()
            bm.create_model()

    # def test_fit_raises_not_implemented_error(self):
    #     with pytest.raises(NotImplementedError):
    #         bm = BayesianModel()
    #         bm.fit()
    #
    # def test_predict_raises_not_implemented_error(self):
    #     with pytest.raises(NotImplementedError):
    #         bm = BayesianModel()
    #         bm.predict()
    #
    # def test_score_raises_not_implemented_error(self):
    #     with pytest.raises(NotImplementedError):
    #         bm = BayesianModel()
    #         bm.score()