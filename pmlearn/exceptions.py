
"""
The :mod:`pmlearn.exceptions` module includes all custom warnings and error
classes used across pymc-learn.
"""

__all__ = ['NotFittedError']


class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if estimator is used before fitting.
    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.
    Examples
    --------
    >>> from pmlearn.gaussian_process import GaussianProcessRegressor
    >>> from pmlearn.exceptions import NotFittedError
    >>> try:
    ...     GaussianProcessRegressor().predict([[1, 2], [2, 3], [3, 4]])
    ... except NotFittedError as e:
    ...     print(repr(e))
    ...                        # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    NotFittedError('This GaussianProcessRegressor instance is not fitted yet'.)
    """
