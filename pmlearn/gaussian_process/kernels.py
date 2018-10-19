"""Kernels for Gaussian process regression and classification.
"""

# Authors: Daniel Emaasit <daniel.emaasit@gmail.com>
#
# License: BSD 3 clause

import pymc3 as pm


class RBF(pm.gp.cov.ExpQuad):
    """Radial-basis function kernel from ``pymc3.gp.cov.ExpQuad``

    The RBF kernel is a stationary kernel. It is also known as the
    "squared exponential" kernel. It is parameterized by a length-scale
    parameter length_scale>0, which can either be a scalar (isotropic variant
    of the kernel) or a vector with the same number of dimensions as the inputs
    X (anisotropic variant of the kernel).

    The kernel is given by:

    k(x_i, x_j) = exp(-1 / 2 d(x_i / length_scale, x_j / length_scale)^2)

    This kernel is infinitely differentiable, which implies that GPs with this
    kernel as covariance function have mean square derivatives of all orders,
    and are thus very smooth.
    """


class DotProduct(pm.gp.cov.Exponential):
    """Dot-Product kernel from ``pymc3.gp.cov.Exponential``

    The DotProduct kernel is non-stationary and can be obtained from linear
    regression by putting N(0, 1) priors on the coefficients of x_d (d = 1, . .
    . , D) and a prior of N(0, \sigma_0^2) on the bias. The DotProduct kernel
    is invariant to a rotation of the coordinates about the origin, but not
    translations. It is parameterized by a parameter sigma_0^2. For
    sigma_0^2 =0, the kernel is called the homogeneous linear kernel, otherwise
    it is inhomogeneous.

    The kernel is given by
    k(x_i, x_j) = sigma_0 ^ 2 + x_i \cdot x_j

    The DotProduct kernel is commonly combined with exponentiation.
    """


class WhiteKernel(pm.gp.cov.WhiteNoise):
    """White kernel from ``pymc3.gp.cov.WhiteNoise..

    The main use-case of this kernel is as part of a sum-kernel where it
    explains the noise-component of the signal. Tuning its parameter
    corresponds to estimating the noise-level.

    k(x_1, x_2) = noise_level if x_1 == x_2 else 0

    """
    def __init__(self, noise_level=1.0):
        super(WhiteKernel, self).__init__(sigma=noise_level)
