"""Credibility models and primitives.

The credibility tools a pricing or experience-rating actuary reaches for: the
credibility-weighting primitive plus greatest-accuracy (Bühlmann and
Bühlmann-Straub) credibility models.

The ``Buhlmann`` and ``BuhlmannStraub`` models were previously part of the
``lossmodels`` package and were moved here, where credibility sits next to the
experience and ratemaking workflows that consume it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_SCALAR_TYPES = (int, float, np.number)


def credibility_weighted_estimate(observed: Any, complement: Any, z: Any) -> Any:
    """Blend an observed estimate with its complement at credibility ``z``.

    Returns ``z * observed + (1 - z) * complement``. Scalar inputs return a
    native ``float``; ``pandas.Series`` inputs return a ``Series`` with the index
    preserved; other array-like inputs return a ``numpy.ndarray``. This is the
    atomic credibility operation; the ``z`` may come from a model below, a filed
    credibility formula, or any other source.
    """
    if isinstance(observed, pd.Series) or isinstance(complement, pd.Series) or isinstance(z, pd.Series):
        return z * observed + (1 - z) * complement
    if isinstance(observed, _SCALAR_TYPES) and isinstance(complement, _SCALAR_TYPES) and isinstance(z, _SCALAR_TYPES):
        return float(z * observed + (1 - z) * complement)
    observed_arr = np.asarray(observed, dtype=float)
    complement_arr = np.asarray(complement, dtype=float)
    z_arr = np.asarray(z, dtype=float)
    return z_arr * observed_arr + (1 - z_arr) * complement_arr


class Buhlmann:
    """Bühlmann credibility model.

    This implementation assumes each risk has the same number of observations.

    Parameters
    ----------
    overall_mean : float
        Estimated collective mean.
    epv : float
        Estimated expected process variance (EPV).
    vhm : float
        Estimated variance of hypothetical means (VHM).
    n_obs : int
        Number of observations per risk.
    """

    def __init__(self, overall_mean: float, epv: float, vhm: float, n_obs: int):
        if n_obs <= 0:
            raise ValueError("n_obs must be positive.")
        if epv < 0:
            raise ValueError("epv must be nonnegative.")
        if vhm < 0:
            raise ValueError("vhm must be nonnegative.")

        self.overall_mean = float(overall_mean)
        self.epv = float(epv)
        self.vhm = float(vhm)
        self.n_obs = int(n_obs)

    @property
    def k(self) -> float:
        """K = EPV / VHM. Returns infinity when VHM = 0."""
        if self.vhm == 0:
            return float("inf")
        return self.epv / self.vhm

    @property
    def z(self) -> float:
        """Credibility factor ``Z = n / (n + K)``. Returns 0 when K is infinite."""
        k = self.k
        if not np.isfinite(k):
            return 0.0
        return self.n_obs / (self.n_obs + k)

    def premium(self, risk_mean: Any) -> Any:
        """Compute the Bühlmann credibility premium ``Z * risk_mean + (1 - Z) * overall_mean``.

        Parameters
        ----------
        risk_mean : float or array-like
            Risk-specific sample mean(s).

        Returns
        -------
        float or numpy.ndarray
            Credibility-weighted premium(s).
        """
        risk_mean = np.asarray(risk_mean, dtype=float)
        premium = self.z * risk_mean + (1.0 - self.z) * self.overall_mean
        return float(premium) if premium.ndim == 0 else premium

    @classmethod
    def fit(cls, data: Any) -> Buhlmann:
        """Fit a Bühlmann credibility model from data.

        Parameters
        ----------
        data : array-like, shape (m, n)
            Observations for m risks, each with n observations.

        Returns
        -------
        Buhlmann
            Fitted Bühlmann model.

        Notes
        -----
        Estimators used:

        - overall_mean = mean of all observations
        - EPV = average of within-risk sample variances
        - VHM = sample variance of risk means minus EPV / n, floored at 0
        """
        data = np.asarray(data, dtype=float)

        if data.ndim != 2:
            raise ValueError("data must be a 2D array with shape (n_risks, n_obs).")

        n_risks, n_obs = data.shape

        if n_risks < 2:
            raise ValueError("data must contain at least two risks.")
        if n_obs < 2:
            raise ValueError("each risk must have at least two observations.")

        risk_means = np.mean(data, axis=1)
        overall_mean = float(np.mean(data))

        within_vars = np.var(data, axis=1, ddof=1)
        epv = float(np.mean(within_vars))

        between_var = float(np.var(risk_means, ddof=1))
        vhm = max(between_var - epv / n_obs, 0.0)

        return cls(overall_mean=overall_mean, epv=epv, vhm=vhm, n_obs=n_obs)

    def __repr__(self) -> str:
        return (
            f"Buhlmann(overall_mean={self.overall_mean}, "
            f"epv={self.epv}, vhm={self.vhm}, n_obs={self.n_obs})"
        )


class BuhlmannStraub:
    """Bühlmann-Straub credibility model.

    This implementation allows different exposure weights by risk and period.

    Parameters
    ----------
    overall_mean : float
        Estimated collective mean.
    epv : float
        Estimated expected process variance (EPV).
    vhm : float
        Estimated variance of hypothetical means (VHM).
    weights : array-like
        Total weight (exposure) for each risk.
    """

    def __init__(self, overall_mean: float, epv: float, vhm: float, weights: Any):
        weights = np.asarray(weights, dtype=float)

        if weights.ndim != 1:
            raise ValueError("weights must be a 1D array.")
        if weights.size == 0:
            raise ValueError("weights must not be empty.")
        if np.any(weights <= 0):
            raise ValueError("weights must be positive.")
        if epv < 0:
            raise ValueError("epv must be nonnegative.")
        if vhm < 0:
            raise ValueError("vhm must be nonnegative.")

        self.overall_mean = float(overall_mean)
        self.epv = float(epv)
        self.vhm = float(vhm)
        self.weights = weights

    @property
    def k(self) -> float:
        """K = EPV / VHM. Returns infinity when VHM = 0."""
        if self.vhm == 0:
            return float("inf")
        return self.epv / self.vhm

    def z(self, weight: Any) -> Any:
        """Credibility factor for a given total risk weight: ``Z_i = w_i / (w_i + K)``.

        Parameters
        ----------
        weight : float or array-like
            Total exposure weight(s).

        Returns
        -------
        float or numpy.ndarray
            Credibility factor(s).
        """
        weight = np.asarray(weight, dtype=float)

        if np.any(weight <= 0):
            raise ValueError("weight must be positive.")

        k = self.k
        if not np.isfinite(k):
            out = np.zeros_like(weight, dtype=float)
        else:
            out = weight / (weight + k)

        return float(out) if out.ndim == 0 else out

    def premium(self, risk_mean: Any, weight: Any) -> Any:
        """Compute the Bühlmann-Straub premium ``Z_i * risk_mean_i + (1 - Z_i) * overall_mean``.

        Parameters
        ----------
        risk_mean : float or array-like
            Risk-specific weighted mean(s).
        weight : float or array-like
            Total exposure weight(s).

        Returns
        -------
        float or numpy.ndarray
            Credibility-weighted premium(s).
        """
        risk_mean = np.asarray(risk_mean, dtype=float)
        z = self.z(weight)
        premium = z * risk_mean + (1.0 - z) * self.overall_mean
        return float(premium) if np.ndim(premium) == 0 else premium

    @classmethod
    def fit(cls, data: Any, weights: Any) -> BuhlmannStraub:
        """Fit a Bühlmann-Straub model from observations and weights.

        Parameters
        ----------
        data : array-like, shape (m, n)
            Observed values X_ij for m risks and n periods.
        weights : array-like, shape (m, n)
            Exposure weights w_ij for m risks and n periods.

        Returns
        -------
        BuhlmannStraub
            Fitted Bühlmann-Straub model.

        Notes
        -----
        Let ``w_i. = sum_j w_ij``, ``Xbar_i = sum_j w_ij X_ij / w_i.``, and
        ``overall_mean = sum_i sum_j w_ij X_ij / sum_i sum_j w_ij``.

        EPV is estimated by ``[sum_i sum_j w_ij (X_ij - Xbar_i)^2] / [m (n - 1)]``.

        VHM is the weighted sample variance of the risk means around the overall
        mean, adjusted by EPV and floored at 0. This is a practical
        implementation intended for equal period counts.
        """
        data = np.asarray(data, dtype=float)
        weights = np.asarray(weights, dtype=float)

        if data.ndim != 2:
            raise ValueError("data must be a 2D array.")
        if weights.ndim != 2:
            raise ValueError("weights must be a 2D array.")
        if data.shape != weights.shape:
            raise ValueError("data and weights must have the same shape.")
        if data.shape[0] < 2:
            raise ValueError("data must contain at least two risks.")
        if data.shape[1] < 2:
            raise ValueError("each risk must have at least two periods.")
        if np.any(weights <= 0):
            raise ValueError("weights must be positive.")

        m, n = data.shape

        risk_weights = np.sum(weights, axis=1)
        weighted_risk_means = np.sum(weights * data, axis=1) / risk_weights

        overall_mean = float(np.sum(weights * data) / np.sum(weights))

        ss_within = np.sum(weights * (data - weighted_risk_means[:, None]) ** 2)
        epv = float(ss_within / (m * (n - 1)))

        mean_risk_weight = float(np.mean(risk_weights))
        between_term = float(
            np.sum(risk_weights * (weighted_risk_means - overall_mean) ** 2) / (m - 1)
        )

        vhm = max((between_term - epv) / mean_risk_weight, 0.0)

        return cls(overall_mean=overall_mean, epv=epv, vhm=vhm, weights=risk_weights)

    def __repr__(self) -> str:
        return (
            f"BuhlmannStraub(overall_mean={self.overall_mean}, "
            f"epv={self.epv}, vhm={self.vhm}, weights={self.weights})"
        )
