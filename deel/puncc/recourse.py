# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements conformal recourse for multivariate regression.

Given a **fixed** pre-trained multi-output linear regression model and an
independent calibration set, :class:`ConformalRecourse` searches for the
cheapest modification of a business record such that the **lower bounds** of
the native Bonferroni prediction intervals — built by split conformal
prediction at a fixed total significance level :math:`\\alpha` — reach
user-specified thresholds on every targeted metric. Targeting interval lower
bounds (instead of point predictions) accounts for predictive uncertainty and
avoids underestimating the risk of missing the thresholds.

The search space is delimited by numeric bounds, immutable fields, allowed
categories and user-given linear business constraints. The total cost of a
modification is the weighted :math:`L_1` norm of the numeric changes (weights
per original unit, all positive) plus the given categorical switching fees.
Because the model is linear and categories enter it through a fixed one-hot
mapping, the problem is cast as a mixed-integer linear program (MILP) and
solved to a **certified global minimum** (HiGHS branch-and-bound, zero gap).

Every suggested record is re-verified end-to-end through the real model and
the real calibrator before being returned; if prediction, solving or
verification fails, an exception is raised and no partial suggestion is
returned.
"""

import logging
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from functools import partial
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np

from deel.puncc.api.corrections import bonferroni
from deel.puncc.api.prediction import BasePredictor
from deel.puncc.regression import SplitCP

try:
    from scipy.optimize import Bounds
    from scipy.optimize import LinearConstraint
    from scipy.optimize import milp
except ImportError:  # pragma: no cover
    milp = None

logger = logging.getLogger(__name__)

# Tolerance on the verification of constraints and interval targets.
_FEAS_TOL = 1e-6
# Tolerance on the integrality of the one-hot decision variables.
_INT_TOL = 1e-4


class RecourseError(Exception):
    """Base class for conformal recourse errors."""


class InsufficientCalibrationError(RecourseError, ValueError):
    """The calibration set is too small for the fixed total alpha.

    Raised when the Bonferroni-corrected level :math:`\\alpha/K` is
    inconsistent with the size :math:`n` of the calibration set, i.e. when
    :math:`\\alpha/K < 1/(n+1)` (same condition as
    :func:`deel.puncc.api.utils.alpha_calib_check`).
    """


class InfeasibleRecourseError(RecourseError):
    """No feasible record satisfies the constraints and interval targets."""


class RecourseSolverError(RecourseError):
    """The MIP solver failed to certify a globally optimal solution."""


class RecourseVerificationError(RecourseError):
    """The candidate record failed the end-to-end verification through the
    real model and calibrator."""


class OneHotFeatureMap:
    """Fixed mapping between business records and model feature vectors.

    The feature vector stacks, in order:

    1. the numeric fields, in the order of ``numeric_fields``;
    2. for each categorical field (in the order of ``categorical_fields``), a
       full one-hot block over its declared categories, in declared order.

    The mapping is frozen at construction: calibration records, the record to
    modify and the suggested record are all encoded through the same mapping,
    so categories always enter the **same** model through the **same** one-hot
    encoding.

    :param Sequence[str] numeric_fields: ordered names of the numeric fields.
    :param Mapping[str, Sequence] categorical_fields: mapping of a categorical
        field name to the ordered sequence of its possible categories.

    .. _example onehotfeaturemap:

    Example::

        fmap = OneHotFeatureMap(
            numeric_fields=["age", "income"],
            categorical_fields={"plan": ["A", "B", "C"]},
        )
        x = fmap.encode({"age": 35.0, "income": 4.2, "plan": "B"})
        # x = array([35. ,  4.2,  0. ,  1. ,  0. ])
    """

    def __init__(
        self,
        numeric_fields: Sequence[str],
        categorical_fields: Mapping[str, Sequence[Any]],
    ):
        numeric_fields = list(numeric_fields)
        categorical_fields = {k: list(v) for k, v in categorical_fields.items()}

        if len(set(numeric_fields)) != len(numeric_fields):
            raise ValueError("Duplicate numeric field names.")
        if set(numeric_fields) & set(categorical_fields):
            raise ValueError(
                "Field names must be either numeric or categorical, not both."
            )
        for f, cats in categorical_fields.items():
            if len(cats) == 0:
                raise ValueError(f"Categorical field {f!r} has no category.")
            if len(set(cats)) != len(cats):
                raise ValueError(f"Duplicate categories in field {f!r}.")

        self._numeric_fields = tuple(numeric_fields)
        self._categorical_fields = {
            k: tuple(v) for k, v in categorical_fields.items()
        }

        self._num_pos = {f: i for i, f in enumerate(numeric_fields)}
        self._cat_pos = {}
        offset = len(numeric_fields)
        for f, cats in categorical_fields.items():
            self._cat_pos[f] = {c: offset + j for j, c in enumerate(cats)}
            offset += len(cats)
        self._n_features = offset

    @property
    def numeric_fields(self) -> Tuple[str, ...]:
        """Ordered numeric field names."""
        return self._numeric_fields

    @property
    def categorical_fields(self) -> Dict[str, Tuple[Any, ...]]:
        """Mapping of categorical field names to their ordered categories."""
        return dict(self._categorical_fields)

    @property
    def n_features(self) -> int:
        """Length of the encoded feature vector."""
        return self._n_features

    @property
    def feature_names(self) -> List[str]:
        """Ordered names of the model input features."""
        names = list(self._numeric_fields)
        for f, cats in self._categorical_fields.items():
            names.extend([f"{f}={c!r}" for c in cats])
        return names

    def numeric_position(self, field: str) -> int:
        """Column index of a numeric field in the feature vector."""
        return self._num_pos[field]

    def category_column(self, field: str, category: Any) -> int:
        """Column index of a category of a categorical field."""
        return self._cat_pos[field][category]

    def categorical_block(self, field: str) -> List[int]:
        """Column indices of the one-hot block of a categorical field."""
        return [self._cat_pos[field][c] for c in self._categorical_fields[field]]

    def categories_of(self, field: str) -> Tuple[Any, ...]:
        """Ordered categories of a categorical field."""
        return self._categorical_fields[field]

    def encode(self, record: Mapping[str, Any]) -> np.ndarray:
        """Encode a record (dict of original fields) into a feature vector.

        :param record: mapping of field names to values. Must contain exactly
            the declared numeric and categorical fields.
        :returns: feature vector of length :attr:`n_features`.
        :rtype: ndarray

        :raises ValueError: missing/extra fields, non-numeric values or
            unknown categories.
        """
        if not isinstance(record, Mapping):
            raise TypeError("record must be a mapping of field names to values.")
        expected = set(self._numeric_fields) | set(self._categorical_fields)
        keys = set(record.keys())
        if keys != expected:
            raise ValueError(
                f"Record fields {sorted(keys)} do not match the feature map "
                + f"fields {sorted(expected)}."
            )
        vec = np.zeros(self._n_features)
        for f in self._numeric_fields:
            try:
                value = float(record[f])
            except (TypeError, ValueError):
                raise ValueError(
                    f"Numeric field {f!r} has non-numeric value {record[f]!r}."
                )
            if not np.isfinite(value):
                raise ValueError(f"Numeric field {f!r} must be finite.")
            vec[self._num_pos[f]] = value
        for f, cats in self._categorical_fields.items():
            value = record[f]
            if value not in self._cat_pos[f]:
                raise ValueError(
                    f"Unknown category {value!r} for field {f!r}; "
                    + f"allowed: {list(cats)}."
                )
            vec[self._cat_pos[f][value]] = 1.0
        return vec

    def encode_many(self, records: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """Encode a sequence of records into a 2D feature matrix."""
        return np.array([self.encode(r) for r in records])

    def decode(self, vector: Sequence[float]) -> Dict[str, Any]:
        """Decode a feature vector back into a record (argmax per one-hot
        block)."""
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (self._n_features,):
            raise ValueError(
                f"Expected a vector of length {self._n_features}, "
                + f"got shape {vector.shape}."
            )
        record = {}
        for f in self._numeric_fields:
            record[f] = float(vector[self._num_pos[f]])
        for f, cats in self._categorical_fields.items():
            cols = [self._cat_pos[f][c] for c in cats]
            record[f] = cats[int(np.argmax(vector[cols]))]
        return record


@dataclass(frozen=True)
class BusinessConstraint:
    r"""Linear business constraint on the numeric fields of the new record.

    The constraint reads ``sum_f coefficients[f] * value_f <sense> rhs``,
    where ``value_f`` is the value of numeric field ``f`` **in its original
    unit** in the suggested record.

    :param Mapping[str, float] coefficients: numeric field name -> coefficient.
    :param str sense: one of ``"<="``, ``">="`` or ``"=="``.
    :param float rhs: right-hand side.
    :param str name: optional human-readable name (used in error messages).
    """

    coefficients: Mapping[str, float]
    sense: str
    rhs: float
    name: str = ""

    def __post_init__(self):
        if self.sense not in ("<=", ">=", "=="):
            raise ValueError(
                f"Invalid sense {self.sense!r}; expected '<=', '>=' or '=='."
            )
        if not np.isfinite(self.rhs):
            raise ValueError("rhs must be finite.")
        for f, v in self.coefficients.items():
            if not np.isfinite(v):
                raise ValueError(f"Coefficient of field {f!r} must be finite.")

    def __str__(self):
        return self.name or f"{dict(self.coefficients)} {self.sense} {self.rhs}"


@dataclass
class RecourseProblem:
    """Feasible domain and cost structure of a recourse search.

    :param Dict[str, float] targets: mapping of metric name to threshold. For
        each targeted metric, the **lower bound** of its Bonferroni interval
        must be greater than or equal to the threshold.
    :param Dict[str, Tuple[float, float]] numeric_bounds: mutable numeric
        fields, mapped to their (lower, upper) bounds. Numeric fields of the
        feature map that are not listed here are immutable.
    :param Dict[str, float] l1_weights: positive cost weight per original unit
        of change, for every mutable numeric field.
    :param Optional[Sequence[str]] immutable_fields: explicit list of immutable
        fields. If None, every numeric field not in ``numeric_bounds`` is
        immutable. If given, it must be disjoint from ``numeric_bounds`` and
        cover exactly the remaining numeric fields.
    :param Dict[str, Sequence] allowed_categories: categorical field name ->
        categories allowed in the new record. Categorical fields not listed
        here are immutable. A field with a single allowed category different
        from the current one is forced to switch (the switching fee applies).
    :param Dict[str, Union[float, Mapping]] switch_costs: switching fee per
        categorical field that may change. Either a non-negative scalar (flat
        fee for any change) or a mapping ``{(from_cat, to_cat): fee}`` (nested
        ``{from_cat: {to_cat: fee}}`` is also accepted).
    :param Sequence[BusinessConstraint] linear_constraints: linear business
        constraints over the numeric fields (original units).
    """

    targets: Dict[str, float]
    numeric_bounds: Dict[str, Tuple[float, float]] = dataclass_field(
        default_factory=dict
    )
    l1_weights: Dict[str, float] = dataclass_field(default_factory=dict)
    immutable_fields: Optional[Sequence[str]] = None
    allowed_categories: Dict[str, Sequence[Any]] = dataclass_field(
        default_factory=dict
    )
    switch_costs: Dict[str, Union[float, Mapping]] = dataclass_field(
        default_factory=dict
    )
    linear_constraints: Sequence[BusinessConstraint] = dataclass_field(
        default_factory=tuple
    )

    def __post_init__(self):
        if len(self.targets) == 0:
            raise ValueError("targets must not be empty.")
        for name, t in self.targets.items():
            if not np.isfinite(t):
                raise ValueError(f"Threshold of metric {name!r} must be finite.")
        for f, bnd in self.numeric_bounds.items():
            try:
                lo, hi = bnd
            except (TypeError, ValueError):
                raise ValueError(
                    f"Invalid bounds for field {f!r}: expected (lower, upper)."
                )
            if not (np.isfinite(lo) and np.isfinite(hi)):
                raise ValueError(f"Bounds of field {f!r} must be finite.")
            if lo > hi:
                raise ValueError(f"Empty bounds for field {f!r}: {lo} > {hi}.")
        for f, w in self.l1_weights.items():
            if not np.isfinite(w) or w <= 0:
                raise ValueError(
                    f"L1 weight of field {f!r} must be positive, got {w!r}."
                )


@dataclass(frozen=True)
class RecourseResult:
    """Verified minimal-cost recourse suggestion.

    :param Dict record: suggested record, expressed in the original fields.
    :param Dict changes: per-field change details
        ``{field: {"old": ..., "new": ..., "cost": ...}}`` for changed fields
        only.
    :param float total_cost: total modification cost (weighted L1 of numeric
        changes in original units plus categorical switching fees), certified
        globally minimal by the MIP solver.
    :param Dict[str, float] point_prediction: metric name -> point prediction
        recomputed with the real model on the suggested record.
    :param Dict[str, Tuple[float, float]] intervals: metric name -> (lower,
        upper) of the native Bonferroni prediction interval at the fixed total
        alpha, recomputed with the real calibrator on the suggested record.
    :param Dict[str, bool] targets_met: targeted metric name -> True (the
        interval lower bound reaches the threshold); all targets are met when
        a result is returned.
    :param float alpha: fixed total significance level of the intervals.
    :param str solver_status: status message of the MIP solver.
    """

    record: Dict[str, Any]
    changes: Dict[str, Dict[str, Any]]
    total_cost: float
    point_prediction: Dict[str, float]
    intervals: Dict[str, Tuple[float, float]]
    targets_met: Dict[str, bool]
    alpha: float
    solver_status: str


@dataclass
class _Domain:
    """Normalized, feature-map-aware view of a RecourseProblem for a record."""

    mutable: List[str]
    x0_num: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    weights: np.ndarray
    switchable: List[Tuple[str, List[Any]]]
    cost_maps: Dict[str, Dict[Tuple[Any, Any], float]]
    forced: Dict[str, Any]
    const_cat_cost: float
    target_idx: List[int]
    target_vals: List[float]


class ConformalRecourse:
    r"""Minimal-cost recourse under native Bonferroni prediction intervals.

    Given a **fixed** pre-trained multi-output linear regression model and an
    independent calibration set, this class:

    1. calibrates split-conformal prediction intervals with puncc's native
       Bonferroni correction at a **fixed** total significance level
       :math:`\alpha` (each of the :math:`K` metrics is calibrated at level
       :math:`\alpha/K`);
    2. solves a mixed-integer linear program for a new record minimizing the
       total modification cost — weighted :math:`L_1` norm of numeric changes
       (original units) plus categorical switching fees — subject to numeric
       bounds, immutable fields, allowed categories, user-given linear
       business constraints, and to every targeted metric's interval **lower
       bound** reaching its threshold:
       :math:`\hat{y}_j(x') - q_j \geq t_j`;
    3. re-verifies the suggested record end-to-end through the real model and
       the real calibrator before returning it.

    The total significance level is fixed at construction and cannot be
    changed by the search. If the problem is infeasible, the calibration set
    too small, or the solver/model fails, an exception is raised and **no
    partial suggestion is returned**.

    :param model: pre-trained multi-output linear regression. It must expose
        ``predict`` and, unless ``coef``/``intercept`` are given explicitly,
        the sklearn-style attributes ``coef_`` (shape ``(K, p)`` or ``(p,)``)
        and ``intercept_``.
    :param OneHotFeatureMap feature_map: fixed mapping between records and
        model feature vectors. The column layout must match the model input.
    :param float alpha: fixed total significance level in (0, 1). The
        simultaneous coverage of the K Bonferroni intervals is at least
        ``1 - alpha``.
    :param Optional[Sequence[str]] metric_names: names of the K model outputs,
        in column order. Defaults to ``["metric_0", ...]``.
    :param Optional[ndarray] coef: explicit linear coefficients, shape
        ``(K, p)`` or ``(p,)``. Defaults to ``model.coef_``.
    :param Optional[ndarray] intercept: explicit intercept(s), shape ``(K,)``
        or scalar. Defaults to ``model.intercept_``.

    .. _example conformalrecourse:

    Example::

        import numpy as np
        from sklearn.linear_model import LinearRegression

        from deel.puncc.recourse import (
            BusinessConstraint,
            ConformalRecourse,
            OneHotFeatureMap,
            RecourseProblem,
        )

        # Fixed one-hot mapping between records and model features
        fmap = OneHotFeatureMap(
            numeric_fields=["age", "income"],
            categorical_fields={"plan": ["A", "B", "C"]},
        )

        # Pre-trained multi-output linear model on encoded features
        model = LinearRegression().fit(X_train, y_train)  # y_train: (n, 2)

        recourse = ConformalRecourse(
            model, fmap, alpha=0.1, metric_names=["sales", "risk"]
        )
        recourse.fit(X_calib, y_calib)  # independent calibration set

        problem = RecourseProblem(
            targets={"sales": 5.0, "risk": 3.0},
            numeric_bounds={"age": (18.0, 70.0), "income": (0.0, 20.0)},
            l1_weights={"age": 1.0, "income": 2.0},
            allowed_categories={"plan": ["A", "B", "C"]},
            switch_costs={"plan": 3.0},
            linear_constraints=[
                BusinessConstraint({"age": 1.0, "income": 1.0}, "<=", 60.0)
            ],
        )
        result = recourse.suggest(
            {"age": 30.0, "income": 5.0, "plan": "A"}, problem
        )
        print(result.record, result.total_cost, result.intervals)
    """

    def __init__(
        self,
        model: Any,
        feature_map: OneHotFeatureMap,
        *,
        alpha: float,
        metric_names: Optional[Sequence[str]] = None,
        coef: Optional[Sequence[Sequence[float]]] = None,
        intercept: Optional[Sequence[float]] = None,
    ):
        if milp is None:  # pragma: no cover
            raise ImportError(
                "ConformalRecourse requires scipy (>= 1.9) to solve the "
                "underlying mixed-integer linear program."
            )
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        if not isinstance(feature_map, OneHotFeatureMap):
            raise TypeError("feature_map must be a OneHotFeatureMap.")

        if coef is None:
            coef = getattr(model, "coef_", None)
            if coef is None:
                raise ValueError(
                    "Linear coefficients could not be read from "
                    "model.coef_; pass them explicitly (coef=...)."
                )
        coef = np.asarray(coef, dtype=float)
        if coef.ndim == 1:
            coef = coef.reshape(1, -1)
        if coef.ndim != 2 or coef.shape[1] != feature_map.n_features:
            raise ValueError(
                f"coef has shape {coef.shape}; expected (K, "
                f"{feature_map.n_features})."
            )
        if intercept is None:
            intercept = getattr(model, "intercept_", 0.0)
        intercept = np.atleast_1d(np.asarray(intercept, dtype=float))
        n_metrics = coef.shape[0]
        if intercept.size == 1 and n_metrics > 1:
            intercept = np.full(n_metrics, intercept.item())
        if intercept.size != n_metrics:
            raise ValueError(
                f"intercept has {intercept.size} entries; expected {n_metrics}."
            )

        if metric_names is None:
            metric_names = [f"metric_{j}" for j in range(n_metrics)]
        metric_names = list(metric_names)
        if len(metric_names) != n_metrics or len(set(metric_names)) != n_metrics:
            raise ValueError(
                f"metric_names must contain {n_metrics} distinct names."
            )

        self._model = model
        self._feature_map = feature_map
        self._alpha = float(alpha)
        self._coef = coef
        self._intercept = intercept
        self._metric_names = tuple(metric_names)
        self._n_metrics = n_metrics
        self._correction = partial(bonferroni, nvars=n_metrics)
        # The model is fixed and pre-trained: it is never retrained here.
        self._cp = SplitCP(BasePredictor(model, is_trained=True), train=False)
        self._is_fitted = False

    @property
    def alpha(self) -> float:
        """Fixed total significance level of the Bonferroni intervals."""
        return self._alpha

    @property
    def metric_names(self) -> Tuple[str, ...]:
        """Names of the model output metrics, in column order."""
        return self._metric_names

    def fit(self, X_calib, y_calib) -> "ConformalRecourse":
        """Calibrate the Bonferroni intervals on an independent calibration set.

        The pre-trained model is **not** retrained: split conformal prediction
        is run with ``train=False`` and only the nonconformity scores of the
        calibration set are computed.

        :param X_calib: calibration features, either a 2D array-like of
            encoded feature vectors (columns ordered as
            :attr:`OneHotFeatureMap.feature_names`) or a sequence of record
            dicts to encode through the feature map.
        :param y_calib: calibration targets, shape ``(n,)`` for a single
            metric or ``(n, K)``.

        :raises InsufficientCalibrationError: if the calibration set is too
            small for the fixed total alpha, i.e. if
            :math:`\\alpha/K < 1/(n+1)`.
        """
        X = self._encode_calib(X_calib)
        y = self._check_y(y_calib, X.shape[0])
        n = X.shape[0]
        K = self._n_metrics
        if n < 2:
            raise InsufficientCalibrationError(
                f"Calibration set too small: n={n}. At least 2 calibration "
                "samples are required."
            )
        # Same consistency condition as puncc's alpha_calib_check, applied to
        # the Bonferroni-corrected level alpha / K.
        if self._alpha / K < 1.0 / (n + 1):
            min_n = int(np.ceil(K / self._alpha - 1 - 1e-9))
            while min_n > 1 and not self._alpha / K < 1.0 / min_n:
                min_n -= 1
            while self._alpha / K < 1.0 / (min_n + 1):
                min_n += 1
            raise InsufficientCalibrationError(
                f"Calibration set too small: n={n} samples for K={K} metrics "
                f"at fixed total alpha={self._alpha}. The Bonferroni level "
                f"alpha/K={self._alpha / K:.6g} requires n >= {min_n} "
                "calibration samples. Provide a larger independent "
                "calibration set (alpha is fixed and cannot be increased)."
            )
        self._cp.fit(X_calib=X, y_calib=y)
        self._is_fitted = True
        logger.info(f"ConformalRecourse calibrated on n={n} samples.")
        return self

    def interval_margins(self) -> Dict[str, float]:
        """Half-widths :math:`q_j` of the calibrated Bonferroni intervals.

        :returns: mapping of metric name to the calibrated quantile of the
            nonconformity scores at level :math:`\\alpha/K`.
        :rtype: Dict[str, float]
        """
        self._check_fitted()
        return {
            name: float(q_j)
            for name, q_j in zip(self._metric_names, self._quantiles())
        }

    def predict_intervals(
        self, record: Mapping[str, Any]
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """Point prediction and native Bonferroni interval of a record.

        The record is encoded through the fixed feature map and evaluated
        through the real model and calibrator at the fixed total alpha.

        :param record: mapping of field names to values.

        :returns: (point prediction, lower bounds, upper bounds), each a
            mapping of metric name to value.
        :rtype: Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]
        """
        self._check_fitted()
        x = self._feature_map.encode(record)
        y_pred, y_lo, y_hi = self._predict_arrays(x)
        return (
            self._as_metric_dict(y_pred),
            self._as_metric_dict(y_lo),
            self._as_metric_dict(y_hi),
        )

    def suggest(
        self,
        record: Mapping[str, Any],
        problem: RecourseProblem,
        *,
        time_limit: Optional[float] = None,
        solver_options: Optional[Mapping] = None,
    ) -> RecourseResult:
        """Search the cheapest record whose interval lower bounds meet the
        targets.

        The search minimizes the total modification cost (weighted L1 of
        numeric changes in original units plus categorical switching fees)
        over the feasible domain defined by ``problem``, subject to the
        Bonferroni interval lower bound of every targeted metric reaching its
        threshold. The mixed-integer linear program is solved to a certified
        global minimum; the suggestion is then re-verified through the real
        model and calibrator.

        :param record: current record, as a mapping of the original fields.
        :param RecourseProblem problem: feasible domain and cost structure.
        :param Optional[float] time_limit: optional solver time limit in
            seconds. If the limit is reached, a :class:`RecourseSolverError`
            is raised (no suboptimal suggestion is returned).
        :param Optional[Mapping] solver_options: extra options forwarded to
            :func:`scipy.optimize.milp` (defaults set a zero MIP gap).

        :returns: verified minimal-cost suggestion.
        :rtype: RecourseResult

        :raises InfeasibleRecourseError: no feasible record meets the targets.
        :raises RecourseSolverError: the solver failed to certify a global
            optimum.
        :raises RecourseVerificationError: the candidate failed verification
            through the real model/calibrator (e.g. the model is not the
            linear model described by the coefficients).
        """
        self._check_fitted()
        if not isinstance(problem, RecourseProblem):
            raise TypeError("problem must be a RecourseProblem.")
        x0 = self._feature_map.encode(record)
        domain = self._build_domain(problem, record)
        q = self._quantiles()
        new_record, total_cost, status = self._optimize(
            record, x0, problem, domain, q, time_limit, solver_options
        )
        return self._finalize(record, new_record, total_cost, status, problem, domain)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Internals ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _check_fitted(self):
        """Raise if the calibrator has not been fitted yet."""
        if not self._is_fitted:
            raise RecourseError("Call `fit` on a calibration set first.")

    def _as_metric_dict(self, values) -> Dict[str, float]:
        """Map a metric-indexed array to a {metric name: value} dict."""
        return {
            name: float(values[j]) for j, name in enumerate(self._metric_names)
        }

    def _encode_calib(self, X_calib) -> np.ndarray:
        """Encode calibration features given as records or as a matrix."""
        if (
            isinstance(X_calib, (list, tuple))
            and len(X_calib) > 0
            and isinstance(X_calib[0], Mapping)
        ):
            return self._feature_map.encode_many(X_calib)
        X = np.asarray(X_calib, dtype=float)
        if X.ndim != 2 or X.shape[1] != self._feature_map.n_features:
            raise ValueError(
                f"X_calib must have shape (n, {self._feature_map.n_features}), "
                + f"got {X.shape}."
            )
        if X.shape[0] < 1:
            raise ValueError("Empty calibration set.")
        return X

    def _check_y(self, y_calib, n: int) -> np.ndarray:
        """Validate and normalize the calibration targets."""
        y = np.asarray(y_calib, dtype=float)
        if self._n_metrics == 1:
            y = y.reshape(-1)
            if y.shape[0] != n:
                raise ValueError(
                    f"y_calib has {y.shape[0]} samples; expected {n}."
                )
            return y
        if y.ndim != 2 or y.shape != (n, self._n_metrics):
            raise ValueError(
                f"y_calib must have shape ({n}, {self._n_metrics}), "
                + f"got {y.shape}."
            )
        return y

    def _quantiles(self) -> np.ndarray:
        """Calibrated per-metric quantiles q_j at the Bonferroni level."""
        q = self._cp.calibrator.compute_quantile(
            alpha=self._alpha, correction=self._correction
        )
        q = np.atleast_1d(np.asarray(q, dtype=float)).reshape(-1)
        if q.size != self._n_metrics:
            raise RecourseError(
                f"Calibrator returned {q.size} quantiles; "
                + f"expected {self._n_metrics}."
            )
        return q

    def _predict_arrays(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Point prediction and Bonferroni interval of one encoded record,
        computed through the real model and calibrator."""
        y_pred, y_lo, y_hi = self._cp.conformal_predictor.predict(
            x.reshape(1, -1), self._alpha, self._correction
        )
        out = []
        for arr in (y_pred, y_lo, y_hi):
            v = np.atleast_1d(np.asarray(arr, dtype=float)).reshape(-1)
            if v.size != self._n_metrics:
                raise RecourseError(
                    f"Model output has {v.size} entries; "
                    + f"expected {self._n_metrics}."
                )
            out.append(v)
        return out[0], out[1], out[2]

    # ~~~~~~~~~~~~~~~~~~~~ Problem validation ~~~~~~~~~~~~~~~~~~~~

    def _build_domain(
        self, problem: RecourseProblem, record: Mapping[str, Any]
    ) -> _Domain:
        """Validate the problem against the feature map and the record, and
        normalize it into a `_Domain`."""
        fmap = self._feature_map
        target_idx, target_vals = [], []
        for name, t in problem.targets.items():
            if name not in self._metric_names:
                raise ValueError(
                    f"Unknown target metric {name!r}; "
                    + f"known metrics: {list(self._metric_names)}."
                )
            target_idx.append(self._metric_names.index(name))
            target_vals.append(float(t))

        mutable, lo, hi, weights, x0_num = [], [], [], [], []
        numeric_fields = set(fmap.numeric_fields)
        for f in fmap.numeric_fields:
            if f not in problem.numeric_bounds:
                continue
            b_lo, b_hi = problem.numeric_bounds[f]
            mutable.append(f)
            lo.append(float(b_lo))
            hi.append(float(b_hi))
            weights.append(float(problem.l1_weights.get(f, np.nan)))
            x0_num.append(float(record[f]))
        self._check_immutable(problem, set(mutable), numeric_fields)
        if set(problem.l1_weights) != set(mutable):
            raise ValueError(
                "l1_weights must be provided for every mutable numeric field "
                + f"{mutable}; got {sorted(problem.l1_weights)}."
            )
        for bc in problem.linear_constraints:
            if not isinstance(bc, BusinessConstraint):
                raise TypeError("linear_constraints must be BusinessConstraint.")
            unknown = set(bc.coefficients) - numeric_fields
            if unknown:
                raise ValueError(
                    f"Business constraint {bc} references non-numeric or "
                    + f"unknown fields {sorted(unknown)}."
                )
        switchable, cost_maps, forced, const_cost = self._build_categoricals(
            problem, record
        )
        return _Domain(
            mutable=mutable,
            x0_num=np.array(x0_num),
            lo=np.array(lo),
            hi=np.array(hi),
            weights=np.array(weights),
            switchable=switchable,
            cost_maps=cost_maps,
            forced=forced,
            const_cat_cost=const_cost,
            target_idx=target_idx,
            target_vals=target_vals,
        )

    def _check_immutable(self, problem, mutable_set, numeric_fields):
        """Validate the immutable/mutable partition of the numeric fields."""
        unknown_bounds = set(problem.numeric_bounds) - numeric_fields
        if unknown_bounds:
            raise ValueError(
                "numeric_bounds reference unknown or non-numeric fields "
                + f"{sorted(unknown_bounds)}."
            )
        if problem.immutable_fields is None:
            return
        immutable = set(problem.immutable_fields)
        categorical_fields = set(self._feature_map.categorical_fields)
        unknown = immutable - numeric_fields - categorical_fields
        if unknown:
            raise ValueError(f"Unknown immutable fields {sorted(unknown)}.")
        if immutable & mutable_set:
            raise ValueError(
                "Fields cannot be both mutable and immutable: "
                + f"{sorted(immutable & mutable_set)}."
            )
        missing = numeric_fields - mutable_set - immutable
        if missing:
            raise ValueError(
                f"Numeric fields {sorted(missing)} are neither mutable "
                + "(numeric_bounds) nor immutable (immutable_fields)."
            )
        if immutable & set(problem.allowed_categories):
            raise ValueError(
                "Fields cannot be both immutable and have allowed_categories: "
                + f"{sorted(immutable & set(problem.allowed_categories))}."
            )

    def _build_categoricals(self, problem, record):
        """Validate allowed categories and switching fees; split categorical
        fields into immutable, forced and switchable ones."""
        fmap = self._feature_map
        switchable, cost_maps, forced = [], {}, {}
        const_cost = 0.0
        for f, cats in problem.allowed_categories.items():
            if f not in fmap.categorical_fields:
                raise ValueError(f"Unknown categorical field {f!r}.")
            cats = list(cats)
            if len(cats) == 0:
                raise ValueError(f"allowed_categories of {f!r} is empty.")
            if len(set(cats)) != len(cats):
                raise ValueError(f"Duplicate allowed categories in {f!r}.")
            unknown = set(cats) - set(fmap.categories_of(f))
            if unknown:
                raise ValueError(
                    f"Allowed categories {sorted(unknown)} of {f!r} are not "
                    + f"in the feature map {list(fmap.categories_of(f))}."
                )
        changeable = []
        for f in fmap.categorical_fields:
            cats = problem.allowed_categories.get(f)
            if cats is None:
                continue  # immutable categorical field
            cats = list(cats)
            current = record[f]
            if len(cats) == 1 and cats[0] == current:
                continue  # effectively immutable
            changeable.append((f, cats, current))
        unknown_costs = set(problem.switch_costs) - {f for f, _, _ in changeable}
        if unknown_costs:
            raise ValueError(
                "switch_costs given for immutable or unknown fields "
                + f"{sorted(unknown_costs)}."
            )
        for f, cats, current in changeable:
            if f not in problem.switch_costs:
                raise ValueError(
                    f"Missing switching fee for categorical field {f!r}."
                )
            needed = {(current, c) for c in cats if c != current}
            cost_maps[f] = _normalize_switch_costs(
                f, problem.switch_costs[f], needed
            )
            if len(cats) == 1:
                forced[f] = cats[0]
                const_cost += cost_maps[f].get((current, cats[0]), 0.0)
            else:
                switchable.append((f, cats))
        return switchable, cost_maps, forced, const_cost

    # ~~~~~~~~~~~~~~~~~~~~ Optimization ~~~~~~~~~~~~~~~~~~~~

    def _optimize(self, record, x0, problem, domain, q, time_limit, solver_options):
        """Run the degenerate check or solve the MILP; return the new record,
        its recomputed total cost and a solver status message."""
        n_z = sum(len(cats) for _, cats in domain.switchable)
        m = len(domain.mutable)
        if 3 * m + n_z == 0:
            # No decision variable: the only candidate is the (possibly
            # forced) current record. Failures here are infeasibilities.
            new_record = {**record, **domain.forced}
            self._verify_domain(new_record, problem, domain, InfeasibleRecourseError)
            _, y_lo, _ = self._predict_arrays(self._feature_map.encode(new_record))
            self._verify_targets(y_lo, problem, InfeasibleRecourseError)
            return new_record, domain.const_cat_cost, "trivial: no decision variables"
        return self._solve_milp(record, x0, problem, domain, q, time_limit, solver_options)

    def _solve_milp(
        self, record, x0, problem, domain, q, time_limit, solver_options
    ):
        """Assemble and solve the recourse MILP with scipy/HiGHS."""
        z_offset, n_vars = _milp_layout(domain)
        const = self._constant_features(x0, domain)
        rows, lbs, ubs = self._milp_constraints(
            record, problem, domain, q, const, z_offset, n_vars
        )
        cost = _milp_objective(record, domain, z_offset, n_vars)
        lb, ub, integrality = _milp_bounds(domain, n_vars)

        options = {"mip_rel_gap": 0.0}
        if time_limit is not None:
            options["time_limit"] = float(time_limit)
        if solver_options:
            options.update(solver_options)
        logger.info("Solving recourse MILP with HiGHS ...")
        res = milp(
            c=cost,
            constraints=LinearConstraint(
                np.array(rows), np.array(lbs), np.array(ubs)
            ),
            integrality=integrality,
            bounds=Bounds(lb, ub),
            options=options,
        )
        if res.status == 2:
            raise InfeasibleRecourseError(
                "No record within the given bounds, allowed categories and "
                "linear constraints can make every targeted interval lower "
                "bound reach its threshold."
            )
        if res.status != 0 or res.x is None:
            raise RecourseSolverError(
                "The MIP solver failed to certify a global optimum "
                + f"(status {res.status}): {res.message}"
            )
        return self._decode(record, res, domain, z_offset)

    def _constant_features(self, x0, domain):
        """Feature vector of the immutable and forced parts of the record
        (mutable numerics and switchable one-hot blocks are zeroed out)."""
        fmap = self._feature_map
        const = x0.copy()
        for f in domain.mutable:
            const[fmap.numeric_position(f)] = 0.0
        for f, _ in domain.switchable:
            for c in fmap.categories_of(f):
                const[fmap.category_column(f, c)] = 0.0
        for f, cat in domain.forced.items():
            for c in fmap.categories_of(f):
                const[fmap.category_column(f, c)] = 0.0
            const[fmap.category_column(f, cat)] = 1.0
        return const

    def _milp_constraints(
        self, record, problem, domain, q, const, z_offset, n_vars
    ):
        """Linear constraints of the recourse MILP: deviation definitions,
        one-hot constraints, interval lower-bound targets and business
        constraints."""
        fmap = self._feature_map
        mutable = domain.mutable
        m = len(mutable)
        rows, lbs, ubs = [], [], []
        # Deviation definition: x_i - d+_i + d-_i = x0_i
        for i in range(m):
            row = np.zeros(n_vars)
            row[i], row[m + i], row[2 * m + i] = 1.0, -1.0, 1.0
            rows.append(row)
            lbs.append(domain.x0_num[i])
            ubs.append(domain.x0_num[i])
        # One-hot constraints: sum_a z_{f,a} = 1
        for f, cats in domain.switchable:
            row = np.zeros(n_vars)
            row[z_offset[f] : z_offset[f] + len(cats)] = 1.0
            rows.append(row)
            lbs.append(1.0)
            ubs.append(1.0)
        # Interval lower-bound targets: W_j x + b_j - q_j >= t_j
        for j, t in zip(domain.target_idx, domain.target_vals):
            row = np.zeros(n_vars)
            w_row = self._coef[j]
            for i, f in enumerate(mutable):
                row[i] = w_row[fmap.numeric_position(f)]
            for f, cats in domain.switchable:
                cols = [fmap.category_column(f, c) for c in cats]
                row[z_offset[f] : z_offset[f] + len(cats)] = w_row[cols]
            rows.append(row)
            lbs.append(t + q[j] - (w_row @ const + self._intercept[j]))
            ubs.append(np.inf)
        # Linear business constraints (immutable fields folded into the rhs)
        x_col = {f: i for i, f in enumerate(mutable)}
        for bc in problem.linear_constraints:
            row = np.zeros(n_vars)
            rhs = float(bc.rhs)
            for f, c in bc.coefficients.items():
                if f in x_col:
                    row[x_col[f]] += c
                else:
                    rhs -= c * float(record[f])
            rows.append(row)
            if bc.sense == "<=":
                lbs.append(-np.inf)
                ubs.append(rhs)
            elif bc.sense == ">=":
                lbs.append(rhs)
                ubs.append(np.inf)
            else:
                lbs.append(rhs)
                ubs.append(rhs)
        return rows, lbs, ubs

    def _decode(self, record, res, domain, z_offset):
        """Decode the solver output into a record and recompute its cost."""
        xsol = res.x
        new_record = dict(record)
        for i, f in enumerate(domain.mutable):
            v = float(xsol[i])
            if v < domain.lo[i] - _FEAS_TOL or v > domain.hi[i] + _FEAS_TOL:
                raise RecourseSolverError(
                    f"Solver returned {f}={v} outside its bounds "
                    + f"[{domain.lo[i]}, {domain.hi[i]}]."
                )
            new_record[f] = min(max(v, domain.lo[i]), domain.hi[i])
        for f, cats in domain.switchable:
            block = xsol[z_offset[f] : z_offset[f] + len(cats)]
            if abs(block.sum() - 1.0) > _INT_TOL or block.max() < 1.0 - _INT_TOL:
                raise RecourseSolverError(
                    f"Solver returned a non-integral category choice for {f!r}."
                )
            new_record[f] = cats[int(np.argmax(block))]
        new_record.update(domain.forced)

        # Recompute the cost from the decoded record (source of truth).
        total = domain.const_cat_cost
        for i, f in enumerate(domain.mutable):
            total += domain.weights[i] * abs(new_record[f] - domain.x0_num[i])
        for f, _ in domain.switchable:
            if new_record[f] != record[f]:
                total += domain.cost_maps[f].get(
                    (record[f], new_record[f]), 0.0
                )
        if not np.isclose(
            total, res.fun + domain.const_cat_cost, rtol=1e-4, atol=1e-6
        ):
            raise RecourseSolverError(
                "The decoded record cost is inconsistent with the solver "
                "objective; aborting without suggestion."
            )
        return new_record, float(total), str(res.message)

    # ~~~~~~~~~~~~~~~~~~~~ Verification ~~~~~~~~~~~~~~~~~~~~

    def _verify_domain(self, new_record, problem, domain, err_cls):
        """Re-check bounds and business constraints on a decoded record."""
        for i, f in enumerate(domain.mutable):
            v = float(new_record[f])
            if v < domain.lo[i] - _FEAS_TOL or v > domain.hi[i] + _FEAS_TOL:
                raise err_cls(
                    f"Field {f!r}={v} violates its bounds "
                    + f"[{domain.lo[i]}, {domain.hi[i]}]."
                )
        for bc in problem.linear_constraints:
            lhs = sum(c * float(new_record[f]) for f, c in bc.coefficients.items())
            tol = _FEAS_TOL * (1.0 + abs(bc.rhs))
            violated = (
                (bc.sense == "<=" and lhs > bc.rhs + tol)
                or (bc.sense == ">=" and lhs < bc.rhs - tol)
                or (bc.sense == "==" and abs(lhs - bc.rhs) > tol)
            )
            if violated:
                raise err_cls(
                    f"Business constraint violated: {bc} (lhs={lhs:.6g})."
                )

    def _verify_targets(self, y_lo, problem, err_cls):
        """Re-check that every targeted interval lower bound meets its
        threshold."""
        for name, t in problem.targets.items():
            j = self._metric_names.index(name)
            if y_lo[j] < t - _FEAS_TOL * (1.0 + abs(t)):
                raise err_cls(
                    f"Interval lower bound of metric {name!r} is "
                    f"{y_lo[j]:.6g} < threshold {t:.6g}."
                )

    def _finalize(self, record, new_record, total_cost, status, problem, domain):
        """End-to-end verification through the real model and calibrator, then
        build the result. No partial suggestion is returned on failure."""
        x_new = self._feature_map.encode(new_record)
        y_pred, y_lo, y_hi = self._predict_arrays(x_new)

        # The model must be the linear model described by the coefficients.
        lin_pred = self._coef @ x_new + self._intercept
        if not np.allclose(lin_pred, y_pred, rtol=1e-5, atol=1e-6):
            raise RecourseVerificationError(
                "The model predictions are inconsistent with the linear "
                "coefficients used for the search; the model must be the "
                "fixed multi-output linear regression described by "
                "coef/intercept."
            )
        # The given model object must still be the model that was calibrated
        # (puncc copies predictors at fit time; a mutated/retrained model
        # object invalidates the calibrated intervals).
        live_pred = np.atleast_1d(
            np.asarray(self._model.predict(x_new.reshape(1, -1)), dtype=float)
        ).reshape(-1)
        if live_pred.size != self._n_metrics or not np.allclose(
            live_pred, y_pred, rtol=1e-5, atol=1e-6
        ):
            raise RecourseVerificationError(
                "The model object differs from the model that was calibrated "
                "in `fit`; rebuild the ConformalRecourse (or refit) with the "
                "current model."
            )
        # Re-check the feasible domain and the interval targets on the
        # actual outputs of the real model and calibrator.
        self._verify_domain(new_record, problem, domain, RecourseVerificationError)
        self._verify_targets(y_lo, problem, RecourseVerificationError)

        changes = {}
        for i, f in enumerate(domain.mutable):
            delta = float(new_record[f]) - float(record[f])
            if abs(delta) > 1e-9:
                changes[f] = {
                    "old": record[f],
                    "new": new_record[f],
                    "cost": float(domain.weights[i] * abs(delta)),
                }
        for f, _ in domain.switchable:
            if new_record[f] != record[f]:
                changes[f] = {
                    "old": record[f],
                    "new": new_record[f],
                    "cost": float(
                        domain.cost_maps[f].get((record[f], new_record[f]), 0.0)
                    ),
                }
        for f, cat in domain.forced.items():
            if cat != record[f]:
                changes[f] = {
                    "old": record[f],
                    "new": cat,
                    "cost": float(domain.cost_maps[f].get((record[f], cat), 0.0)),
                }

        to_dict = self._as_metric_dict
        return RecourseResult(
            record=new_record,
            changes=changes,
            total_cost=float(total_cost),
            point_prediction=to_dict(y_pred),
            intervals={
                name: (float(y_lo[j]), float(y_hi[j]))
                for j, name in enumerate(self._metric_names)
            },
            targets_met={name: True for name in problem.targets},
            alpha=self._alpha,
            solver_status=status,
        )


def _milp_layout(domain):
    """Variable layout of the recourse MILP.

    Variables are ordered as: numeric values ``x`` on ``[0, m)``, positive
    deviations ``d+`` on ``[m, 2m)``, negative deviations ``d-`` on
    ``[2m, 3m)``, then one binary variable per allowed category on
    ``[3m, 3m + n_z)``.

    :returns: (mapping of categorical field to the offset of its one-hot
        block, total number of variables).
    """
    m = len(domain.mutable)
    z_offset = {}
    offset = 3 * m
    for f, cats in domain.switchable:
        z_offset[f] = offset
        offset += len(cats)
    return z_offset, offset


def _milp_objective(record, domain, z_offset, n_vars):
    """Cost vector: weighted L1 on deviations plus switching fees."""
    m = len(domain.mutable)
    cost = np.zeros(n_vars)
    cost[m : 2 * m] = domain.weights
    cost[2 * m : 3 * m] = domain.weights
    for f, cats in domain.switchable:
        current = record[f]
        cost[z_offset[f] : z_offset[f] + len(cats)] = [
            domain.cost_maps[f].get((current, c), 0.0) for c in cats
        ]
    return cost


def _milp_bounds(domain, n_vars):
    """Variable bounds and integrality flags of the recourse MILP."""
    m = len(domain.mutable)
    lb = np.zeros(n_vars)
    ub = np.zeros(n_vars)
    lb[:m], ub[:m] = domain.lo, domain.hi
    ub[m : 2 * m] = np.maximum(domain.hi - domain.x0_num, 0.0)
    ub[2 * m : 3 * m] = np.maximum(domain.x0_num - domain.lo, 0.0)
    ub[3 * m :] = 1.0
    integrality = np.zeros(n_vars)
    integrality[3 * m :] = 1
    return lb, ub, integrality


def _normalize_switch_costs(field, spec, needed_pairs):
    """Normalize a switching-fee specification into a cost map
    ``{(from_cat, to_cat): fee}`` covering ``needed_pairs``."""
    if isinstance(spec, (int, float, np.integer, np.floating)) and not isinstance(
        spec, bool
    ):
        costs = {pair: float(spec) for pair in needed_pairs}
    elif isinstance(spec, Mapping):
        if all(isinstance(k, tuple) for k in spec):
            flat = dict(spec)
        elif all(isinstance(v, Mapping) for v in spec.values()):
            flat = {
                (frm, to): fee for frm, sub in spec.items() for to, fee in sub.items()
            }
        else:
            raise ValueError(
                f"Invalid switch cost mapping for field {field!r}: use either "
                + "{(from_cat, to_cat): fee} or {from_cat: {to_cat: fee}}."
            )
        missing = needed_pairs - set(flat)
        if missing:
            raise ValueError(
                f"Missing switch costs for field {field!r}: {sorted(missing)}."
            )
        costs = {pair: float(flat[pair]) for pair in needed_pairs}
    else:
        raise ValueError(
            f"Invalid switch cost for field {field!r}: {spec!r}."
        )
    for pair, fee in costs.items():
        if not np.isfinite(fee) or fee < 0:
            raise ValueError(
                f"Switch cost {fee} for field {field!r} and pair {pair} must "
                + "be non-negative and finite."
            )
    return costs
