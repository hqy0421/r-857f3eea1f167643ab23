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

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

import deel.puncc.recourse as recourse_module
from deel.puncc.recourse import BusinessConstraint
from deel.puncc.recourse import ConformalRecourse
from deel.puncc.recourse import InfeasibleRecourseError
from deel.puncc.recourse import InsufficientCalibrationError
from deel.puncc.recourse import OneHotFeatureMap
from deel.puncc.recourse import RecourseError
from deel.puncc.recourse import RecourseProblem
from deel.puncc.recourse import RecourseResult
from deel.puncc.recourse import RecourseSolverError
from deel.puncc.recourse import RecourseVerificationError

# Ground truth of the synthetic data generating process. Features are
# [age, income, plan=A, plan=B, plan=C]; two metrics.
W_TRUE = np.array(
    [
        [1.5, -0.5, 2.0, 0.0, -1.0],
        [0.5, 1.0, -1.0, 1.0, 0.0],
    ]
)
B_TRUE = np.array([0.3, -0.2])

RECORD = {"age": 4.0, "income": 6.0, "plan": "B"}


def make_data(n=400, seed=0, noise=0.5):
    rng = np.random.RandomState(seed)
    x_num = rng.uniform(0, 10, size=(n, 2))
    cat = rng.randint(0, 3, size=n)
    X = np.hstack([x_num, np.eye(3)[cat]])
    y = X @ W_TRUE.T + B_TRUE + rng.normal(0, noise, size=(n, 2))
    return X, y


def make_fmap():
    return OneHotFeatureMap(
        numeric_fields=["age", "income"],
        categorical_fields={"plan": ["A", "B", "C"]},
    )


def base_problem(**overrides):
    kwargs = dict(
        targets={"m0": 5.0, "m1": 3.0},
        numeric_bounds={"age": (0.0, 10.0), "income": (0.0, 10.0)},
        l1_weights={"age": 2.0, "income": 1.0},
        allowed_categories={"plan": ["A", "B", "C"]},
        switch_costs={"plan": 3.0},
    )
    kwargs.update(overrides)
    return RecourseProblem(**kwargs)


@pytest.fixture
def fitted_recourse():
    X, y = make_data()
    model = LinearRegression().fit(X[:300], y[:300])
    rec = ConformalRecourse(
        model, make_fmap(), alpha=0.1, metric_names=["m0", "m1"]
    )
    rec.fit(X[300:], y[300:])
    return rec, model


def test_suggest_returns_verified_record_in_original_fields(fitted_recourse):
    rec, model = fitted_recourse
    result = rec.suggest(RECORD, base_problem())

    assert isinstance(result, RecourseResult)
    # Record is expressed in the original fields.
    assert set(result.record) == {"age", "income", "plan"}
    assert result.record["plan"] in ("A", "B", "C")
    # All interval lower bounds meet their thresholds.
    for name, t in base_problem().targets.items():
        assert result.intervals[name][0] >= t - 1e-6
        assert result.targets_met[name] is True
    # Point predictions and intervals are recomputed through the real
    # model and calibrator: they match a direct call on the same record.
    point, lower, upper = rec.predict_intervals(result.record)
    assert point == pytest.approx(result.point_prediction)
    assert lower == pytest.approx(
        {k: v[0] for k, v in result.intervals.items()}
    )
    assert upper == pytest.approx({k: v[1] for k, v in result.intervals.items()})
    # The real model agrees with the returned point predictions.
    x_new = make_fmap().encode(result.record)
    real = model.predict(x_new.reshape(1, -1))[0]
    assert result.point_prediction["m0"] == pytest.approx(real[0])
    assert result.point_prediction["m1"] == pytest.approx(real[1])
    # Total cost is the sum of the declared change costs.
    assert result.total_cost == pytest.approx(
        sum(c["cost"] for c in result.changes.values())
    )
    # Changes detail old/new values.
    for f, change in result.changes.items():
        assert change["old"] == RECORD[f]
        assert change["new"] == result.record[f]


def test_suggest_is_globally_minimal(fitted_recourse):
    rec, model = fitted_recourse
    problem = base_problem()
    result = rec.suggest(RECORD, problem)
    margins = rec.interval_margins()

    # Brute-force search over a fine grid (numerics) x categories, using the
    # same fitted linear model as the MILP.
    W_fit = np.atleast_2d(model.coef_)
    b_fit = np.atleast_1d(model.intercept_)
    best = np.inf
    for age in np.arange(0.0, 10.0001, 0.02):
        for income in np.arange(0.0, 10.0001, 0.02):
            for z, plan in enumerate(("A", "B", "C")):
                x = np.array([age, income] + list(np.eye(3)[z]))
                pred = W_fit @ x + b_fit
                if np.all(
                    pred - np.array([margins["m0"], margins["m1"]])
                    >= np.array([5.0, 3.0]) - 1e-9
                ):
                    cost = 2.0 * abs(age - 4.0) + abs(income - 6.0)
                    cost += 0.0 if plan == "B" else 3.0
                    best = min(best, cost)
    # The MILP solution is at least as good as any grid point (continuous
    # optimum) and matches it up to the grid resolution.
    assert result.total_cost <= best + 1e-6
    assert result.total_cost >= best - 0.05


def test_interval_lower_bound_not_point_prediction(fitted_recourse):
    rec, _ = fitted_recourse
    # Thresholds exactly at the current point predictions: a point-only
    # search would not change anything, but the interval lower bounds fall
    # short by the calibrated margins.
    point, lower, _ = rec.predict_intervals(RECORD)
    problem = base_problem(targets=point)
    result = rec.suggest(RECORD, problem)

    assert lower["m0"] < point["m0"]  # current record fails the targets
    assert result.total_cost > 0
    for name, t in point.items():
        assert result.intervals[name][0] >= t - 1e-6


def test_business_constraint_and_immutable_fields(fitted_recourse):
    rec, _ = fitted_recourse
    # With income immutable and the cap on age + income, the unconstrained
    # optimum (age ~ 5.7, plan B) is excluded: the search must switch plan.
    problem = base_problem(
        numeric_bounds={"age": (0.0, 10.0)},
        l1_weights={"age": 2.0},
        immutable_fields=["income"],
        linear_constraints=[
            BusinessConstraint(
                {"age": 1.0, "income": 1.0}, "<=", 11.0, name="cap"
            )
        ],
    )
    result = rec.suggest(RECORD, problem)
    # Immutable field is untouched.
    assert result.record["income"] == RECORD["income"]
    assert "income" not in result.changes
    # The linear business constraint holds on the suggested record.
    assert result.record["age"] + result.record["income"] <= 11.0 + 1e-6
    # The constraint binds: staying with plan B would require age > 5.
    assert result.record["plan"] == "A"
    for name, t in problem.targets.items():
        assert result.intervals[name][0] >= t - 1e-6


def test_categorical_switch_when_cheaper(fitted_recourse):
    rec, _ = fitted_recourse
    # Numeric changes are prohibitively expensive: switching plan B -> A
    # (+2 on m0 for a flat fee of 3) is the cheap direction.
    problem = base_problem(l1_weights={"age": 100.0, "income": 100.0})
    result = rec.suggest(RECORD, problem)
    assert result.record["plan"] == "A"
    assert result.changes["plan"] == {"old": "B", "new": "A", "cost": 3.0}


def test_no_change_when_already_compliant(fitted_recourse):
    rec, _ = fitted_recourse
    result = rec.suggest(RECORD, base_problem(targets={"m0": 1.0, "m1": 1.0}))
    assert result.total_cost == 0.0
    assert result.changes == {}
    assert result.record == RECORD


def test_forced_categorical_switch(fitted_recourse):
    rec, _ = fitted_recourse
    problem = base_problem(
        targets={"m0": 1.0, "m1": 1.0},
        allowed_categories={"plan": ["A"]},
        switch_costs={"plan": 2.5},
    )
    result = rec.suggest(RECORD, problem)
    assert result.record["plan"] == "A"
    assert result.changes["plan"]["cost"] == pytest.approx(2.5)
    assert result.total_cost == pytest.approx(2.5)


def test_switch_cost_mappings(fitted_recourse):
    rec, _ = fitted_recourse
    # Per-pair flat mapping: B->A is cheap, B->C is expensive.
    problem = base_problem(
        l1_weights={"age": 100.0, "income": 100.0},
        switch_costs={"plan": {("B", "A"): 1.0, ("B", "C"): 5.0}},
    )
    result = rec.suggest(RECORD, problem)
    assert result.record["plan"] == "A"
    assert result.changes["plan"]["cost"] == pytest.approx(1.0)
    # Nested mapping form is also accepted.
    problem = base_problem(
        l1_weights={"age": 100.0, "income": 100.0},
        switch_costs={"plan": {"B": {"A": 1.0, "C": 5.0}}},
    )
    result = rec.suggest(RECORD, problem)
    assert result.record["plan"] == "A"
    assert result.changes["plan"]["cost"] == pytest.approx(1.0)


def test_infeasible_is_distinct_from_insufficient_calibration(fitted_recourse):
    rec, _ = fitted_recourse
    # Infeasible: enough calibration data, but the threshold is unreachable.
    with pytest.raises(InfeasibleRecourseError) as exc_info:
        rec.suggest(RECORD, base_problem(targets={"m0": 1e6, "m1": 3.0}))
    assert not isinstance(exc_info.value, InsufficientCalibrationError)

    # Insufficient calibration: n must satisfy alpha / K >= 1 / (n + 1).
    X, y = make_data()
    model = LinearRegression().fit(X[:300], y[:300])
    small = ConformalRecourse(
        model, make_fmap(), alpha=0.1, metric_names=["m0", "m1"]
    )
    with pytest.raises(InsufficientCalibrationError) as exc_info2:
        small.fit(X[300:310], y[300:310])  # n=10 < 19 required
    assert isinstance(exc_info2.value, ValueError)
    assert "n >= 19" in str(exc_info2.value)

    # A single calibration sample is never enough.
    with pytest.raises(InsufficientCalibrationError, match="At least 2"):
        small.fit(X[300:301], y[300:301])


def test_solver_failure_returns_no_partial_suggestion(
    fitted_recourse, monkeypatch
):
    rec, _ = fitted_recourse

    class FakeMilp:
        def __init__(self, status, message):
            self.status = status
            self.message = message
            self.x = None
            self.fun = np.inf

    monkeypatch.setattr(
        recourse_module, "milp", lambda **kw: FakeMilp(1, "time limit reached")
    )
    with pytest.raises(RecourseSolverError):
        rec.suggest(RECORD, base_problem())

    monkeypatch.setattr(
        recourse_module, "milp", lambda **kw: FakeMilp(2, "infeasible")
    )
    with pytest.raises(InfeasibleRecourseError):
        rec.suggest(RECORD, base_problem())


def test_verification_fails_if_model_differs_from_coefficients(fitted_recourse):
    rec, model = fitted_recourse
    # The model drifts away from the coefficients used for the search.
    model.coef_ = model.coef_ + 1.0
    with pytest.raises(RecourseVerificationError):
        rec.suggest(RECORD, base_problem())


def test_verification_fails_on_wrong_explicit_coefficients():
    X, y = make_data()
    model = LinearRegression().fit(X[:300], y[:300])
    # Explicit coefficients that do not match the real model.
    rec = ConformalRecourse(
        model,
        make_fmap(),
        alpha=0.1,
        metric_names=["m0", "m1"],
        coef=model.coef_ + 1.0,
        intercept=model.intercept_,
    )
    rec.fit(X[300:], y[300:])
    with pytest.raises(RecourseVerificationError):
        rec.suggest(RECORD, base_problem())


def test_single_output_model():
    rng = np.random.RandomState(1)
    X, _ = make_data(n=400, seed=1)
    w = np.array([1.0, 0.5, 0.0, 2.0, -1.0])
    y1 = X @ w + rng.normal(0, 0.3, size=len(X))
    model = LinearRegression().fit(X[:300], y1[:300])
    rec = ConformalRecourse(model, make_fmap(), alpha=0.1, metric_names=["y"])
    rec.fit(X[300:], y1[300:])
    problem = RecourseProblem(
        targets={"y": 8.0},
        numeric_bounds={"age": (0.0, 10.0), "income": (0.0, 10.0)},
        l1_weights={"age": 1.0, "income": 1.0},
        allowed_categories={"plan": ["A", "B", "C"]},
        switch_costs={"plan": 2.0},
    )
    result = rec.suggest(RECORD, problem)
    assert set(result.intervals) == {"y"}
    assert result.intervals["y"][0] >= 8.0 - 1e-6


def test_alpha_is_fixed_and_intervals_are_native_bonferroni(fitted_recourse):
    rec, _ = fitted_recourse
    assert rec.alpha == 0.1
    # alpha cannot be changed through the search API.
    assert "alpha" not in inspect.signature(rec.suggest).parameters
    result = rec.suggest(RECORD, base_problem())
    assert result.alpha == 0.1
    # Native Bonferroni intervals are constant-width: half-widths equal the
    # calibrated quantiles at level alpha / K.
    margins = rec.interval_margins()
    for name, (lo, hi) in result.intervals.items():
        assert (hi - lo) / 2 == pytest.approx(margins[name], rel=1e-9)
        assert result.point_prediction[name] == pytest.approx((lo + hi) / 2)


def test_degenerate_problem_without_decision_variables(fitted_recourse):
    rec, _ = fitted_recourse
    # Nothing can change: the current record is the only candidate.
    trivial = RecourseProblem(targets={"m0": 1.0, "m1": 1.0})
    result = rec.suggest(RECORD, trivial)
    assert result.total_cost == 0.0
    assert result.changes == {}
    assert result.record == RECORD

    impossible = RecourseProblem(targets={"m0": 1e6})
    with pytest.raises(InfeasibleRecourseError):
        rec.suggest(RECORD, impossible)


def test_fit_accepts_records_as_calibration_set():
    X, y = make_data()
    model = LinearRegression().fit(X[:300], y[:300])
    fmap = make_fmap()
    records = [fmap.decode(row) for row in X[300:]]
    rec = ConformalRecourse(model, fmap, alpha=0.1, metric_names=["m0", "m1"])
    rec.fit(records, y[300:])
    result = rec.suggest(RECORD, base_problem())
    assert result.targets_met == {"m0": True, "m1": True}


def test_suggest_before_fit_raises():
    X, y = make_data()
    model = LinearRegression().fit(X[:300], y[:300])
    rec = ConformalRecourse(model, make_fmap(), alpha=0.1)
    with pytest.raises(RecourseError, match="fit"):
        rec.suggest(RECORD, base_problem(targets={"metric_0": 1.0}))


def test_invalid_specs_raise_value_error(fitted_recourse):
    rec, _ = fitted_recourse

    # Unknown target metric.
    with pytest.raises(ValueError, match="Unknown target metric"):
        rec.suggest(RECORD, base_problem(targets={"unknown": 1.0}))
    # Missing L1 weight for a mutable field.
    with pytest.raises(ValueError, match="l1_weights"):
        rec.suggest(RECORD, base_problem(l1_weights={"age": 1.0}))
    # Unknown category in the record.
    with pytest.raises(ValueError, match="Unknown category"):
        rec.suggest({"age": 4.0, "income": 6.0, "plan": "Z"}, base_problem())
    # Allowed category outside the fixed one-hot mapping.
    with pytest.raises(ValueError, match="not in the feature map"):
        rec.suggest(RECORD, base_problem(allowed_categories={"plan": ["A", "Z"]}))
    # Missing switching fee for a switchable field.
    with pytest.raises(ValueError, match="Missing switching fee"):
        rec.suggest(RECORD, base_problem(switch_costs={}))
    # Switching fee given for an immutable field.
    with pytest.raises(ValueError, match="immutable or unknown"):
        rec.suggest(
            RECORD,
            base_problem(
                allowed_categories={}, switch_costs={"plan": 1.0}
            ),
        )
    # Business constraint on a categorical field.
    with pytest.raises(ValueError, match="non-numeric"):
        rec.suggest(
            RECORD,
            base_problem(
                linear_constraints=[BusinessConstraint({"plan": 1.0}, "<=", 1.0)]
            ),
        )
    # Field both mutable and immutable.
    with pytest.raises(ValueError, match="both mutable and immutable"):
        rec.suggest(
            RECORD,
            base_problem(
                immutable_fields=["age", "income"],
            ),
        )
    # Numeric field neither mutable nor immutable.
    with pytest.raises(ValueError, match="neither mutable"):
        rec.suggest(
            RECORD,
            base_problem(
                numeric_bounds={"age": (0.0, 10.0)},
                l1_weights={"age": 2.0},
                immutable_fields=[],
            ),
        )
    # Missing switch cost pair.
    with pytest.raises(ValueError, match="Missing switch costs"):
        rec.suggest(RECORD, base_problem(switch_costs={"plan": {("B", "A"): 1.0}}))


def test_problem_dataclass_validation():
    with pytest.raises(ValueError, match="must not be empty"):
        RecourseProblem(targets={})
    with pytest.raises(ValueError, match="must be positive"):
        RecourseProblem(targets={"m0": 1.0}, l1_weights={"age": 0.0})
    with pytest.raises(ValueError, match="Empty bounds"):
        RecourseProblem(targets={"m0": 1.0}, numeric_bounds={"age": (2.0, 1.0)})
    with pytest.raises(ValueError, match="Invalid sense"):
        BusinessConstraint({"age": 1.0}, "<", 1.0)


def test_constructor_validation():
    X, y = make_data()
    model = LinearRegression().fit(X[:300], y[:300])
    with pytest.raises(ValueError, match="alpha"):
        ConformalRecourse(model, make_fmap(), alpha=1.5)
    with pytest.raises(ValueError, match="coef"):
        ConformalRecourse(
            model, make_fmap(), alpha=0.1, coef=np.zeros((2, 3))
        )
    with pytest.raises(ValueError, match="metric_names"):
        ConformalRecourse(
            model, make_fmap(), alpha=0.1, metric_names=["only_one"]
        )


def test_feature_map_encode_decode():
    fmap = make_fmap()
    x = fmap.encode(RECORD)
    assert x.tolist() == [4.0, 6.0, 0.0, 1.0, 0.0]
    assert fmap.feature_names == [
        "age",
        "income",
        "plan='A'",
        "plan='B'",
        "plan='C'",
    ]
    assert fmap.decode(x) == RECORD
    with pytest.raises(ValueError, match="do not match"):
        fmap.encode({"age": 4.0, "plan": "B"})
    with pytest.raises(ValueError, match="Duplicate categories"):
        OneHotFeatureMap(["a"], {"c": ["x", "x"]})
