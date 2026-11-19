.. _recourse:

🎯 Conformal recourse
=====================

This module searches for minimal-cost modifications of a business record so
that the **lower bounds** of the native Bonferroni prediction intervals of a
**fixed** pre-trained multi-output linear model reach user-given thresholds,
at a fixed total significance level :math:`\alpha`.

Given an independent calibration set, :class:`ConformalRecourse` calibrates
split-conformal Bonferroni intervals (each of the :math:`K` metrics at level
:math:`\alpha/K`), then solves a mixed-integer linear program that minimizes
the total modification cost — weighted :math:`L_1` norm of the numeric changes
(in original units) plus categorical switching fees — subject to numeric
bounds, immutable fields, allowed categories and linear business constraints.
Categories enter the model through a fixed one-hot mapping
(:class:`OneHotFeatureMap`). Every suggestion is re-verified end-to-end
through the real model and calibrator before being returned.

.. autoclass:: deel.puncc.recourse.ConformalRecourse

.. autoclass:: deel.puncc.recourse.OneHotFeatureMap

.. autoclass:: deel.puncc.recourse.RecourseProblem

.. autoclass:: deel.puncc.recourse.BusinessConstraint

.. autoclass:: deel.puncc.recourse.RecourseResult

.. autoclass:: deel.puncc.recourse.RecourseError

.. autoclass:: deel.puncc.recourse.InsufficientCalibrationError

.. autoclass:: deel.puncc.recourse.InfeasibleRecourseError

.. autoclass:: deel.puncc.recourse.RecourseSolverError

.. autoclass:: deel.puncc.recourse.RecourseVerificationError
