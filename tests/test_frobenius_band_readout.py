"""CPU-only checks for the norm-band report's comparison denominators."""

import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "band_readout", Path(__file__).parents[1] / "scripts/manifold/report_frobenius_band_20261122.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def test_summary_ignores_null_angles_but_not_nonfinite_values():
    assert report.distribution([None, 1, 2, 3])["median"] == 2
    with pytest.raises(ValueError):
        report.distribution([1, float("nan")])


def test_endpoint_is_not_a_substitute_for_the_predeclared_window():
    reference = {s: 4 for s in report.WINDOW}
    treatment = {s: 4.01 for s in report.WINDOW}
    treatment[3000] = 3.999
    effect = report.loss_effect(reference, treatment)
    assert effect["endpoint_delta"] < 0
    assert effect["window_mean_delta"] > 0
    assert effect["window_wins"] == 1


def test_mismatched_validation_grids_are_rejected():
    with pytest.raises(ValueError):
        report.loss_effect({200: 4}, {400: 4})


def records():
    return [dict(step=s, parameter=f"layer.{i}", component="q",
                 **{m: (1.0 if "norm_over_R" in m else .01) for m in report.METRICS})
            for s in report.SAMPLES for i in range(196)]


def test_complete_grid_has_7056_rows():
    rows = records()
    assert len(rows) == 7056
    assert len(report.verify_rows(rows, .9, 1.1)) == 196


@pytest.mark.parametrize("problem", ["missing", "duplicate", "infeasible", "nan", "null_displacement"])
def test_invalid_diagnostics_are_rejected(problem):
    rows = records()
    if problem == "missing":
        rows.pop()
    elif problem == "duplicate":
        rows.append(rows[0])
    elif problem == "infeasible":
        rows[0]["returned_norm_over_R"] = 1.2
    elif problem == "nan":
        rows[0]["polar_angle"] = float("nan")
    else:
        rows[0]["aus"] = None
    with pytest.raises(ValueError):
        report.verify_rows(rows, .9, 1.1)


def test_aus_ratios_are_paired_before_taking_the_median():
    rows = records()
    reference = {(r["step"], r["parameter"], r["component"]): dict(r) for r in rows}
    for r in rows:
        r["aus"] *= 2
    summary, _ = report.summarize_rows(rows, .9, 1.1, reference)
    assert all(r["paired_aus_ratio"]["median"] == 2 for r in summary)


def test_zero_lr_has_no_defined_relative_aus():
    rows = records()
    for r in rows:
        if r["step"] == 1:
            r["aus"] = r["trial_elr"] = 0
    reference = {(r["step"], r["parameter"], r["component"]): r for r in rows}
    summary, _ = report.summarize_rows(rows, .9, 1.1, reference)
    assert summary[0]["paired_aus_ratio"]["count"] == 0
    assert summary[0]["paired_aus_ratio"]["median"] is None
