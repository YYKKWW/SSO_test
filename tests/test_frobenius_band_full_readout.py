"""Small CPU tests for full-budget comparisons and completed-trace checks."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location("full_band", Path(__file__).parents[1] / "scripts/manifold/report_frobenius_band_20261125.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def test_early_win_does_not_override_full_budget_or_late_window():
    ref = {s: 4.0 for s in (3000, *report.WINDOW)}
    other = {s: 4.001 for s in ref}
    other[3000] = 3.99
    result = report.effect(ref, other)
    assert result["endpoint_delta"] > 0 and result["late_mean_delta"] > 0
    assert result["last_logged_win"] == 3000 and result["late_wins"] == 0


def test_missing_final_validation_is_rejected():
    with pytest.raises(ValueError):
        report.effect({3000: 4}, {3000: 4})


def fixture(directory):
    layout = [dict(parameter=f"w{i}", radius=2.) for i in range(196)]
    aus = np.full((3, 196), .01, dtype=np.float32)
    norms = np.full((3, 196, 2), 2., dtype=np.float32)
    np.save(directory / "aus.npy", aus)
    np.save(directory / "matrix_norms.npy", norms)
    (directory / "aus_trace.json").write_text(json.dumps(dict(steps=3, matrices=layout, sha256=report.digest(directory / "aus.npy"))))
    meta = dict(steps=3, fields=["before_update", "after_return"], band=[.9, 1.1], weight_storage="FP32 master",
                sha256=report.digest(directory / "matrix_norms.npy"), layout_sha256=report.digest(directory / "aus_trace.json"))
    (directory / "matrix_norms.json").write_text(json.dumps(meta))
    (directory / "band_trace_progress.json").write_text(json.dumps(dict(steps_written=3, completed=True)))
    return layout


def test_verified_trace_counts_all_matrices_and_both_norms(tmp_path):
    layout = fixture(tmp_path)
    _, _, receipt = report.verify_arrays(tmp_path, 3, layout, .9, 1.1)
    assert receipt["norm_entries"] == 1176 and receipt["aus_entries"] == 588
    assert receipt["max_relative_continuity_error"] == 0


@pytest.mark.parametrize("problem", ["hash", "unfinished", "layout", "bounds", "continuity"])
def test_invalid_trace_is_not_promoted_to_complete(tmp_path, problem):
    layout = fixture(tmp_path)
    if problem == "unfinished":
        (tmp_path / "band_trace_progress.json").write_text(json.dumps(dict(steps_written=2, completed=False)))
    elif problem == "layout":
        layout = list(reversed(layout))
    else:
        norms = np.load(tmp_path / "matrix_norms.npy")
        norms[0, 0, 1] = 3. if problem == "bounds" else 2.01
        np.save(tmp_path / "matrix_norms.npy", norms)
        if problem != "hash":
            f = tmp_path / "matrix_norms.json"
            meta = json.loads(f.read_text())
            meta["sha256"] = report.digest(tmp_path / "matrix_norms.npy")
            f.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        report.verify_arrays(tmp_path, 3, layout, .9, 1.1)
