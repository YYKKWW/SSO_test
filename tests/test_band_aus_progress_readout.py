"""The running-prefix report must not consume unwritten array tails."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location("progress", Path(__file__).parents[1]/"scripts/manifold/report_band_aus_progress_20261125.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def test_prefix_is_minimum_of_both_flush_markers():
    assert report.safe_prefix(dict(steps_written=300), dict(steps_written=200), 11445) == 200


@pytest.mark.parametrize("value", [0, -1, 20000, 2.5, True])
def test_invalid_flush_marker_rejected(value):
    with pytest.raises(ValueError):
        report.safe_prefix(dict(steps_written=value), dict(steps_written=200), 11445)


def arrays():
    target = np.full((10, 3), .01, dtype=np.float32)
    norms = np.full((10, 3, 2), 2., dtype=np.float32)
    replay = np.stack((target, target*1.1, target, target*0), axis=2)
    return target.copy(), norms, replay, target, np.full(3, 2.)


def test_valid_prefix_counts_matrix_decisions_and_norm_continuity():
    result = report.verify_prefix(*arrays(), .9, 1.1)
    assert result["matrix_decisions"] == 30 and result["max_abs_error"] == 0
    assert result["max_norm_continuity_error"] == 0


@pytest.mark.parametrize("fault", ["target", "error", "nan", "norm", "continuity", "dtype"])
def test_invalid_prefix_not_published_as_success(fault):
    aus, norms, replay, target, radii = arrays()
    if fault == "target":
        replay[0,0,0] += .001
    elif fault == "error":
        aus[0,0] += .001
    elif fault == "nan":
        aus[0,0] = np.nan
    elif fault == "norm":
        norms[0,0,0] = 3
    elif fault == "continuity":
        norms[0,0,1] += .01
    else:
        aus = aus.astype(np.float64)
    with pytest.raises(ValueError):
        report.verify_prefix(aus, norms, replay, target, radii, .9, 1.1)
