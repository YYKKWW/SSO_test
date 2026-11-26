"""Full-budget evidence must not be replaced by a prefix or endpoint alone."""

import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).parents[1] / "scripts/manifold"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("final_readout", SCRIPTS / "report_band_aus_final_20261126.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def test_completed_receipts_required():
    receipt = dict(exit_code=0, training_exit_code=0)
    progress = dict(completed=True, steps_written=report.STOP)
    report.require_complete("COMPLETED", receipt, progress, progress, report.STOP)


@pytest.mark.parametrize("fault", ["running", "exit", "training_exit", "unflushed", "incomplete"])
def test_unsuccessful_run_not_published_as_final(fault):
    state = "RUNNING" if fault == "running" else "COMPLETED"
    receipt = dict(exit_code=int(fault == "exit"), training_exit_code=int(fault == "training_exit"))
    band = dict(completed=fault != "incomplete", steps_written=report.STOP - int(fault == "unflushed"))
    replay = dict(completed=True, steps_written=report.STOP)
    with pytest.raises(ValueError):
        report.require_complete(state, receipt, band, replay, report.STOP)


def test_endpoint_win_does_not_imply_late_window_win():
    reference = {s: 3.5 for s in report.WINDOW}
    treatment = {s: 3.5002 for s in report.WINDOW}
    treatment[report.STOP] = 3.4999
    result = report.compare_curves(reference, treatment)
    assert result["endpoint_delta"] < 0 < result["late_mean_delta"]
    assert result["late_wins"] == 1 and result["late_points"] == 9


def test_partial_or_different_validation_grids_rejected():
    with pytest.raises(ValueError):
        report.compare_curves({200: 4.}, {200: 3.9})
    with pytest.raises(ValueError):
        report.compare_curves({s: 3.5 for s in report.WINDOW}, {report.STOP: 3.4})
