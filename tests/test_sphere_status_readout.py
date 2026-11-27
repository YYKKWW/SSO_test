import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location("status", Path(__file__).parents[1]/"scripts/manifold/report_sphere_status_20261127.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def row():
    return dict(state="COMPLETED", scheduled_steps=11445, gate=False, trace_steps=11445,
                completion=dict(exit_code=0, training_exit_code=0), validation={"11445": 3.46},
                early_exit=None, nan_iterations=0, skipped_iterations=0, replay_max_abs_error=1e-8)


def test_complete_endpoint():
    assert report.full_endpoint(row())


@pytest.mark.parametrize("fault", ["running", "gate", "partial", "failure", "missing_receipt", "missing_validation",
                                  "nan", "skipped", "early_exit", "bad_replay"])
def test_gate_or_unverified_value_is_not_a_full_endpoint(fault):
    item = row()
    if fault == "running": item["state"] = "RUNNING"
    if fault == "gate": item["gate"] = True
    if fault == "partial": item["trace_steps"] = 6
    if fault == "failure": item["completion"]["training_exit_code"] = 1
    if fault == "missing_receipt": item["completion"] = None
    if fault == "missing_validation": item["validation"] = {"3000": 3.77}
    if fault == "nan": item["nan_iterations"] = 1
    if fault == "skipped": item["skipped_iterations"] = 1
    if fault == "early_exit": item["early_exit"] = 3000
    if fault == "bad_replay": item["replay_max_abs_error"] = .01
    assert not report.full_endpoint(item)
