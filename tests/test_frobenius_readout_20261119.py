"""Regression checks for the published analysis, independent of CUDA."""

import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "readout", Path(__file__).parents[1] / "scripts/manifold/report_frobenius_20261119.py")
readout = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(readout)


def test_endpoint_and_window_can_disagree():
    effect = readout.projection_effect({1: 4, 2: 4}, {1: 4.01, 2: 3.999}, (1, 2))
    assert effect["endpoint_delta"] < 0
    assert effect["window_mean_delta"] > 0
    assert effect["projected_wins"] == 1


def pair():
    base = dict(step=3000, source="G", layer=0, component="attention.q", stalled=False,
                baseline_losses=[4.0, 4.5], target_chord=.01, effective_model_chord=.011)
    return [dict(base, preproject=False, candidate_losses=[3.99, 4.51]),
            dict(base, preproject=True, candidate_losses=[3.98, 4.52])]


def test_microbatches_are_paired_before_block_aggregation():
    result, = readout.audit_pairs(pair())
    assert result["batch_deltas"] == pytest.approx([-.01, .01])
    assert result["mean_delta"] == pytest.approx(0)


@pytest.mark.parametrize("problem", ["missing", "duplicate", "baseline", "target", "stall"])
def test_invalid_audits_fail_closed(problem):
    rows = pair()
    if problem == "missing":
        rows.pop()
    elif problem == "duplicate":
        rows.append(rows[0])
    elif problem == "baseline":
        rows[1]["baseline_losses"] = [4.01, 4.5]
    elif problem == "target":
        rows[1]["target_chord"] = .02
    else:
        rows[1]["stalled"] = True
    with pytest.raises(ValueError):
        readout.audit_pairs(rows)
