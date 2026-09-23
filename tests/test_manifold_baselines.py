"""Independent equation, state, and launch protocol checks."""

import copy

import pytest
import torch
from emerging_optimizers.orthogonalized_optimizers.manifold_baselines import (
    ManifoldBaseline, baseline_update, imuon_direction)
from emerging_optimizers.orthogonalized_optimizers.manifold_geometry import (
    constraint_defect, initialize_matrix, partial_polar)


def test_muonh_matches_hyperball_equation():
    torch.manual_seed(6)
    w, values = initialize_matrix(torch.randn(5, 3), "frobenius")
    m = torch.randn_like(w)
    u = partial_polar(m)
    trial = w - .02 * values.radius * u / torch.linalg.vector_norm(u)
    expected = values.radius * trial / torch.linalg.vector_norm(trial)
    actual = baseline_update(w.clone(), m, values, baseline="muonh", lr=.02, lmo_mode="exact")
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("shape", [(7, 3), (3, 7), (4, 4)])
def test_imuon_is_tangent_and_structural_zero_is_not_amplified(shape):
    torch.manual_seed(6)
    w, values = initialize_matrix(torch.randn(shape), "stiefel")
    m = torch.randn_like(w)
    d = imuon_direction(w, m, values.radius, "exact")
    q, v = (w.T, d.T) if shape[0] < shape[1] else (w, d)
    torch.testing.assert_close(q.T @ v + v.T @ q, torch.zeros(q.shape[1], q.shape[1]), atol=2e-5, rtol=0)
    assert torch.linalg.vector_norm(imuon_direction(w, w, values.radius, "exact")) == 0
    assert torch.count_nonzero(imuon_direction(w, torch.zeros_like(w), values.radius, "ns")) == 0


@pytest.mark.parametrize("baseline,geometry", [("muonh", "frobenius"), ("imuon", "stiefel")])
def test_baseline_state_resume(baseline, geometry):
    torch.manual_seed(18)
    w, values = initialize_matrix(torch.randn(5, 3), geometry)
    p = torch.nn.Parameter(w)
    spec = [dict(label="test", rows=tuple(range(5)), radius=values.radius, scale=values.scale)]
    p.manifold_spec = spec
    kwargs = dict(lr=.01, geometry=geometry, baseline=baseline, use_nesterov=False, lmo_mode="exact", stiefel_return_mode="exact")
    opt = ManifoldBaseline([p], **kwargs)
    p.grad = torch.randn_like(p)
    opt.step()
    saved = copy.deepcopy(opt.state_dict())
    restored = torch.nn.Parameter(p.detach().clone())
    restored.manifold_spec = spec
    other = ManifoldBaseline([restored], **kwargs)
    other.load_state_dict(saved)
    g = torch.randn_like(p)
    p.grad, restored.grad = g.clone(), g.clone()
    opt.step()
    other.step()
    torch.testing.assert_close(p, restored, atol=0, rtol=0)
    assert constraint_defect(p, values) < 1e-5


def test_reject_wrong_geometry():
    with pytest.raises(ValueError, match="geometry"):
        ManifoldBaseline([torch.nn.Parameter(torch.eye(2))], baseline="muonh", geometry="spectral", lr=.01)


@pytest.mark.parametrize("baseline", ["sso", "muonsphere"])
def test_spectral_adapter_matches_existing_kernel(monkeypatch, baseline):
    from emerging_optimizers.orthogonalized_optimizers import muon_ball
    from emerging_optimizers.orthogonalized_optimizers import \
        spectral_ball_utils as kernels

    # Test the same PI arithmetic without paying CPU torch.compile startup cost.
    pi = getattr(kernels.power_iteration, "_torchdynamo_orig_callable", kernels.power_iteration)
    monkeypatch.setattr(kernels, "power_iteration", pi)
    monkeypatch.setattr(muon_ball, "power_iteration", pi)
    torch.manual_seed(8)
    w, values = initialize_matrix(torch.randn(5, 3), "spectral")
    m = torch.randn_like(w)
    work = w.clone()
    if baseline == "sso":
        direction, _, _ = kernels.compute_spectral_ball_update(work, m, values.radius, 10, 8, "bisection", 2e-4, 20)
    else:
        direction, _, _ = muon_ball.compute_muon_ball_update(work, m, values.radius, 10, 8)
    expected = work - .02 * values.scale * direction.float()
    actual = baseline_update(w.clone(), m, values, baseline=baseline, lr=.02)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
