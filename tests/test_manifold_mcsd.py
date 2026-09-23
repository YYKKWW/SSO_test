"""Standalone CPU reference checks for three-geometry MCSD updates."""

import copy

import pytest
import torch

from emerging_optimizers.orthogonalized_optimizers.manifold_geometry import (
    GeometryFailure,
    constraint_defect,
    full_polar_msign,
    initialize_matrix,
    partial_polar,
    return_to_manifold,
    tangent_project,
)
from emerging_optimizers.orthogonalized_optimizers.manifold_layout import (
    components_for_parameter,
    validate_layout,
)
from emerging_optimizers.orthogonalized_optimizers.manifold_mcsd import (
    ManifoldMCSD,
    prepare_manifold_model,
)
from emerging_optimizers.orthogonalized_optimizers.manifold_polar import polar_express_msign


@pytest.mark.parametrize("shape", [(5, 3), (3, 5), (4, 4)])
@pytest.mark.parametrize("geometry", ["frobenius", "spectral", "stiefel"])
def test_tangent_projection_is_orthogonal_projector(shape, geometry):
    torch.manual_seed(21)
    raw = torch.randn(shape, dtype=torch.float64)
    w, values = initialize_matrix(raw, geometry)
    x, y = torch.randn_like(w), torch.randn_like(w)
    px = tangent_project(w, x, values)
    py = tangent_project(w, y, values)
    assert torch.allclose(tangent_project(w, px, values), px, atol=1e-10)
    assert torch.allclose(torch.sum(px * y), torch.sum(x * py), atol=1e-10)
    if geometry == "frobenius":
        assert abs(torch.sum(px * w).item()) < 1e-10
    elif geometry == "stiefel":
        tall_w, tall_px = (w.T, px.T) if w.shape[0] < w.shape[1] else (w, px)
        sym = tall_w.T @ tall_px + tall_px.T @ tall_w
        assert torch.linalg.vector_norm(sym).item() < 1e-10
    else:
        u, _, vh = torch.linalg.svd(w, full_matrices=False)
        assert abs(torch.sum(px * torch.outer(u[:, 0], vh[0])).item()) < 1e-10


@pytest.mark.parametrize("shape", [(5, 3), (3, 5), (4, 4)])
@pytest.mark.parametrize("geometry", ["frobenius", "spectral", "stiefel"])
@pytest.mark.parametrize("method", ["mcsd", "mcsd_tp"])
def test_return_feasible(shape, geometry, method):
    torch.manual_seed(22)
    raw = torch.randn(shape, dtype=torch.float64)
    w, values = initialize_matrix(raw, geometry)
    step = torch.randn_like(w)
    if method == "mcsd_tp":
        step = tangent_project(w, step, values)
    mode = "radial" if geometry == "spectral" and method == "mcsd_tp" else "projection"
    updated, _ = return_to_manifold(w + 0.01 * step, values, mode=mode)
    assert constraint_defect(updated, values) < 1e-10


def test_exact_partial_and_full_msign_are_distinct():
    x = torch.diag(torch.tensor([2.0, 0.0], dtype=torch.float64))
    assert torch.equal(partial_polar(x), torch.diag(torch.tensor([1.0, 0.0], dtype=x.dtype)))
    with pytest.raises(GeometryFailure, match="rank_deficient_trial"):
        full_polar_msign(x)
    y = torch.tensor([[2.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=x.dtype)
    p = full_polar_msign(y)
    assert torch.allclose(p.T @ p, torch.eye(2, dtype=x.dtype), atol=1e-12)


def test_spectral_projection_is_not_radial_and_tie_does_not_stop():
    w = torch.diag(torch.tensor([1.0, 0.5], dtype=torch.float64))
    _, values = initialize_matrix(w, "spectral")
    trial = torch.diag(torch.tensor([2.0, 1.0], dtype=torch.float64))
    projected, info = return_to_manifold(trial, values, mode="projection")
    radial, _ = return_to_manifold(trial, values, mode="radial")
    assert torch.allclose(projected, torch.eye(2, dtype=w.dtype), atol=1e-12)
    assert torch.allclose(radial, w, atol=1e-12)
    assert info.top_gap_below_warning
    # A chosen SVD normal is only an empirical continuation at this tie.
    assert torch.isfinite(tangent_project(projected, w, values)).all()


def test_hypersphere_zero_trial_fails():
    w = torch.eye(2, dtype=torch.float64)
    _, values = initialize_matrix(w, "frobenius")
    with pytest.raises(GeometryFailure, match="zero_trial"):
        return_to_manifold(torch.zeros_like(w), values, mode="projection")


def test_gqa_and_gate_up_round_trip():
    kwargs = dict(
        hidden_size=384,
        ffn_hidden_size=1152,
        num_attention_heads=6,
        num_query_groups=3,
        kv_channels=64,
    )
    for name, shape in (
        ("decoder.layers.0.self_attention.linear_qkv.weight", (768, 384)),
        ("decoder.layers.0.mlp.linear_fc1.weight", (2304, 384)),
    ):
        components = components_for_parameter(name, shape, **kwargs)
        validate_layout(components, shape[0])
        physical = torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(shape)
        rebuilt = torch.zeros_like(physical)
        for component in components:
            component.write(rebuilt, component.read(physical))
        assert torch.equal(rebuilt, physical)
    qkv = components_for_parameter(
        "linear_qkv.weight", (768, 384), **kwargs
    )
    assert [len(item.rows) for item in qkv] == [384, 192, 192]
    assert qkv[0].rows[128] == 256
    assert qkv[1].rows[64] == 384


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_proj = torch.nn.Linear(4, 4, bias=False)


@pytest.mark.parametrize("geometry", ["frobenius", "spectral", "stiefel"])
@pytest.mark.parametrize("method", ["mcsd", "mcsd_tp"])
def test_optimizer_two_steps_and_resume(geometry, method):
    torch.manual_seed(23)
    module = _Tiny().float()
    manifest = prepare_manifold_model(
        module,
        geometry=geometry,
        hidden_size=4,
        ffn_hidden_size=12,
        num_attention_heads=1,
        num_query_groups=1,
        kv_channels=4,
        initialize=True,
    )
    assert len(manifest) == 1
    param = module.linear_proj.weight
    reference = dict(lmo_mode="exact", stiefel_return_mode="exact", spectral_solver="exact")
    optimizer = ManifoldMCSD([param], lr=0.02, geometry=geometry, method=method, **reference)
    param.grad = torch.randn_like(param)
    optimizer.step()
    state_before = copy.deepcopy(optimizer.state_dict())
    after_first = param.detach().clone()
    gradient = torch.randn_like(param)
    param.grad = gradient.clone()
    optimizer.step()
    expected = param.detach().clone()

    resumed = torch.nn.Parameter(after_first.clone())
    resumed.manifold_spec = param.manifold_spec
    resumed_optimizer = ManifoldMCSD(
        [resumed], lr=0.02, geometry=geometry, method=method, **reference
    )
    resumed_optimizer.load_state_dict(state_before)
    resumed.grad = gradient.clone()
    resumed_optimizer.step()
    assert torch.allclose(resumed, expected, atol=2e-6)
    values = optimizer.state[param]["components"][0]
    from emerging_optimizers.orthogonalized_optimizers.manifold_geometry import GeometryParameters

    params = GeometryParameters(geometry, values["radius"], values["scale"])
    assert constraint_defect(param, params) < 2e-6
    assert optimizer.last_step_stats["components"] == 1


def test_prepared_fp32_initialization_survives_bf16_model_conversion():
    torch.manual_seed(24)
    model = _Tiny()
    prepare_manifold_model(
        model,
        geometry="stiefel",
        hidden_size=4,
        ffn_hidden_size=12,
        num_attention_heads=1,
        num_query_groups=1,
        kv_channels=4,
        initialize=True,
    )
    fp32 = model.linear_proj.weight.manifold_initial_fp32.clone()
    model.bfloat16()
    assert model.linear_proj.weight.manifold_initial_fp32.dtype == torch.float32
    assert torch.equal(fp32, model.linear_proj.weight.manifold_initial_fp32)


@pytest.mark.parametrize("shape", [(16, 8), (8, 16), (16, 16)])
def test_practical_full_msign_is_nearly_orthogonal(shape):
    torch.manual_seed(25)
    x = torch.randn(shape, dtype=torch.float32)
    y = polar_express_msign(x, steps=10, polish_steps=3)
    tall = y.T if shape[0] < shape[1] else y
    gram = tall.T @ tall
    assert torch.linalg.vector_norm(gram - torch.eye(gram.shape[0])).item() < 1e-4


@pytest.mark.parametrize("geometry", ["frobenius", "spectral", "stiefel"])
def test_practical_step_does_not_call_svd(monkeypatch, geometry):
    torch.manual_seed(26)
    module = _Tiny().float()
    prepare_manifold_model(
        module,
        geometry=geometry,
        hidden_size=4,
        ffn_hidden_size=12,
        num_attention_heads=1,
        num_query_groups=1,
        kv_channels=4,
        initialize=True,
        polar_mode="ns",
    )
    param = module.linear_proj.weight
    optimizer = ManifoldMCSD([param], lr=0.01, geometry=geometry, method="mcsd_tp")
    param.grad = torch.randn_like(param)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("full SVD entered the practical training step")

    monkeypatch.setattr(torch.linalg, "svd", forbidden)
    monkeypatch.setattr(torch.linalg, "svdvals", forbidden)
    optimizer.step()
    assert torch.isfinite(param).all()
