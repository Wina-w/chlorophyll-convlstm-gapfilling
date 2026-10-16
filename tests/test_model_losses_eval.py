from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch

from chl_gapfill.config import ExperimentConfig, SplitConfig
from chl_gapfill.evaluate import (
    bootstrap_ci,
    compare_same_pixels,
    interp_fill,
    overlaps,
    paired_rmse_diff,
    resample_days,
    skill,
)
from chl_gapfill.losses import GapFillLoss, gradient_loss, masked_mse
from chl_gapfill.model import ConvLSTM, PartialConvUNet, build_model
from chl_gapfill.train import assert_no_overlap, check_split_windows, load_weights, split_positions


# --- models -------------------------------------------------------------------------
def test_convlstm_sequence_and_legacy_forward_and_grads():
    torch.manual_seed(0)
    m = ConvLSTM(
        input_channels=4, hidden_channels=[8, 4], kernel_size=3, image_size=(8, 8), step=3, use_norm=True
    )
    y = m(torch.randn(2, 3, 4, 8, 8))
    assert y.shape == (2, 1, 8, 8)
    y.sum().backward()
    assert all(p.grad is not None for p in m.parameters())
    assert m(torch.randn(2, 4, 8, 8)).shape == (2, 1, 8, 8)  # legacy re-fed mode
    n_params = sum(p.numel() for p in m.parameters())
    assert 0 < n_params < 50_000


def test_pconv_unet_forward_and_factory():
    m = PartialConvUNet(input_channels=6, base=16, depth=2)
    out = m(torch.randn(2, 6, 8, 8), torch.rand(2, 1, 8, 8).round())
    assert out.shape == (2, 1, 8, 8)
    assert isinstance(build_model("pconv_unet", 6, [64, 32], 3, (8, 8)), PartialConvUNet)
    assert isinstance(build_model("convlstm", 4, [8, 4], 3, (8, 8)), ConvLSTM)
    with pytest.raises(ValueError):
        build_model("nope", 4, [8], 3, (8, 8))


def test_published_convlstm_size_is_about_387k():
    m = build_model("convlstm", 6, [64, 32], 3, (20, 20), use_norm=True)
    n = sum(p.numel() for p in m.parameters())
    assert 350_000 < n < 420_000, n


class _Payload:  # any non-tensor object: pickled by reference, so loading it could run code
    pass


def test_checkpoint_loading_is_strict_and_weights_only(tmp_path, capsys):
    cpu = torch.device("cpu")
    trained = build_model("convlstm", 4, [8, 4], 3, (8, 8), use_norm=False)
    torch.save(trained.state_dict(), tmp_path / "bare.pth")
    torch.save({"model": trained.state_dict(), "epoch": 3}, tmp_path / "full.pt")
    for name in ("bare.pth", "full.pt"):  # both formats load into the matching model
        m = load_weights(build_model("convlstm", 4, [8, 4], 3, (8, 8), use_norm=False), tmp_path / name, cpu)
        assert all(torch.equal(m.state_dict()[k], v) for k, v in trained.state_dict().items())
    # use_norm=True has GroupNorm weights the file lacks: scoring this model would score random weights
    with pytest.raises(ValueError, match="does not match"):
        load_weights(build_model("convlstm", 4, [8, 4], 3, (8, 8), use_norm=True), tmp_path / "bare.pth", cpu)
    with pytest.raises(ValueError, match="shape mismatch"):
        load_weights(
            build_model("convlstm", 6, [8, 4], 3, (8, 8), use_norm=False), tmp_path / "bare.pth", cpu
        )
    load_weights(
        build_model("convlstm", 4, [8, 4], 3, (8, 8), use_norm=True), tmp_path / "bare.pth", cpu, strict=False
    )
    assert "partial load" in capsys.readouterr().out
    torch.save({"model": trained.state_dict(), "extra": _Payload()}, tmp_path / "untrusted.pt")
    with pytest.raises(pickle.UnpicklingError):
        load_weights(build_model("convlstm", 4, [8, 4], 3, (8, 8)), tmp_path / "untrusted.pt", cpu)


# --- losses -------------------------------------------------------------------------
def test_masked_losses_score_only_hole_pixels():
    pred = torch.zeros(1, 1, 4, 4)
    target = torch.ones(1, 1, 4, 4)
    hole = torch.zeros(1, 1, 4, 4)
    hole[..., 0, 0] = 1
    assert masked_mse(pred, target, hole).item() == pytest.approx(1.0)
    pred2 = pred.clone()
    pred2[..., 3, 3] = 100.0  # error outside the hole must not count
    assert masked_mse(pred2, target, hole).item() == pytest.approx(1.0)
    assert gradient_loss(target, target).item() == 0.0
    loss = GapFillLoss(w_hole=1.0, w_valid=0.0, w_grad=0.0)
    assert loss(pred, target, hole).item() == pytest.approx(1.0)


# --- splits -------------------------------------------------------------------------
def _dates(n=120):
    base = np.datetime64("2022-01-01")
    out = []
    for i in range(n):
        d = (base + np.timedelta64(i, "D")).astype(object)
        out.append((d.year, d.month, d.day))
    return np.array(out)


def test_split_positions_inclusive_bounds_and_gap():
    split = SplitConfig(
        train_end="2022-03-21",
        val_start="2022-03-22",
        val_end="2022-04-10",
        test_start="2022-04-11",
        test_end="2022-04-30",
        gap_days=0,
    )
    tr, va, te = split_positions(_dates(), split, 3)
    assert len(tr) == 80 and len(va) == 20 and len(te) == 20
    split2 = split.model_copy(update={"gap_days": 2})
    tr2, va2, te2 = split_positions(_dates(), split2, 3)
    assert len(tr2) == 80 and len(va2) == 18 and len(te2) == 18
    assert_no_overlap(_dates(), tr2, va2, input_days=3)
    with pytest.raises(ValueError, match="overlap"):
        assert_no_overlap(_dates(), tr, va, input_days=3)


def test_split_windows_are_checked_on_both_boundaries(capsys):
    split = SplitConfig(
        train_end="2022-03-21",
        val_start="2022-03-22",
        val_end="2022-04-10",
        test_start="2022-04-11",
        test_end="2022-04-30",
        gap_days=0,
    )
    published = ExperimentConfig(split=split)
    tr, va, te = split_positions(_dates(), split, 3)
    check_split_windows(_dates(), tr, va, te, published)  # gap_days=0 as published: reported, not raised
    out = capsys.readouterr().out
    assert "train->val" in out and "val->test" in out
    strict = ExperimentConfig(split=split.model_copy(update={"gap_days": 2}))
    check_split_windows(_dates(), *split_positions(_dates(), strict.split, 3), strict)  # the gap fixes both
    assert "WARN" not in capsys.readouterr().out
    with pytest.raises(ValueError, match="overlap"):
        check_split_windows(_dates(), tr, va, te, strict)


# --- evaluation ---------------------------------------------------------------------
def test_skill_bootstrap_overlaps_and_interp():
    assert skill([1, 2], [1, 2])["rmse"] == 0 and skill([np.nan], [1])["n"] == 0
    rng = np.random.default_rng(0)
    truth = [rng.normal(size=50) for _ in range(20)]
    pred = [t + rng.normal(0, 0.2, 50) for t in truth]
    ci = bootstrap_ci(pred, truth, n_boot=100, seed=0)
    assert ci["rmse"]["lo"] <= ci["rmse"]["point"] <= ci["rmse"]["hi"] and not ci["degenerate"]
    assert bootstrap_ci(pred[:1], truth[:1])["degenerate"]
    assert overlaps({"lo": 0, "hi": 1}, {"lo": 0.5, "hi": 2}) and not overlaps(
        {"lo": 0, "hi": 1}, {"lo": 1.5, "hi": 2}
    )
    field = np.arange(16, dtype=float).reshape(4, 4)
    hole = np.zeros((4, 4), bool)
    hole[1, 1] = True
    filled = interp_fill(field, ~hole, hole)
    assert filled.shape == (1,) and abs(filled[0] - 5.0) < 1e-6  # linear interpolation of a plane is exact


def test_day_bootstrap_is_unchanged_and_blocks_are_consecutive():
    rng = np.random.default_rng(0)
    truth = [rng.normal(size=30) for _ in range(25)]
    pred = [t + rng.normal(0, 0.3, 30) for t in truth]
    # block_days=1 draws what the original day bootstrap drew, so published CIs reproduce
    ref, draw = [], np.random.default_rng(7)
    for _ in range(200):
        idx = draw.integers(0, 25, 25)
        ref.append(
            skill(np.concatenate([pred[i] for i in idx]), np.concatenate([truth[i] for i in idx]))["rmse"]
        )
    ci = bootstrap_ci(pred, truth, n_boot=200, seed=7)
    assert (ci["rmse"]["lo"], ci["rmse"]["hi"]) == tuple(np.percentile(ref, [2.5, 97.5]))
    idx = resample_days(np.random.default_rng(1), 25, 5)
    assert len(idx) == 25 and all(np.diff(b).tolist() == [1] * 4 for b in idx.reshape(5, 5))
    assert bootstrap_ci(pred, truth, n_boot=200, seed=7, block_days=5)["rmse"] != ci["rmse"]


def test_paired_difference_resolves_what_overlapping_cis_cannot():
    rng = np.random.default_rng(3)
    hard = rng.uniform(0.2, 2.0, 30)  # day difficulty varies a lot and is shared by both methods
    truth = [np.zeros(40) for _ in hard]
    a = [rng.normal(0, h, 40) for h in hard]
    b = [x * 1.15 for x in a]  # b is 15% worse on every pixel of every day
    ca, cb = bootstrap_ci(a, truth, n_boot=300, seed=0), bootstrap_ci(b, truth, n_boot=300, seed=0)
    assert overlaps(ca["rmse"], cb["rmse"])  # the marginal CIs overlap...
    d = paired_rmse_diff(a, b, truth, n_boot=300, seed=0)
    assert d["point"] < 0 and d["hi"] < 0  # ...but the paired difference is clearly negative
    same = paired_rmse_diff(a, a, truth, n_boot=50, seed=0)
    assert same["point"] == same["lo"] == same["hi"] == 0.0
    assert paired_rmse_diff(a[:1], b[:1], truth[:1])["degenerate"]
    with pytest.raises(ValueError):
        paired_rmse_diff(a, b[:-1], truth)


def test_compare_same_pixels_scores_every_method_on_one_pixel_set():
    rng = np.random.default_rng(5)
    truth = [rng.normal(size=20) for _ in range(12)]
    good = [t + rng.normal(0, 0.1, 20) for t in truth]
    bad = [t + rng.normal(0, 0.5, 20) for t in truth]
    # a baseline that only fills half the pixels, and nothing at all on day 0
    partial = [np.where(np.arange(20) % 2 == 0, t, np.nan) for t in bad]
    partial[0][:] = np.nan
    r = compare_same_pixels({"model": good, "persistence": partial}, truth, "persistence", n_boot=200, seed=0)
    assert r["n"] == 11 * 10 and r["n_days"] == 11  # the empty day is dropped, not resampled
    keep = [np.isfinite(p) for p in partial[1:]]
    sub = lambda per_day: [x[k] for x, k in zip(per_day[1:], keep, strict=True)]  # noqa: E731
    assert (
        r["methods"]["model"]["rmse"]["point"]
        == skill(np.concatenate(sub(good)), np.concatenate(sub(truth)))["rmse"]
    )
    d = r["diff"]["model_minus_persistence"]
    assert d == {
        **paired_rmse_diff(sub(good), sub(partial), sub(truth), n_boot=200, seed=0),
        "resolved": True,
    }
    assert d["hi"] < 0 and "persistence_minus_persistence" not in r["diff"]
    assert compare_same_pixels({"a": [np.full(3, np.nan)], "b": [np.zeros(3)]}, [np.zeros(3)], "b") == {
        "n": 0,
        "n_days": 0,
    }
