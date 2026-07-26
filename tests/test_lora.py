"""Tests for Module 13B — LoRA (g2c/lora/).

Suggested order to implement & turn green:

1. `LoRALinear.forward` (g2c/lora/layer.py)
   -> test_forward_is_exact_noop_at_init
   -> test_forward_matches_manual_delta
   -> test_forward_handles_batched_sequences
   -> test_gradient_asymmetry_at_init
   -> test_gradient_flows_to_A_once_B_moves

2. `LoRALinear.merge` (g2c/lora/layer.py)
   -> test_merge_preserves_the_function
   -> test_merge_is_idempotent

3. `LoRALinear.unmerge` (g2c/lora/layer.py)
   -> test_unmerge_restores_the_base_weight

4. `mark_only_lora_trainable` (g2c/lora/inject.py)
   -> test_mark_only_lora_trainable_freezes_exactly_the_right_set
   -> test_frozen_base_is_bit_identical_after_training
   -> test_adapter_state_dict_round_trip_transfers_behavior

Construction, injection, counting, and state-dict plumbing are
implemented boilerplate, so those tests pass from the start and sanity-
check this file itself.

Everything here runs on tiny synthetic torch models — no BaseLM
download, no MPS requirement.
"""
from __future__ import annotations

import copy

import pytest
import torch

from g2c.lora import (
    LoRALinear,
    LoRAModel,
    count_parameters,
    inject_lora,
    load_lora_state_dict,
    lora_state_dict,
    mark_only_lora_trainable,
)

# GQA-style dimensions on purpose: q_proj is square, v_proj is not.
# A transposed merge that "works" on square layers fails loudly here.
D_MODEL = 8
D_KV = 4


class TinyBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(D_MODEL, D_MODEL)
        self.v_proj = torch.nn.Linear(D_MODEL, D_KV)
        self.out = torch.nn.Linear(D_KV, D_MODEL)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(torch.tanh(self.v_proj(torch.tanh(self.q_proj(x)))))


class TinyNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([TinyBlock(), TinyBlock()])
        self.head = torch.nn.Linear(D_MODEL, 12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = x + block(x)
        return self.head(x)


def make_net(seed: int = 0) -> TinyNet:
    torch.manual_seed(seed)
    return TinyNet()


def make_input(seed: int = 1, *, shape: tuple[int, ...] = (5, D_MODEL)) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=generator)


def randomize_adapter(lora: LoRALinear, seed: int = 2) -> None:
    """Give A and B non-trivial values so the delta path actually matters."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        lora.lora_A.copy_(
            torch.randn(*lora.lora_A.shape, generator=generator) * 0.3
        )
        lora.lora_B.copy_(
            torch.randn(*lora.lora_B.shape, generator=generator) * 0.3
        )


# ----------------------------------------------------------------------
# Boilerplate: construction and initialization (pass from the start)
# ----------------------------------------------------------------------


def test_lora_linear_shapes_and_init():
    base = torch.nn.Linear(D_MODEL, D_KV)
    lora = LoRALinear(base, rank=3)
    assert lora.lora_A.shape == (D_MODEL, 3)
    assert lora.lora_B.shape == (3, D_KV)
    # B starts at exactly zero; A starts random (nonzero).
    assert torch.equal(lora.lora_B, torch.zeros(3, D_KV))
    assert lora.lora_A.abs().sum() > 0
    assert lora.merged is False
    assert lora.base is base  # wrapped, not copied


def test_lora_linear_scaling_defaults_and_alpha():
    base = torch.nn.Linear(D_MODEL, D_MODEL)
    assert LoRALinear(base, rank=8).scaling == pytest.approx(1.0)
    assert LoRALinear(base, rank=8, alpha=16).scaling == pytest.approx(2.0)


def test_lora_linear_rejects_bad_arguments():
    with pytest.raises(TypeError):
        LoRALinear(torch.nn.ReLU(), rank=4)
    with pytest.raises(ValueError):
        LoRALinear(torch.nn.Linear(4, 4), rank=0)


# ----------------------------------------------------------------------
# Boilerplate: injection and counting (pass from the start)
# ----------------------------------------------------------------------


def test_inject_lora_replaces_exactly_the_named_targets():
    net = make_net()
    original_weights = {
        name: module.weight
        for name, module in net.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    replaced = inject_lora(net, {"q_proj", "v_proj"}, rank=4)
    assert sorted(replaced) == [
        "blocks.0.q_proj",
        "blocks.0.v_proj",
        "blocks.1.q_proj",
        "blocks.1.v_proj",
    ]
    for block in net.blocks:
        assert isinstance(block.q_proj, LoRALinear)
        assert isinstance(block.v_proj, LoRALinear)
        assert isinstance(block.out, torch.nn.Linear)  # untouched
        # The wrapped layers are the SAME objects, not re-initialized copies.
        assert block.q_proj.base.weight is not None
    assert isinstance(net.head, torch.nn.Linear)
    assert net.blocks[0].q_proj.base.weight is original_weights["blocks.0.q_proj"]


def test_inject_lora_raises_on_typo():
    net = make_net()
    with pytest.raises(ValueError, match="matched no"):
        inject_lora(net, {"qproj"}, rank=4)


def test_inject_lora_refuses_silent_double_injection():
    net = make_net()
    inject_lora(net, {"q_proj"}, rank=4)
    # Already-wrapped layers are not eligible targets, so a second pass
    # matches nothing and fails loudly instead of double-wrapping.
    with pytest.raises(ValueError, match="matched no"):
        inject_lora(net, {"q_proj"}, rank=4)


def test_count_parameters_tracks_requires_grad():
    net = make_net()
    trainable, total = count_parameters(net)
    assert trainable == total > 0
    net.head.weight.requires_grad_(False)
    trainable_after, total_after = count_parameters(net)
    assert total_after == total
    assert trainable_after == total - net.head.weight.numel()


# ----------------------------------------------------------------------
# Boilerplate: adapter state dict and the trainer view (pass from start)
# ----------------------------------------------------------------------


def test_lora_state_dict_contains_only_adapter_tensors():
    net = make_net()
    inject_lora(net, {"q_proj", "v_proj"}, rank=4)
    state = lora_state_dict(net)
    assert len(state) == 8  # 4 injected layers x (A, B)
    for name, tensor in state.items():
        assert name.endswith("lora_A") or name.endswith("lora_B")
        assert tensor.device.type == "cpu"
        assert not tensor.requires_grad


def test_load_lora_state_dict_rejects_mismatched_adapters():
    donor = make_net()
    inject_lora(donor, {"q_proj", "v_proj"}, rank=2)
    state = lora_state_dict(donor)

    wrong_rank = make_net()
    inject_lora(wrong_rank, {"q_proj", "v_proj"}, rank=4)
    with pytest.raises(ValueError):
        load_lora_state_dict(wrong_rank, state)

    wrong_targets = make_net()
    inject_lora(wrong_targets, {"q_proj"}, rank=2)
    with pytest.raises(ValueError, match="mismatch"):
        load_lora_state_dict(wrong_targets, state)


def test_lora_model_view_exposes_only_trainable_parameters():
    net = make_net()
    for p in net.parameters():
        p.requires_grad_(False)
    net.head.bias.requires_grad_(True)
    view = LoRAModel(net)
    assert view.parameters() == [net.head.bias]
    # Attribute access and forward fall through to the wrapped model.
    x = make_input()
    assert torch.equal(view(x), net(x))
    assert view.to("cpu") is view


# ----------------------------------------------------------------------
# Step 1: LoRALinear.forward
# ----------------------------------------------------------------------


def test_forward_is_exact_noop_at_init():
    net = make_net()
    pristine = copy.deepcopy(net)
    inject_lora(net, {"q_proj", "v_proj"}, rank=4)
    x = make_input()
    # B is zero, so the delta is exactly zero: not "close", identical.
    assert torch.equal(net(x), pristine(x))


def test_forward_matches_manual_delta():
    torch.manual_seed(3)
    base = torch.nn.Linear(D_MODEL, D_KV)
    lora = LoRALinear(base, rank=3, alpha=6)  # scaling = 2.0
    randomize_adapter(lora)
    x = make_input()
    expected = base(x) + (x @ lora.lora_A @ lora.lora_B) * 2.0
    assert torch.allclose(lora(x), expected, atol=1e-6)


def test_forward_handles_batched_sequences():
    torch.manual_seed(4)
    base = torch.nn.Linear(D_MODEL, D_KV)
    lora = LoRALinear(base, rank=2)
    randomize_adapter(lora)
    x = make_input(shape=(2, 5, D_MODEL))  # (B, T, d) like a real projection
    out = lora(x)
    assert out.shape == (2, 5, D_KV)
    assert torch.allclose(
        out[1, 3], lora(x[1, 3:4])[0], atol=1e-6
    )  # position-wise, no cross-talk


def test_gradient_asymmetry_at_init():
    torch.manual_seed(5)
    lora = LoRALinear(torch.nn.Linear(D_MODEL, D_KV), rank=3)
    lora(make_input()).sum().backward()
    # dL/dA flows THROUGH B. B is zero at init, so A's gradient is exactly
    # zero while B's is not — the reason exactly one matrix starts at zero.
    assert torch.equal(lora.lora_A.grad, torch.zeros_like(lora.lora_A))
    assert lora.lora_B.grad.abs().sum() > 0


def test_gradient_flows_to_A_once_B_moves():
    torch.manual_seed(6)
    lora = LoRALinear(torch.nn.Linear(D_MODEL, D_KV), rank=3)
    randomize_adapter(lora)
    lora(make_input()).sum().backward()
    assert lora.lora_A.grad.abs().sum() > 0


# ----------------------------------------------------------------------
# Steps 2-3: merge / unmerge
# ----------------------------------------------------------------------


def test_merge_preserves_the_function():
    torch.manual_seed(7)
    lora = LoRALinear(torch.nn.Linear(D_MODEL, D_KV), rank=3)
    randomize_adapter(lora)
    x = make_input()
    before = lora(x)
    weight_before = lora.base.weight.detach().clone()
    lora.merge()
    assert lora.merged is True
    assert not torch.equal(lora.base.weight, weight_before)  # folded in
    assert torch.allclose(lora(x), before, atol=1e-5)


def test_merge_is_idempotent():
    torch.manual_seed(8)
    lora = LoRALinear(torch.nn.Linear(D_MODEL, D_KV), rank=3)
    randomize_adapter(lora)
    lora.merge()
    weight_once = lora.base.weight.detach().clone()
    lora.merge()  # must be a no-op, not a double-add
    assert torch.equal(lora.base.weight, weight_once)


def test_unmerge_restores_the_base_weight():
    torch.manual_seed(9)
    lora = LoRALinear(torch.nn.Linear(D_MODEL, D_KV), rank=3)
    randomize_adapter(lora)
    weight_original = lora.base.weight.detach().clone()
    x = make_input()
    before = lora(x)
    lora.merge()
    lora.unmerge()
    assert lora.merged is False
    # Float add-then-subtract is not bitwise exact — but it is very close.
    assert torch.allclose(lora.base.weight, weight_original, atol=1e-6)
    assert torch.allclose(lora(x), before, atol=1e-5)
    lora.unmerge()  # no-op on an unmerged layer
    assert lora.merged is False


# ----------------------------------------------------------------------
# Step 4: mark_only_lora_trainable, and the guarantees it buys
# ----------------------------------------------------------------------


def test_mark_only_lora_trainable_freezes_exactly_the_right_set():
    net = make_net()
    inject_lora(net, {"q_proj", "v_proj"}, rank=4)
    trainable, total = mark_only_lora_trainable(net)
    assert (trainable, total) == count_parameters(net)
    expected_trainable = 2 * (
        4 * (D_MODEL + D_MODEL)  # q_proj: A (8,4) + B (4,8)
        + 4 * (D_MODEL + D_KV)  # v_proj: A (8,4) + B (4,4)
    )
    assert trainable == expected_trainable
    for name, p in net.named_parameters():
        is_adapter = name.endswith("lora_A") or name.endswith("lora_B")
        assert p.requires_grad is is_adapter, name


def test_frozen_base_is_bit_identical_after_training():
    net = make_net()
    inject_lora(net, {"q_proj", "v_proj"}, rank=4)
    # The target is produced by a teacher with the SAME frozen base and a
    # different adapter — a function the student adapter can actually
    # reach, which is exactly LoRA's claim about fine-tuning deltas.
    teacher = copy.deepcopy(net)
    for module in teacher.modules():
        if isinstance(module, LoRALinear):
            randomize_adapter(module, seed=13)
    mark_only_lora_trainable(net)
    frozen_snapshot = {
        name: p.detach().clone()
        for name, p in net.named_parameters()
        if not (name.endswith("lora_A") or name.endswith("lora_B"))
    }

    x = make_input(seed=10, shape=(16, D_MODEL))
    with torch.no_grad():
        target = teacher(x)
    view = LoRAModel(net)
    params = view.parameters()
    losses = []
    for _ in range(60):
        loss = torch.nn.functional.mse_loss(view(x), target)
        losses.append(loss.item())
        for p in params:
            if p.grad is not None:
                p.grad = None
        loss.backward()
        with torch.no_grad():
            for p in params:
                p -= 0.5 * p.grad
    assert losses[-1] < losses[0] * 0.8  # training moved the adapter

    changed = 0
    for name, p in net.named_parameters():
        if name.endswith("lora_A") or name.endswith("lora_B"):
            changed += 1
            continue
        # The LoRA guarantee: not "roughly unchanged" — untouched.
        assert torch.equal(p, frozen_snapshot[name]), name
    assert changed == 8


def test_adapter_state_dict_round_trip_transfers_behavior():
    net = make_net(seed=42)
    inject_lora(net, {"q_proj", "v_proj"}, rank=4)
    mark_only_lora_trainable(net)
    for module in net.modules():
        if isinstance(module, LoRALinear):
            randomize_adapter(module, seed=13)
    x = make_input()
    trained_out = net(x)

    fresh = make_net(seed=42)  # same base weights by construction
    inject_lora(fresh, {"q_proj", "v_proj"}, rank=4)
    assert not torch.allclose(fresh(x), trained_out)  # adapter not loaded yet
    load_lora_state_dict(fresh, lora_state_dict(net))
    assert torch.allclose(fresh(x), trained_out, atol=1e-6)
