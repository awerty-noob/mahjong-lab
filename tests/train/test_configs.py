"""Strict explicit VRPO horizons and reference budgets; no training or models."""

import math
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from train.configs import (
    ALGORITHM_KIND,
    VRPOConfig,
    paper_reference_preset,
    tiny_debug_preset,
)
from train.update import damping


def test_damping_horizons_have_no_silent_production_default():
    with pytest.raises(TypeError):
        VRPOConfig()
    with pytest.raises(TypeError):
        VRPOConfig(t_eta=100)
    with pytest.raises(TypeError):
        paper_reference_preset(t_eta=100, t_alpha=100)


def test_default_whole_trajectory_reference_budgets_and_algorithm_metadata():
    cfg = VRPOConfig(t_eta=101, t_alpha=203)
    assert cfg.actor_epochs == cfg.critic_epochs == cfg.trajectory_minibatches == 4
    assert cfg.actor_epochs * cfg.trajectory_minibatches == 16
    assert cfg.critic_epochs * cfg.trajectory_minibatches == 16
    assert cfg.rollout_games == 2048 and cfg.num_envs == 32 and cfg.games_per_minibatch == 512
    assert cfg.actor_lr == cfg.critic_lr == 4e-4
    assert cfg.clip0 == 0.02 and cfg.uniform_alpha0 == 0.1 and cfg.weight_decay == 0.01
    assert (cfg.actor_microbatch_size, cfg.critic_microbatch_size, cfg.inference_microbatch_size) == (64, 32, 64)
    assert cfg.execution_block_steps == 64 and cfg.page_rows == 2048
    assert cfg.max_bytes == 8 * 1024**3
    assert cfg.calibration_max_epochs == 64 and cfg.calibration_min_relative_improvement == 0.01
    assert cfg.optimizer == "muon" and cfg.muon_momentum == 0.95
    assert cfg.muon_ns_steps == 5 and cfg.muon_ns_dtype == "bfloat16"
    assert not hasattr(cfg, "muon_lr_multiplier")
    assert cfg.bc_reference_coef == 0.0
    contract = cfg.algorithm_config()
    assert contract["version"] == 2 and contract["kind"] == ALGORITHM_KIND
    assert contract["t_eta"] == 101 and contract["t_alpha"] == 203
    assert contract["trace_clock"] == "raw_environment_transition"
    assert contract["payoff_seats"] == "absolute3"
    assert contract["actor_reduction"] == "sum_rows_mean_games"
    assert contract["critic_targets"] == "sampled_action_only"
    assert contract["training_seats"] == "all"
    assert contract["critic_reduction"] == "half_sum_rows_mean_games"
    assert contract["payoff_channel_reduction"] == "mean_absolute3"
    for name in ("advantage_normalization", "critic_auxiliary", "replay"):
        assert contract[name] is False
    contract["actor_lr"] = 9.0
    assert cfg.actor_lr == 4e-4
    with pytest.raises(FrozenInstanceError):
        cfg.actor_lr = 9.0


@pytest.mark.parametrize("iteration", [0, 1, 2, 4, 8, 128])
def test_damping_exact_formulas_no_warmup(iteration):
    cfg = VRPOConfig(t_eta=2, t_alpha=8, clip0=0.2, uniform_alpha0=0.04)
    schedule = damping(cfg, iteration)
    eta = min(1.0, 2 / max(iteration, 1))
    alpha = min(1.0, 8 / max(iteration, 1))
    np.testing.assert_allclose(schedule, [eta, np.sqrt(eta), 0.2 * eta, 0.04 * np.sqrt(alpha), 1.0], rtol=1e-6)
    if iteration in (0, 1):
        assert float(schedule.actor_multiplier) == float(schedule.critic_multiplier) == 1.0
        assert float(schedule.clip) == pytest.approx(cfg.clip0)
        assert float(schedule.uniform_alpha) == pytest.approx(cfg.uniform_alpha0)


def test_presets_are_explicit_labeled_debug_or_reference_not_training_evidence():
    tiny = tiny_debug_preset()
    assert tiny.preset == "tiny_debug" and tiny.rollout_games == 4
    assert tiny.actor_epochs == tiny.critic_epochs == 1
    assert tiny.trajectory_minibatches == tiny.games_per_minibatch == 2
    assert tiny_debug_preset(t_eta=17, optimizer="muon").t_eta == 17
    reference = paper_reference_preset(
        t_eta=500, t_alpha=800, actor_lr=1e-5, critic_lr=3e-5, clip0=0.15, uniform_alpha0=0.02
    )
    assert reference.preset == "paper_reference" and reference.t_eta == 500
    assert reference.t_alpha == 800 and reference.actor_lr == 1e-5 and reference.critic_lr == 3e-5
    assert reference.actor_epochs * reference.trajectory_minibatches == 16
    assert reference.critic_epochs * reference.trajectory_minibatches == 16


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("version", 1.0),
        ("version", 1),
        ("kind", "ppo"),
        ("preset", "production_tuned"),
        ("gamma", 0.99),
        ("gamma", True),
        ("trace_lambda", -1.0),
        ("trace_lambda", 1.1),
        ("t_eta", 0),
        ("t_alpha", -1),
        ("t_eta", 2.0),
        ("t_alpha", True),
        ("actor_epochs", 0),
        ("critic_epochs", -1),
        ("trajectory_minibatches", 3),
        ("num_envs", 3),
        ("rollout_games", 7),
        ("actor_microbatch_size", 0),
        ("critic_microbatch_size", False),
        ("inference_microbatch_size", -1),
        ("execution_block_steps", 0),
        ("time_bucket", 0),
        ("max_bytes", 0),
        ("max_game_steps", 0),
        ("actor_lr", 0.0),
        ("critic_lr", -1.0),
        ("clip0", 0.0),
        ("clip0", 1.0),
        ("uniform_alpha0", -0.1),
        ("temperature", 0.0),
        ("target_kl", 0.0),
        ("actor_max_grad_norm", 0.0),
        ("critic_max_grad_norm", -1.0),
        ("weight_decay", -0.01),
        ("bc_reference_coef", -0.01),
        ("optimizer", "adamw"),
        ("optimizer", "sgd"),
        ("muon_momentum", 1.0),
        ("muon_momentum", -0.1),
        ("muon_ns_steps", 0),
        ("muon_ns_dtype", "float64"),
        ("calibration_train_games", 0),
        ("calibration_heldout_games", 0),
        ("calibration_max_epochs", 0),
        ("calibration_min_relative_improvement", 1.0),
        ("actor_lr", float("nan")),
        ("uniform_alpha0", float("inf")),
    ],
)
def test_invalid_configuration_rejected(field, value):
    with pytest.raises(ValueError):
        replace(VRPOConfig(t_eta=2, t_alpha=3), **{field: value})


@pytest.mark.parametrize(
    "actor,critic,inference,envs,steps", [(16, 8, 16, 8, 256), (3, 4, 5, 2, 7), (7, 11, 2, 2, 3), (2, 2, 2, 2, 8)]
)
def test_page_rows_round_whole_execution_block_to_microbatch_lcm(actor, critic, inference, envs, steps):
    cfg = tiny_debug_preset(
        actor_microbatch_size=actor,
        critic_microbatch_size=critic,
        inference_microbatch_size=inference,
        num_envs=envs,
        rollout_games=envs * 2,
        execution_block_steps=steps,
    )
    alignment = math.lcm(actor, critic, inference)
    assert cfg.storage_page_rows is None and cfg.page_alignment == alignment
    assert cfg.page_rows == math.ceil(envs * steps / alignment) * alignment
    assert cfg.page_rows >= envs * steps
    assert all(cfg.page_rows % size == 0 for size in (actor, critic, inference))
    assert cfg.algorithm_config()["execution_layout"] == {
        "version": 2,
        "kind": "paged_interleaved_games",
        "page_rows": cfg.page_rows,
    }


@pytest.mark.parametrize("rows", [16, 32, 48])
def test_explicit_storage_page_rows_is_preserved_and_in_semantic_contract(rows):
    cfg = tiny_debug_preset(storage_page_rows=rows)
    assert cfg.page_rows == rows and cfg.algorithm_config()["storage_page_rows"] == rows
    assert cfg.algorithm_config()["execution_layout"]["page_rows"] == rows
    assert replace(cfg, time_bucket=4096, max_game_steps=99999).page_rows == rows


@pytest.mark.parametrize("rows", [True, False, 0, -1, 16.0, "16", 2, 15, 17])
def test_invalid_page_rows_type_size_or_alignment_is_rejected(rows):
    with pytest.raises(ValueError, match="page"):
        tiny_debug_preset(storage_page_rows=rows)


def test_execution_block_and_microbatch_changes_revalidate_explicit_page_size():
    cfg = tiny_debug_preset(storage_page_rows=16)
    with pytest.raises(ValueError, match="page"):
        replace(cfg, execution_block_steps=9)
    with pytest.raises(ValueError, match="page"):
        replace(cfg, actor_microbatch_size=3)
    auto = replace(cfg, storage_page_rows=None, actor_microbatch_size=3)
    assert auto.page_rows == 18 and auto.page_alignment == 6


def test_removed_multiplier_is_not_a_supported_compatibility_alias():
    with pytest.raises(TypeError):
        VRPOConfig(t_eta=2, t_alpha=3, muon_lr_multiplier=1.0)


def test_tempo_filter_schedule_matches_epsilon_prism_formula():
    cfg = VRPOConfig(
        t_eta=2,
        t_alpha=8,
        clip0=0.2,
        uniform_alpha0=0.04,
        tempo_anchor=True,
        tempo_filter_gain=0.9,
        tempo_filter_tau=4.0,
        tempo_filter_power=2.0 / 3.0,
        tempo_lr_floor=0.25,
    )
    seen = []
    for iteration in (1, 2, 10, 50, 200):
        schedule = damping(cfg, iteration)
        keep = 0.9 * (1.0 + max(iteration - 1, 0) / 4.0) ** (-2.0 / 3.0)
        eta = min(1.0, 2 / iteration)
        assert float(schedule.tempo_keep) == pytest.approx(keep, rel=1e-6)
        assert float(schedule.actor_multiplier) == pytest.approx(eta * min(1.0, keep / 0.25), rel=1e-6)
        seen.append(float(schedule.tempo_keep))
    assert all(a > b for a, b in zip(seen, seen[1:]))  # gain decays monotonically


def test_tempo_anchor_keeps_actor_step_identity_at_unit_gain():
    cfg = VRPOConfig(t_eta=2, t_alpha=8, clip0=0.2, uniform_alpha0=0.04)
    schedule = damping(cfg, 3)
    assert float(schedule.tempo_keep) == 1.0
    assert float(schedule.actor_multiplier) == pytest.approx(min(1.0, 2 / 3), rel=1e-6)
