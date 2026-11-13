"""Unit and small real-environment contracts for ``train.eval``."""

import hashlib
import pickle

import jax
import numpy as np
import pytest

from networks.mahjong_3p.configs import DEFAULT_READING_CONFIG
from networks.mahjong_3p.network import build_actor, tokenize_state
from train import eval as evaluation
from train.configs import default_reward_config, rule_spec, tiny_debug_preset
from train.selfplay import VRPORunConfig


def _pass_result(*, scores, winds, initial_scores, focal, stats):
    return {
        "valid": True,
        "stats": stats,
        "scores": np.asarray(scores, np.int32),
        "winds": np.asarray(winds, np.int32),
        "initial_scores": np.asarray(initial_scores, np.int32),
        "focal": np.asarray(focal, np.int32),
    }


def test_metric_schema_and_reducer_units():
    result = _pass_result(
        scores=[[450, 300, 250], [300, 250, 450]],
        winds=[[0, 1, 2], [1, 2, 0]],
        initial_scores=[[350, 350, 350], [350, 350, 350]],
        focal=[0, 1],
        stats={
            "hands": 4,
            "wins": 2,
            "deal_ins": 1,
            "riichi_hands": 2,
            "meld_hands": 3,
            "draws": 2,
            "tsumo_hands": 1,
            "damaten_hands": 1,
            "win_pts": 1600,
            "deal_in_pts": 800,
            "turn_sum": 40,
        },
    )
    summary = evaluation._summarize(result, nums_eval=2, order_points=[200, 0, -200])
    assert summary["valid"] is True
    assert summary["games"] == 2
    assert summary["hands"] == 4
    assert summary["avg_rank"] == 2.0
    assert summary["avg_rank_pt"] == 0.0
    assert summary["win_rate"] == 0.5
    assert summary["deal_in_rate"] == 0.25
    assert summary["riichi_rate"] == 0.5
    assert summary["meld_rate"] == 0.75
    assert summary["avg_win_pt"] == 800.0
    assert summary["avg_deal_in_pt"] == 800.0
    assert summary["draw_rate"] == 0.5
    assert summary["bankruptcy_rate"] == 0.0
    assert summary["tsumo_rate"] == 0.5
    assert summary["damaten_rate"] == 0.5
    assert summary["avg_turn"] == 10.0
    assert set(summary) == {"valid", "games", "hands", *evaluation.METRIC_KEYS}


def test_invalid_pass_has_null_metrics():
    result = _pass_result(
        scores=[[350, 350, 350]],
        winds=[[0, 1, 2]],
        initial_scores=[[350, 350, 350]],
        focal=[0],
        stats={"hands": 1},
    )
    result["valid"] = False
    summary = evaluation._summarize(result, nums_eval=1, order_points=[200, 0, -200])
    assert summary["valid"] is False
    assert summary["games"] == 1
    assert summary["hands"] == 0
    assert all(summary[key] is None for key in evaluation.METRIC_KEYS)


def test_eval_config_validation():
    base = dict(
        algorithm=tiny_debug_preset(),
        bc_path="bc.pkl",
        bc_sha256="a" * 64,
        output_dir="out",
        run_id="run",
    )
    options = VRPORunConfig(**base)
    assert options.eval_interval == 5
    assert options.baseline_model is None
    assert options.nums_eval == 30

    options = VRPORunConfig(**base, eval_interval=2, nums_eval=4)
    assert options.eval_interval == 2
    assert options.nums_eval == 4

    with pytest.raises(ValueError):
        VRPORunConfig(**base, eval_interval=0)
    with pytest.raises(ValueError):
        VRPORunConfig(**base, nums_eval=0)
    with pytest.raises(ValueError):
        VRPORunConfig(**base, baseline_model="bc.pkl")
    with pytest.raises(ValueError):
        VRPORunConfig(**base, baseline_sha256="a" * 64)


def test_rule_spec_round_end_is_explicit():
    env = rule_spec("sanma").make_env("half", round_end="confirm")
    assert env.round_end == "confirm"
    with pytest.raises(ValueError):
        rule_spec("sanma").make_env("half", round_end="invalid")


def test_default_baseline_is_the_existing_params():
    params = object()
    state = type("State", (), {"params": params})()
    options = type("Options", (), {"baseline_model": None})()
    assert evaluation.load_baseline_params(options, None, state) is params


def test_training_tensorboard_flattens_nested_metrics(monkeypatch, tmp_path):
    captured = {}

    def capture(directory, step, prefix, values):
        captured.update(directory=directory, step=step, prefix=prefix, values=dict(values))

    monkeypatch.setattr(evaluation, "_write_scalars", capture)
    row = {
        "progress": {"completed_updates": 3, "envsteps": 900},
        "update": {"actor_steps": 2, "kl": 0.125},
        "execution": {"cache": {"hits": 4}},
        "wall_seconds": 1.5,
        "eval": {"learner": {"win_rate": 0.5}},
        "note": "ignored",
        "flag": True,
    }

    assert evaluation.write_training_tensorboard(tmp_path, 3, row) is True
    assert captured["directory"] == tmp_path / "learner"
    assert captured["step"] == 3
    assert captured["prefix"] == "train"
    assert captured["values"] == {
        "progress/completed_updates": 3.0,
        "progress/envsteps": 900.0,
        "update/actor_steps": 2.0,
        "update/kl": 0.125,
        "execution/cache/hits": 4.0,
        "wall_seconds": 1.5,
    }


def test_dual_evaluation_smoke():
    env = rule_spec("sanma").make_env(
        "half",
        reward_config=default_reward_config("sanma"),
        round_end="confirm",
    )
    state = env.init(jax.random.PRNGKey(0))
    observation = tokenize_state(env, state)
    model = build_actor({"architecture": "v5", "num_layers": 1, "d_model": 8, "num_heads": 2, "mlp_dim": 16})
    params = model.init(jax.random.PRNGKey(1), observation, method=model.get_action_logits)

    result = evaluation.run_dual_evaluation(
        env,
        model,
        params,
        params,
        nums_eval=1,
        seed=0,
        max_steps=5_000,
    )

    assert result["valid"] is True
    assert result["nums_eval"] == 1
    assert result["games"] == 2
    assert result["hand_stats_available"] is True
    for view in ("learner", "baseline"):
        assert result[view]["valid"] is True
        assert result[view]["games"] == 1
        assert set(result[view]) == {"valid", "games", "hands", *evaluation.METRIC_KEYS}


def test_explicit_baseline_checkpoint_loads_from_a_real_env(tmp_path):
    tiny = {"architecture": "v5", "num_layers": 1, "d_model": 8, "num_heads": 2, "mlp_dim": 16}
    head = {"version": 1, "hidden_dim": 8}
    env = rule_spec("sanma").make_env("half", reward_config=default_reward_config("sanma"))
    state = env.init(jax.random.PRNGKey(0))
    observation = tokenize_state(env, state)
    model = build_actor(tiny, head_config=head)
    params = model.init(jax.random.PRNGKey(1), observation, method=model.get_action_logits)
    payload = {
        "schema_version": 1,
        "checkpoint_kind": "bc",
        "kind": "bc",
        "env_name": "sanma",
        "round_mode": "half",
        "ruleset": "riichienv-standard-sanma",
        "network_config": tiny,
        "head_config": head,
        "reading_config": dict(DEFAULT_READING_CONFIG),
        "params": params,
        "critic_trained": False,
    }
    data = pickle.dumps(payload)
    path = tmp_path / "bc.pkl"
    path.write_bytes(data)
    options = type("Options", (), {"baseline_model": str(path), "baseline_sha256": hashlib.sha256(data).hexdigest()})()
    online = type("State", (), {"params": params})()
    loaded = evaluation.load_baseline_params(
        options,
        env,
        online,
        sample_state=state,
        contract={"network_config": tiny, "head_config": head, "reading_config": dict(DEFAULT_READING_CONFIG)},
    )
    assert set(loaded["params"]) == {"policy_extractor", "policy_mlp"}
    assert evaluation.baseline_identity(options)["source"] == "checkpoint"
    assert evaluation.baseline_identity(options)["sha256"] == options.baseline_sha256
