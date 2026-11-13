"""Bidirectional greedy self-play evaluation for accepted VRPO actors.

The simulator and the network use different action/observation schemas.  This
module is deliberately the only place in the evaluation path that translates
between them: the simulator exposes 88 actions and canonical 34-tile ids, while
the Actor consumes 87 actions and compact 27-tile ids.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from networks.mahjong_3p.network import (
    compact_legal_mask,
    to_simulator_action,
    tokenize_state,
)
from train.configs import settlement_components

METRIC_KEYS = (
    "avg_rank",
    "avg_rank_pt",
    "win_rate",
    "deal_in_rate",
    "riichi_rate",
    "meld_rate",
    "avg_win_pt",
    "avg_deal_in_pt",
    "draw_rate",
    "bankruptcy_rate",
    "tsumo_rate",
    "damaten_rate",
    "avg_turn",
)


def _zero_stats():
    return {
        "hands": jnp.asarray(0.0, jnp.float32),
        "wins": jnp.asarray(0.0, jnp.float32),
        "deal_ins": jnp.asarray(0.0, jnp.float32),
        "riichi_hands": jnp.asarray(0.0, jnp.float32),
        "meld_hands": jnp.asarray(0.0, jnp.float32),
        "draws": jnp.asarray(0.0, jnp.float32),
        "tsumo_hands": jnp.asarray(0.0, jnp.float32),
        "damaten_hands": jnp.asarray(0.0, jnp.float32),
        "win_pts": jnp.asarray(0.0, jnp.float32),
        "deal_in_pts": jnp.asarray(0.0, jnp.float32),
        "turn_sum": jnp.asarray(0.0, jnp.float32),
    }


def _add(stats, name, value, mask):
    return stats | {name: stats[name] + jnp.sum(jnp.where(mask, value, 0.0))}


def _run_focal_pass(
    env,
    model,
    focal_params,
    opponent_params,
    *,
    nums_eval,
    seed,
    max_steps,
):
    """Play one focal model against the other model, returning host-side sums."""
    n = int(nums_eval)
    if n <= 0:
        raise ValueError("nums_eval must be positive")
    base = jax.random.PRNGKey(int(seed))
    init_keys = jax.random.split(base, n)
    initial = jax.vmap(env.init)(init_keys)
    initial_scores = initial.round_state.score
    keys = jax.random.split(jax.random.fold_in(base, 1), n)
    focal = jnp.arange(n, dtype=jnp.int32) % 3
    focal_params = {"params": focal_params["params"]}
    opponent_params = {"params": opponent_params["params"]}

    def observe(state):
        return jax.vmap(lambda s: tokenize_state(env, s))(state)

    def logits(params, observation):
        return model.apply(params, observation, method=model.get_action_logits)

    def step_all(state, actions, step_keys):
        return jax.vmap(lambda s, a, k: env.step(s, a, k))(state, actions, step_keys)

    def body(carry):
        state = carry["state"]
        active = ~carry["done"]
        observation = observe(state)
        learner_logits = logits(focal_params, observation)
        opponent_logits = logits(opponent_params, observation)
        logits_now = jnp.where((state.current_player == focal)[:, None], learner_logits, opponent_logits)
        mask = jax.vmap(lambda s: compact_legal_mask(s.legal_action_mask))(state)
        chosen = jnp.argmax(jnp.where(mask, logits_now, -jnp.inf), axis=-1)
        split = jax.vmap(lambda k: jax.random.split(k, 2))(carry["keys"])
        next_keys, step_keys = split[:, 0], split[:, 1]
        next_state = step_all(state, to_simulator_action(chosen), step_keys)

        prev_terminated = state.round_state.terminated_round
        hand_end = next_state.round_state.terminated_round & ~prev_terminated & active
        rewards = next_state.rewards
        rows = jnp.arange(n)
        focal_reward = rewards[rows, focal]
        # Tenpai exhaustive-draw payments are +20/-10/-10, not a win.
        won = hand_end & next_state.players.has_won.any(axis=-1)
        n_negative = jnp.sum(rewards < 0, axis=-1)
        ron = won & (n_negative == 1)
        focal_win = won & (focal_reward > 0)
        deal_in = ron & (focal_reward < 0)
        focal_riichi = next_state.players.riichi[rows, focal]
        focal_concealed = next_state.players.is_hand_concealed[rows, focal]
        turn = jnp.sum(next_state.players.discard_counts, axis=-1) / 3.0

        stats = carry["stats"]
        stats = _add(stats, "hands", 1.0, hand_end)
        stats = _add(stats, "wins", 1.0, focal_win)
        stats = _add(stats, "deal_ins", 1.0, deal_in)
        stats = _add(stats, "riichi_hands", 1.0, hand_end & focal_riichi)
        stats = _add(stats, "meld_hands", 1.0, hand_end & ~focal_concealed)
        stats = _add(stats, "draws", 1.0, hand_end & ~won)
        stats = _add(stats, "tsumo_hands", 1.0, focal_win & (n_negative > 1))
        stats = _add(stats, "damaten_hands", 1.0, focal_win & ~focal_riichi & focal_concealed)
        stats = _add(stats, "win_pts", focal_reward * 100.0, focal_win)
        stats = _add(stats, "deal_in_pts", -focal_reward * 100.0, deal_in)
        stats = _add(stats, "turn_sum", turn, hand_end)

        next_done = next_state.terminated | next_state.truncated | (carry["steps"] + 1 >= max_steps)
        next_failed = carry["failed"] | (
            active & (next_state.truncated | ((carry["steps"] + 1 >= max_steps) & ~next_state.terminated))
        )
        done = carry["done"] | next_done

        def freeze(old, new):
            shape = (n,) + (1,) * max(0, getattr(old, "ndim", 0) - 1)
            return jnp.where(carry["done"].reshape(shape), old, new)

        state = jax.tree.map(freeze, state, next_state)
        return {
            "state": state,
            "keys": next_keys,
            "done": done,
            "failed": next_failed,
            "steps": carry["steps"] + 1,
            "stats": stats,
        }

    carry = {
        "state": initial,
        "keys": keys,
        "done": jnp.zeros(n, jnp.bool_),
        "failed": jnp.zeros(n, jnp.bool_),
        "steps": jnp.asarray(0, jnp.int32),
        "stats": _zero_stats(),
    }

    def run():
        return jax.lax.while_loop(lambda c: ~jnp.all(c["done"]), body, carry)

    result = jax.jit(run)()
    stats = {name: float(np.asarray(value)) for name, value in result["stats"].items()}
    valid = bool(np.all(np.asarray(result["done"] & ~result["failed"])))
    return {
        "stats": stats,
        "valid": valid,
        "scores": np.asarray(result["state"].round_state.score),
        "winds": np.asarray(result["state"].round_state.init_wind),
        "initial_scores": np.asarray(initial_scores),
        "focal": np.asarray(focal),
    }


def _rate(numerator, denominator):
    if denominator <= 0:
        return None
    return numerator / denominator


def _summarize(pass_result, *, nums_eval, order_points):
    """Convert one pass's hand sums and final scores into the public schema."""
    empty = {key: None for key in METRIC_KEYS}
    if not pass_result["valid"]:
        return {"valid": False, "games": int(nums_eval), "hands": 0, **empty}
    stats = pass_result["stats"]
    scores = pass_result["scores"]
    winds = pass_result["winds"]
    initial = pass_result["initial_scores"]
    focal = pass_result["focal"]
    rows = np.arange(int(nums_eval))
    components = settlement_components(scores, winds, np.asarray(order_points, np.int32))
    ranks = np.asarray(components["ranks"])[rows, focal] + 1
    settled = np.asarray(components["settled_scores"])[rows, focal]
    raw = np.asarray(components["raw_scores"])[rows, focal]
    initial_focal = initial[rows, focal]
    hands = int(stats["hands"])
    wins = int(stats["wins"])
    deal_ins = int(stats["deal_ins"])
    games = int(nums_eval)
    return {
        "valid": True,
        "games": games,
        "hands": hands,
        "avg_rank": float(np.mean(ranks)),
        "avg_rank_pt": float(np.mean((settled - initial_focal) / 10.0)),
        "win_rate": _rate(wins, hands),
        "deal_in_rate": _rate(deal_ins, hands),
        "riichi_rate": _rate(stats["riichi_hands"], hands),
        "meld_rate": _rate(stats["meld_hands"], hands),
        "avg_win_pt": _rate(stats["win_pts"], wins),
        "avg_deal_in_pt": _rate(stats["deal_in_pts"], deal_ins),
        "draw_rate": _rate(stats["draws"], hands),
        "bankruptcy_rate": float(np.mean(raw < 0)),
        "tsumo_rate": _rate(stats["tsumo_hands"], wins),
        "damaten_rate": _rate(stats["damaten_hands"], wins),
        "avg_turn": _rate(stats["turn_sum"], hands),
    }


def run_dual_evaluation(
    env,
    model,
    learner_params,
    baseline_params,
    *,
    nums_eval,
    seed,
    max_steps=20_000,
):
    """Evaluate learner and baseline in mirrored focal passes with greedy actions."""
    if type(nums_eval) is not int or nums_eval <= 0:
        raise ValueError("nums_eval must be a positive integer")
    if type(max_steps) is not int or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    learner_pass = _run_focal_pass(
        env,
        model,
        learner_params,
        baseline_params,
        nums_eval=nums_eval,
        seed=seed,
        max_steps=max_steps,
    )
    baseline_pass = _run_focal_pass(
        env,
        model,
        baseline_params,
        learner_params,
        nums_eval=nums_eval,
        seed=seed,
        max_steps=max_steps,
    )
    learner = _summarize(learner_pass, nums_eval=nums_eval, order_points=env.order_points)
    baseline = _summarize(baseline_pass, nums_eval=nums_eval, order_points=env.order_points)
    return {
        "valid": learner["valid"] and baseline["valid"],
        "nums_eval": nums_eval,
        "games": nums_eval * 2,
        "hand_stats_available": learner["valid"] and baseline["valid"],
        "learner": learner,
        "baseline": baseline,
    }


def load_baseline_params(options, env, state, *, sample_state=None, contract=None, default_params=None):
    """Resolve the online baseline, keeping only Actor roots for inference.

    ``state`` is the online training state (parameters/optimizer), not an
    environment state.  A small sample state is used solely to instantiate the
    checkpoint's observation schema.
    """
    if options.baseline_model is None:
        return state.params if default_params is None else default_params
    from train.selfplay import _bc_actor, load_verified, load_vrpo_actor

    checkpoint = load_verified(options.baseline_model, options.baseline_sha256)
    if contract is not None:
        from train.continuation import _equal_contract

        for name in ("network_config", "head_config", "reading_config"):
            if name in checkpoint:
                _equal_contract(checkpoint.get(name), contract[name], "baseline checkpoint contract mismatch: " + name)
    if sample_state is None:
        sample_state = env.init(jax.random.PRNGKey(0))
    observation = tokenize_state(env, sample_state)
    if checkpoint.get("checkpoint_kind") == "bc":
        actor, _, _ = _bc_actor(
            checkpoint,
            observation,
            reading_auxiliary=checkpoint.get("reading_config") is not None,
        )
    else:
        _, actor = load_vrpo_actor(checkpoint, observation)
    merged = dict(state.params["params"])
    merged.update(actor["params"])
    return {"params": merged}


def baseline_identity(options):
    """Stable provenance for the evaluation baseline."""
    if options.baseline_model is None:
        return {"source": "bc", "sha256": options.bc_sha256}
    return {
        "source": "checkpoint",
        "path": str(options.baseline_model),
        "sha256": options.baseline_sha256,
    }


def _iter_training_scalars(row, prefix=""):
    """Yield nested numeric training fields as TensorBoard tag paths."""
    if not isinstance(row, Mapping):
        return
    for key, value in row.items():
        if key == "eval":
            continue
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            yield from _iter_training_scalars(value, path + "/")
        elif isinstance(value, numbers.Real) and not isinstance(value, bool):
            if math.isfinite(float(value)):
                yield path, float(value)


def _write_scalars(directory, step, prefix, values):
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    writer = EventFileWriter(str(directory))
    try:
        for key, value in values:
            summary = Summary(value=[Summary.Value(tag=prefix + "/" + key, simple_value=float(value))])
            writer.add_event(Event(step=int(step), summary=summary))
    finally:
        writer.flush()
        writer.close()


def write_tensorboard(output, update, result):
    """Best-effort scalar logging: two runs with identical tags give one chart."""
    if not result.get("valid"):
        return False
    try:
        root = Path(output)
        for name in ("learner", "baseline"):
            values = [
                (key, value)
                for key, value in result[name].items()
                if isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(float(value))
            ]
            _write_scalars(root / name, update, "eval", values)
    except Exception:
        return False
    return True


def write_training_tensorboard(output, update, row):
    """Best-effort ``train/*`` scalars for the online learner run."""
    try:
        values = list(_iter_training_scalars(row))
        _write_scalars(Path(output) / "learner", update, "train", values)
    except Exception:
        return False
    return True
