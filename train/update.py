"""Paged actor/critic update kernels and traces."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict, freeze

from train.cache import bounded_jit, fixed_page_scratch_bytes
from train.configs import PagedRawQBatch, QBoostLabels, VRPOTrainingState
from train.optimizer import make_vrpo_muon

# --- from vrpo_trace.py ---


def public_q_input(raw):
    """Assemble the Q input without duplicating public observations in storage."""
    return {"observation": raw.observation, "privileged": raw.critic_observation}


def _probability_valid(probabilities, mask, action):
    safe = jnp.clip(action, 0, 86)
    selected = jnp.take_along_axis(probabilities, safe[..., None], axis=-1)[..., 0]
    return (
        jnp.all(jnp.isfinite(probabilities), axis=-1)
        & jnp.all(probabilities >= 0, axis=-1)
        & jnp.all(jnp.where(mask, True, probabilities == 0), axis=-1)
        & (jnp.abs(jnp.sum(probabilities, axis=-1) - 1) <= 1e-5)
        & (selected > 0)
    )


def raw_row_validity(raw):
    """Device boolean per row; unexecuted placeholders are deliberately ignored."""
    safe = jnp.clip(raw.action, 0, 86)
    legal = jnp.take_along_axis(raw.action_mask, safe[..., None], axis=-1)[..., 0]
    count = jnp.sum(raw.action_mask, axis=-1)
    valid = (
        (raw.action >= 0)
        & (raw.action < 87)
        & legal
        & (count > 0)
        & (raw.is_decision == (count > 1))
        & (raw.current_player >= 0)
        & (raw.current_player < 3)
        & jnp.all(jnp.isfinite(raw.reward), axis=-1)
        & _probability_valid(raw.behavior_probs, raw.action_mask, raw.action)
    )
    if raw.reading_targets is not None:
        from .auxhead import validate_reading_targets

        valid &= validate_reading_targets(raw.reading_targets, raw.executed.shape)
    return ~raw.executed | valid


def _check_feature_tree(tree):
    # Prepared features are explicit dictionaries, never an environment State or
    # PRNG stream. Check structure/metadata only; feature arrays stay on device.
    if not isinstance(tree, Mapping):
        raise ValueError("raw Q features must be explicit prepared feature dictionaries, not State")
    forbidden = {"state", "full_state", "complete_state", "rng", "rng_key", "_rng_key"}
    for name, value in tree.items():
        if str(name).lower() in forbidden:
            raise ValueError("environment State/RNG must not enter raw Q feature storage")
        if isinstance(value, Mapping):
            _check_feature_tree(value)
        elif not hasattr(value, "shape") or not hasattr(value, "dtype"):
            raise ValueError("prepared feature dictionaries must contain only arrays/dictionaries, not State")
        elif jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
            raise ValueError("PRNG keys must not enter raw Q feature storage")


def _check_shapes(raw, q_values=None, probabilities=None):
    shape = raw.executed.shape
    for name in ("action", "current_player", "done", "is_decision", "is_learner", "policy_id"):
        if getattr(raw, name).shape != shape:
            raise ValueError(f"{name} must have raw row shape {shape}")
    if raw.reward.shape != shape + (3,):
        raise ValueError("reward must contain all three absolute-seat payoffs")
    for name in ("action_mask", "behavior_probs"):
        if getattr(raw, name).shape != shape + (87,):
            raise ValueError(f"{name} must have 87 actions per raw row")
    if not jnp.issubdtype(raw.action.dtype, jnp.integer):
        raise ValueError("action must be integer")
    if not jnp.issubdtype(raw.current_player.dtype, jnp.integer):
        raise ValueError("current_player must be integer")
    for name in ("done", "is_decision", "is_learner", "executed", "action_mask"):
        if getattr(raw, name).dtype != jnp.bool_:
            raise ValueError(f"{name} must be boolean")
    for tree in (raw.observation, raw.critic_observation):
        _check_feature_tree(tree)
        for leaf in jax.tree.leaves(tree):
            if leaf.shape[: len(shape)] != shape:
                raise ValueError("feature leaves must have the same leading raw row shape")
    if raw.reading_targets is not None:
        from .auxhead import reading_target_shape

        reading_target_shape(raw.reading_targets, shape)  # also abstract admission inputs
    if q_values is not None and q_values.shape != shape + (3, 87):
        raise ValueError("q_values must have shape [..., 3, 87]")
    if probabilities is not None and probabilities.shape != shape + (87,):
        raise ValueError("probabilities must have shape [..., 87]")


def _eager_check(ok, message):
    if not isinstance(ok, jax.core.Tracer) and not bool(np.asarray(ok)):
        raise ValueError(message)


def _layout_validity(raw, offsets, lengths):
    """Complete contiguous game prefix, with no terminal hidden inside a game."""
    n = raw.executed.shape[0]
    expected_offsets = jnp.cumsum(lengths) - lengths
    total = jnp.sum(lengths)
    real = jnp.arange(n) < total
    # Zero-length slots are allowed solely for padded whole-game minibatches.
    end_positions = jnp.where(lengths > 0, offsets + lengths - 1, n)
    ends = jnp.zeros(n, jnp.int32).at[end_positions].add((lengths > 0).astype(jnp.int32), mode="drop") > 0
    return (
        jnp.all(lengths >= 0)
        & jnp.all(offsets == expected_offsets)
        & (total <= n)
        & jnp.all(raw.executed == real)
        & jnp.all(jnp.where(real, raw.done == ends, True))
    )


@bounded_jit(max_entries=8)
def _qboost_labels(raw, q_values, probabilities, gamma, trace_lambda, offsets, lengths):
    real = raw.executed
    valid = (
        _layout_validity(raw, offsets, lengths)
        & jnp.all(raw_row_validity(raw))
        & jnp.all(~real | _probability_valid(probabilities, raw.action_mask, raw.action))
        & jnp.all(~real | jnp.all(jnp.isfinite(q_values), axis=(-2, -1)))
        & jnp.isfinite(gamma)
        & (gamma >= 0)
        & (gamma <= 1)
        & jnp.isfinite(trace_lambda)
        & (trace_lambda >= 0)
        & (trace_lambda <= 1)
    )
    dtype = jnp.result_type(q_values, raw.reward, jnp.float32)
    q = jnp.where(real[:, None, None], q_values, 0).astype(dtype)
    probs = jnp.where(real[:, None], probabilities, 0).astype(dtype)
    values = jnp.sum(q * probs[:, None, :], axis=-1)
    action = jnp.clip(raw.action, 0, 86)
    taken = jnp.take_along_axis(q, action[:, None, None], axis=-1)[..., 0]
    reward = jnp.where(real[:, None], raw.reward, 0).astype(dtype)
    zero = jnp.zeros(3, dtype)

    def step(carry, row):
        next_value, next_z = carry
        r, v, qt, terminal, executed = row
        continuation = executed & ~terminal
        delta = r + gamma * jnp.where(continuation, next_value, zero) - qt
        z = delta + gamma * trace_lambda * jnp.where(continuation, next_z, zero)
        z = jnp.where(executed, z, zero)
        return (v, z), z

    _, z = jax.lax.scan(step, (zero, zero), (reward, values, taken, raw.done, real), reverse=True)
    advantage = jnp.where(real[:, None], taken - values + z, 0)
    target = jnp.where(real[:, None], taken + z, 0)
    valid &= (
        jnp.all(jnp.isfinite(advantage))
        & jnp.all(jnp.isfinite(target))
        & jnp.all(jnp.isfinite(values))
        & jnp.all(jnp.isfinite(taken))
    )

    def poison(x):
        return jnp.where(valid, x, jnp.full_like(x, jnp.nan))

    labels = QBoostLabels(
        poison(advantage), poison(target), poison(values), poison(taken), real & raw.is_learner & raw.is_decision, real
    )
    return labels, valid


def qboost_labels(
    transitions, q_values, *, probabilities=None, gamma=1.0, trace_lambda=0.95, game_offsets=None, game_lengths=None
):
    """Return all-seat Q(lambda) labels without removing foreign or forced rows.

    Inputs are flat ``[bucket,...]``. Metadata describes a packed complete-game
    prefix; omitted metadata means one game (an empty/all-padding game is valid).
    Current-policy probabilities may replace behavior probabilities, but *both*
    distributions are validated, including support of the recorded action.
    Invalid data raises ValueError eagerly, or yields NaN labels while tracing.
    """
    raw = transitions
    if raw.executed.ndim != 1:
        raise ValueError("qboost_labels expects flat raw transitions")
    _check_shapes(raw, q_values, probabilities)
    probabilities = raw.behavior_probs if probabilities is None else probabilities
    if (game_offsets is None) != (game_lengths is None):
        raise ValueError("game_offsets and game_lengths must be provided together")
    if game_lengths is None:
        game_lengths = jnp.sum(raw.executed, dtype=jnp.int32)[None]
        game_offsets = jnp.zeros(1, jnp.int32)
    else:
        game_lengths = jnp.asarray(game_lengths)
        game_offsets = jnp.asarray(game_offsets)
        if (
            game_lengths.ndim != 1
            or game_offsets.shape != game_lengths.shape
            or not jnp.issubdtype(game_lengths.dtype, jnp.integer)
            or not jnp.issubdtype(game_offsets.dtype, jnp.integer)
        ):
            raise ValueError("game metadata must be equal-shaped integer vectors")
    labels, ok = _qboost_labels(raw, q_values, probabilities, gamma, trace_lambda, game_offsets, game_lengths)
    _eager_check(ok, "invalid raw Q trace: incomplete game, layout, Q or probability support")
    return labels


# Explicit bounded-cache management for integration and lifecycle tests.
qboost_labels.clear_cache = _qboost_labels.clear_cache
qboost_labels.cache_info = _qboost_labels.cache_info

# --- from vrpo_update.py ---

ROOTS = {"actor": ("policy_extractor", "policy_mlp"), "critic": ("critic_extractor", "value_critic_mlp")}


class Damping(NamedTuple):
    actor_multiplier: object
    critic_multiplier: object
    clip: object
    uniform_alpha: object
    tempo_keep: object


def damping(config, iteration):
    """T is the completed rollout iteration (first online rollout is T=1).

    ``tempo_keep`` is the Epsilon-Prism TEMPO gain K_t: the Actor objective keeps
    only K_t of the surrogate and pulls the rest back to the iteration that
    collected the batch, while the Actor step is scaled by min(1, K_t/floor).
    """
    t = jnp.maximum(jnp.asarray(iteration, jnp.float32), 1.0)
    eta = jnp.minimum(1.0, config.t_eta / t)
    alpha = jnp.minimum(1.0, config.t_alpha / t)
    if config.tempo_anchor:
        steps = jnp.maximum(t - 1.0, 0.0)
        keep = config.tempo_filter_gain * jnp.power(1.0 + steps / config.tempo_filter_tau, -config.tempo_filter_power)
        actor_multiplier = eta * jnp.minimum(1.0, keep / config.tempo_lr_floor)
    else:
        keep = jnp.asarray(1.0, jnp.float32)
        actor_multiplier = eta
    return Damping(actor_multiplier, jnp.sqrt(eta), config.clip0 * eta, config.uniform_alpha0 * jnp.sqrt(alpha), keep)


def phase_params(params, phase):
    if set(params) != {"params"} or set(params["params"]) != set(sum(ROOTS.values(), ())):
        raise ValueError("VRPO variables require exactly the four explicit Actor/Q roots")
    result = {"params": {name: params["params"][name] for name in ROOTS[phase]}}
    return freeze(result) if isinstance(params, FrozenDict) else result


def replace_phase(params, replacement, phase):
    roots = dict(params["params"])
    roots.update({name: replacement["params"][name] for name in ROOTS[phase]})
    result = {"params": roots}
    return freeze(result) if isinstance(params, FrozenDict) else result


def make_phase_optimizer(params, config, phase):
    """No inactive partition is evaluated, decayed, clipped or advanced."""
    return make_vrpo_muon(phase_params(params, phase), config, phase)


def create_training_state(params, config):
    if config.reading_auxiliary and "reading" not in params["params"]["policy_mlp"]:
        raise ValueError("V5 training requires initialized policy_mlp/reading; no implicit V4 transfer")
    actor = make_phase_optimizer(params, config, "actor")
    critic = make_phase_optimizer(params, config, "critic")
    state = VRPOTrainingState(
        params, actor.init(phase_params(params, "actor")), critic.init(phase_params(params, "critic"))
    )
    return state, actor, critic


def masked_law(logits, mask, temperature=1.0):
    """Finite safe placeholders for padding; caller still validates real masks."""
    fallback = jnp.arange(logits.shape[-1]) == 0
    legal = jnp.where(jnp.any(mask, axis=-1, keepdims=True), mask, fallback)
    logp = jax.nn.log_softmax(jnp.where(legal, logits / temperature, -1e30), axis=-1)
    return jnp.where(legal, jnp.exp(logp), 0.0), jnp.where(legal, logp, 0.0)


def finite_tree(tree):
    leaves = [x for x in jax.tree.leaves(tree) if hasattr(x, "dtype")]
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in leaves])) if leaves else jnp.bool_(True)


def _host_finite(tree, stage, *, summary=None):
    summary = finite_tree if summary is None else summary
    if not bool(np.asarray(summary(tree))):
        raise FloatingPointError(f"VRPO nonfinite {stage}; prior complete boundary retained")


def _micro_tree(tree, size):
    n = jax.tree.leaves(tree)[0].shape[0]
    if n == 0:
        raise ValueError("empty inference/gradient arena")
    padded = math.ceil(n / size) * size

    def pack(x):
        tail = jnp.broadcast_to(x[:1], (padded - n,) + x.shape[1:])
        return jnp.concatenate((x, tail), axis=0).reshape((-1, size) + x.shape[1:])

    return jax.tree.map(pack, tree), n


def micro_predict(apply, params, inputs, size):
    blocks, n = _micro_tree(inputs, size)
    _, values = jax.lax.scan(lambda carry, x: (carry, apply(params, x)), None, blocks)
    return jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:])[:n], values)


@bounded_jit(max_entries=8)
def gather_row_values(
    values, source_offsets, source_lengths, indices, selected_offsets, selected_lengths, output_template
):
    """Repack cached Q/targets with the SAME prefix layout as whole-game gather."""
    n = output_template.shape[0]
    row = jnp.arange(n)
    game = jnp.searchsorted(jnp.cumsum(selected_lengths), row, side="right")
    safe = jnp.minimum(game, indices.shape[0] - 1)
    source_game = jnp.clip(indices[safe], 0, source_lengths.shape[0] - 1)
    source = source_offsets[source_game] + row - selected_offsets[safe]
    result = values[jnp.clip(source, 0, values.shape[0] - 1)]
    valid = row < jnp.sum(selected_lengths)
    return jnp.where(valid.reshape((n,) + (1,) * (result.ndim - 1)), result, 0.0)


def _actor_loss(
    actor,
    network,
    data,
    game_count,
    temperature,
    clip,
    alpha,
    bc_coef,
    reading_coef=0.0,
    reading_counts=None,
    tempo_keep=None,
):
    obs, mask, action, old_probs, advantage, weight, reference = data[:7]
    if reading_coef:
        if len(data) != 8 or data[7] is None or reading_counts is None:
            raise ValueError("V5 Actor requires explicit reading targets and global task counts")
        logits, predictions = network.apply(actor, obs, method=network.get_action_and_reading)
    else:
        logits = network.apply(actor, obs, method=network.get_action_logits)
    probs, logp = masked_law(logits, mask, temperature)
    taken = jnp.take_along_axis(logp, action[:, None], axis=-1)[:, 0]
    old_taken = jnp.take_along_axis(old_probs, action[:, None], axis=-1)[:, 0]
    ratio = jnp.exp(taken - jnp.log(jnp.maximum(old_taken, 1e-30)))
    advantage = jax.lax.stop_gradient(advantage)
    surrogate = jnp.minimum(ratio * advantage, jnp.clip(ratio, 1.0 - clip, 1.0 + clip) * advantage)
    count = jnp.maximum(jnp.sum(mask, axis=-1), 1)
    uniform_kl = jnp.sum(probs * (logp + jnp.log(count[:, None])), axis=-1)
    # Reference is fixed BC law, NOT legal-uniform damping. Disabled by default.
    reference_kl = jnp.sum(reference * (jnp.log(jnp.maximum(reference, 1e-30)) - logp), axis=-1)
    if tempo_keep is None:
        loss = -surrogate + alpha * uniform_kl + bc_coef * reference_kl
    else:
        anchor_kl = jnp.sum(old_probs * (jnp.log(jnp.maximum(old_probs, 1e-30)) - logp), axis=-1)
        loss = tempo_keep * (-surrogate) + (1.0 - tempo_keep) * anchor_kl + alpha * uniform_kl + bc_coef * reference_kl
    total = jnp.sum(weight * loss) / game_count
    if reading_coef:
        from .auxhead import reading_loss

        total += reading_coef * reading_loss(predictions, data[7], weight > 0, reading_counts)
    return total


def _joint_reading_diagnostics(actor, network, raw, microbatch_size):
    """One sparse committed Actor pass: raw-row logits plus ten task aggregates.

    Only eligible rows run the encoder. Compaction happens solely for inference;
    callers still rebuild the trace on the complete, original raw clock. Borrowed
    tail observations remain tokenizer-safe and never duplicate labels/counts.
    """
    from .auxhead import reading_task_sums

    if raw.reading_targets is None:
        raise ValueError("V5 final diagnostics require reading sidecars")
    active = raw.executed & raw.is_learner & raw.is_decision
    n = active.shape[0]
    if not n:
        raise ValueError("empty inference/gradient arena")
    capacity = math.ceil(n / microbatch_size) * microbatch_size
    indices = jnp.nonzero(active, size=capacity, fill_value=0)[0]
    count = jnp.sum(active, dtype=jnp.int32)
    weights = jnp.arange(capacity) < count

    def step(carry):
        start, output, total = carry
        block_indices, mask = jax.tree.map(
            lambda x: jax.lax.dynamic_slice_in_dim(x, start, microbatch_size), (indices, weights)
        )
        # Gather only the current micro: a flat rollout must not create a
        # second full-batch observation/target arena for final diagnostics.
        obs, labels = jax.tree.map(lambda x: x[block_indices], (raw.observation, raw.reading_targets))
        logits, predictions = network.apply(actor, obs, method=network.get_action_and_reading)
        sums, counts = reading_task_sums(predictions, labels, mask)
        output = jax.lax.dynamic_update_slice_in_dim(output, logits, start, axis=0)
        return start + microbatch_size, output, total + jnp.stack((sums, counts))

    _, compact_logits, summaries = jax.lax.while_loop(
        lambda c: c[0] < count,
        step,
        (jnp.int32(0), jnp.zeros((capacity, 87), jnp.float32), jnp.zeros((2, 5), jnp.float32)),
    )
    destinations = jnp.where(weights, indices, n)
    logits = jnp.zeros((n, 87), jnp.float32).at[destinations].set(compact_logits, mode="drop")
    return logits, summaries


def _reading_diagnostic_sums(actor, network, raw, microbatch_size):
    """Fixed microblocks; only ten task aggregates, never predictions, persist."""
    from .auxhead import reading_task_sums

    if raw.reading_targets is None:
        raise ValueError("V5 final diagnostics require reading sidecars")
    active = raw.executed & raw.is_learner & raw.is_decision
    blocks, _ = _micro_tree((raw.observation, raw.reading_targets, active), microbatch_size)
    observation, targets, mask = blocks
    # _micro_tree borrows valid observations for padded forwards, not labels.
    valid = jnp.arange(mask.size).reshape(mask.shape) < active.shape[0]
    blocks = (observation, targets, mask & valid)

    def step(total, data):
        observation, targets, mask = data
        _, predictions = network.apply(actor, observation, method=network.get_action_and_reading)
        sums, counts = reading_task_sums(predictions, targets, mask)
        return total + jnp.stack((sums, counts)), None

    return jax.lax.scan(step, jnp.zeros((2, 5), jnp.float32), blocks)[0]


def _reading_metrics(summaries, coef, *, finite_check=None):
    from .auxhead import TASK_NAMES

    finite_check = _host_finite if finite_check is None else finite_check
    finite_check(summaries, "committed Actor reading diagnostics")
    sums, counts = np.asarray(summaries, np.float64)
    losses = sums / np.maximum(counts, 1.0)
    metrics = {f"reading_{name}_loss": float(loss) for name, loss in zip(TASK_NAMES, losses)}
    metrics.update({f"reading_{name}_valid_cells": float(n) for name, n in zip(TASK_NAMES, counts)})
    metrics["actor_final_reading_loss"] = float(coef * losses.sum())
    return metrics


def _diagnostic_sums(raw, q_old, targets, advantages, law, logp, reference, clip):
    """Device scalar reductions only: never return observations or row arrays."""
    real = raw.executed
    actor = real & raw.is_learner & raw.is_decision
    own = jnp.take_along_axis(advantages, raw.current_player[:, None], -1)[:, 0]
    taken_q = jnp.take_along_axis(q_old, raw.action[:, None, None], -1)[..., 0]
    old = raw.behavior_probs
    old_logp = jnp.log(jnp.maximum(old, 1e-30))
    ratio = jnp.take_along_axis(law, raw.action[:, None], -1)[:, 0] / jnp.maximum(
        jnp.take_along_axis(old, raw.action[:, None], -1)[:, 0], 1e-30
    )
    surrogate = jnp.minimum(ratio * own, jnp.clip(ratio, 1.0 - clip, 1.0 + clip) * own)
    legal_count = jnp.maximum(jnp.sum(raw.action_mask, -1), 1)
    uniform = jnp.sum(law * (logp + jnp.log(legal_count[:, None])), -1)
    ref_kl = jnp.sum(reference * (jnp.log(jnp.maximum(reference, 1e-30)) - logp), -1)
    result = dict(
        raw_rows=jnp.sum(real),
        actor_rows=jnp.sum(actor),
        actor_surrogate_sum=-jnp.sum(jnp.where(actor, surrogate, 0.0)),
        actor_uniform_sum=jnp.sum(jnp.where(actor, uniform, 0.0)),
        actor_reference_sum=jnp.sum(jnp.where(actor, ref_kl, 0.0)),
        actor_kl_sum=jnp.sum(jnp.where(actor, jnp.sum(old * (old_logp - logp), -1), 0.0)),
        q_old_squared_error=jnp.sum(jnp.where(real[:, None], jnp.square(taken_q - targets), 0.0)),
    )
    for name, value, mask in (
        ("q_taken", taken_q, real[:, None]),
        ("q_target", targets, real[:, None]),
        ("advantage", own, actor),
    ):
        masked = jnp.where(mask, value, 0.0)
        result[name + "_sum"] = jnp.sum(masked)
        result[name + "_squared_sum"] = jnp.sum(jnp.square(masked))
        result[name + "_absmax"] = jnp.max(jnp.abs(masked))
    return result


def _merge_diagnostic_sums(left, right):
    if left is None:
        return right
    return {
        key: jnp.maximum(left[key], right[key]) if key.endswith("_absmax") else left[key] + right[key] for key in left
    }


def _diagnostic_metrics(sums, games, schedule, bc_coef):
    """Only these scalar summaries cross to host; cache health is separate."""
    values = {key: float(np.asarray(value)) for key, value in sums.items()}
    result = dict(raw_rows=values["raw_rows"], actor_rows=values["actor_rows"], rollout_games=float(games))
    for name in ("q_taken", "q_target", "advantage"):
        count = max(values["actor_rows"] if name == "advantage" else 3 * values["raw_rows"], 1.0)
        result[name + "_mean"] = values[name + "_sum"] / count
        result[name + "_rms"] = math.sqrt(max(values[name + "_squared_sum"] / count, 0.0))
        result[name + "_absmax"] = values[name + "_absmax"]
    result.update(
        actor_final_surrogate_loss=values["actor_surrogate_sum"] / games,
        actor_final_uniform_loss=float(schedule.uniform_alpha) * values["actor_uniform_sum"] / games,
        actor_final_reference_loss=bc_coef * values["actor_reference_sum"] / games,
        actor_final_kl=values["actor_kl_sum"] / max(values["actor_rows"], 1.0),
        q_old_row_mse=values["q_old_squared_error"] / max(3 * values["raw_rows"], 1.0),
    )
    result["actor_final_loss"] = sum(
        result[key] for key in ("actor_final_surrogate_loss", "actor_final_uniform_loss", "actor_final_reference_loss")
    )
    return result


def _critic_loss(critic, network, data, denominator):
    inputs, action, target, weight = data
    q = network.apply(critic, inputs, method=network.get_action_q)
    taken = jnp.take_along_axis(q, action[:, None, None], axis=-1)[..., 0]
    return jnp.sum(weight[:, None] * jnp.square(taken - jax.lax.stop_gradient(target))) / denominator


def _accumulate(loss, params, inputs, microbatch_size):
    blocks, _ = _micro_tree(inputs, microbatch_size)
    zeros = jax.tree.map(jnp.zeros_like, params)

    def step(carry, block):
        total, grads = carry
        value, grad = jax.value_and_grad(loss)(params, block)
        return (total + value, jax.tree.map(jnp.add, grads, grad)), None

    return jax.lax.scan(step, (jnp.float32(0), zeros), blocks)[0]


@dataclass(frozen=True)
class VRPOUpdateResult:
    state: VRPOTrainingState
    rng: object
    metrics: dict


def _actor_attempt_stats(grads, before, after):
    """Two device scalars only; never export observations or gradient trees."""
    delta = jax.tree.map(lambda a, b: a - b, after, before)
    return jnp.stack((optax.global_norm(grads), optax.global_norm(delta)))


class _ActorAttemptLog:
    """Attempt evidence survives rollback; it is NOT committed-policy diagnostics."""

    def __init__(self):
        self.metrics = dict(
            actor_attempts=0,
            actor_candidate_kl_max=0.0,
            actor_candidate_kl_last=0.0,
            actor_candidate_grad_norm_max=0.0,
            actor_candidate_update_norm_max=0.0,
            actor_rejected_kl=0.0,
            actor_rejection_stage=0,
        )

    def record(self, kl, stats, *, finite_check=None):
        finite_check = _host_finite if finite_check is None else finite_check
        finite_check(stats, "Actor attempt statistics")
        grad_norm, update_norm = map(float, np.asarray(stats))
        value = float(np.asarray(kl))
        m = self.metrics
        m["actor_candidate_kl_max"] = value if not m["actor_attempts"] else max(m["actor_candidate_kl_max"], value)
        m["actor_attempts"] += 1
        m["actor_candidate_kl_last"] = value
        m["actor_candidate_grad_norm_max"] = max(m["actor_candidate_grad_norm_max"], grad_norm)
        m["actor_candidate_update_norm_max"] = max(m["actor_candidate_update_norm_max"], update_norm)

    def reject(self, kl, *, epoch=False):
        self.metrics["actor_rejected_kl"] = float(np.asarray(kl))
        # 0 = none; 1 = selected-minibatch candidate; 2 = whole-batch epoch gate.
        self.metrics["actor_rejection_stage"] = 2 if epoch else 1


class VRPOUpdater:
    """Host owner of bounded pure JITs; closures never capture this mutable owner."""

    def __init__(self, network, config):
        self.network, self.config = network, config
        self._paged = None
        # Capture immutable static values only, not owner/state/batches/arrays.
        cfg, net = config, network

        def actor_apply(p, x):
            return net.apply(p, x, method=net.get_action_logits)

        def q_apply(p, x):
            return net.apply(p, x, method=net.get_action_q)

        self.actor_predict = bounded_jit(max_entries=8)(
            lambda p, x: micro_predict(actor_apply, p, x, cfg.inference_microbatch_size)
        )
        self.q_predict = bounded_jit(max_entries=8)(
            lambda p, x: micro_predict(q_apply, p, x, cfg.inference_microbatch_size)
        )

        def actor_grad(p, data, games, clip, alpha, reading_counts=None, tempo_keep=None):
            if tempo_keep is None:
                tempo_keep = jnp.asarray(1.0, jnp.float32)

            def objective(a, b):
                return _actor_loss(
                    a,
                    net,
                    b,
                    games,
                    cfg.temperature,
                    clip,
                    alpha,
                    cfg.bc_reference_coef,
                    cfg.reading_aux_coef if cfg.reading_auxiliary else 0.0,
                    reading_counts,
                    tempo_keep if cfg.tempo_anchor else None,
                )

            return _accumulate(objective, p, data, cfg.actor_microbatch_size)

        def critic_grad(p, data, denom):
            return _accumulate(lambda a, b: _critic_loss(a, net, b, denom), p, data, cfg.critic_microbatch_size)

        self.actor_grad = bounded_jit(max_entries=8)(actor_grad)
        self.critic_grad = bounded_jit(max_entries=8)(critic_grad)
        self.diagnostic_sums = bounded_jit(max_entries=8)(_diagnostic_sums)
        self.reading_diagnostic_sums = bounded_jit(max_entries=8)(
            lambda actor, raw: _reading_diagnostic_sums(actor, net, raw, cfg.actor_microbatch_size)
        )
        self.joint_reading_diagnostics = bounded_jit(max_entries=8)(
            lambda actor, raw: _joint_reading_diagnostics(actor, net, raw, cfg.actor_microbatch_size)
        )
        # Q, targets, labels, phase transactions and scalar diagnostics have
        # distinct tree signatures; retain their small finite-only executables.
        self.finite_summary = bounded_jit(max_entries=16)(finite_tree)

        def optimizer_step(params, grads, slots, multiplier, phase):
            group = phase_params(params, phase)
            tx = make_phase_optimizer(params, cfg, phase)
            updates, next_slots = tx.update(grads, slots, group)
            updates = jax.tree.map(lambda x: x * multiplier, updates)
            return optax.apply_updates(group, updates), next_slots

        self.actor_opt_step = bounded_jit(max_entries=8)(lambda p, g, s, m: optimizer_step(p, g, s, m, "actor"))
        self.critic_opt_step = bounded_jit(max_entries=8)(lambda p, g, s, m: optimizer_step(p, g, s, m, "critic"))
        self.actor_attempt_stats = bounded_jit(max_entries=8)(_actor_attempt_stats)

    def _host_finite(self, tree, stage):
        # Keep the legacy path unchanged; V5 avoids eager per-leaf dispatches.
        if self.config.reading_auxiliary:
            _host_finite(tree, stage, summary=self.finite_summary)
        else:
            _host_finite(tree, stage)

    def _paged_updater(self):
        if self._paged is None:
            from .update import PagedVRPOUpdater

            self._paged = PagedVRPOUpdater(self.network, self.config)
        return self._paged

    def cache_statistics(self):
        if self._paged is not None:
            return self._paged.cache_statistics()
        names = (
            "actor_predict",
            "q_predict",
            "actor_grad",
            "critic_grad",
            "actor_opt_step",
            "critic_opt_step",
            "diagnostic_sums",
            "actor_attempt_stats",
        )
        if self.config.reading_auxiliary:
            names += ("reading_diagnostic_sums", "joint_reading_diagnostics", "finite_summary")
        return {name: getattr(self, name).cache_info()._asdict() for name in names}

    def close(self):
        if self._paged is not None:
            self._paged.close()
        for fn in (
            self.actor_predict,
            self.q_predict,
            self.actor_grad,
            self.critic_grad,
            self.actor_opt_step,
            self.critic_opt_step,
            self.diagnostic_sums,
            self.actor_attempt_stats,
            self.reading_diagnostic_sums,
            self.joint_reading_diagnostics,
            self.finite_summary,
        ):
            fn.clear_cache()

    def admit(self, batch):
        """Conservative logical rollout/cache/minibatch admission, not GPU workspace."""
        if isinstance(batch, PagedRawQBatch):
            return self._paged_updater().admit(batch)
        cfg = self.config
        leaves = jax.tree.leaves(batch.transitions)
        n = batch.transitions.executed.shape[0]
        raw_bytes = sum(x.size * x.dtype.itemsize for x in leaves)
        lengths = np.asarray(batch.game_lengths)  # game metadata only
        largest = math.ceil(int(lengths.max()) / cfg.time_bucket) * cfg.time_bucket
        selected = cfg.games_per_minibatch * largest
        row_bytes = sum(math.prod(x.shape[1:]) * x.dtype.itemsize for x in leaves)
        # Old Q + fixed/current labels/laws; gather and microbatch preparation
        # can coexist. This cap excludes model/optimizer/executable workspaces.
        charge = raw_bytes + n * (3 * 87 * 4 + 3 * 4 * 8 + 87 * 4 * 2)
        charge += selected * (row_bytes * 3 + 3 * 87 * 4 + 3 * 4 * 8 + 87 * 4 * 2)
        if cfg.reading_auxiliary:
            if batch.transitions.reading_targets is None:
                raise ValueError("V5 rollout missing reading target sidecars")
            charge += selected * (2 * 214 * 4 + 40) + 4096
        if charge > cfg.max_bytes:
            raise MemoryError("VRPO rollout/Q-cache/whole-game minibatch exceeds byte budget")
        return charge

    def old_q(self, state, batch):
        if isinstance(batch, PagedRawQBatch):
            return self._paged_updater().old_q(state, batch)
        self.admit(batch)
        q = self.q_predict(phase_params(state.params, "critic"), public_q_input(batch.transitions))
        self._host_finite(q, "Q-old")
        return jax.lax.stop_gradient(q)

    def _current_labels(self, params, batch, q_old):
        raw = batch.transitions
        logits = self.actor_predict(phase_params(params, "actor"), raw.observation)
        law, _ = masked_law(logits, raw.action_mask, self.config.temperature)
        law = jnp.where(raw.is_learner[:, None], law, raw.behavior_probs)
        labels = qboost_labels(
            raw,
            q_old,
            probabilities=law,
            gamma=self.config.gamma,
            trace_lambda=self.config.trace_lambda,
            game_offsets=batch.game_offsets,
            game_lengths=batch.game_lengths,
        )
        return jax.tree.map(jax.lax.stop_gradient, labels)

    def _kl(self, params, batch):
        raw = batch.transitions
        logits = self.actor_predict(phase_params(params, "actor"), raw.observation)
        _, logp = masked_law(logits, raw.action_mask, self.config.temperature)
        old = raw.behavior_probs
        row = jnp.sum(old * (jnp.log(jnp.maximum(old, 1e-30)) - logp), axis=-1)
        mask = raw.executed & raw.is_learner & raw.is_decision
        return jnp.sum(jnp.where(mask, row, 0.0)) / jnp.maximum(jnp.sum(mask), 1)

    def _diagnostics(self, params, batch, q_old, targets, schedule, reference_actor):
        raw = batch.transitions
        reading_sums = None
        if self.config.reading_auxiliary:
            logits, reading_sums = self.joint_reading_diagnostics(phase_params(params, "actor"), raw)
        else:
            logits = self.actor_predict(phase_params(params, "actor"), raw.observation)
        law, logp = masked_law(logits, raw.action_mask, self.config.temperature)
        # A forced learner row has one legal action, so its law is unchanged
        # by the zero placeholder logits. Preserve the flat reference contract.
        law = jnp.where(raw.is_learner[:, None], law, raw.behavior_probs)
        labels = qboost_labels(
            raw,
            q_old,
            probabilities=law,
            gamma=self.config.gamma,
            trace_lambda=self.config.trace_lambda,
            game_offsets=batch.game_offsets,
            game_lengths=batch.game_lengths,
        )
        reference = jnp.zeros_like(law)
        if self.config.bc_reference_coef:
            reference, _ = masked_law(
                self.actor_predict(reference_actor, raw.observation), raw.action_mask, self.config.temperature
            )
        sums = self.diagnostic_sums(raw, q_old, targets, labels.advantages, law, logp, reference, schedule.clip)
        self._host_finite(sums, "final semantic diagnostics")
        metrics = _diagnostic_metrics(sums, batch.game_lengths.shape[0], schedule, self.config.bc_reference_coef)
        if self.config.reading_auxiliary:
            metrics.update(_reading_metrics(reading_sums, self.config.reading_aux_coef, finite_check=self._host_finite))
            metrics["actor_final_loss"] += metrics["actor_final_reading_loss"]
        return metrics

    def update(self, state, rng, batch, *, q_old=None, reference_actor=None):  # noqa: C901 - ported verbatim; refactor would risk training semantics
        """Return a COMPLETE candidate only; exceptions never mutate prior state."""
        if isinstance(batch, PagedRawQBatch):
            return self._paged_updater().update(state, rng, batch, q_old=q_old, reference_actor=reference_actor)
        cfg = self.config
        games = batch.game_lengths.shape[0]
        if games != cfg.rollout_games or games % cfg.trajectory_minibatches:
            raise ValueError("rollout must contain the configured number of complete games")
        lengths = np.asarray(batch.game_lengths)  # tiny game metadata only, NOT observations
        if np.any(lengths <= 0):
            raise ValueError("online rollout may not contain padded/incomplete games")
        if cfg.bc_reference_coef and reference_actor is None:
            raise ValueError("positive BC reference coefficient requires explicit frozen BC Actor")
        self.admit(batch)
        q_old = self.old_q(state, batch) if q_old is None else jax.lax.stop_gradient(q_old)
        labels_old = qboost_labels(
            batch.transitions,
            q_old,
            gamma=cfg.gamma,
            trace_lambda=cfg.trace_lambda,
            game_offsets=batch.game_offsets,
            game_lengths=batch.game_lengths,
        )
        targets = jax.lax.stop_gradient(labels_old.targets)
        self._host_finite(targets, "rollout targets")
        schedule = damping(cfg, state.completed_updates + 1)
        original = state
        losses = {"actor_loss": [], "critic_loss": [], "actor_kl": [], "rejected_epochs": 0}
        attempts = _ActorAttemptLog()

        def selected(indices):
            local_lengths = lengths[np.asarray(indices)]
            time = math.ceil(int(local_lengths.max()) / cfg.time_bucket) * cfg.time_bucket
            part = gather_whole_games(batch, indices, time_bucket=time)

            def rows(value):
                return gather_row_values(
                    value,
                    batch.game_offsets,
                    batch.game_lengths,
                    indices,
                    part.game_offsets,
                    part.game_lengths,
                    part.transitions.executed,
                )

            return part, rows

        # Each epoch permutation contains complete games; no shuffled decision rows.
        for _ in range(cfg.actor_epochs):
            epoch_start = state
            epoch_loss_count, epoch_kl_count = len(losses["actor_loss"]), len(losses["actor_kl"])
            rng, shuffle = jax.random.split(rng)
            order = jax.random.permutation(shuffle, games).reshape(cfg.trajectory_minibatches, -1)
            rejected = False
            for indices in order:
                part, rows = selected(indices)
                raw = part.transitions
                labels = self._current_labels(state.params, part, rows(q_old))
                self._host_finite(labels, "current-policy trace")
                actor_mask = labels.actor_mask
                count = int(np.asarray(jnp.sum(actor_mask)))
                if not count:
                    continue
                # Compaction is AFTER rebuilding the full future raw trace.
                capacity = math.ceil(count / cfg.actor_microbatch_size) * cfg.actor_microbatch_size
                idx = jnp.nonzero(actor_mask, size=capacity, fill_value=0)[0]
                weight = (jnp.arange(capacity) < count).astype(jnp.float32)
                obs = jax.tree.map(lambda x: x[idx], raw.observation)
                reference = jnp.ones((capacity, 87), jnp.float32) / 87
                if cfg.bc_reference_coef:
                    logits = self.actor_predict(reference_actor, obs)
                    reference, _ = masked_law(logits, raw.action_mask[idx], cfg.temperature)
                own_advantage = jnp.take_along_axis(labels.advantages, raw.current_player[:, None], axis=-1)[:, 0]
                data = (
                    obs,
                    raw.action_mask[idx],
                    raw.action[idx],
                    raw.behavior_probs[idx],
                    own_advantage[idx],
                    weight,
                    reference,
                )
                reading_counts = None
                if cfg.reading_auxiliary:
                    from .auxhead import reading_target_counts

                    reading_targets = jax.tree.map(jax.lax.stop_gradient, raw.reading_targets)
                    reading_counts = reading_target_counts(reading_targets, actor_mask)
                    data += (jax.tree.map(lambda x: x[idx], reading_targets),)
                group = phase_params(state.params, "actor")
                loss, grads = self.actor_grad(
                    group,
                    data,
                    jnp.float32(indices.shape[0]),
                    schedule.clip,
                    schedule.uniform_alpha,
                    reading_counts if cfg.reading_auxiliary else None,
                    schedule.tempo_keep,
                )
                updated, slots = self.actor_opt_step(
                    state.params, grads, state.actor_opt_state, schedule.actor_multiplier
                )
                self._host_finite((loss, grads, updated, slots), "Actor transaction")
                candidate = state.replace(
                    params=replace_phase(state.params, updated, "actor"),
                    actor_opt_state=slots,
                    actor_steps=state.actor_steps + 1,
                )
                kl = self._kl(candidate.params, part)
                self._host_finite(kl, "Actor KL")
                attempts.record(kl, self.actor_attempt_stats(grads, group, updated), finite_check=self._host_finite)
                if float(np.asarray(kl)) > cfg.target_kl:
                    attempts.reject(kl)
                    rejected = True
                    break
                state = candidate
                losses["actor_loss"].append(float(np.asarray(loss)))
            if not rejected:
                kl = self._kl(state.params, batch)
                self._host_finite(kl, "epoch KL")
                rejected = float(np.asarray(kl)) > cfg.target_kl
                if rejected:
                    attempts.reject(kl, epoch=True)
                losses["actor_kl"].append(float(np.asarray(kl)))
            if rejected:
                state = epoch_start  # includes every Actor slot AND its counter
                del losses["actor_loss"][epoch_loss_count:]
                del losses["actor_kl"][epoch_kl_count:]
                losses["rejected_epochs"] += 1
                break
        # Fixed rollout targets; Actor changes cannot alter Critic supervision.
        for _ in range(cfg.critic_epochs):
            rng, shuffle = jax.random.split(rng)
            order = jax.random.permutation(shuffle, games).reshape(cfg.trajectory_minibatches, -1)
            for indices in order:
                part, rows = selected(indices)
                raw = part.transitions
                # Paper-style .5 * mean(payoff channels) * sum(raw rows) / mean(games).
                denom = jnp.float32(6 * indices.shape[0])
                data = (public_q_input(raw), raw.action, rows(targets), raw.executed.astype(jnp.float32))
                # Pad weights with zero even if the last microbatch borrows real features.
                n = raw.executed.shape[0]
                padding = (-n) % cfg.critic_microbatch_size
                if padding:
                    data = jax.tree.map(lambda x: jnp.concatenate((x, jnp.repeat(x[:1], padding, axis=0))), data)
                    data = data[:-1] + (data[-1].at[n:].set(0.0),)
                group = phase_params(state.params, "critic")
                loss, grads = self.critic_grad(group, data, denom)
                updated, slots = self.critic_opt_step(
                    state.params, grads, state.critic_opt_state, schedule.critic_multiplier
                )
                self._host_finite((loss, grads, updated, slots), "Critic transaction")
                state = state.replace(
                    params=replace_phase(state.params, updated, "critic"),
                    critic_opt_state=slots,
                    critic_steps=state.critic_steps + 1,
                )
                losses["critic_loss"].append(float(np.asarray(loss)))
        state = state.replace(completed_updates=original.completed_updates + 1)
        self._host_finite(state, "complete iteration")
        metrics = {
            name: float(np.mean(value)) if value else 0.0 for name, value in losses.items() if isinstance(value, list)
        }
        metrics.update(
            rejected_epochs=losses["rejected_epochs"],
            actor_steps=int(np.asarray(state.actor_steps - original.actor_steps)),
            critic_steps=int(np.asarray(state.critic_steps - original.critic_steps)),
            actor_lr=float(cfg.actor_lr * schedule.actor_multiplier),
            critic_lr=float(cfg.critic_lr * schedule.critic_multiplier),
            clip=float(schedule.clip),
            uniform_alpha=float(schedule.uniform_alpha),
        )
        metrics.update(attempts.metrics)
        metrics.update(self._diagnostics(state.params, batch, q_old, targets, schedule, reference_actor))
        return VRPOUpdateResult(state, rng, metrics)


# --- from vrpo_paged_update.py ---


def _selected_rows(page, selected):
    raw = page.transitions
    return raw.executed & selected[jnp.clip(page.game_ids, 0, selected.shape[0] - 1)]


def _compact(tree, active):
    """Fixed P output; the dynamic active count never becomes an array dimension."""
    n = active.shape[0]
    indices = jnp.nonzero(active, size=n, fill_value=0)[0]
    return jax.tree.map(lambda x: x[indices], tree), jnp.sum(active, dtype=jnp.int32), indices


def _sparse_indices(active, size):
    """Compact only row indices, preserving the original stable active order."""
    capacity = math.ceil(active.shape[0] / size) * size
    return (jnp.nonzero(active, size=capacity, fill_value=0)[0], jnp.sum(active, dtype=jnp.int32))


def _sparse_predict(apply, params, inputs, active, size, trailing_shape):
    """Gather the original tree only within each executing fixed microbatch."""
    indices, count = _sparse_indices(active, size)
    n = active.shape[0]
    output = jnp.zeros((indices.shape[0],) + trailing_shape, jnp.float32)

    def body(carry):
        start, result = carry
        rows = jax.lax.dynamic_slice_in_dim(indices, start, size)
        block = jax.tree.map(lambda x: x[rows], inputs)
        value = apply(params, block)
        return start + size, jax.lax.dynamic_update_slice_in_dim(result, value, start, axis=0)

    _, output = jax.lax.while_loop(lambda c: c[0] < count, body, (jnp.int32(0), output))
    destinations = jnp.where(jnp.arange(indices.shape[0]) < count, indices, n)
    return jnp.zeros((n,) + trailing_shape, jnp.float32).at[destinations].set(output, mode="drop")


def _sparse_accumulate(loss, params, data, weight_index, size):
    """Keep page-local zero initialization and micro addition order unchanged.

    Only indices/count are page-sized. Borrowed row0 inputs/labels stay safe for
    tokenization, but their zero weight also excludes every reading task mask.
    Differentiate one fixed micro, never through the dynamic while loop.
    """
    indices, count = _sparse_indices(data[weight_index] > 0, size)
    zeros = jax.tree.map(jnp.zeros_like, params)

    def body(carry):
        start, total, grads = carry
        rows = jax.lax.dynamic_slice_in_dim(indices, start, size)
        block = jax.tree.map(lambda x: x[rows], data)
        weights = jnp.where(start + jnp.arange(size) < count, block[weight_index], 0.0)
        block = block[:weight_index] + (weights,) + block[weight_index + 1 :]
        value, grad = jax.value_and_grad(loss)(params, block)
        return start + size, total + value, jax.tree.map(jnp.add, grads, grad)

    _, total, grads = jax.lax.while_loop(lambda c: c[0] < count, body, (jnp.int32(0), jnp.float32(0), zeros))
    return total, grads


def _trace_status(status, ok, labels, carry, page_index):
    """Sticky device flags and first failure in reverse page execution order."""
    finite = finite_tree((labels, carry))
    failed = ~ok | ~finite
    first = (status[2] < 0) & failed
    return jnp.stack(
        (
            status[0] & ok.astype(jnp.int32),
            status[1] & finite.astype(jnp.int32),
            jnp.where(first, page_index, status[2]),
            jnp.where(first, jnp.where(~ok, 1, 2), status[3]),
        )
    )


def page_resident_bytes(batch):
    """Never trust a caller's understated host ledger to bypass admission."""
    if type(batch.resident_bytes) is not int or batch.resident_bytes < 0:
        raise ValueError("paged resident byte ledger must be a nonnegative integer")
    actual = sum(1024 + sum(x.size * x.dtype.itemsize + 256 for x in jax.tree.leaves(page)) for page in batch.pages)
    actual += 1024 + sum(x.size * x.dtype.itemsize + 256 for x in (batch.game_lengths, batch.learner_seats))
    return max(actual, batch.resident_bytes)


def page_scratch_bytes(batch):
    """Conservative fixed scratch, excluding model/optimizer/XLA workspace memory."""
    leaves = jax.tree.leaves(batch.pages[0].transitions)
    row_bytes = sum(math.prod(x.shape[1:]) * x.dtype.itemsize for x in leaves)
    return fixed_page_scratch_bytes(batch.page_rows, int(row_bytes), batch.num_games, len(leaves))


class PagedVRPOUpdater:
    """Own page-sized kernels only; closures capture net/config, never this owner."""

    def __init__(self, network, config):
        self.network, self.config = network, config
        cfg, net = config, network

        def actor_apply(p, x):
            return net.apply(p, x, method=net.get_action_logits)

        def q_apply(p, x):
            return net.apply(p, x, method=net.get_action_q)

        self.actor_predict = bounded_jit(max_entries=8)(
            lambda p, x, m: _sparse_predict(actor_apply, p, x, m, cfg.inference_microbatch_size, (87,))
        )
        self.q_predict = bounded_jit(max_entries=8)(
            lambda p, x, m: _sparse_predict(q_apply, p, x, m, cfg.inference_microbatch_size, (3, 87))
        )

        def actor_grad(p, data, games, clip, alpha, reading_counts=None, tempo_keep=None):
            def objective(a, b):
                return _actor_loss(
                    a,
                    net,
                    b,
                    games,
                    cfg.temperature,
                    clip,
                    alpha,
                    cfg.bc_reference_coef,
                    cfg.reading_aux_coef if cfg.reading_auxiliary else 0.0,
                    reading_counts,
                    tempo_keep if cfg.tempo_anchor else None,
                )

            return _sparse_accumulate(objective, p, data, 5, cfg.actor_microbatch_size)

        def critic_grad(p, data, denom):
            return _sparse_accumulate(
                lambda a, b: _critic_loss(a, net, b, denom), p, data, 3, cfg.critic_microbatch_size
            )

        self.actor_grad = bounded_jit(max_entries=8)(actor_grad)
        self.critic_grad = bounded_jit(max_entries=8)(critic_grad)
        self.diagnostic_sums = bounded_jit(max_entries=8)(_diagnostic_sums)
        self.reading_diagnostic_sums = bounded_jit(max_entries=8)(
            lambda actor, raw: _reading_diagnostic_sums(actor, net, raw, cfg.actor_microbatch_size)
        )
        self.joint_reading_diagnostics = bounded_jit(max_entries=8)(
            lambda actor, raw: _joint_reading_diagnostics(actor, net, raw, cfg.actor_microbatch_size)
        )
        # More than eight distinct finite-only tree signatures occur per update.
        self.finite_summary = bounded_jit(max_entries=16)(finite_tree)

        def reading_counts(page, selected):
            from .auxhead import reading_target_counts

            raw = page.transitions
            active = _selected_rows(page, selected) & raw.is_learner & raw.is_decision
            return reading_target_counts(raw.reading_targets, active)

        self.reading_counts = bounded_jit(max_entries=8)(reading_counts)

        def optimizer_step(params, grads, slots, multiplier, phase):
            group = phase_params(params, phase)
            tx = make_phase_optimizer(params, cfg, phase)
            updates, next_slots = tx.update(grads, slots, group)
            updates = jax.tree.map(lambda x: x * multiplier, updates)
            return optax.apply_updates(group, updates), next_slots

        self.actor_opt_step = bounded_jit(max_entries=8)(lambda p, g, s, m: optimizer_step(p, g, s, m, "actor"))
        self.critic_opt_step = bounded_jit(max_entries=8)(lambda p, g, s, m: optimizer_step(p, g, s, m, "critic"))
        self.actor_attempt_stats = bounded_jit(max_entries=8)(_actor_attempt_stats)

        def current_law(p, page, selected):
            raw = page.transitions
            active = _selected_rows(page, selected) & raw.is_learner & raw.is_decision
            logits = _sparse_predict(actor_apply, p, raw.observation, active, cfg.inference_microbatch_size, (87,))
            law, _ = masked_law(logits, raw.action_mask, cfg.temperature)
            return jnp.where(active[:, None], law, raw.behavior_probs)

        self.current_law = bounded_jit(max_entries=8)(current_law)

        def kl_page(p, page, selected):
            raw = page.transitions
            active = _selected_rows(page, selected) & raw.is_learner & raw.is_decision
            logits = _sparse_predict(actor_apply, p, raw.observation, active, cfg.inference_microbatch_size, (87,))
            _, logp = masked_law(logits, raw.action_mask, cfg.temperature)
            old = raw.behavior_probs
            row = jnp.sum(old * (jnp.log(jnp.maximum(old, 1e-30)) - logp), axis=-1)
            return jnp.sum(jnp.where(active, row, 0.0)), jnp.sum(active, dtype=jnp.int32)

        self.kl_page = bounded_jit(max_entries=8)(kl_page)

        def actor_data(page, labels, selected, reference):
            raw = page.transitions
            active = _selected_rows(page, selected) & labels.actor_mask
            advantage = jnp.take_along_axis(labels.advantages, raw.current_player[:, None], -1)[:, 0]
            data = (
                raw.observation,
                raw.action_mask,
                raw.action,
                raw.behavior_probs,
                advantage,
                active.astype(jnp.float32),
                reference,
            )
            if cfg.reading_auxiliary:
                if raw.reading_targets is None:
                    raise ValueError("V5 Actor page missing reading target sidecar")
                data += (jax.tree.map(jax.lax.stop_gradient, raw.reading_targets),)
            return data

        self.actor_data = bounded_jit(max_entries=8)(actor_data)

        def critic_data(page, target, selected):
            return (
                public_q_input(page.transitions),
                page.transitions.action,
                target,
                _selected_rows(page, selected).astype(jnp.float32),
            )

        self.critic_data = bounded_jit(max_entries=8)(critic_data)

        def actor_page_execute(
            p,
            page,
            q_old,
            carry,
            selected,
            reference_actor,
            games,
            clip,
            alpha,
            tempo_keep,
            counts,
            total,
            grads,
            count,
            status,
            page_index,
        ):
            law = current_law(p, page, selected)
            labels, carry, ok = qboost_page_device(
                page,
                q_old,
                carry,
                probabilities=law,
                selected_games=selected,
                gamma=cfg.gamma,
                trace_lambda=cfg.trace_lambda,
            )
            labels = jax.tree.map(jax.lax.stop_gradient, labels)
            status = _trace_status(status, ok, labels, carry, page_index)
            reference = jnp.ones_like(law) / 87
            if cfg.bc_reference_coef:
                active = _selected_rows(page, selected) & labels.actor_mask
                logits = _sparse_predict(
                    actor_apply,
                    reference_actor,
                    page.transitions.observation,
                    active,
                    cfg.inference_microbatch_size,
                    (87,),
                )
                reference, _ = masked_law(logits, page.transitions.action_mask, cfg.temperature)
            data = actor_data(page, labels, selected, reference)
            # Do NOT seed the micro carry with incoming cross-page gradients:
            # incoming + (zero + micro0 + micro1 ...) is the original grouping.
            value, page_grad = actor_grad(p, data, games, clip, alpha, counts, tempo_keep)
            return (total + value, jax.tree.map(jnp.add, grads, page_grad), count + jnp.sum(data[5]), carry, status)

        def critic_page_execute(p, page, target, selected, denom, total, grads, count):
            data = critic_data(page, target, selected)
            value, page_grad = critic_grad(p, data, denom)
            return (total + value, jax.tree.map(jnp.add, grads, page_grad), count + jnp.sum(data[3]))

        self.actor_page_execute = bounded_jit(max_entries=8)(actor_page_execute)
        self.critic_page_execute = bounded_jit(max_entries=8)(critic_page_execute)
        self.first_bad_page_trace = None

        def squared_error(p, page, target):
            raw = page.transitions
            q = _sparse_predict(q_apply, p, public_q_input(raw), raw.executed, cfg.inference_microbatch_size, (3, 87))
            taken = jnp.take_along_axis(q, raw.action[:, None, None], -1)[..., 0]
            squared = jnp.where(raw.executed[:, None], jnp.square(taken - target), 0.0)
            return jnp.sum(squared), jnp.all(jnp.isfinite(q)) & jnp.all(jnp.isfinite(squared))

        self.squared_error = bounded_jit(max_entries=8)(squared_error)
        self._kernels = (
            self.actor_predict,
            self.q_predict,
            self.actor_grad,
            self.critic_grad,
            self.actor_opt_step,
            self.critic_opt_step,
            self.current_law,
            self.kl_page,
            self.actor_data,
            self.critic_data,
            self.squared_error,
            self.diagnostic_sums,
            self.actor_attempt_stats,
            self.reading_counts,
            self.reading_diagnostic_sums,
            self.joint_reading_diagnostics,
            self.finite_summary,
            self.actor_page_execute,
            self.critic_page_execute,
        )

    def _host_finite(self, tree, stage):
        if self.config.reading_auxiliary:
            _host_finite(tree, stage, summary=self.finite_summary)
        else:
            _host_finite(tree, stage)

    def cache_statistics(self):
        names = (
            "actor_predict",
            "q_predict",
            "actor_grad",
            "critic_grad",
            "actor_opt_step",
            "critic_opt_step",
            "current_law",
            "kl_page",
            "actor_data",
            "critic_data",
            "squared_error",
            "actor_attempt_stats",
        )
        if self.config.reading_auxiliary:
            names += (
                "reading_counts",
                "reading_diagnostic_sums",
                "joint_reading_diagnostics",
                "finite_summary",
                "actor_page_execute",
                "critic_page_execute",
            )
        return {name: getattr(self, name).cache_info()._asdict() for name in names}

    def close(self):
        for kernel in self._kernels:
            kernel.clear_cache()
        self.first_bad_page_trace = None

    def admit(self, batch):
        if not isinstance(batch, PagedRawQBatch):
            raise TypeError("paged updater requires PagedRawQBatch")
        if batch.page_rows != self.config.page_rows:
            raise ValueError("batch page rows differ from configured execution layout")
        if batch.game_lengths.shape != (self.config.rollout_games,):
            raise ValueError("online pages require configured complete-game count")
        validate_paged_batch(batch, max_game_steps=self.config.max_game_steps)
        # Q-old and immutable sampled-action targets persist for all pages; laws,
        # current traces, compaction and gradient data need only one fixed page.
        charge = (
            page_resident_bytes(batch)
            + len(batch.pages) * batch.page_rows * (3 * 87 * 4 + 3 * 4)
            + page_scratch_bytes(batch)
        )
        if self.config.reading_auxiliary:
            if any(page.transitions.reading_targets is None for page in batch.pages):
                raise ValueError("V5 pages missing reading target sidecars")
            charge += batch.page_rows * (2 * 214 * 4 + 40) + 4096
        if charge > self.config.max_bytes:
            raise MemoryError("VRPO pages/Q-cache/targets/fixed scratch exceed byte budget")
        return charge

    def old_q(self, state, batch):
        self.admit(batch)
        critic = phase_params(state.params, "critic")
        values = []
        for _, page in batch.iter_pages():
            q = self.q_predict(critic, public_q_input(page.transitions), page.transitions.executed)
            self._host_finite(q, "Q-old page")
            values.append(jax.lax.stop_gradient(q))
        return tuple(values)  # HOST collection, never a kernel argument.

    def _check_q(self, batch, q_old):
        if not isinstance(q_old, (tuple, list)) or len(q_old) != len(batch.pages):
            raise ValueError("Q-old must be one fixed array per page")
        for q in q_old:
            if not hasattr(q, "shape") or q.shape != (batch.page_rows, 3, 87):
                raise ValueError("Q-old page has invalid shape")
            if q.dtype != jnp.dtype(jnp.float32):
                raise ValueError("Q-old pages must have exact float32 dtype; no implicit conversion")
        values = []
        for q in q_old:
            q = jax.lax.stop_gradient(jnp.asarray(q))
            self._host_finite(q, "Q-old page")
            values.append(q)
        return tuple(values)

    def targets(self, batch, q_old=None, *, monte_carlo=False):
        """Reverse full games with per-game carries; never concatenate pages."""
        carry = init_page_trace_carry(batch.num_games, jnp.float32)
        selected = jnp.ones((batch.num_games,), jnp.bool_)
        zero = jnp.zeros((batch.page_rows, 3, 87), jnp.float32) if monte_carlo else None
        targets = [None] * len(batch.pages)
        for index, page in batch.iter_pages(reverse=True):
            q = zero if monte_carlo else q_old[index]
            labels, carry = qboost_page(
                page,
                q,
                carry,
                selected_games=selected,
                gamma=1.0 if monte_carlo else self.config.gamma,
                trace_lambda=1.0 if monte_carlo else self.config.trace_lambda,
            )
            target = jax.lax.stop_gradient(labels.targets)
            self._host_finite((target, carry), "Monte Carlo targets" if monte_carlo else "rollout targets")
            targets[index] = target
        return tuple(targets)

    @staticmethod
    def selection(batch, indices=None):
        if indices is None:
            return jnp.ones((batch.num_games,), jnp.bool_)
        host = np.asarray(indices, np.int32)
        mask = np.zeros((batch.num_games,), bool)
        mask[host] = True
        return jnp.asarray(mask)

    def _kl(self, params, batch, indices=None):
        selected = self.selection(batch, indices)
        total, count = jnp.float32(0), jnp.int32(0)
        actor = phase_params(params, "actor")
        for _, page in batch.iter_pages(game_indices=indices):
            value, rows = self.kl_page(actor, page, selected)
            total, count = total + value, count + rows
        return total / jnp.maximum(count, 1)

    def _check_actor_minibatch(self, status, total, grads, batch, q_old, actor, indices):
        # Exactly one small D2H at the complete-MB boundary, before optimizer.
        summary = np.asarray(jnp.concatenate((status, self.finite_summary((total, grads)).astype(jnp.int32)[None])))
        if int(summary[2]) >= 0:
            page_index, stage = map(int, summary[2:4])
            if self.first_bad_page_trace is None:
                # Retain existing immutable device/host references, not copies.
                # Replaying an incoming carry also requires the caller-held full
                # batch and frozen Q-old suffix (or its frozen Critic params).
                self.first_bad_page_trace = dict(
                    page_index=page_index,
                    stage=stage,
                    page=batch.pages[page_index],
                    q_old=q_old[page_index],
                    actor=actor,
                    selected_indices=indices,
                )
            evidence = f" (first bad page={page_index}, stage={stage})"
            if stage == 1:
                raise ValueError("invalid raw Q page trace: game metadata, Q or probability support" + evidence)
            raise FloatingPointError(
                "VRPO nonfinite current-policy page trace; prior complete boundary retained" + evidence
            )
        if not bool(summary[4]):
            raise FloatingPointError("VRPO nonfinite Actor transaction; prior complete boundary retained")

    def actor_minibatch(self, params, batch, q_old, indices, schedule, reference_actor=None):
        selected = self.selection(batch, indices)
        actor = phase_params(params, "actor")
        grads = jax.tree.map(jnp.zeros_like, actor)
        total, count = jnp.float32(0), jnp.float32(0)
        carry = init_page_trace_carry(batch.num_games, jnp.float32)
        reading_counts = None
        if self.config.reading_auxiliary:
            reading_counts = jnp.zeros(5, jnp.float32)
            for _, page in batch.iter_pages(game_indices=indices):
                reading_counts += self.reading_counts(page, selected)
            status = jnp.array([1, 1, -1, 0], jnp.int32)
            for index, page in batch.iter_pages(reverse=True, game_indices=indices):
                total, grads, count, carry, status = self.actor_page_execute(
                    actor,
                    page,
                    q_old[index],
                    carry,
                    selected,
                    reference_actor,
                    jnp.float32(len(indices)),
                    schedule.clip,
                    schedule.uniform_alpha,
                    schedule.tempo_keep,
                    reading_counts,
                    total,
                    grads,
                    count,
                    status,
                    jnp.int32(index),
                )
            self._check_actor_minibatch(status, total, grads, batch, q_old, actor, indices)
            return total, grads, count
        for index, page in batch.iter_pages(reverse=True, game_indices=indices):
            law = self.current_law(actor, page, selected)
            labels, carry = qboost_page(
                page,
                q_old[index],
                carry,
                probabilities=law,
                selected_games=selected,
                gamma=self.config.gamma,
                trace_lambda=self.config.trace_lambda,
            )
            labels = jax.tree.map(jax.lax.stop_gradient, labels)
            self._host_finite((labels, carry), "current-policy page trace")
            reference = jnp.ones((batch.page_rows, 87), jnp.float32) / 87
            if self.config.bc_reference_coef:
                active = _selected_rows(page, selected) & labels.actor_mask
                logits = self.actor_predict(reference_actor, page.transitions.observation, active)
                reference, _ = masked_law(logits, page.transitions.action_mask, self.config.temperature)
            data = self.actor_data(page, labels, selected, reference)
            value, grad = self.actor_grad(
                actor,
                data,
                jnp.float32(len(indices)),
                schedule.clip,
                schedule.uniform_alpha,
                reading_counts if self.config.reading_auxiliary else None,
                schedule.tempo_keep,
            )
            total, grads = total + value, jax.tree.map(jnp.add, grads, grad)
            count = count + jnp.sum(data[5])
        return total, grads, count

    def critic_minibatch(self, params, batch, targets, indices):
        selected = self.selection(batch, indices)
        denominator = jnp.float32(6 * len(indices))
        critic = phase_params(params, "critic")
        total, grads = jnp.float32(0), jax.tree.map(jnp.zeros_like, critic)
        if self.config.reading_auxiliary:
            count = jnp.float32(0)
            for index, page in batch.iter_pages(game_indices=indices):
                total, grads, count = self.critic_page_execute(
                    critic, page, targets[index], selected, denominator, total, grads, count
                )
            self._host_finite((total, grads), "Critic transaction")
            return total, grads
        for index, page in batch.iter_pages(game_indices=indices):
            data = self.critic_data(page, targets[index], selected)
            value, grad = self.critic_grad(critic, data, denominator)
            total, grads = total + value, jax.tree.map(jnp.add, grads, grad)
        return total, grads

    def _diagnostics(self, params, batch, q_old, targets, schedule, reference_actor):
        """One reverse page pass; only constant-size scalar aggregates persist."""
        selected = self.selection(batch)
        actor = phase_params(params, "actor")
        carry = init_page_trace_carry(batch.num_games, jnp.float32)
        sums = None
        reading_sums = jnp.zeros((2, 5), jnp.float32)
        for index, page in batch.iter_pages(reverse=True):
            raw = page.transitions
            active = raw.executed & raw.is_learner & raw.is_decision
            if self.config.reading_auxiliary:
                logits, reading_part = self.joint_reading_diagnostics(actor, raw)
                reading_sums += reading_part
            else:
                logits = self.actor_predict(actor, raw.observation, active)
            law, logp = masked_law(logits, raw.action_mask, self.config.temperature)
            law = jnp.where(active[:, None], law, raw.behavior_probs)
            labels, carry = qboost_page(
                page,
                q_old[index],
                carry,
                probabilities=law,
                selected_games=selected,
                gamma=self.config.gamma,
                trace_lambda=self.config.trace_lambda,
            )
            reference = jnp.zeros_like(law)
            if self.config.bc_reference_coef:
                reference, _ = masked_law(
                    self.actor_predict(reference_actor, raw.observation, active),
                    raw.action_mask,
                    self.config.temperature,
                )
            part = self.diagnostic_sums(
                raw, q_old[index], targets[index], labels.advantages, law, logp, reference, schedule.clip
            )
            sums = _merge_diagnostic_sums(sums, part)
        self._host_finite(sums, "final semantic diagnostics")
        metrics = _diagnostic_metrics(sums, batch.num_games, schedule, self.config.bc_reference_coef)
        if self.config.reading_auxiliary:
            metrics.update(_reading_metrics(reading_sums, self.config.reading_aux_coef, finite_check=self._host_finite))
            metrics["actor_final_loss"] += metrics["actor_final_reading_loss"]
        return metrics

    def update(self, state, rng, batch, *, q_old=None, reference_actor=None):
        cfg = self.config
        if batch.num_games != cfg.rollout_games or batch.num_games % cfg.trajectory_minibatches:
            raise ValueError("rollout must contain configured number of complete games")
        if cfg.bc_reference_coef and reference_actor is None:
            raise ValueError("positive BC reference coefficient requires explicit frozen BC Actor")
        self.admit(batch)  # BEFORE any optimizer transaction/cache allocation.
        q_old = self.old_q(state, batch) if q_old is None else self._check_q(batch, q_old)
        targets = self.targets(batch, q_old)
        schedule = damping(cfg, state.completed_updates + 1)
        original = state
        losses = {"actor_loss": [], "critic_loss": [], "actor_kl": [], "rejected_epochs": 0}
        attempts = _ActorAttemptLog()
        for _ in range(cfg.actor_epochs):
            epoch_start = state
            epoch_loss_count, epoch_kl_count = len(losses["actor_loss"]), len(losses["actor_kl"])
            rng, shuffle = jax.random.split(rng)
            order = np.asarray(jax.random.permutation(shuffle, batch.num_games)).reshape(cfg.trajectory_minibatches, -1)
            rejected = False
            for indices in order:
                loss, grads, count = self.actor_minibatch(
                    state.params, batch, q_old, indices, schedule, reference_actor
                )
                if not int(np.asarray(count)):
                    continue
                updated, slots = self.actor_opt_step(
                    state.params, grads, state.actor_opt_state, schedule.actor_multiplier
                )
                self._host_finite((loss, grads, updated, slots), "Actor transaction")
                candidate = state.replace(
                    params=replace_phase(state.params, updated, "actor"),
                    actor_opt_state=slots,
                    actor_steps=state.actor_steps + 1,
                )
                kl = self._kl(candidate.params, batch, indices)
                self._host_finite(kl, "Actor KL")
                attempts.record(
                    kl,
                    self.actor_attempt_stats(grads, phase_params(state.params, "actor"), updated),
                    finite_check=self._host_finite,
                )
                if float(np.asarray(kl)) > cfg.target_kl:
                    attempts.reject(kl)
                    rejected = True
                    break
                state = candidate
                losses["actor_loss"].append(float(np.asarray(loss)))
            if not rejected:
                kl = self._kl(state.params, batch)
                self._host_finite(kl, "epoch KL")
                rejected = float(np.asarray(kl)) > cfg.target_kl
                if rejected:
                    attempts.reject(kl, epoch=True)
                losses["actor_kl"].append(float(np.asarray(kl)))
            if rejected:
                state = epoch_start
                del losses["actor_loss"][epoch_loss_count:]
                del losses["actor_kl"][epoch_kl_count:]
                losses["rejected_epochs"] += 1
                break
        for _ in range(cfg.critic_epochs):
            rng, shuffle = jax.random.split(rng)
            order = np.asarray(jax.random.permutation(shuffle, batch.num_games)).reshape(cfg.trajectory_minibatches, -1)
            for indices in order:
                loss, grads = self.critic_minibatch(state.params, batch, targets, indices)
                updated, slots = self.critic_opt_step(
                    state.params, grads, state.critic_opt_state, schedule.critic_multiplier
                )
                self._host_finite((loss, grads, updated, slots), "Critic transaction")
                state = state.replace(
                    params=replace_phase(state.params, updated, "critic"),
                    critic_opt_state=slots,
                    critic_steps=state.critic_steps + 1,
                )
                losses["critic_loss"].append(float(np.asarray(loss)))
        state = state.replace(completed_updates=original.completed_updates + 1)
        self._host_finite(state, "complete iteration")
        metrics = {
            name: float(np.mean(value)) if value else 0.0 for name, value in losses.items() if isinstance(value, list)
        }
        metrics.update(
            rejected_epochs=losses["rejected_epochs"],
            actor_steps=int(np.asarray(state.actor_steps - original.actor_steps)),
            critic_steps=int(np.asarray(state.critic_steps - original.critic_steps)),
            actor_lr=float(cfg.actor_lr * schedule.actor_multiplier),
            critic_lr=float(cfg.critic_lr * schedule.critic_multiplier),
            clip=float(schedule.clip),
            uniform_alpha=float(schedule.uniform_alpha),
        )
        metrics.update(attempts.metrics)
        metrics.update(self._diagnostics(state.params, batch, q_old, targets, schedule, reference_actor))
        return VRPOUpdateResult(state, rng, metrics)


# Import-time cycle: ``selfplay`` re-uses the batch primitives above, so these
# lazily forward to it instead of importing it at module scope.
def _forward(name):
    def call(*args, **kwargs):
        from . import selfplay

        return getattr(selfplay, name)(*args, **kwargs)

    call.__name__ = name
    return call


gather_whole_games = _forward("gather_whole_games")
init_page_trace_carry = _forward("init_page_trace_carry")
qboost_page = _forward("qboost_page")
qboost_page_device = _forward("qboost_page_device")
validate_paged_batch = _forward("validate_paged_batch")
