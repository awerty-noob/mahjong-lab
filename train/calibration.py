"""Q calibration against a frozen Actor."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from train.configs import PagedRawQBatch, RawQBatch, VRPOConfig, VRPOTrainingState
from train.optimizer import vrpo_optimizer_metadata
from train.selfplay import (
    _batch_summary,
    _validate_batch_shape,
    gather_whole_games,
    init_page_trace_carry,
    qboost_page,
    validate_paged_batch,
)
from train.update import (
    PagedVRPOUpdater,
    VRPOUpdater,
    create_training_state,
    finite_tree,
    gather_row_values,
    page_resident_bytes,
    page_scratch_bytes,
    phase_params,
    public_q_input,
    qboost_labels,
    replace_phase,
)

# --- from vrpo_calibration.py ---


@dataclass(frozen=True)
class CalibrationResult:
    state: VRPOTrainingState
    rng: object
    report: dict


class CalibrationFailure(ValueError):
    """Fail closed: no online state or publishable artifact accompanies failure."""

    def __init__(self, report):
        self.report = dict(report)
        super().__init__(f"Q calibration failed: {report.get('reason', 'held-out acceptance')}")


def actor_parameter_fingerprint(params):
    """Byte-exact Actor-only SHA256; this one warmup parameter D2H is deliberate."""
    digest = hashlib.sha256()
    for path, leaf in jax.tree.flatten_with_path(phase_params(params, "actor"))[0]:
        array = np.asarray(leaf)
        for item in (str(path), str(array.dtype), str(array.shape)):
            encoded = item.encode("utf8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _finite(tree, stage):
    if not bool(np.asarray(finite_tree(tree))):
        raise ValueError(f"nonfinite {stage}")


def _game_ids(ids, expected, name):
    if ids is None or isinstance(ids, (str, bytes, Mapping)):
        raise ValueError(f"{name} requires explicit whole-game IDs")
    if isinstance(ids, jax.Array):
        ids = np.asarray(ids)  # IDs are explicitly small game metadata.
    ids = tuple(ids)
    if len(ids) != expected:
        raise ValueError(f"{name} must contain exactly {expected} game IDs")
    normalized = []
    for value in ids:
        if isinstance(value, np.integer) and not isinstance(value, np.bool_):
            value = int(value)
        if type(value) not in (str, int) or (isinstance(value, str) and not value.strip()):
            raise ValueError(f"{name} has a missing/invalid game ID")
        normalized.append(value)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} repeats a whole-game ID")
    return tuple(normalized)


def _validate_partition(batch, expected, config, name):
    if not isinstance(batch, RawQBatch):
        raise ValueError(f"{name} requires an explicit complete-game RawQBatch")
    _validate_batch_shape(batch)
    lengths = np.asarray(batch.game_lengths)  # O(games), never observation payloads
    if lengths.shape != (expected,) or np.any(lengths <= 0):
        raise ValueError(f"{name} requires exactly {expected} nonempty complete games")
    if np.any(lengths > config.max_game_steps):
        raise ValueError(f"{name} exceeds the whole-game step safety limit")
    if int(np.asarray(_batch_summary(batch))[0]) != 1:
        raise ValueError(f"{name} has invalid, missing or truncated complete games")
    return lengths


def _admit(train, heldout, config, train_lengths, heldout_lengths):
    """Conservative live input/MC scratch/minibatch admission, not GPU workspace."""
    resident = sum(int(x.nbytes) for b in (train, heldout) for x in jax.tree.leaves(b))
    rows = [b.transitions.executed.shape[0] for b in (train, heldout)]
    # Fixed MC targets for both partitions, plus Q-zero/trace construction scratch.
    resident += sum(rows) * 3 * 4
    scratch = max(rows) * (3 * 87 * 4 + 3 * 4 * 8)
    selected = 0
    for batch, lengths in ((train, train_lengths), (heldout, heldout_lengths)):
        width = min(config.games_per_minibatch, len(lengths))
        time = math.ceil(int(lengths.max()) / config.time_bucket) * config.time_bucket
        n = width * time + config.critic_microbatch_size
        row_bytes = sum(math.prod(x.shape[1:]) * x.dtype.itemsize for x in jax.tree.leaves(batch.transitions))
        selected = max(selected, n * (3 * row_bytes + 3 * 87 * 4 + 3 * 4 * 8))
    if resident + scratch + selected > config.max_bytes:
        raise ValueError("calibration whole-game/MC memory admission exceeds max_bytes")


def _mc_targets(batch):
    raw = batch.transitions
    # Q=0 and gamma=lambda=1 is the exact reverse raw-reward MC sum. Only the
    # sampled action receives this target: no labels for the other 86 actions.
    labels = qboost_labels(
        raw,
        jnp.zeros(raw.executed.shape + (3, 87), jnp.float32),
        gamma=1.0,
        trace_lambda=1.0,
        game_offsets=batch.game_offsets,
        game_lengths=batch.game_lengths,
    )
    targets = jax.lax.stop_gradient(labels.targets)
    _finite(targets, "Monte Carlo targets")
    return targets


def _parts(batch, targets, lengths, order, config):
    """Only whole games are shuffled; fixed game count with bounded rounded T."""
    width = min(config.games_per_minibatch, len(lengths))
    for start in range(0, len(lengths), width):
        indices = np.asarray(order[start : start + width], np.int32)
        time = math.ceil(int(lengths[indices].max()) / config.time_bucket) * config.time_bucket
        indices = jnp.asarray(np.pad(indices, (0, width - len(indices)), constant_values=-1))
        part = gather_whole_games(batch, indices, time_bucket=time)
        values = gather_row_values(
            targets,
            batch.game_offsets,
            batch.game_lengths,
            indices,
            part.game_offsets,
            part.game_lengths,
            part.transitions.executed,
        )
        yield part, values


def _heldout_mse(updater, params, batch, targets, lengths, config):
    error, count = 0.0, 0
    for part, target in _parts(batch, targets, lengths, np.arange(len(lengths)), config):
        raw = part.transitions
        q = updater.q_predict(phase_params(params, "critic"), public_q_input(raw))
        taken = jnp.take_along_axis(q, raw.action[:, None, None], axis=-1)[..., 0]
        squared = jnp.where(raw.executed[:, None], jnp.square(taken - target), 0.0)
        _finite((q, squared), "held-out sampled-action predictions")
        error += float(np.asarray(jnp.sum(squared)))
        count += int(np.asarray(jnp.sum(raw.executed))) * 3
    mse = error / count
    if not math.isfinite(mse):
        raise ValueError("nonfinite held-out MSE")
    return mse


def calibrate_q(network, config, state, rng, train_batch, heldout_batch, *, train_game_ids, heldout_game_ids):
    """Fit Critic only, accept strictly against zero AND initial held-out MSE.

    Input optimizer slots/counters are never used or advanced. Calibration owns
    separate fresh slots; success returns *another* fresh pair of ONLINE slots,
    all counters zero (including completed_updates), not a resume boundary.
    """
    if isinstance(train_batch, PagedRawQBatch) or isinstance(heldout_batch, PagedRawQBatch):
        from .calibration import calibrate_paged_q

        return calibrate_paged_q(
            network,
            config,
            state,
            rng,
            train_batch,
            heldout_batch,
            train_game_ids=train_game_ids,
            heldout_game_ids=heldout_game_ids,
        )
    report = dict(
        accepted=False,
        phase="original_bc_q_calibration",
        gamma=1.0,
        trace_lambda=1.0,
        target_source="all_raw_environment_rewards",
        payoff_seats="absolute3",
        regression="sampled_action_only",
        advantage_normalization=False,
        critic_auxiliary=False,
        replay=False,
        epochs_completed=0,
        best_epoch=0,
        calibration_critic_steps=0,
        online_completed_updates=0,
        optimizer_lineage="fresh_optimizer_slots",
        resume_semantics="initialization_only",
        epoch_heldout_mse=[],
    )
    updater = None
    try:
        if not isinstance(config, VRPOConfig):
            raise ValueError("config must be a validated VRPOConfig")
        config.__post_init__()
        report.update(
            training_reduction="half_mean3_sum_raw_rows_mean_games_v2",
            heldout_reduction="mean_raw_rows_mean3_unhalved",
            optimizer=vrpo_optimizer_metadata(config),
        )
        if not isinstance(state, VRPOTrainingState):
            raise ValueError("state must be VRPOTrainingState")
        train_ids = _game_ids(train_game_ids, config.calibration_train_games, "train_game_ids")
        heldout_ids = _game_ids(heldout_game_ids, config.calibration_heldout_games, "heldout_game_ids")
        if set(train_ids) & set(heldout_ids):
            raise ValueError("train and held-out whole-game IDs overlap")
        train_lengths = _validate_partition(train_batch, len(train_ids), config, "train")
        heldout_lengths = _validate_partition(heldout_batch, len(heldout_ids), config, "heldout")
        _admit(train_batch, heldout_batch, config, train_lengths, heldout_lengths)
        _finite(state.params, "initial parameters")
        report.update(
            train_game_ids=list(train_ids),
            heldout_game_ids=list(heldout_ids),
            train_games=len(train_ids),
            heldout_games=len(heldout_ids),
            train_raw_rows=int(train_lengths.sum()),
            heldout_raw_rows=int(heldout_lengths.sum()),
            critic_microbatch_size=config.critic_microbatch_size,
            min_relative_improvement=config.calibration_min_relative_improvement,
            max_epochs=config.calibration_max_epochs,
            actor_sha256_before=actor_parameter_fingerprint(state.params),
        )
        train_targets, heldout_targets = _mc_targets(train_batch), _mc_targets(heldout_batch)
        mask = heldout_batch.transitions.executed[:, None]
        zero_mse = float(
            np.asarray(jnp.sum(jnp.where(mask, jnp.square(heldout_targets), 0.0)) / (heldout_lengths.sum() * 3))
        )
        if not math.isfinite(zero_mse):
            raise ValueError("nonfinite zero baseline")
        updater = VRPOUpdater(network, config)
        initial_mse = _heldout_mse(updater, state.params, heldout_batch, heldout_targets, heldout_lengths, config)
        report.update(zero_baseline_mse=zero_mse, initial_critic_mse=initial_mse, best_heldout_mse=initial_mse)
        calibration, _, _ = create_training_state(state.params, config)
        for epoch in range(1, config.calibration_max_epochs + 1):
            rng, shuffle = jax.random.split(rng)
            order = np.asarray(jax.random.permutation(shuffle, len(train_lengths)))  # game metadata
            for part, target in _parts(train_batch, train_targets, train_lengths, order, config):
                raw = part.transitions
                # The last fixed-width arena may include -1 padded game IDs.
                # They have zero lengths and are NOT games in the paper mean.
                denominator = (6 * jnp.sum(part.game_lengths > 0)).astype(jnp.float32)
                data = (public_q_input(raw), raw.action, target, raw.executed.astype(jnp.float32))
                n = raw.executed.shape[0]
                padding = (-n) % config.critic_microbatch_size
                if padding:
                    data = jax.tree.map(lambda x: jnp.concatenate((x, jnp.repeat(x[:1], padding, axis=0))), data)
                    data = data[:-1] + (data[-1].at[n:].set(0.0),)
                loss, grads = updater.critic_grad(phase_params(calibration.params, "critic"), data, denominator)
                updated, slots = updater.critic_opt_step(
                    calibration.params, grads, calibration.critic_opt_state, jnp.float32(1.0)
                )
                _finite((loss, grads, updated, slots), "calibration Critic transaction")
                calibration = calibration.replace(
                    params=replace_phase(calibration.params, updated, "critic"),
                    critic_opt_state=slots,
                    critic_steps=calibration.critic_steps + 1,
                )
                report["calibration_critic_steps"] += 1
            mse = _heldout_mse(updater, calibration.params, heldout_batch, heldout_targets, heldout_lengths, config)
            report["epochs_completed"] = epoch
            report["epoch_heldout_mse"].append(mse)
        # Select a pre-registered final epoch instead of taking the minimum over
        # the same held-out set used for acceptance.  The latter is optimistically
        # biased and makes the zero/initial comparisons look better than they are.
        best_params = calibration.params
        best_mse = report["epoch_heldout_mse"][-1]
        report.update(best_epoch=config.calibration_max_epochs, best_heldout_mse=best_mse)
        report["actor_sha256_after"] = actor_parameter_fingerprint(calibration.params)
        if report["actor_sha256_after"] != report["actor_sha256_before"]:
            raise ValueError("original BC Actor fingerprint changed")
        relative = config.calibration_min_relative_improvement
        # The explicit strict comparisons matter when relative=0 or baseline=0.
        report["beats_zero_baseline"] = bool(best_mse < zero_mse and best_mse < zero_mse * (1.0 - relative))
        report["beats_initial_critic"] = bool(best_mse < initial_mse and best_mse < initial_mse * (1.0 - relative))
        if not report["beats_zero_baseline"] or not report["beats_initial_critic"]:
            raise ValueError("held-out MSE must strictly beat both zero baseline and initial Critic")
        online, _, _ = create_training_state(best_params, config)
        _finite(online, "fresh online state")
        report.update(
            accepted=True, reason="passed", fresh_optimizer_slots=True, online_actor_steps=0, online_critic_steps=0
        )
        return CalibrationResult(online, rng, report)
    except (ValueError, TypeError, FloatingPointError, OverflowError) as error:
        report["reason"] = str(error)
        raise CalibrationFailure(report) from error
    finally:
        if updater is not None:
            updater.close()


def build_calibrated_model(result, *, contract, metadata):
    """Build an ordinary initialization artifact; caller authenticates/publishes it.

    Caller-supplied ``bc_source_sha256`` identifies the original BC bytes. This is
    provenance, not a claim that this pure function has read/authenticated a file.
    """
    from .checkpoint import build_vrpo_model

    if not isinstance(result, CalibrationResult) or result.report.get("accepted") is not True:
        raise ValueError("only accepted calibration may produce an artifact")
    if not isinstance(metadata, Mapping):
        raise ValueError("caller metadata with bc_source_sha256 is required")
    source = metadata.get("bc_source_sha256")
    if not isinstance(source, str) or len(source) != 64 or any(c not in "0123456789abcdef" for c in source):
        raise ValueError("caller must supply original bc_source_sha256 identity")
    if actor_parameter_fingerprint(result.state.params) != result.report.get("actor_sha256_before"):
        raise ValueError("calibrated Actor no longer matches original BC fingerprint")
    for name in ("actor_steps", "critic_steps", "completed_updates"):
        if int(np.asarray(getattr(result.state, name))) != 0:
            raise ValueError("calibration artifact requires fresh online counters")
    saved_metadata = dict(metadata)
    saved_metadata.update(
        critic_trained=True,
        calibration_report=result.report,
        optimizer_lineage="fresh_optimizer_slots",
        fresh_optimizer_slots=True,
    )
    model = build_vrpo_model(result.state.params, contract=contract, metadata=saved_metadata)
    model.update(checkpoint_kind="vrpo_q_calibrated", critic_trained=True)
    return model


# --- from vrpo_paged_calibration.py ---


def _validate_paged_partition(batch, expected, config, name):
    if not isinstance(batch, PagedRawQBatch):
        raise ValueError(f"{name} requires an explicit complete-game PagedRawQBatch")
    if batch.page_rows != config.page_rows:
        raise ValueError(f"{name} page rows differ from configured execution layout")
    if batch.game_lengths.shape != (expected,):
        raise ValueError(f"{name} requires exactly {expected} nonempty complete games")
    lengths = validate_paged_batch(batch, max_game_steps=config.max_game_steps)
    if np.any(lengths <= 0):
        raise ValueError(f"{name} requires exactly {expected} nonempty complete games")
    return lengths


def _admit_paged(train, heldout, config):
    # Both partitions and ALL immutable MC targets coexist, but trace/prediction/
    # compaction scratch is fixed to ONE page. No max_game_steps-sized arena.
    resident = page_resident_bytes(train) + page_resident_bytes(heldout)
    resident += sum(len(b.pages) * b.page_rows * 3 * 4 for b in (train, heldout))
    charge = resident + max(page_scratch_bytes(b) for b in (train, heldout))
    if charge > config.max_bytes:
        raise ValueError("calibration pages/MC targets/fixed scratch exceed max_bytes")
    return charge


def _paged_mc_targets(batch):
    """Exact all-raw Monte Carlo rewards, including foreign and forced transitions."""
    carry = init_page_trace_carry(batch.num_games, jnp.float32)
    selected = jnp.ones((batch.num_games,), jnp.bool_)
    zero = jnp.zeros((batch.page_rows, 3, 87), jnp.float32)
    targets = [None] * len(batch.pages)
    for index, page in batch.iter_pages(reverse=True):
        labels, carry = qboost_page(page, zero, carry, selected_games=selected, gamma=1.0, trace_lambda=1.0)
        target = jax.lax.stop_gradient(labels.targets)
        _finite((target, carry), "Monte Carlo page targets")
        targets[index] = target
    return tuple(targets)


def _paged_heldout_mse(updater, params, batch, targets):
    from .update import phase_params

    error = 0.0
    critic = phase_params(params, "critic")
    for index, page in batch.iter_pages():
        squared, finite = updater.squared_error(critic, page, targets[index])
        if not bool(np.asarray(finite)):
            raise ValueError("nonfinite held-out sampled-action predictions")
        error += float(np.asarray(squared))
    mse = error / (batch.raw_count * 3)
    if not math.isfinite(mse):
        raise ValueError("nonfinite held-out MSE")
    return mse


def calibrate_paged_q(network, config, state, rng, train_batch, heldout_batch, *, train_game_ids, heldout_game_ids):
    """One optimizer step per whole-game minibatch, never per page or raw row."""
    report = dict(
        accepted=False,
        phase="original_bc_q_calibration",
        gamma=1.0,
        trace_lambda=1.0,
        target_source="all_raw_environment_rewards",
        payoff_seats="absolute3",
        regression="sampled_action_only",
        advantage_normalization=False,
        critic_auxiliary=False,
        replay=False,
        epochs_completed=0,
        best_epoch=0,
        calibration_critic_steps=0,
        online_completed_updates=0,
        optimizer_lineage="fresh_optimizer_slots",
        resume_semantics="initialization_only",
        epoch_heldout_mse=[],
    )
    updater = None
    try:
        if not isinstance(config, VRPOConfig):
            raise ValueError("config must be a validated VRPOConfig")
        config.__post_init__()
        report.update(
            training_reduction="half_mean3_sum_raw_rows_mean_games_v2",
            heldout_reduction="mean_raw_rows_mean3_unhalved",
            optimizer=vrpo_optimizer_metadata(config),
        )
        if not isinstance(state, VRPOTrainingState):
            raise ValueError("state must be VRPOTrainingState")
        train_ids = _game_ids(train_game_ids, config.calibration_train_games, "train_game_ids")
        heldout_ids = _game_ids(heldout_game_ids, config.calibration_heldout_games, "heldout_game_ids")
        if set(train_ids) & set(heldout_ids):
            raise ValueError("train and held-out whole-game IDs overlap")
        train_lengths = _validate_paged_partition(train_batch, len(train_ids), config, "train")
        heldout_lengths = _validate_paged_partition(heldout_batch, len(heldout_ids), config, "heldout")
        _admit_paged(train_batch, heldout_batch, config)  # BEFORE target allocation or any fit step.
        _finite(state.params, "initial parameters")
        report.update(
            train_game_ids=list(train_ids),
            heldout_game_ids=list(heldout_ids),
            train_games=len(train_ids),
            heldout_games=len(heldout_ids),
            train_raw_rows=int(train_lengths.sum()),
            heldout_raw_rows=int(heldout_lengths.sum()),
            critic_microbatch_size=config.critic_microbatch_size,
            min_relative_improvement=config.calibration_min_relative_improvement,
            max_epochs=config.calibration_max_epochs,
            actor_sha256_before=actor_parameter_fingerprint(state.params),
            execution_layout_version=2,
            page_rows=config.page_rows,
        )
        train_targets, heldout_targets = _paged_mc_targets(train_batch), _paged_mc_targets(heldout_batch)
        squared = 0.0
        for index, page in heldout_batch.iter_pages():
            squared += float(
                np.asarray(
                    jnp.sum(jnp.where(page.transitions.executed[:, None], jnp.square(heldout_targets[index]), 0.0))
                )
            )
        zero_mse = squared / (heldout_batch.raw_count * 3)
        if not math.isfinite(zero_mse):
            raise ValueError("nonfinite zero baseline")
        updater = PagedVRPOUpdater(network, config)
        initial_mse = _paged_heldout_mse(updater, state.params, heldout_batch, heldout_targets)
        report.update(zero_baseline_mse=zero_mse, initial_critic_mse=initial_mse, best_heldout_mse=initial_mse)
        calibration, _, _ = create_training_state(state.params, config)
        best_params, best_mse = state.params, initial_mse
        width = config.games_per_minibatch
        for epoch in range(1, config.calibration_max_epochs + 1):
            rng, shuffle = jax.random.split(rng)
            order = np.asarray(jax.random.permutation(shuffle, train_batch.num_games))
            for start in range(0, train_batch.num_games, width):
                indices = order[start : start + width]  # HOST metadata, never a kernel shape.
                loss, grads = updater.critic_minibatch(calibration.params, train_batch, train_targets, indices)
                updated, slots = updater.critic_opt_step(
                    calibration.params, grads, calibration.critic_opt_state, jnp.float32(1.0)
                )
                _finite((loss, grads, updated, slots), "calibration Critic transaction")
                calibration = calibration.replace(
                    params=replace_phase(calibration.params, updated, "critic"),
                    critic_opt_state=slots,
                    critic_steps=calibration.critic_steps + 1,
                )
                report["calibration_critic_steps"] += 1
            mse = _paged_heldout_mse(updater, calibration.params, heldout_batch, heldout_targets)
            report["epochs_completed"] = epoch
            report["epoch_heldout_mse"].append(mse)
        # Select a pre-registered final epoch instead of taking the minimum over
        # the same held-out set used for acceptance.  The latter is optimistically
        # biased and makes the zero/initial comparisons look better than they are.
        best_params = calibration.params
        best_mse = report["epoch_heldout_mse"][-1]
        report.update(best_epoch=config.calibration_max_epochs, best_heldout_mse=best_mse)
        report["actor_sha256_after"] = actor_parameter_fingerprint(calibration.params)
        if report["actor_sha256_after"] != report["actor_sha256_before"]:
            raise ValueError("original BC Actor fingerprint changed")
        relative = config.calibration_min_relative_improvement
        report["beats_zero_baseline"] = bool(best_mse < zero_mse and best_mse < zero_mse * (1.0 - relative))
        report["beats_initial_critic"] = bool(best_mse < initial_mse and best_mse < initial_mse * (1.0 - relative))
        if not report["beats_zero_baseline"] or not report["beats_initial_critic"]:
            raise ValueError("held-out MSE must strictly beat both zero baseline and initial Critic")
        online, _, _ = create_training_state(best_params, config)
        _finite(online, "fresh online state")
        report.update(
            accepted=True, reason="passed", fresh_optimizer_slots=True, online_actor_steps=0, online_critic_steps=0
        )
        return CalibrationResult(online, rng, report)
    except (ValueError, TypeError, FloatingPointError, OverflowError, MemoryError) as error:
        report["reason"] = str(error)
        raise CalibrationFailure(report) from error
    finally:
        if updater is not None:
            updater.close()
