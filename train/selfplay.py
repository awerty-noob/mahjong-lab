"""Sealed-page self-play collection and the run driver."""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import pickle
import shutil
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import unfreeze

from networks.mahjong_3p.configs import (
    DEFAULT_CENTRAL_Q_CRITIC_CONFIG,
    DEFAULT_CRITIC_CONFIG,
    DISABLED_HEAD_CONFIG,
    RULESET,
    checkpoint_critic_config,
    checkpoint_head_config,
    checkpoint_network_config,
    checkpoint_reading_config,
    validate_reading_config,
)
from networks.mahjong_3p.feature import POLICY_CONTEXT_SHAPES, build_central_q_input
from networks.mahjong_3p.network import (
    build_actor,
    build_central_q,
    compact_legal_mask,
    compact_observation,
    to_simulator_action,
)
from networks.mahjong_3p.transformer import clone_actor_and_public_encoder
from train.cache import bounded_jit, fixed_page_scratch_bytes
from train.configs import (
    PagedRawQBatch,
    PageTraceCarry,
    QBoostLabels,
    RawQBatch,
    RawQPage,
    RawQTransition,
    VRPOConfig,
    default_reward_config,
    rule_spec,
)
from train.update import (
    _check_shapes,
    _eager_check,
    _layout_validity,
    _probability_valid,
    create_training_state,
    phase_params,
    raw_row_validity,
)

# --- from vrpo_buffer.py ---


def _charge(tree):
    return 1024 + sum(int(x.nbytes) + 256 for x in jax.tree.leaves(tree))


def _shape_charge(tree, leading, old_leading):
    return 1024 + sum(
        int(np.prod(leading + x.shape[old_leading:])) * x.dtype.itemsize + 256 for x in jax.tree.leaves(tree)
    )


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _rounded(count, quantum):
    return max(quantum, ((int(count) + quantum - 1) // quantum) * quantum)


def _borrow_feature_padding(raw, real, source):
    """Tokenizer-safe features, while every padding label/mask stays invalid."""

    def borrow(x, first):
        return jnp.where(real.reshape(real.shape + (1,) * (x.ndim - real.ndim)), x, first)

    return raw._replace(
        observation=jax.tree.map(borrow, raw.observation, source.observation),
        critic_observation=jax.tree.map(borrow, raw.critic_observation, source.critic_observation),
    )


@bounded_jit(max_entries=8)
def _block_summary(raw, seats):
    executed = raw.executed
    width = executed.shape[1]
    count = executed.sum(axis=1, dtype=jnp.int32)
    step = jnp.arange(width)[None, :]
    owned = (seats[:, None] < 0) | (raw.current_player == seats[:, None])
    return jnp.stack(
        (
            count,
            jnp.any(executed != (step < count[:, None]), axis=1),
            jnp.any(executed & ~raw_row_validity(raw), axis=1),
            jnp.any(executed & (raw.is_learner != owned), axis=1),
            jnp.any(executed & raw.done & (step != count[:, None] - 1), axis=1),
            jnp.any(executed & raw.done, axis=1),
            jnp.sum(executed & raw.is_learner & raw.is_decision, axis=1, dtype=jnp.int32),
        ),
        axis=1,
    )


def _block_destinations(blocks, counts, lengths, bucket):
    offsets = jnp.cumsum(lengths) - lengths
    previous = jnp.zeros_like(lengths)
    destinations = []
    for i, block in enumerate(blocks):
        steps = jnp.arange(block.executed.shape[1])[None, :]
        destination = offsets[:, None] + previous[:, None] + steps
        destinations.append(jnp.where(block.executed, destination, bucket).reshape(-1))
        previous = previous + counts[i]
    return destinations, offsets


@bounded_jit(max_entries=8, static_argnames=("bucket",))
def _pack_execution_blocks(blocks, counts, lengths, *, bucket):
    """Static block topology, dynamic lengths; invalid tails scatter out of bounds."""
    destinations, offsets = _block_destinations(blocks, counts, lengths, bucket)

    def pack(*xs):
        trailing = xs[0].shape[2:]
        out = jnp.zeros((bucket,) + trailing, xs[0].dtype)
        for dest, x in zip(destinations, xs):
            out = out.at[dest].set(x.reshape((-1,) + trailing), mode="drop")
        return out

    raw = jax.tree.map(pack, *blocks)
    # The first real row need not be lane zero (a zero-length padded lane can be
    # represented in generic metadata), so take it from the packed output.
    first = jax.tree.map(lambda x: x[0], raw)
    raw = _borrow_feature_padding(raw, raw.executed, first)
    return raw, offsets


@bounded_jit(max_entries=8, static_argnames=("bucket",))
def _pack_old_q(blocks, q_blocks, counts, lengths, *, bucket):
    destinations, _ = _block_destinations(blocks, counts, lengths, bucket)
    out = jnp.zeros((bucket, 3, 87), q_blocks[0].dtype)
    for dest, q in zip(destinations, q_blocks):
        out = out.at[dest].set(q.reshape((-1, 3, 87)), mode="drop")
    return out


@bounded_jit(max_entries=8)
def _batch_summary(batch):
    raw = batch.transitions
    games = batch.game_lengths.shape[0]
    actor = raw.executed & raw.is_learner & raw.is_decision
    n = raw.executed.shape[0]
    game_id = jnp.searchsorted(batch.game_offsets, jnp.arange(n), side="right") - 1
    safe_id = jnp.clip(game_id, 0, max(games - 1, 0))
    if games:
        seats = batch.learner_seats[safe_id]
        owned = (seats < 0) | (raw.current_player == seats)
        seat_ok = jnp.all((batch.learner_seats >= -1) & (batch.learner_seats < 3))
    else:
        owned = jnp.zeros(n, bool)
        seat_ok = jnp.asarray(True)
    ok = (
        _layout_validity(raw, batch.game_offsets, batch.game_lengths)
        & jnp.all(raw_row_validity(raw))
        & seat_ok
        & jnp.all(~raw.executed | (raw.is_learner == owned))
    )
    return jnp.stack((ok.astype(jnp.int32), raw.executed.sum(dtype=jnp.int32), actor.sum(dtype=jnp.int32)))


def _validate_batch_shape(batch):
    raw = batch.transitions
    if raw.executed.ndim != 1:
        raise ValueError("complete batch must use a flat raw arena")
    _check_shapes(raw)
    shape = batch.game_lengths.shape
    if (
        len(shape) != 1
        or batch.game_offsets.shape != shape
        or batch.learner_seats.shape != shape
        or any(
            not jnp.issubdtype(x.dtype, jnp.integer)
            for x in (batch.game_offsets, batch.game_lengths, batch.learner_seats)
        )
    ):
        raise ValueError("game offsets, lengths and learner seats must be integer vectors")


class RawQBuffer:
    """One fixed-lane cohort, retaining blocks rather than safety-limit matrices.

    Each lane contributes one complete game. ``lengths`` passed to append is the
    optional expected *block* executed count; ``completed`` is the cumulative
    completion vector. Finished lanes must remain frozen. Learner seat -1 means
    all-owned self-play. ``max_game_steps`` is only a rejection/safety limit.
    Default output capacity is observed raw count rounded to ``bucket_granularity``;
    explicit ``bucket_size`` instead requests that fixed capacity and acts as a cap.
    """

    def __init__(
        self,
        num_envs,
        learner_seats,
        *,
        max_game_steps,
        max_bytes,
        bucket_size=None,
        bucket_granularity=128,
        cache_old_q=False,
    ):
        self.num_envs = _positive(num_envs, "num_envs")
        self.max_game_steps = _positive(max_game_steps, "max_game_steps")
        self.max_bytes = _positive(max_bytes, "max_bytes")
        self.bucket_size = None if bucket_size is None else _positive(bucket_size, "bucket_size")
        self.bucket_granularity = _positive(bucket_granularity, "bucket_granularity")
        seats = np.asarray(learner_seats, dtype=np.int32)
        if seats.shape != (num_envs,) or np.any((seats < -1) | (seats > 2)):
            raise ValueError("learner_seats must have one seat (-1 or 0..2) per lane")
        self.learner_seats = seats.copy()
        self.cache_old_q = bool(cache_old_q)
        self.lengths = np.zeros(num_envs, np.int32)
        self.actor_counts = np.zeros(num_envs, np.int32)
        self.completed = np.zeros(num_envs, bool)
        self._blocks = []
        self._counts = []
        self._q_blocks = []
        self._schema = None
        self.resident_bytes = 0

    @property
    def raw_count(self):
        return int(self.lengths.sum())

    @property
    def actor_count(self):
        return int(self.actor_counts.sum())

    @property
    def complete(self):
        return bool(np.all(self.completed))

    def _capacity(self, count):
        return self.bucket_size if self.bucket_size is not None else _rounded(count, self.bucket_granularity)

    def _output_charge(self, raw, count, q=None):
        bucket = self._capacity(count)
        charge = _shape_charge(raw, (bucket,), 2) + 12 * self.num_envs
        if q is not None:
            charge += _shape_charge(q, (bucket,), 2)
        return charge

    def append_execution_block(self, raw, *, lengths=None, completed=None, truncated=None, old_q=None):
        _check_shapes(raw, old_q)
        if raw.executed.ndim != 2 or raw.executed.shape[0] != self.num_envs or raw.executed.shape[1] == 0:
            raise ValueError("execution blocks must have shape [num_envs, positive T, ...]")
        if truncated is not None and bool(np.asarray(jnp.any(jnp.asarray(truncated)))):
            raise ValueError("truncated games cannot enter raw Q storage")
        if self.cache_old_q != (old_q is not None):
            raise ValueError("old_q must be supplied exactly when cache_old_q is enabled")
        summary = np.asarray(_block_summary(raw, jnp.asarray(self.learner_seats)))
        count = summary[:, 0].astype(np.int32)
        if np.any(summary[:, 1:5]):
            raise ValueError("invalid execution block: prefix, probability/action, ownership or terminal")
        if not count.any():
            raise ValueError("empty execution block makes no progress")
        if np.any(self.completed & (count > 0)):
            raise ValueError("cannot append after a lane terminal")
        if lengths is not None and not np.array_equal(np.asarray(lengths), count):
            raise ValueError("execution length context does not match raw executed rows")
        new_lengths = self.lengths + count
        new_completed = self.completed | summary[:, 5].astype(bool)
        if completed is not None and not np.array_equal(np.asarray(completed), new_completed):
            raise ValueError("completion context does not match true raw terminals")
        total = int(new_lengths.sum())
        if np.any(new_lengths > self.max_game_steps) or (self.bucket_size is not None and total > self.bucket_size):
            raise MemoryError("raw game safety/capacity exceeded; never truncate or compact actor rows")
        if old_q is not None:
            _eager_check(jnp.all(~raw.executed | jnp.all(jnp.isfinite(old_q), axis=(-2, -1))), "nonfinite old_q cache")
        schema = (jax.tree.structure(raw), tuple((x.shape[2:], x.dtype) for x in jax.tree.leaves(raw)))
        if self._schema is not None and schema != self._schema:
            raise ValueError("raw feature schema/dtype changed within a cohort")
        incoming = _charge(raw) + count.nbytes + (0 if old_q is None else _charge(old_q))
        required = self.resident_bytes + incoming + self._output_charge(raw, total, old_q)
        if required > self.max_bytes:
            raise MemoryError(f"raw Q memory admission requires {required} bytes > {self.max_bytes}")
        # No device allocation/copy of a safety-limit arena, no host observation traffic.
        self._blocks.append(raw)
        self._counts.append(count.copy())
        if old_q is not None:
            self._q_blocks.append(old_q)
        self._schema = schema
        self.lengths = new_lengths
        self.completed = new_completed
        self.actor_counts = self.actor_counts + summary[:, 6].astype(np.int32)
        self.resident_bytes += incoming

    def export_complete_batch(self):
        if not self._blocks or not self.complete:
            raise ValueError("only complete terminal games may be exported")
        raw = self._blocks[0]
        q = self._q_blocks[0] if self._q_blocks else None
        if self.resident_bytes + self._output_charge(raw, self.raw_count, q) > self.max_bytes:
            raise MemoryError("raw Q export exceeds declared memory budget")
        packed, offsets = _pack_execution_blocks(
            tuple(self._blocks),
            jnp.asarray(np.stack(self._counts)),
            jnp.asarray(self.lengths),
            bucket=self._capacity(self.raw_count),
        )
        batch = RawQBatch(packed, offsets, jnp.asarray(self.lengths), jnp.asarray(self.learner_seats))
        summary = np.asarray(_batch_summary(batch))
        if tuple(summary) != (1, self.raw_count, self.actor_count):
            raise ValueError("raw packing lost/overwrote executed or original Actor rows")
        return batch

    def export_old_q(self):
        if not self._q_blocks or not self.complete:
            raise ValueError("no complete old-Q cache")
        if (
            self.resident_bytes + self._output_charge(self._blocks[0], self.raw_count, self._q_blocks[0])
            > self.max_bytes
        ):
            raise MemoryError("old-Q export exceeds declared memory budget")
        return _pack_old_q(
            tuple(self._blocks),
            tuple(self._q_blocks),
            jnp.asarray(np.stack(self._counts)),
            jnp.asarray(self.lengths),
            bucket=self._capacity(self.raw_count),
        )


@bounded_jit(max_entries=8, static_argnames=("bucket",))
def _merge_batches(batches, *, bucket):
    lengths = jnp.concatenate(tuple(b.game_lengths for b in batches))
    seats = jnp.concatenate(tuple(b.learner_seats for b in batches))
    offsets = jnp.cumsum(lengths) - lengths
    starts = jnp.cumsum(jnp.stack(tuple(jnp.sum(b.game_lengths) for b in batches)))
    starts = jnp.concatenate((jnp.zeros(1, jnp.int32), starts[:-1]))
    destinations = []
    for i, b in enumerate(batches):
        index = jnp.arange(b.transitions.executed.shape[0])
        destinations.append(jnp.where(index < b.game_lengths.sum(), starts[i] + index, bucket))

    def merge(*xs):
        out = jnp.zeros((bucket,) + xs[0].shape[1:], xs[0].dtype)
        for pos, x in zip(destinations, xs):
            out = out.at[pos].set(x, mode="drop")
        return out

    raw = jax.tree.map(merge, *(b.transitions for b in batches))
    raw = _borrow_feature_padding(raw, raw.executed, jax.tree.map(lambda x: x[0], raw))
    return RawQBatch(raw, offsets, lengths, seats)


class RawQCohortStager:
    """Retain ordered complete cohorts, reserving an observed-size final output."""

    def __init__(self, *, max_bytes, bucket_size=None, bucket_granularity=128):
        self.max_bytes = _positive(max_bytes, "max_bytes")
        self.bucket_size = None if bucket_size is None else _positive(bucket_size, "bucket_size")
        self.bucket_granularity = _positive(bucket_granularity, "bucket_granularity")
        self._batches = []
        self.raw_count = self.actor_count = self.resident_bytes = 0
        self._schema = None

    def _capacity(self, count):
        return self.bucket_size if self.bucket_size is not None else _rounded(count, self.bucket_granularity)

    def append_complete_batch(self, batch, *, raw_count=None, actor_count=None):
        _validate_batch_shape(batch)
        summary = np.asarray(_batch_summary(batch))
        if summary[0] != 1:
            raise ValueError("invalid complete raw batch")
        if raw_count is not None and int(raw_count) != int(summary[1]):
            raise ValueError("raw row count differs from original collection count")
        if actor_count is not None and int(actor_count) != int(summary[2]):
            raise ValueError("Actor row count differs from original collection count")
        if summary[1] == 0:
            raise ValueError("cannot stage an empty cohort")
        schema = (
            jax.tree.structure(batch.transitions),
            tuple((x.shape[1:], x.dtype) for x in jax.tree.leaves(batch.transitions)),
        )
        if self._schema is not None and schema != self._schema:
            raise ValueError("cohort feature schema/dtype mismatch")
        total = self.raw_count + int(summary[1])
        charge = _charge(batch)
        output = _shape_charge(batch.transitions, (self._capacity(total),), 1)
        metadata = 12 * sum(b.game_lengths.shape[0] for b in self._batches + [batch])
        if self.bucket_size is not None and total > self.bucket_size:
            raise MemoryError("staged raw row capacity exceeded")
        if self.resident_bytes + charge + output + metadata > self.max_bytes:
            raise MemoryError("cohort memory admission exceeds declared byte budget")
        self._batches.append(batch)
        self._schema = schema
        self.raw_count = total
        self.actor_count += int(summary[2])
        self.resident_bytes += charge

    append_batch = append_complete_batch

    def export_complete_batch(self):
        if not self._batches:
            raise ValueError("no complete cohorts staged")
        batch = _merge_batches(tuple(self._batches), bucket=self._capacity(self.raw_count))
        summary = np.asarray(_batch_summary(batch))
        if tuple(summary) != (1, self.raw_count, self.actor_count):
            raise ValueError("cohort merge lost original raw/Actor rows")
        return batch


@bounded_jit(max_entries=8, static_argnames=("time_bucket",))
def _gather_whole_games(batch, indices, *, time_bucket):
    n = batch.transitions.executed.shape[0]
    games = batch.game_lengths.shape[0]
    safe = jnp.clip(indices, 0, max(games - 1, 0))
    chosen = (indices >= 0) & (indices < games)
    if games:
        lengths = jnp.where(chosen, batch.game_lengths[safe], 0)
        source_offsets = batch.game_offsets[safe]
        seats = jnp.where(chosen, batch.learner_seats[safe], -1)
    else:
        lengths = jnp.zeros_like(indices)
        source_offsets = jnp.zeros_like(indices)
        seats = jnp.full_like(indices, -1)
    steps = jnp.arange(time_bucket)[None, :]
    offsets = jnp.cumsum(lengths) - lengths
    valid = steps < lengths[:, None]
    bucket = indices.shape[0] * time_bucket
    destination = jnp.where(valid, offsets[:, None] + steps, bucket).reshape(-1)
    source = jnp.clip(source_offsets[:, None] + steps, 0, max(n - 1, 0)).reshape(-1)

    def gather(x):
        out = jnp.zeros((bucket,) + x.shape[1:], x.dtype)
        if n:
            out = out.at[destination].set(x[source], mode="drop")
        return out

    raw = jax.tree.map(gather, batch.transitions)
    if n:
        raw = _borrow_feature_padding(raw, raw.executed, jax.tree.map(lambda x: x[0], batch.transitions))
    ok = (
        jnp.all((indices == -1) | chosen)
        & jnp.all(lengths <= time_bucket)
        & _layout_validity(batch.transitions, batch.game_offsets, batch.game_lengths)
    )
    raw = raw._replace(reward=jnp.where(ok, raw.reward, jnp.full_like(raw.reward, jnp.nan)))
    return RawQBatch(raw, offsets, lengths, seats), ok


def gather_whole_games(batch, game_indices, *, time_bucket):
    """Fixed game-count/T-bucket minibatch; -1 indices are padded games.

    Complete selected games are repacked into a contiguous raw prefix in requested
    order. No clipped gather index is reused for invalid-tail scatter. Per-game
    lengths/maxima are dynamic values, never compilation keys; the caller chooses
    a rounded T bucket (typically 128) from the current minibatch's maximum length.
    """
    _validate_batch_shape(batch)
    _positive(time_bucket, "time_bucket")
    indices = jnp.asarray(game_indices)
    if indices.ndim != 1 or not jnp.issubdtype(indices.dtype, jnp.integer):
        raise ValueError("game_indices must be a fixed-size integer vector")
    if indices.shape[0] == 0:
        raise ValueError("whole-game minibatch needs a positive fixed game count")
    result, ok = _gather_whole_games(batch, indices, time_bucket=time_bucket)
    _eager_check(ok, "whole-game gather: invalid game index, incomplete layout or T bucket too small")
    return result


gather_whole_games.clear_cache = _gather_whole_games.clear_cache
gather_whole_games.cache_info = _gather_whole_games.cache_info

# --- from vrpo_pages.py ---


def _charge(tree):
    """Works on arrays AND ShapeDtypeStruct without allocating/reading payloads."""
    return 1024 + sum(math.prod(x.shape) * np.dtype(x.dtype).itemsize + 256 for x in jax.tree.leaves(tree))


def _schema(raw):
    return (jax.tree.structure(raw), tuple((x.shape[2:], x.dtype) for x in jax.tree.leaves(raw)))


def _page_charge(raw, rows):
    return (
        1024
        + sum(rows * math.prod(x.shape[2:]) * np.dtype(x.dtype).itemsize + 256 for x in jax.tree.leaves(raw))
        + 2 * (rows * 4 + 256)
    )


@bounded_jit(max_entries=8, static_argnames=("page_rows",))
def _initial_page(raw, *, page_rows):
    flat = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), raw)
    first_index = jnp.argmax(flat.executed)
    first = jax.tree.map(lambda x: x[first_index], flat)
    zero = jax.tree.map(lambda x: jnp.zeros((page_rows,) + x.shape[2:], x.dtype), raw)
    zero = _borrow_feature_padding(zero, zero.executed, first)
    return RawQPage(zero, jnp.full(page_rows, -1, jnp.int32), jnp.full(page_rows, -1, jnp.int32))


def _empty_like(page):
    zero = jax.tree.map(jnp.zeros_like, page.transitions)
    first = jax.tree.map(lambda x: x[0], page.transitions)
    zero = _borrow_feature_padding(zero, zero.executed, first)
    return RawQPage(zero, jnp.full_like(page.game_ids, -1), jnp.full_like(page.game_steps, -1))


@bounded_jit(max_entries=8)
def _append_page_block(current, raw, game_ids, game_starts, used):
    """Stable compaction into current+overflow; invalid rows scatter OOB/drop."""
    rows = current.game_ids.shape[0]
    width = raw.executed.shape[1]
    flat = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), raw)
    real = flat.executed
    rank = jnp.cumsum(real, dtype=jnp.int32) - 1
    destination = used + rank
    first_dest = jnp.where(real & (destination < rows), destination, rows)
    second_dest = jnp.where(real & (destination >= rows), destination - rows, rows)
    empty = _empty_like(current)
    base = jax.tree.map(lambda old, blank: jnp.where(used > 0, old, blank), current, empty)
    gids = jnp.broadcast_to(game_ids[:, None], raw.executed.shape).reshape(-1)
    steps = (game_starts[:, None] + jnp.arange(width, dtype=jnp.int32)[None, :]).reshape(-1)
    incoming = RawQPage(flat, gids, steps)
    first = jax.tree.map(lambda out, x: out.at[first_dest].set(x, mode="drop"), base, incoming)
    second = jax.tree.map(lambda out, x: out.at[second_dest].set(x, mode="drop"), empty, incoming)
    return first, second


@bounded_jit(max_entries=8)
def _packed_summary(page):
    raw = page.transitions
    count = raw.executed.sum(dtype=jnp.int32)
    prefix = jnp.all(raw.executed == (jnp.arange(raw.executed.shape[0]) < count))
    metadata = jnp.all(
        jnp.where(
            raw.executed, (page.game_ids >= 0) & (page.game_steps >= 0), (page.game_ids == -1) & (page.game_steps == -1)
        )
    )
    return jnp.stack(
        (
            prefix & metadata & jnp.all(raw_row_validity(raw)),
            count,
            jnp.sum(raw.executed & raw.is_learner & raw.is_decision, dtype=jnp.int32),
        )
    )


@bounded_jit(max_entries=8)
def _certificate_page(page, counts, completed, actors, seats):
    """Forward streaming certificate; only O(G) metadata/summaries leave device."""
    raw = page.transitions
    games = counts.shape[0]
    gid_ok = (page.game_ids >= 0) & (page.game_ids < games)
    safe_id = jnp.clip(page.game_ids, 0, games - 1)
    real_count = raw.executed.sum(dtype=jnp.int32)
    valid = (
        jnp.all(raw_row_validity(raw))
        & jnp.all(raw.executed == (jnp.arange(raw.executed.shape[0]) < real_count))
        & jnp.all(jnp.where(raw.executed, gid_ok, (page.game_ids == -1) & (page.game_steps == -1)))
        & jnp.all(raw.executed | (~raw.done & ~raw.is_learner & ~raw.is_decision))
    )
    present = jnp.zeros(games, jnp.bool_)

    def step(carry, row):
        count, done, actor, seen, ok = carry
        gid, clock, terminal, player, learner, decision, executed, good_id = row

        def update(state):
            count, done, actor, seen, ok = state
            owned = (seats[gid] < 0) | (player == seats[gid])
            ok &= good_id & ~done[gid] & (clock == count[gid]) & (learner == owned)
            return (
                count.at[gid].add(1),
                done.at[gid].set(terminal),
                actor.at[gid].add((learner & decision).astype(jnp.int32)),
                seen.at[gid].set(True),
                ok,
            )

        return jax.lax.cond(executed, update, lambda state: state, carry), None

    result, _ = jax.lax.scan(
        step,
        (counts, completed, actors, present, valid),
        (safe_id, page.game_steps, raw.done, raw.current_player, raw.is_learner, raw.is_decision, raw.executed, gid_ok),
    )
    counts, completed, actors, present, valid = result
    return counts, completed, actors, present, jnp.stack((valid, real_count))


def _integer_vector(value, size, name):
    arr = np.asarray(value)
    if arr.shape != (size,) or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{name} must be an integer [{size}] vector")
    if np.any((arr < np.iinfo(np.int32).min) | (arr > np.iinfo(np.int32).max)):
        raise ValueError(f"{name} does not fit int32")
    return arr.astype(np.int32, copy=False)


def validate_paged_batch(batch, *, max_game_steps=None):
    """Certify chronological complete games AND the page-selection directory.

    Does not flatten pages or transfer either feature tree to host. Returns host
    int32 game lengths. The host owner/page tuple is never submitted to JIT.
    """
    if not isinstance(batch, PagedRawQBatch) or type(batch.page_rows) is not int or batch.page_rows <= 0:
        raise ValueError("expected a PagedRawQBatch with positive page_rows")
    if (
        any(type(x) is not int for x in (batch.raw_count, batch.actor_count, batch.resident_bytes))
        or batch.raw_count <= 0
        or batch.actor_count < 0
        or batch.actor_count > batch.raw_count
        or batch.resident_bytes <= 0
    ):
        raise ValueError("paged raw/Actor/byte counts must be canonical nonnegative Python integers")
    lengths = np.asarray(batch.game_lengths)
    if lengths.ndim != 1 or lengths.size == 0 or lengths.dtype != np.dtype(np.int32):
        raise ValueError("paged game_lengths must be a nonempty canonical int32 vector")
    games = lengths.size
    lengths = _integer_vector(lengths, games, "game_lengths")
    seats = np.asarray(batch.learner_seats)
    if seats.dtype != np.dtype(np.int32):
        raise ValueError("paged learner_seats must be a canonical int32 vector")
    seats = _integer_vector(seats, games, "learner_seats")
    if np.any(lengths <= 0) or np.any((seats < -1) | (seats > 2)):
        raise ValueError("every paged game must be nonempty with learner seat -1 or 0..2")
    if max_game_steps is not None:
        _positive(max_game_steps, "max_game_steps")
        if np.any(lengths > max_game_steps):
            raise ValueError("paged game exceeds max_game_steps safety limit")
    if not batch.pages or len(batch.pages) != len(batch.page_game_ids):
        raise ValueError("page tuple and game directory must be nonempty and aligned")
    minimum_bytes = sum(_charge(page) for page in batch.pages) + lengths.nbytes + seats.nbytes
    if batch.resident_bytes < minimum_bytes:
        raise ValueError("paged resident_bytes understates the live raw pages and game metadata")
    counts = jnp.zeros(games, jnp.int32)
    completed = jnp.zeros(games, jnp.bool_)
    actors = jnp.zeros(games, jnp.int32)
    device_seats = jnp.asarray(seats)
    schema = None
    for index, page in enumerate(batch.pages):
        raw = page.transitions
        _check_shapes(raw)
        if (
            raw.executed.shape != (batch.page_rows,)
            or page.game_ids.shape != raw.executed.shape
            or page.game_steps.shape != raw.executed.shape
            or page.game_ids.dtype != jnp.int32
            or page.game_steps.dtype != jnp.int32
        ):
            raise ValueError("every page must have fixed [P] raw rows and int32 ids/steps")
        current_schema = (jax.tree.structure(raw), tuple((x.shape[1:], x.dtype) for x in jax.tree.leaves(raw)))
        if schema is not None and current_schema != schema:
            raise ValueError("page feature schema/dtype changed")
        schema = current_schema
        counts, completed, actors, present, summary = _certificate_page(page, counts, completed, actors, device_seats)
        summary = np.asarray(summary)
        if not summary[0] or summary[1] <= 0 or (index < len(batch.pages) - 1 and summary[1] != batch.page_rows):
            raise ValueError("invalid paged raw layout, chronology, ownership, terminal or row law")
        directory = batch.page_game_ids[index]
        if (
            not isinstance(directory, tuple)
            or any(type(g) is not int for g in directory)
            or tuple(sorted(set(directory))) != directory
            or directory != tuple(np.flatnonzero(np.asarray(present)).tolist())
        ):
            raise ValueError("page game directory does not match true executed game ids")
    if (
        not np.array_equal(np.asarray(counts), lengths)
        or not np.all(np.asarray(completed))
        or int(lengths.sum(dtype=np.int64)) != batch.raw_count
        or int(np.asarray(actors).sum(dtype=np.int64)) != batch.actor_count
    ):
        raise ValueError("paged complete-game lengths, terminals or raw/Actor counts disagree")
    return lengths.copy()


class RawQPageStore:
    """One host collection owner; successful appends publish immutable pages.

    Source blocks are released by the caller after append succeeds. Allocation
    admission includes retained pages, cache reservation per allocated page row,
    incoming E*H source, old/new current and overflow, and bounded page scratch.
    max_game_steps is ONLY a per-game guard, never an allocation dimension.
    """

    def __init__(
        self,
        num_games,
        *,
        num_envs,
        execution_block_steps,
        page_rows,
        max_game_steps,
        max_bytes,
        reserve_bytes_per_row=1056,
    ):
        self.num_games = _positive(num_games, "num_games")
        self.num_envs = _positive(num_envs, "num_envs")
        self.execution_block_steps = _positive(execution_block_steps, "execution_block_steps")
        self.page_rows = _positive(page_rows, "page_rows")
        self.max_game_steps = _positive(max_game_steps, "max_game_steps")
        self.max_bytes = _positive(max_bytes, "max_bytes")
        if self.page_rows < self.num_envs * self.execution_block_steps:
            raise ValueError("page_rows must be at least num_envs*execution_block_steps")
        if (
            isinstance(reserve_bytes_per_row, bool)
            or type(reserve_bytes_per_row) is not int
            or reserve_bytes_per_row < 0
        ):
            raise ValueError("reserve_bytes_per_row must be a nonnegative integer")
        if self.max_game_steps > np.iinfo(np.int32).max:
            raise ValueError("max_game_steps must fit the int32 game clock")
        self.reserve_bytes_per_row = reserve_bytes_per_row
        self.lengths = np.zeros(num_games, np.int32)
        self.completed = np.zeros(num_games, bool)
        self.actor_counts = np.zeros(num_games, np.int32)
        self.learner_seats = np.full(num_games, -2, np.int32)
        self._pages = []
        self._page_game_ids = []
        self._used = 0
        self._schema = None
        self._page_bytes = None
        self._block_bytes = None

    @property
    def raw_count(self):
        return int(self.lengths.sum(dtype=np.int64))

    @property
    def actor_count(self):
        return int(self.actor_counts.sum(dtype=np.int64))

    @property
    def complete(self):
        return bool(np.all(self.completed))

    @property
    def resident_bytes(self):
        # Account host and eventual device game metadata, not just observations.
        metadata = 2048 + self.num_games * 32 + sum(256 + 4 * len(x) for x in self._page_game_ids)
        return metadata + len(self._pages) * (self._page_bytes or 0)

    def admit_next_block(self, abstract_or_real_raw):
        """Check BEFORE executing/allocating a source block; establish schema only."""
        raw = abstract_or_real_raw
        _check_shapes(raw)
        if raw.executed.shape != (self.num_envs, self.execution_block_steps):
            raise ValueError("execution block shape must be the configured fixed [E,H]")
        schema = _schema(raw)
        if self._schema is not None and schema != self._schema:
            raise ValueError("raw feature schema/dtype changed within collection")
        page_bytes = _page_charge(raw, self.page_rows)
        block_bytes = _charge(raw)
        # At most one extra retained page for a P>=E*H append. Three candidate
        # pages conservatively include first initialization + both kernel outputs.
        future_pages = len(self._pages) + 1
        cache_reserve = future_pages * self.page_rows * self.reserve_bytes_per_row
        # Known trajectory intermediates: Q, law, four all-seat label arrays,
        # indices/masks; model activations/gradients/executables are not claimed.
        scratch = (
            self.page_rows * (3 * 87 * 4 + 87 * 4 + 4 * 3 * 4 + 16)
            + self.num_games * 64
            + self.num_envs * self.execution_block_steps * 16
        )
        append_peak = (
            self.resident_bytes
            + block_bytes
            + 3 * page_bytes
            + cache_reserve
            + scratch
            + 2 * (256 + 4 * self.num_games)
        )
        leaves = jax.tree.leaves(raw)
        raw_row_bytes = sum(math.prod(x.shape[2:]) * np.dtype(x.dtype).itemsize for x in leaves)
        shared_scratch = fixed_page_scratch_bytes(self.page_rows, raw_row_bytes, self.num_games, len(leaves))
        # An admitted source must also fit the subsequent update/calibration
        # phase. In particular P may be much larger than E*H, so an append-only
        # incoming-block allowance cannot stand in for page-sized phase scratch.
        future_raw = self.resident_bytes + page_bytes + 256 + 4 * self.num_games
        phase_peak = future_raw + cache_reserve + shared_scratch
        required = max(append_peak, phase_peak)
        if required > self.max_bytes:
            raise MemoryError(f"raw page admission requires {required} bytes > {self.max_bytes}; never resize/truncate")
        self._schema, self._page_bytes, self._block_bytes = schema, page_bytes, block_bytes
        return required

    def append_execution_block(self, raw, *, game_ids, learner_seats, lengths=None, completed=None, truncated=None):
        _check_shapes(raw)
        if raw.executed.shape != (self.num_envs, self.execution_block_steps):
            raise ValueError("execution block shape must be the configured fixed [E,H]")
        if self._schema is not None and _schema(raw) != self._schema:
            raise ValueError("raw feature schema/dtype changed within collection")
        gids = _integer_vector(game_ids, self.num_envs, "game_ids")
        seats = _integer_vector(learner_seats, self.num_envs, "learner_seats")
        active = gids >= 0
        if np.any(gids < -1) or np.any(gids >= self.num_games) or len(set(gids[active].tolist())) != int(active.sum()):
            raise ValueError("game_ids must be distinct in-range game ids, or -1 padded lanes")
        if np.any((seats < -1) | (seats > 2)):
            raise ValueError("learner_seats must be -1 or 0..2")
        if truncated is not None:
            truncated = np.asarray(truncated)
            if truncated.shape != (self.num_envs,) or np.any(truncated):
                raise ValueError("truncated games cannot enter raw page storage")
        summary = np.asarray(_block_summary(raw, jnp.asarray(seats)))
        count = summary[:, 0].astype(np.int32)
        if np.any(summary[:, 1:5]) or not count.any() or np.any(count[~active]):
            raise ValueError("invalid/empty execution block: prefix, law, ownership, terminal or padded lane")
        safe = np.maximum(gids, 0)
        previous = np.where(active, self.lengths[safe], 0)
        was_completed = np.where(active, self.completed[safe], False)
        if np.any(was_completed & (count > 0)):
            raise ValueError("cannot append after a game terminal")
        known = self.learner_seats[safe]
        if np.any(active & (known != -2) & (known != seats)):
            raise ValueError("learner seat changed within a game")
        if lengths is not None and not np.array_equal(np.asarray(lengths), count):
            raise ValueError("execution length context does not match raw rows")
        new_lengths = previous.astype(np.int64) + count
        new_completed = was_completed | summary[:, 5].astype(bool)
        if completed is not None and not np.array_equal(np.asarray(completed), new_completed):
            raise ValueError("completion context does not match true terminals")
        if np.any(new_lengths > self.max_game_steps):
            raise MemoryError("raw game safety limit exceeded; never truncate/drop a game")
        self.admit_next_block(raw)
        current = self._pages[-1] if self._pages else _initial_page(raw, page_rows=self.page_rows)
        first, overflow = _append_page_block(
            current, raw, jnp.asarray(gids), jnp.asarray(previous, jnp.int32), jnp.int32(self._used)
        )
        added = int(count.sum())
        combined = self._used + added
        first_summary = np.asarray(_packed_summary(first))
        overflow_summary = np.asarray(_packed_summary(overflow))
        expected_first, expected_overflow = min(combined, self.page_rows), max(combined - self.page_rows, 0)
        old_actors = 0 if not self._used else int(np.asarray(_packed_summary(current))[2])
        if (
            not first_summary[0]
            or not overflow_summary[0]
            or first_summary[1] != expected_first
            or overflow_summary[1] != expected_overflow
            or first_summary[2] + overflow_summary[2] != old_actors + int(summary[:, 6].sum())
        ):
            raise ValueError("page packing changed raw/Actor counts or validity")
        # Device summaries synchronize candidate writes. No committed host/game
        # metadata or old page is replaced before every validation succeeds.
        row_gids = np.repeat(gids, count)
        split = self.page_rows - self._used
        initial_ids = set(self._page_game_ids[-1]) if self._used else set()
        first_ids = tuple(sorted(initial_ids | set(row_gids[:split].tolist())))
        overflow_ids = tuple(sorted(set(row_gids[split:].tolist())))
        if self._used:
            self._pages[-1] = first
            self._page_game_ids[-1] = first_ids
        else:
            self._pages.append(first)
            self._page_game_ids.append(first_ids)
        if expected_overflow:
            self._pages.append(overflow)
            self._page_game_ids.append(overflow_ids)
        self._used = combined % self.page_rows
        self.lengths[gids[active]] = new_lengths[active].astype(np.int32)
        self.completed[gids[active]] = new_completed[active]
        self.actor_counts[gids[active]] += summary[active, 6].astype(np.int32)
        self.learner_seats[gids[active]] = seats[active]

    def export_complete_batch(self):
        if not self._pages or not self.complete:
            raise ValueError("only complete terminal games may be exported")
        batch = PagedRawQBatch(
            tuple(self._pages),
            jnp.asarray(self.lengths),
            jnp.asarray(self.learner_seats),
            tuple(self._page_game_ids),
            self.page_rows,
            self.raw_count,
            self.actor_count,
            self.resident_bytes,
        )
        validate_paged_batch(batch, max_game_steps=self.max_game_steps)
        return batch

    def cache_statistics(self):
        return {
            name: kernel.cache_info()._asdict()
            for name, kernel in (
                ("initial_page", _initial_page),
                ("pack_page", _append_page_block),
                ("block_summary", _block_summary),
                ("packed_summary", _packed_summary),
                ("certificate", _certificate_page),
            )
        }

    def to_diagnostic(self):
        return dict(
            kind="raw_q_page_store",
            storage_version=2,
            pages=tuple(self._pages),
            game_lengths=self.lengths.copy(),
            completed=self.completed.copy(),
            actor_counts=self.actor_counts.copy(),
            learner_seats=self.learner_seats.copy(),
            page_game_ids=tuple(self._page_game_ids),
            page_rows=self.page_rows,
            current_used=self._used,
            raw_count=self.raw_count,
            actor_count=self.actor_count,
            resident_bytes=self.resident_bytes,
        )


# Publicly inspectable, owner-free bounded kernels for compilation-invariance tests.
pack_execution_page = _append_page_block
certify_raw_page = _certificate_page

# --- from vrpo_page_trace.py ---


def init_page_trace_carry(num_games, dtype=jnp.float32):
    if isinstance(num_games, bool) or not isinstance(num_games, int) or num_games <= 0:
        raise ValueError("num_games must be a positive integer")
    zero = jnp.zeros((num_games, 3), dtype)
    return PageTraceCarry(zero, zero)


def _qboost_page_serial_impl(page, q_values, carry, probabilities, gamma, trace_lambda, selected_games):
    """Original row-serial CPU oracle and exact single-present-game fallback."""
    raw = page.transitions
    games = carry.next_value.shape[0]
    safe_id = jnp.clip(page.game_ids, 0, games - 1)
    gid_valid = (page.game_ids >= 0) & (page.game_ids < games)
    real = raw.executed & gid_valid & selected_games[safe_id]
    # A replacement current policy is meaningful only for learner decisions.
    # Foreign AND forced transitions retain their actual collection law.
    current_decision = raw.is_learner & raw.is_decision
    law = jnp.where(current_decision[:, None], probabilities, raw.behavior_probs)
    valid = (
        jnp.all(raw_row_validity(raw))
        & jnp.all(~raw.executed | (gid_valid & (page.game_steps >= 0)))
        & jnp.all(raw.executed | ((page.game_ids == -1) & (page.game_steps == -1)))
        & jnp.all(~real | _probability_valid(law, raw.action_mask, raw.action))
        & jnp.all(~real | jnp.all(jnp.isfinite(q_values), axis=(-2, -1)))
        & jnp.all(jnp.isfinite(carry.next_value))
        & jnp.all(jnp.isfinite(carry.next_z))
        & jnp.isfinite(gamma)
        & (gamma >= 0)
        & (gamma <= 1)
        & jnp.isfinite(trace_lambda)
        & (trace_lambda >= 0)
        & (trace_lambda <= 1)
    )
    dtype = jnp.result_type(q_values, raw.reward, jnp.float32)
    q = jnp.where(real[:, None, None], q_values, 0).astype(dtype)
    probs = jnp.where(real[:, None], law, 0).astype(dtype)
    values = jnp.sum(q * probs[:, None, :], axis=-1)
    action = jnp.clip(raw.action, 0, 86)
    taken = jnp.take_along_axis(q, action[:, None, None], axis=-1)[..., 0]
    reward = jnp.where(real[:, None], raw.reward, 0).astype(dtype)
    zero = jnp.zeros(3, dtype)

    def step(state, row):
        gid, r, value, qt, terminal, executed = row

        def update(c):
            continuation = ~terminal
            delta = r + gamma * jnp.where(continuation, c.next_value[gid], zero) - qt
            z = delta + gamma * trace_lambda * jnp.where(continuation, c.next_z[gid], zero)
            return PageTraceCarry(c.next_value.at[gid].set(value), c.next_z.at[gid].set(z)), z

        # In particular, padding with clipped id=0 MUST NOT clear game zero's
        # carry between pages. Unselected games obey the same no-op contract.
        return jax.lax.cond(executed, update, lambda c: (c, zero), state)

    new_carry, z = jax.lax.scan(step, carry, (safe_id, reward, values, taken, raw.done, real), reverse=True)
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
    return labels, jax.tree.map(poison, new_carry), valid


# The CPU oracle has its own bounded reference cache; production inlines the
# same implementation into its branch and retains only the page owner's cache.
_qboost_page_serial = bounded_jit(max_entries=8)(_qboost_page_serial_impl)


def _stable_game_rows(safe_id, real, games, counts=None):
    """Stable grouped row permutation [P] and counts/offsets [G], never [G,P].

    Invalid/padding/unselected rows sort after every selected game. The stable
    sort preserves the original raw chronology *within* every game; game_steps
    are certification metadata, not a reason to reorder floating-point updates.
    """
    keys = jnp.where(real, safe_id, games).astype(jnp.int32)
    rowperm = jnp.argsort(keys, stable=True).astype(jnp.int32)
    if counts is None:
        counts = jnp.zeros(games, jnp.int32).at[keys].add(real.astype(jnp.int32), mode="drop")
    offsets = jnp.cumsum(counts, dtype=jnp.int32) - counts
    return rowperm, counts, offsets


def _qboost_page_parallel(page, q_values, carry, probabilities, gamma, trace_lambda, selected_games, counts):
    raw = page.transitions
    games = carry.next_value.shape[0]
    rows = raw.executed.shape[0]
    safe_id = jnp.clip(page.game_ids, 0, games - 1)
    gid_valid = (page.game_ids >= 0) & (page.game_ids < games)
    real = raw.executed & gid_valid & selected_games[safe_id]
    current_decision = raw.is_learner & raw.is_decision
    law = jnp.where(current_decision[:, None], probabilities, raw.behavior_probs)
    # Keep the serial reference's validation scope: raw behavior/reading checks
    # apply even to unselected executed games; replacement law/Q only to real.
    valid = (
        jnp.all(raw_row_validity(raw))
        & jnp.all(~raw.executed | (gid_valid & (page.game_steps >= 0)))
        & jnp.all(raw.executed | ((page.game_ids == -1) & (page.game_steps == -1)))
        & jnp.all(~real | _probability_valid(law, raw.action_mask, raw.action))
        & jnp.all(~real | jnp.all(jnp.isfinite(q_values), axis=(-2, -1)))
        & jnp.all(jnp.isfinite(carry.next_value))
        & jnp.all(jnp.isfinite(carry.next_z))
        & jnp.isfinite(gamma)
        & (gamma >= 0)
        & (gamma <= 1)
        & jnp.isfinite(trace_lambda)
        & (trace_lambda >= 0)
        & (trace_lambda <= 1)
    )
    dtype = jnp.result_type(q_values, raw.reward, jnp.float32)
    q = jnp.where(real[:, None, None], q_values, 0).astype(dtype)
    probs = jnp.where(real[:, None], law, 0).astype(dtype)
    values = jnp.sum(q * probs[:, None, :], axis=-1)
    action = jnp.clip(raw.action, 0, 86)
    taken = jnp.take_along_axis(q, action[:, None, None], axis=-1)[..., 0]
    reward = jnp.where(real[:, None], raw.reward, 0).astype(dtype)
    zero = jnp.zeros(3, dtype)
    rowperm, counts, offsets = _stable_game_rows(safe_id, real, games, counts)
    limit = jnp.max(counts)

    def step(state):
        rank, c, zs = state
        active = rank < counts
        grouped_index = jnp.clip(offsets + counts - 1 - rank, 0, rows - 1)
        row = rowperm[grouped_index]
        continuation = ~raw.done[row]
        # Exactly the original FP32 recurrence and operation order, with one
        # independent lane per game. No associative scan or 64-row truncation.
        delta = reward[row] + gamma * jnp.where(continuation[:, None], c.next_value, zero) - taken[row]
        z = delta + gamma * trace_lambda * jnp.where(continuation[:, None], c.next_z, zero)
        new_c = PageTraceCarry(
            jnp.where(active[:, None], values[row], c.next_value), jnp.where(active[:, None], z, c.next_z)
        )
        # Inactive lanes must not overwrite any real row (including row zero).
        output_row = jnp.where(active, row, rows)
        zs = zs.at[output_row].set(z, mode="drop")
        return rank + 1, new_c, zs

    z = jnp.zeros((rows, 3), dtype)
    if rows:  # Static empty-page compatibility; never depends on a game length.
        _, new_carry, z = jax.lax.while_loop(lambda s: s[0] < limit, step, (jnp.int32(0), carry, z))
    else:
        new_carry = carry
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
    return labels, jax.tree.map(poison, new_carry), valid


@bounded_jit(max_entries=8)
def _qboost_page(page, q_values, carry, probabilities, gamma, trace_lambda, selected_games):
    games = carry.next_value.shape[0]
    safe_id = jnp.clip(page.game_ids, 0, games - 1)
    real = page.transitions.executed & (page.game_ids >= 0) & (page.game_ids < games) & selected_games[safe_id]
    keys = jnp.where(real, safe_id, games).astype(jnp.int32)
    counts = jnp.zeros(games, jnp.int32).at[keys].add(real.astype(jnp.int32), mode="drop")
    single_present_game = jnp.sum(counts > 0, dtype=jnp.int32) == 1
    # Advancing G lanes for P iterations wastes work on a true single-game
    # long tail. Only this dynamic case uses the original serial recurrence;
    # multiple present games always retain the parallel inverse-local-rank path.
    # Count first so the single-game branch also avoids sorting the page.
    return jax.lax.cond(
        single_present_game,
        lambda _: _qboost_page_serial_impl(page, q_values, carry, probabilities, gamma, trace_lambda, selected_games),
        lambda _: _qboost_page_parallel(
            page, q_values, carry, probabilities, gamma, trace_lambda, selected_games, counts
        ),
        operand=None,
    )


# Additional conservative logical scratch charge, on top of the existing trace
# ledger: 32*P + (128 + 48*D)*G bytes, D=promoted trace dtype itemsize (FP32: 4).
# P: eight int32-sized row buffers cover stable-sort keys/permutation/work copies.
# G: 32 int32-sized slots cover counts/offsets/prefix/sort indices, rank/gather/
# scatter indices and boolean masks (each conservatively charged as int32);
# 48*D covers sixteen [G,3] gathered/recurrence vectors including both sides of
# carry updates and selection temporaries. This is logical array scratch,
# NOT a bound on backend sort/compiler workspace, allocator, or network memory.


def qboost_page_device(page, q_values, carry, *, probabilities=None, gamma=1.0, trace_lambda=0.95, selected_games=None):
    """Return ``(labels, carry, ok)`` without a host numerical check.

    Static shape/dtype checks also run when embedded in an owner's page JIT.
    ``ok`` is device-resident; callers must sticky-reduce and reject it before
    consuming a minibatch. Invalid numerics poison labels/carry as usual.
    """
    raw = page.transitions
    if raw.executed.ndim != 1:
        raise ValueError("qboost_page expects one flat fixed-size page")
    _check_shapes(raw, q_values, probabilities)
    n = raw.executed.shape[0]
    if (
        page.game_ids.shape != (n,)
        or page.game_steps.shape != (n,)
        or not jnp.issubdtype(page.game_ids.dtype, jnp.integer)
        or not jnp.issubdtype(page.game_steps.dtype, jnp.integer)
    ):
        raise ValueError("page game ids and steps must be integer row vectors")
    shape = carry.next_value.shape
    if len(shape) != 2 or shape[0] == 0 or shape[1] != 3 or carry.next_z.shape != shape:
        raise ValueError("page trace carries must have shape [positive G,3]")
    dtype = jnp.result_type(q_values, raw.reward, jnp.float32)
    if carry.next_value.dtype != dtype or carry.next_z.dtype != dtype:
        raise ValueError("page trace carry dtype must match the promoted Q/reward dtype")
    if selected_games is None:
        selected_games = jnp.ones(shape[0], jnp.bool_)
    else:
        selected_games = jnp.asarray(selected_games)
        if selected_games.shape != (shape[0],) or selected_games.dtype != jnp.bool_:
            raise ValueError("selected_games must be a boolean [G] vector")
    probabilities = raw.behavior_probs if probabilities is None else probabilities
    return _qboost_page(
        page,
        q_values,
        carry,
        probabilities,
        jnp.asarray(gamma, dtype),
        jnp.asarray(trace_lambda, dtype),
        selected_games,
    )


def qboost_page(page, q_values, carry, *, probabilities=None, gamma=1.0, trace_lambda=0.95, selected_games=None):
    """Return ``(labels, carry)`` for one page, visiting pages in reverse order.

    ``selected_games`` is a dynamic bool[G] whole-game mask, never a row subset.
    Current probabilities replace learner decision laws only. No stop_gradient or
    loss normalization is applied here. Malformed numerical inputs raise eagerly
    and poison outputs under an outer JIT, as in the flat reference trace.
    """
    labels, new_carry, ok = qboost_page_device(
        page,
        q_values,
        carry,
        probabilities=probabilities,
        gamma=gamma,
        trace_lambda=trace_lambda,
        selected_games=selected_games,
    )
    _eager_check(ok, "invalid raw Q page trace: game metadata, Q or probability support")
    return labels, new_carry


qboost_page.clear_cache = _qboost_page.clear_cache
qboost_page.cache_info = _qboost_page.cache_info
qboost_page_device.clear_cache = _qboost_page.clear_cache
qboost_page_device.cache_info = _qboost_page.cache_info

# --- from vrpo_rollout.py ---


class RawCollectorState(NamedTuple):
    env_state: object
    rng: object
    finished: object
    steps: object
    failed: object

    @property
    def completed(self):
        return self.finished & self.env_state.terminated & ~self.env_state.truncated & ~self.failed

    @property
    def truncated(self):
        return self.env_state.truncated


def init_raw_collector(env_state, rng):
    """Start fresh games from a batched state; accept a root or per-lane keys.

    ``steps`` counts this fresh collection, not an arbitrary custom state's
    counters. The reward envelope is proved only for fresh sanma factory games.
    """
    shape = jnp.shape(env_state.current_player)
    if len(shape) != 1 or shape[0] < 1:
        raise ValueError("raw collector requires a nonempty batched environment")
    e = shape[0]
    rng = jnp.asarray(rng)
    typed = jax.dtypes.issubdtype(rng.dtype, jax.dtypes.prng_key)
    if (typed and rng.shape == ()) or (not typed and rng.shape == (2,)):
        rng = jax.random.split(rng, e)
    if rng.shape != ((e,) if typed else (e, 2)):
        raise ValueError("rng must be one root key or one key per environment")
    terminated, truncated = env_state.terminated, env_state.truncated
    if terminated.shape != shape or truncated.shape != shape:
        raise ValueError("terminal flags must match the environment batch")
    return RawCollectorState(env_state, rng, terminated | truncated, jnp.zeros(e, jnp.int32), jnp.zeros(e, jnp.bool_))


init_runner = init_raw_collector


def raw_block_context(runner, raw):
    """Device-only append context; failed lanes are rejected like truncations.

    Pass this dictionary directly to ``RawQBuffer.append_execution_block``.
    Actual environment truncation remains separately available on the runner.
    """
    return dict(
        lengths=raw.executed.sum(axis=1, dtype=jnp.int32),
        completed=runner.completed,
        truncated=runner.truncated | runner.failed,
    )


def _context(policy_context, learner_seats):
    e = learner_seats.shape[0]
    if policy_context is None:
        policy_context = {}
    if not isinstance(policy_context, Mapping) or set(policy_context) - set(POLICY_CONTEXT_SHAPES):
        raise ValueError("policy_context only accepts fixed identity fields, never RNG/labels")
    defaults = dict(
        learner_seat=learner_seats,
        policy_ids=jnp.zeros((e, 3), jnp.int32),
        policy_versions=jnp.zeros((e, 3), jnp.int32),
        role_ids=jnp.zeros((e, 3), jnp.int32),
    )
    out = {}
    for key, shape in POLICY_CONTEXT_SHAPES.items():
        value = jnp.asarray(policy_context.get(key, defaults[key]))
        if value.shape != (e,) + shape or value.dtype.kind not in "iu":
            raise ValueError(f"policy_context {key} must have integer shape {(e,) + shape}")
        out[key] = value.astype(jnp.int32)
    valid = (
        (out["learner_seat"] == learner_seats)
        & (learner_seats >= -1)
        & (learner_seats < 3)
        & jnp.all(out["policy_ids"] >= 0, axis=-1)
    )
    return out, valid


def make_raw_execution_block(  # noqa: C901 - ported verbatim; refactor would risk training semantics
    env,
    *,
    actor_logits_fn,
    foreign_logits_fn=None,  # noqa: C901 - ported verbatim; refactor would risk training semantics
    temperature=1.0,
    num_steps=256,
    policy_context_fn=None,
    input_builder_fn=None,
    foreign_player_arg=False,
    all_seats=False,
    reading_targets_fn=None,
    lazy_env_step=False,
):
    """Build a pure, shape-bounded (8 entries) jitted execution callback.

    Returned signature::

      collect(params, foreign_params, foreign_context, runner, learner_seats,
              policy_context=None) -> (runner, RawQTransition[E,T,...])

    Actor callback: ``(params, prepared_public[E,...]) -> logits[E,87]``.
    Foreign callback: ``(foreign_params, foreign_context, prepared_public)`` with
    the same output. It must route the already-fixed per-game frozen policies;
    rule policies are unsupported. Foreign temperature is exactly one (a callback
    may explicitly return pre-scaled logits for another frozen policy protocol).
    With ``foreign_player_arg=True``, the foreign callback additionally receives
    ``player[E], mask[E,87]`` after public observations. These PRE-state public
    metadata enable absolute-seat bank routing without injecting Actor features.
    The learner temperature applies only to owned decisions. Forced rows store
    exact one-hot laws. Both ownership and cached laws refer to the PRE-state.
    Static ``all_seats=True`` requires seat -1 and never traces a foreign callback.
    Optional ``reading_targets_fn(pre_state)`` is a training-only current-state
    oracle, evaluated only on active owned multi-option rows. Its output lives
    solely in ``reading_targets`` and is NEVER passed to either network callback.
    ``lazy_env_step=True`` uses scalar conditional mapping and ``env.step_lazy``
    when available (otherwise ``env.step``), without changing per-lane RNG splits.

    ``policy_context_fn(context, learner_seats)`` optionally constructs the fixed
    allowlisted identity dictionary. ``input_builder_fn(state, public, context)``
    defaults to the reviewed sanma central builder; override only for pure toy
    fixtures, not to add RNG or labels. No callback receives private features.
    """
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError("num_steps must be a positive integer")
    if (
        isinstance(temperature, bool)
        or not np.isscalar(temperature)
        or not np.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("temperature must be finite and positive")
    if not isinstance(foreign_player_arg, bool):
        raise ValueError("foreign_player_arg must be bool")
    if type(all_seats) is not bool:
        raise ValueError("all_seats must be bool")
    if type(lazy_env_step) is not bool:
        raise ValueError("lazy_env_step must be bool")
    if not callable(actor_logits_fn) or (not all_seats and not callable(foreign_logits_fn)):
        raise TypeError("actor callback and, outside joint self-play, foreign callback required")
    if reading_targets_fn is not None and not callable(reading_targets_fn):
        raise TypeError("reading_targets_fn must be a pure current-state oracle or None")
    builder = build_central_q_input if input_builder_fn is None else input_builder_fn
    step_env = getattr(env, "step_lazy", env.step) if lazy_env_step else env.step
    # Environment order_points are in HUNDREDS, as are env_state.rewards.
    uma = np.asarray(getattr(env, "order_points", [200, 0, -200]), np.float64)
    if uma.shape != (3,) or not np.isfinite(uma).all():
        raise ValueError("sanma order_points must contain three finite hundred-point values")
    bound_base = (1344000 + 1000 + float(np.max(np.abs(uma))) * 100.0) / 32000.0
    bound_growth = 2400.0 / 32000.0

    @bounded_jit(max_entries=8)
    def collect(params, foreign_params, foreign_context, runner, learner_seats, policy_context=None):
        learner_seats = jnp.asarray(learner_seats)
        e = runner.finished.shape[0]
        if learner_seats.shape != (e,) or learner_seats.dtype.kind not in "iu":
            raise ValueError("learner_seats must be an integer vector per environment")
        context = policy_context if policy_context_fn is None else policy_context_fn(policy_context, learner_seats)
        context, context_ok = _context(context, learner_seats)
        if all_seats:
            context_ok &= learner_seats == -1

        def step(carry, _):
            state, keys, finished, steps, failed = carry
            active = ~(finished | failed | state.terminated | state.truncated)
            player = state.current_player.astype(jnp.int32)
            raw_mask = state.legal_action_mask
            simulator_mask = raw_mask.shape[-1] == 88
            mask = compact_legal_mask(raw_mask) if simulator_mask else raw_mask
            if mask.shape != (e, 87) or mask.dtype != jnp.bool_:
                raise ValueError("environment legal mask must be bool[E,87]")
            # Build directly from the PRE-action state. The generated current wall
            # is allowed; the RNG keys used for future walls are never passed.
            public = jax.vmap(lambda s: compact_observation(env.observe(s)))(state)
            central = jax.vmap(builder)(state, public, context)
            if set(central) != {"observation", "privileged"}:
                raise ValueError("input builder must return observation and privileged only")
            public, private = central["observation"], central["privileged"]
            owned = (learner_seats < 0) | (learner_seats == player)
            decision = mask.sum(axis=-1) > 1
            reading_targets = None
            if reading_targets_fn is not None:
                from .auxhead import empty_reading_targets

                # Sequential map retains conditional skipping of expensive oracle
                # scoring for inactive/forced rows; vmap(cond) would select both.
                reading_targets = jax.lax.map(
                    lambda pair: jax.lax.cond(pair[1], reading_targets_fn, lambda _: empty_reading_targets(), pair[0]),
                    (state, active & owned & decision),
                )
            zeros = jnp.zeros((e, 87), jnp.float32)
            actor_logits = jax.lax.cond(
                jnp.any(active & owned & decision), lambda: actor_logits_fn(params, public), lambda: zeros
            )

            def evaluate_foreign():
                if foreign_player_arg:
                    return foreign_logits_fn(foreign_params, foreign_context, public, player, mask)
                return foreign_logits_fn(foreign_params, foreign_context, public)

            foreign_logits = (
                zeros
                if all_seats
                else jax.lax.cond(jnp.any(active & ~owned & decision), evaluate_foreign, lambda: zeros)
            )
            if actor_logits.shape != (e, 87) or foreign_logits.shape != (e, 87):
                raise ValueError("policy callbacks must return logits[E,87]")
            logits = jnp.where(owned[:, None], actor_logits / temperature, foreign_logits)
            finite = jnp.all(~mask | jnp.isfinite(logits), axis=-1)
            logits = jnp.where(mask, logits, -jnp.inf)
            probabilities = jax.nn.softmax(logits, axis=-1)
            forced = jax.nn.one_hot(jnp.argmax(mask, axis=-1), 87, dtype=jnp.float32)
            probabilities = jnp.where(decision[:, None], probabilities, forced)
            # Keep a fixed split schedule even on forced rows. Inactive lanes keep
            # their key entirely unchanged and never start a replacement game.
            split = jax.vmap(lambda key: jax.random.split(key, 3))(keys)
            next_keys, action_keys, env_keys = split[:, 0], split[:, 1], split[:, 2]
            action = jax.vmap(jax.random.categorical)(
                action_keys, jnp.where(probabilities > 0, jnp.log(probabilities), -jnp.inf)
            )
            action = action.astype(jnp.int32)
            safe_player = jnp.clip(player, 0, 2)
            policy_id = jnp.take_along_axis(context["policy_ids"], safe_player[:, None], axis=1)[:, 0]
            mask_ok = mask.any(axis=-1)
            # Assert that the central builder did not pair another state's mask.
            if "legal_action_mask" in private:
                mask_ok &= jnp.all(private["legal_action_mask"] == raw_mask, axis=-1)
            behavior_ok = (
                context_ok
                & mask_ok
                & (player >= 0)
                & (player < 3)
                & (~decision | finite)
                & jnp.all(jnp.isfinite(probabilities), axis=-1)
                & jnp.take_along_axis(mask, action[:, None], axis=1)[:, 0]
            )
            execute = active & behavior_ok
            if lazy_env_step:
                # Scalar map preserves lazy action dispatch and inactive skipping;
                # vmap(cond/switch) would turn these into eager batched selects.
                simulator_action = to_simulator_action(action) if simulator_mask else action
                next_state = jax.lax.map(
                    lambda x: jax.lax.cond(x[3], lambda: step_env(x[0], x[1], x[2]), lambda: x[0]),
                    (state, simulator_action, env_keys, execute),
                )
            else:
                simulator_action = to_simulator_action(action) if simulator_mask else action
                next_state = jax.vmap(lambda s, a, k, go: jax.lax.cond(go, lambda: step_env(s, a, k), lambda: s))(
                    state, simulator_action, env_keys, execute
                )
            reward = jnp.asarray(next_state.rewards, jnp.float32) / 320.0
            if reward.shape != (e, 3):
                raise ValueError("environment rewards must be absolute-seat [E,3]")
            reward_ok = jnp.all(jnp.isfinite(reward), axis=-1)
            bad = active & (~behavior_ok | ~reward_ok | next_state.truncated)
            next_failed = failed | bad
            next_finished = finished | next_state.terminated | next_state.truncated | next_failed
            row = RawQTransition(
                public,
                private,
                jnp.where(active, action, 0),
                mask,
                jnp.where(active[:, None], jnp.where(~bad[:, None], probabilities, jnp.nan), 0.0),
                jnp.where(active[:, None], jnp.where(behavior_ok[:, None], reward, jnp.nan), 0.0),
                player,
                active & next_state.terminated & ~next_state.truncated & ~bad,
                active & decision,
                active & owned,
                active,
                policy_id,
                reading_targets,
            )
            next_keys = jax.tree.map(
                lambda new, old: jnp.where(active.reshape((e,) + (1,) * (new.ndim - 1)), new, old), next_keys, keys
            )
            carry = RawCollectorState(
                next_state, next_keys, next_finished, steps + active.astype(jnp.int32), next_failed
            )
            return carry, row

        next_runner, raw = jax.lax.scan(step, runner, None, length=num_steps)
        raw = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), raw)
        # Match the legacy proof: each lane's allowance uses actual cumulative
        # executed length AFTER this block, never width/padding/safety limits.
        bounds = bound_base + bound_growth * next_runner.steps
        tolerance = 8 * jnp.finfo(raw.reward.dtype).eps * jnp.maximum(bounds, 1.0)
        excess = raw.executed & jnp.any(jnp.abs(raw.reward) > (bounds + tolerance)[:, None, None], axis=-1)
        failed = next_runner.failed | excess.any(axis=1)
        # Retain the ORIGINAL finite payout for failed-block forensic replay;
        # poison its law (rather than replacing the evidence with NaN).
        raw = raw._replace(behavior_probs=jnp.where(excess[..., None], jnp.nan, raw.behavior_probs))
        next_runner = next_runner._replace(failed=failed, finished=next_runner.finished | failed)
        return next_runner, raw

    return collect


# --- from vrpo_selfplay.py ---

SELFPLAY_KIND = "current_joint_all_seats"
_MAX_GAME_ID = 2**63 - 1


def initial_collection_state():
    return {"version": 1, "kind": SELFPLAY_KIND, "next_game_id": 0}


def actor_sha256(params) -> str:
    """Hash the parameter tree; the calibration fingerprint uses the same walk."""
    import hashlib

    import jax

    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(params):
        digest.update(str(getattr(leaf, "shape", ())).encode())
        digest.update(str(getattr(leaf, "dtype", "")).encode())
        digest.update(np.asarray(leaf).tobytes())
    return digest.hexdigest()


def checked_collection_state(value, *, expected_next_game_id=None):
    """Validate without mutation or inferred migration; return a detached copy."""
    if (
        not isinstance(value, Mapping)
        or set(value) != {"version", "kind", "next_game_id"}
        or type(value["version"]) is not int
        or value["version"] != 1
        or value["kind"] != SELFPLAY_KIND
    ):
        raise ValueError("explicit current-joint-all-seats collection state required")
    count = value["next_game_id"]
    if type(count) is not int or not 0 <= count <= _MAX_GAME_ID:
        raise ValueError("next_game_id must be a nonnegative int64-range host integer")
    if expected_next_game_id is not None:
        if (
            type(expected_next_game_id) is not int
            or not 0 <= expected_next_game_id <= _MAX_GAME_ID
            or count != expected_next_game_id
        ):
            raise ValueError("collection game clock differs from complete update boundary")
    return {"version": 1, "kind": SELFPLAY_KIND, "next_game_id": count}


def advance_collection_state(value, games):
    candidate = checked_collection_state(value)
    if type(games) is not int or games <= 0:
        raise ValueError("collection game increment must be a positive integer")
    candidate["next_game_id"] += games
    return checked_collection_state(candidate)


# --- from vrpo_runtime.py ---


@dataclass(frozen=True, kw_only=True)
class VRPORunConfig:
    algorithm: VRPOConfig
    bc_path: str
    bc_sha256: str
    output_dir: str
    run_id: str
    metrics_path: str | None = None
    tensorboard_dir: str | None = None
    eval_interval: int = 5
    baseline_model: str | None = None
    baseline_sha256: str | None = None
    nums_eval: int = 30
    seed: int = 0
    max_updates: int | None = None
    max_wall_seconds: float | None = None
    min_free_bytes: int = 4 * 1024**3
    actor_stall_limit: int = 3

    def __post_init__(self):
        if not isinstance(self.algorithm, VRPOConfig):
            raise ValueError("explicit VRPOConfig required")
        for name in ("bc_path", "output_dir", "run_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(name + " must be explicit")
        if self.metrics_path is not None and (not isinstance(self.metrics_path, str) or not self.metrics_path.strip()):
            raise ValueError("metrics_path must be a nonempty path string or null")
        if self.tensorboard_dir is not None and (
            not isinstance(self.tensorboard_dir, str) or not self.tensorboard_dir.strip()
        ):
            raise ValueError("tensorboard_dir must be a nonempty path string or null")
        for name in ("eval_interval", "nums_eval"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(name + " must be a positive integer")
        if self.baseline_model is not None and (
            not isinstance(self.baseline_model, str) or not self.baseline_model.strip()
        ):
            raise ValueError("baseline_model must be a nonempty path string or null")
        if self.baseline_model is None:
            if self.baseline_sha256 is not None:
                raise ValueError("baseline_sha256 requires baseline_model")
        else:
            _digest(self.baseline_sha256)
        _digest(self.bc_sha256)
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be a uint32-range integer")
        for name in ("min_free_bytes", "actor_stall_limit"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(name + " must be positive")
        if self.max_updates is not None and (type(self.max_updates) is not int or self.max_updates <= 0):
            raise ValueError("max_updates must be a positive integer")
        if self.max_wall_seconds is not None and (
            isinstance(self.max_wall_seconds, bool)
            or not isinstance(self.max_wall_seconds, (int, float))
            or not math.isfinite(self.max_wall_seconds)
            or self.max_wall_seconds <= 0
        ):
            raise ValueError("max_wall_seconds must be finite and positive")

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or not isinstance(data.get("algorithm"), dict):
            raise ValueError("configuration requires explicit algorithm mapping")
        return cls(**(data | {"algorithm": VRPOConfig(**data["algorithm"])}))

    def to_dict(self):
        return asdict(self)


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(x not in "0123456789abcdef" for x in value):
        raise ValueError("explicit lowercase SHA256 required before deserialization")
    return value


def load_verified(path, sha256):
    expected = _digest(sha256)
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("VRPO SHA256 mismatch BEFORE deserialization")
    return pickle.loads(data)


def checked_parameters(loaded, template):
    if jax.tree.structure(loaded) != jax.tree.structure(template):
        raise ValueError("parameter tree mismatch; no implicit Actor/Q migration")
    for actual, expected in zip(jax.tree.leaves(loaded), jax.tree.leaves(template), strict=True):
        x = np.asarray(actual)
        if x.shape != expected.shape or x.dtype != expected.dtype or not np.isfinite(x).all():
            raise ValueError("parameter shape/dtype/finite mismatch")
    return jax.tree.map(jnp.asarray, loaded)


def _bc_actor(checkpoint, observation, *, debug=False, reading_auxiliary=False):
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("checkpoint_kind") != "bc"
        or type(checkpoint.get("schema_version")) is not int
        or checkpoint["schema_version"] != 1
    ):
        raise ValueError("VRPO starts from an explicit original BC envelope, not PPO weights")
    for name, value in (("env_name", "sanma"), ("round_mode", "half"), ("ruleset", RULESET)):
        if checkpoint.get(name) != value:
            raise ValueError("BC " + name + " mismatch")
    network_config = checkpoint_network_config(checkpoint)
    # Legacy BC envelopes may omit the head; disabled (version 0) is the only
    # safe legacy default.  Metadata is never promoted to root provenance.
    head = (
        checkpoint_head_config(checkpoint) if checkpoint.get("head_config") is not None else dict(DISABLED_HEAD_CONFIG)
    )
    reading = _checked_bc_reading(checkpoint, network_config, reading_auxiliary)
    # Size and head schema come from the authenticated BC root, never a preset
    # whitelist. ``debug`` remains a source-compatible no-op; checks are identical.
    variables = checkpoint.get("params")
    if not isinstance(variables, Mapping):
        raise ValueError("BC params missing")
    variables = variables if "params" in variables else {"params": variables}
    required = ("policy_extractor", "policy_mlp")
    if not isinstance(variables["params"], Mapping) or not all(x in variables["params"] for x in required):
        raise ValueError("BC Actor roots missing")
    actor = {"params": {name: unfreeze(variables["params"][name]) for name in required}}
    factory_kwargs = {"head_config": head}
    if reading is not None:
        factory_kwargs["reading_config"] = reading
    model = get_network_cls("sanma", **network_config, **factory_kwargs)()
    abstract = jax.eval_shape(
        lambda key, obs: model.init(key, obs, method=model.get_action_logits), jax.random.PRNGKey(0), observation
    )
    actor = checked_parameters(actor, abstract)
    return actor, network_config, head


def _checked_bc_reading(checkpoint, network_config, reading_auxiliary):
    """Require same-architecture BC, with explicit V5 root provenance first."""
    if type(reading_auxiliary) is not bool:
        raise ValueError("reading_auxiliary must be bool")
    # The consolidated stack has one architecture.  Legacy algorithm schemas
    # simply do not attach the reading head; they still use v5 weights.
    if network_config["architecture"] != "v5":
        raise ValueError("BC Actor must be V5")
    if reading_auxiliary and checkpoint.get("head_config") is None:
        raise ValueError("V5 BC requires explicit root head_config; no legacy head defaults")
    return checkpoint_reading_config(checkpoint) if reading_auxiliary else None


def initialize_bc(checkpoint, options):
    """Clone the complete same-architecture BC Actor/public encoder; Q is fresh."""
    network_config = checkpoint_network_config(checkpoint)
    reading = _checked_bc_reading(checkpoint, network_config, options.algorithm.reading_auxiliary)
    env = rule_spec("sanma").make_env("half", reward_config=default_reward_config("sanma"))
    root = jax.random.PRNGKey(options.seed)
    root, env_key, network_key = jax.random.split(root, 3)
    initial = env.init(env_key)
    central = build_central_q_input(initial, compact_observation(env.observe(initial)))
    actor, network_config, heads = _bc_actor(
        checkpoint,
        central["observation"],
        debug=options.algorithm.preset == "tiny_debug",
        reading_auxiliary=options.algorithm.reading_auxiliary,
    )
    model = build_central_q(
        network_config, head_config=heads, critic_config=dict(DEFAULT_CRITIC_CONFIG), reading_config=reading
    )
    variables = model.init(network_key, central["observation"], central)
    variables = clone_actor_and_public_encoder(actor, variables)
    if actor_sha256(actor) != actor_sha256(phase_params(variables, "actor")):
        raise ValueError("BC Actor changed during Q initialization")
    state, _, _ = create_training_state(variables, options.algorithm)
    return env, model, state, root, actor, network_config, heads


def source_versions():
    root = Path(__file__).resolve().parents[1]
    sources = {}
    for directory in ("train", "networks", "mahjax"):
        for path in sorted((root / directory).rglob("*.py")):
            sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "packages": {name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "flax", "optax")},
        "implementation_sha256": sources,
    }


def make_contract(options, network_config, head_config):
    config = options.algorithm
    if network_config["architecture"] != "v5":
        raise ValueError("network architecture must be v5")
    from .optimizer import vrpo_optimizer_metadata

    algorithm = config.algorithm_config() | {
        "collection_profile": {"kind": SELFPLAY_KIND, "seed": options.seed, "bc_sha256": options.bc_sha256}
    }
    return dict(
        network_config=dict(network_config),
        head_config=dict(head_config),
        **({"reading_config": validate_reading_config(None, "v5")} if config.reading_auxiliary else {}),
        critic_config=dict(DEFAULT_CENTRAL_Q_CRITIC_CONFIG),
        algorithm_config=algorithm,
        reward_config=default_reward_config("sanma"),
        optimizer_config=vrpo_optimizer_metadata(config),
        environment_config={"env_name": "sanma", "round_mode": "half", "ruleset": RULESET},
        source_versions=source_versions(),
        retention_enabled=config.bc_reference_coef > 0,
    )


def checked_vrpo_collection_state(value, options, *, completed_updates):
    """Bind the host identity clock to a COMPLETE online update, never a pool."""
    if type(completed_updates) is not int or completed_updates < 0:
        raise ValueError("invalid complete VRPO update counter")
    return checked_collection_state(value, expected_next_game_id=completed_updates * options.algorithm.rollout_games)


def diagnostic_batch(value):
    return value.to_diagnostic() if isinstance(value, PagedRawQBatch) else value


def _staging_evidence(value):
    from .selfplay import RawQPageStore

    if isinstance(value, RawQPageStore):
        return value.to_diagnostic()
    if isinstance(value, tuple):
        return tuple(_staging_evidence(x) for x in value)
    if isinstance(value, RawQCohortStager):
        return dict(
            kind="completed_cohorts",
            batches=tuple(value._batches),
            raw_count=value.raw_count,
            actor_count=value.actor_count,
        )
    if isinstance(value, RawQBuffer):
        return dict(
            kind="partial_cohort",
            blocks=tuple(value._blocks),
            counts=tuple(value._counts),
            lengths=value.lengths.copy(),
            completed=value.completed.copy(),
            learner_seats=value.learner_seats.copy(),
        )
    raise TypeError("unsupported diagnostic staging owner")


class RawCollectionFailure(RuntimeError):
    def __init__(self, message, *, runner, raw, params, staged, collection_state=None):
        super().__init__(message)
        self.evidence = dict(
            pre_runner=runner,
            first_bad_block=raw,
            frozen_params=params,
            staged=_staging_evidence(staged),
            collection_state=(None if collection_state is None else checked_collection_state(collection_state)),
        )


class CompleteGameCollector:
    """No mutable owner enters a JIT closure; all profiles freeze for one batch."""

    def __init__(self, env, model, options):
        self.env, self.options = env, options
        config = options.algorithm
        reading_kwargs = {}
        if config.reading_auxiliary:
            from .auxhead import build_reading_targets

            reading_kwargs["reading_targets_fn"] = build_reading_targets
            reading_kwargs["lazy_env_step"] = True
        self.block = make_raw_execution_block(
            env,
            actor_logits_fn=lambda variables, obs: model.apply(variables, obs, method=model.get_action_logits),
            all_seats=True,
            temperature=config.temperature,
            num_steps=config.execution_block_steps,
            **reading_kwargs,
        )
        self.reset = bounded_jit(max_entries=8)(lambda keys: jax.vmap(env.init)(keys))
        self._block_abstract = None
        self._last_storage_statistics = {}

    def cache_statistics(self):
        return {
            "reset": self.reset.cache_info()._asdict(),
            "block": self.block.cache_info()._asdict(),
            "storage": self._last_storage_statistics,
        }

    def close(self):
        self.block.clear_cache()
        self.reset.clear_cache()

    def collect(
        self, state, rng, collection_state, *, num_games=None, stop_requested=None, reserved_bytes=0, purpose="online"
    ):
        from .selfplay import RawQPageStore

        cfg = self.options.algorithm
        collection_state = checked_collection_state(collection_state)
        if type(reserved_bytes) is not int or reserved_bytes < 0 or reserved_bytes >= cfg.max_bytes:
            raise MemoryError("invalid/exhausted external trajectory byte reservation")
        if purpose not in ("online", "calibration"):
            raise ValueError("unknown paged collection resource profile")
        available_bytes = cfg.max_bytes - reserved_bytes
        games = cfg.rollout_games if num_games is None else num_games
        if type(games) is not int or games <= 0 or games % cfg.num_envs:
            raise ValueError("complete game count must divide into configured cohorts")
        frozen = jax.tree.map(jax.lax.stop_gradient, phase_params(state.params, "actor"))
        next_collection = advance_collection_state(collection_state, games)
        first_game_id = collection_state["next_game_id"]
        seats = np.full(cfg.num_envs, -1, np.int32)
        device_seats = jnp.asarray(seats)
        policy_ids = jnp.zeros((cfg.num_envs, 3), jnp.int32)
        context = dict(
            learner_seat=device_seats,
            policy_ids=policy_ids,
            policy_versions=jnp.full((cfg.num_envs, 3), state.completed_updates, jnp.int32),
            role_ids=jnp.zeros((cfg.num_envs, 3), jnp.int32),
        )
        # Raw pages plus future Q-old/targets (online) or MC targets (calibration).
        reserve = 3 * 87 * 4 + 3 * 4 if purpose == "online" else 3 * 4
        store = RawQPageStore(
            games,
            num_envs=cfg.num_envs,
            execution_block_steps=cfg.execution_block_steps,
            page_rows=cfg.page_rows,
            max_game_steps=cfg.max_game_steps,
            max_bytes=available_bytes,
            reserve_bytes_per_row=reserve,
        )
        ids = []
        before = raw = runner = None
        try:
            for cohort in range(games // cfg.num_envs):
                if stop_requested is not None and stop_requested():
                    raise RawCollectionFailure(
                        "stop during collection; not a resume boundary",
                        runner=None,
                        raw=None,
                        params=frozen,
                        staged=store,
                        collection_state=collection_state,
                    )
                rng, init_key, play_key = jax.random.split(rng, 3)
                env_state = self.reset(jax.random.split(init_key, cfg.num_envs))
                runner = init_raw_collector(env_state, jax.random.split(play_key, cfg.num_envs))
                game_ids = np.arange(cohort * cfg.num_envs, (cohort + 1) * cfg.num_envs, dtype=np.int32)
                ids.extend(f"{self.options.seed}:{first_game_id + int(i)}" for i in game_ids)
                while not np.all(store.completed[game_ids]):
                    if stop_requested is not None and stop_requested():
                        raise RawCollectionFailure(
                            "stop during collection; not a resume boundary",
                            runner=runner,
                            raw=None,
                            params=frozen,
                            staged=store,
                            collection_state=collection_state,
                        )
                    before, raw = runner, None
                    if self._block_abstract is None:
                        self._block_abstract = jax.eval_shape(
                            self.block, frozen, (), policy_ids, runner, device_seats, context
                        )[1]
                    store.admit_next_block(self._block_abstract)
                    runner, raw = self.block(frozen, (), policy_ids, runner, device_seats, context)
                    if np.any(np.asarray(runner.failed)):
                        raise ValueError("collector law/reward guard failed")
                    store.append_execution_block(
                        raw, game_ids=game_ids, learner_seats=seats, **raw_block_context(runner, raw)
                    )
                    # A committed page owns its values; do not retain successful source blocks.
                    before = raw = None
                runner = None
                del env_state
            batch = store.export_complete_batch()
            self._last_storage_statistics = store.cache_statistics()
            return (
                batch,
                rng,
                next_collection,
                dict(
                    game_ids=ids,
                    raw_rows=store.raw_count,
                    actor_rows=store.actor_count,
                    storage_version=2,
                    page_rows=cfg.page_rows,
                    page_count=len(batch.pages),
                    storage_bytes=batch.resident_bytes,
                ),
            )
        except RawCollectionFailure:
            raise
        except Exception as error:
            raise RawCollectionFailure(
                str(error),
                runner=before if before is not None else runner,
                raw=raw,
                params=frozen,
                staged=store,
                collection_state=collection_state,
            ) from error


def write_new_pickle(path, payload, *, min_free_bytes):
    """Exclusive standalone calibration/diagnostic artifact, never replaces data."""
    path = Path(path)
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    if shutil.disk_usage(path.parent).free < len(data) + min_free_bytes:
        raise OSError("VRPO standalone artifact would violate disk reserve")
    import os

    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(data).hexdigest()


def load_vrpo_actor(checkpoint, observation):
    """Public-only inference for new Q artifacts; never constructs/loads a Q trunk."""
    from .configs import ALGORITHM_KIND

    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("checkpoint_kind") not in ("vrpo_model", "vrpo_q_calibrated")
        or type(checkpoint.get("schema_version")) is not int
        or checkpoint["schema_version"] != 1
    ):
        raise ValueError("Actor inference requires an explicit ordinary VRPO model")
    for name, expected in (("env_name", "sanma"), ("round_mode", "half"), ("ruleset", RULESET)):
        if checkpoint.get(name) != expected:
            raise ValueError("VRPO inference environment mismatch")
    if checkpoint.get("head_config") is None or checkpoint_critic_config(checkpoint) != DEFAULT_CENTRAL_Q_CRITIC_CONFIG:
        raise ValueError("new VRPO artifacts require explicit Actor and Q schemas")
    algorithm = checkpoint.get("algorithm_config")
    config, head = checkpoint_network_config(checkpoint), checkpoint_head_config(checkpoint)
    reading_kwargs = {}
    legacy = (
        isinstance(algorithm, Mapping)
        and type(algorithm.get("version")) is int
        and (algorithm["version"], algorithm.get("kind"))
        in ((1, "vrpo_qboost_chance_residual_v1"), (2, ALGORITHM_KIND))
        and algorithm.get("reading_auxiliary", False) is False
    )
    reading = isinstance(algorithm, Mapping) and algorithm.get("reading_auxiliary") is True
    if reading:
        from .checkpoint import validate_v5_reading_algorithm

        validate_v5_reading_algorithm(algorithm)
        reading_kwargs["reading_config"] = checkpoint_reading_config(checkpoint)
    elif not legacy:
        raise ValueError("unsupported VRPO algorithm/network schema")
    model = build_actor(config, head_config=head, **reading_kwargs)
    params = checkpoint.get("params")
    if not isinstance(params, Mapping) or "params" not in params or not isinstance(params["params"], Mapping):
        raise ValueError("VRPO variables missing")
    roots = ("policy_extractor", "policy_mlp")
    if not all(name in params["params"] for name in roots):
        raise ValueError("VRPO Actor roots missing")
    actor = {"params": {name: unfreeze(params["params"][name]) for name in roots}}
    abstract = jax.eval_shape(
        lambda key, obs: model.init(key, obs, method=model.get_action_logits), jax.random.PRNGKey(0), observation
    )
    return model, checked_parameters(actor, abstract)


# Re-exports used by tests.
from networks.mahjong_3p.network import (  # noqa: E402, F401
    get_central_q_network_cls,
    get_network_cls,
)
