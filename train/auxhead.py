"""Opponent-reading auxiliary targets and their oracle."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from mahjax.common.hand import Hand
from mahjax.common.meld import EMPTY_MELD
from mahjax.common.shanten import Shanten
from mahjax.common.tile import River, Tile
from mahjax.common.yaku import Yaku
from mahjax.mahjong_3p.action import Action

TILE_TYPES = 34
NORTH_TYPE = 30

TILE_TYPES = 34
NORTH_TYPE = 30

# --- from reading.py ---

READING_SCHEMA_VERSION = 1
POINTS_SCALE = 8000
TASK_NAMES = ("tenpai", "shanten", "danger", "waits", "danger_points")
CANONICAL_TILE_EXISTS = (True,) + (False,) * 7 + (True,) * 26


class ReadingTargets(NamedTuple):
    tenpai: jax.Array  # bool[..., 2]
    shanten: jax.Array  # int32[..., 2], 0, 1, 2, 3+
    danger: jax.Array  # bool[..., 2, 34], legal independent ron
    waits: jax.Array  # bool[..., 2, 34], shape only
    danger_points: jax.Array  # float32[..., 2, 34], raw ron income / 8000
    tenpai_valid: jax.Array  # bool, same shape as corresponding label
    shanten_valid: jax.Array
    danger_valid: jax.Array
    waits_valid: jax.Array
    danger_points_valid: jax.Array


def empty_reading_targets(leading_shape=()):
    if isinstance(leading_shape, int):
        leading_shape = (leading_shape,)
    leading_shape = tuple(leading_shape)
    seats, tiles = leading_shape + (2,), leading_shape + (2, 34)
    return ReadingTargets(
        jnp.zeros(seats, jnp.bool_),
        jnp.zeros(seats, jnp.int32),
        jnp.zeros(tiles, jnp.bool_),
        jnp.zeros(tiles, jnp.bool_),
        jnp.zeros(tiles, jnp.float32),
        jnp.zeros(seats, jnp.bool_),
        jnp.zeros(seats, jnp.bool_),
        jnp.zeros(tiles, jnp.bool_),
        jnp.zeros(tiles, jnp.bool_),
        jnp.zeros(tiles, jnp.bool_),
    )


def reading_target_shape(targets, leading_shape=None):
    """Validate static schema/dtypes, including ShapeDtypeStruct admission inputs."""
    if not isinstance(targets, ReadingTargets):
        raise ValueError("reading targets require ReadingTargets schema")
    if leading_shape is None:
        if len(targets.tenpai.shape) < 1 or targets.tenpai.shape[-1] != 2:
            raise ValueError("reading targets require opponent axis of size 2")
        leading_shape = targets.tenpai.shape[:-1]
    if isinstance(leading_shape, int):
        leading_shape = (leading_shape,)
    leading_shape = tuple(leading_shape)
    for name, value in zip(ReadingTargets._fields, targets):
        task = name.removesuffix("_valid")
        expected = leading_shape + ((2,) if task in TASK_NAMES[:2] else (2, 34))
        if value.shape != expected:
            raise ValueError(f"reading {name} shape must be {expected}, got {value.shape}")
        dtype = np.dtype("int32" if name == "shanten" else "float32" if name == "danger_points" else "bool")
        if np.dtype(value.dtype) != dtype:
            raise ValueError(f"reading {name} dtype must be {dtype}")
    return leading_shape


def validate_reading_targets(targets, leading_shape=None):
    """Raise on static schema errors; return device bool per leading row.

    No arrays are copied to host. Active numeric labels must be finite/in range;
    inactive payload can be arbitrary (including NaN), except nonexistent manzu
    slots must always have false labels/masks and zero points. A false mask is
    unknown/not applicable, never a negative safety label.
    """
    leading_shape = reading_target_shape(targets, leading_shape)
    t = targets
    ok = jnp.all(~t.shanten_valid | ((t.shanten >= 0) & (t.shanten <= 3)), axis=-1)
    ok &= jnp.all(~t.danger_points_valid | (jnp.isfinite(t.danger_points) & (t.danger_points >= 0)), axis=(-2, -1))
    ok &= jnp.all(~t.waits_valid | (t.tenpai_valid & t.tenpai)[..., None], axis=(-2, -1))
    ok &= jnp.all(~t.danger_points_valid | (t.danger_valid & t.danger), axis=(-2, -1))
    # Legal ron implies a shape wait, regardless of the independent waits mask.
    ok &= jnp.all(~(t.danger_valid & t.danger) | t.waits, axis=(-2, -1))
    absent = ~jnp.asarray(CANONICAL_TILE_EXISTS)
    for value in (t.danger, t.waits, t.danger_valid, t.waits_valid, t.danger_points_valid):
        ok &= jnp.all(~value | ~absent, axis=(-2, -1))
    ok &= jnp.all(jnp.where(absent, t.danger_points == 0, True), axis=(-2, -1))
    return ok.reshape(leading_shape)


def _row_mask(targets, row_mask):
    shape = reading_target_shape(targets)
    if row_mask is None:
        return jnp.ones(shape, jnp.bool_)
    if row_mask.shape != shape or np.dtype(row_mask.dtype) != np.dtype("bool"):
        raise ValueError(f"reading row_mask must be bool{shape}")
    return jnp.asarray(row_mask)


def _task_masks(targets, row_mask):
    return tuple(
        getattr(targets, name + "_valid")
        & row_mask.reshape(row_mask.shape + (1,) * (1 if name in TASK_NAMES[:2] else 2))
        for name in TASK_NAMES
    )


def reading_target_counts(targets, row_mask=None):
    """Return float32[5] valid-cell counts, additive across pages/micros."""
    row_mask = _row_mask(targets, row_mask)
    return jnp.stack([jnp.sum(m, dtype=jnp.float32) for m in _task_masks(targets, row_mask)])


def reading_task_sums(predictions, targets, row_mask=None):
    """Return (float32[5] numerators, float32[5] valid-cell counts).

    Sanitize inactive labels AND predictions before nonlinear operations, so
    padded NaNs have exactly zero gradient. Active nonfinite inputs poison the
    numerator; callers must reject nonfinite loss/gradients before committing.
    """
    row_mask = _row_mask(targets, row_mask)
    masks = _task_masks(targets, row_mask)
    expected_keys = {"tenpai_logits", "shanten_logits", "danger_logits", "waits_logits", "danger_points"}
    if set(predictions) != expected_keys:
        raise ValueError(f"reading predictions require exactly {sorted(expected_keys)}")
    sums = []
    schema_ok = jnp.all(~row_mask | validate_reading_targets(targets))
    for name, mask in zip(TASK_NAMES, masks):
        key = name if name == "danger_points" else name + "_logits"
        pred = predictions[key]
        expected = mask.shape + ((4,) if name == "shanten" else ())
        if pred.shape != expected or not jnp.issubdtype(pred.dtype, jnp.floating):
            raise ValueError(f"reading prediction {key} must be floating {expected}")
        label = getattr(targets, name)
        p = jnp.where(mask[..., None] if name == "shanten" else mask, pred, 0).astype(jnp.float32)
        y = jnp.where(mask, label, 0)
        if name == "shanten":
            valid_label = (y >= 0) & (y < 4)
            loss = optax.softmax_cross_entropy_with_integer_labels(p, jnp.where(valid_label, y, 0))
            finite = valid_label & jnp.all(jnp.isfinite(p), axis=-1)
        elif name == "danger_points":
            loss = jnp.square(p - y)
            finite = jnp.isfinite(p) & jnp.isfinite(y) & (y >= 0)
        else:
            loss = optax.sigmoid_binary_cross_entropy(p, y.astype(jnp.float32))
            finite = jnp.isfinite(p)
        loss = jnp.where(mask, jnp.where(finite, loss, jnp.nan), 0)
        sums.append(jnp.where(schema_ok, jnp.sum(loss, dtype=jnp.float32), jnp.nan))
    return jnp.stack(sums), jnp.stack([jnp.sum(m, dtype=jnp.float32) for m in masks])


def reading_loss(predictions, targets, row_mask=None, global_counts=None):
    """Sum five independent valid means, using global counts when accumulating.

    A global zero-count task contributes zero; inconsistent/nonfinite/negative
    counts poison the loss rather than hiding an active task. No .01 coefficient
    is applied here. Never average these already-normalized microbatch losses.
    """
    sums, local_counts = reading_task_sums(predictions, targets, row_mask)
    counts = local_counts if global_counts is None else jnp.asarray(global_counts)
    if counts.shape != (5,) or not jnp.issubdtype(counts.dtype, jnp.floating):
        raise ValueError("reading global_counts must be floating[5]")
    counts_ok = jnp.all(jnp.isfinite(counts) & (counts >= local_counts) & (counts >= 0))
    terms = sums / jnp.where(counts > 0, counts, 1)
    return jnp.where(counts_ok, jnp.sum(terms), jnp.nan)


# --- from reading_oracle.py ---


def _score_wait_candidates(pending, judge):
    """Score every eligible wait exactly once, without a speculative tile batch.

    A scalar while predicate is essential: vmap(cond) would run the expensive
    judge on nonwaits too. No wait-count cap (kokushi can have thirteen waits).
    Outputs retain all 27 slots; unscored candidates are known zero here, not a
    replacement for the separate context/validity masks in ReadingTargets.
    """
    if pending.shape != (TILE_TYPES,) or pending.dtype != jnp.bool_:
        raise ValueError(f"reading score candidates must be bool[{TILE_TYPES}]")

    def step(carry):
        remaining, danger, points = carry
        tile = jnp.argmax(remaining).astype(jnp.int32)
        legal, income = judge(tile)
        return (remaining.at[tile].set(False), danger.at[tile].set(legal), points.at[tile].set(income))

    _, danger, points = jax.lax.while_loop(
        lambda c: c[0].any(), step, (pending, jnp.zeros(TILE_TYPES, jnp.bool_), jnp.zeros(TILE_TYPES, jnp.float32))
    )
    return danger, points


def _discard_action_available(legal_action_mask):
    """Whether a draw decision can discard an actual tile (raw 88-action mask)."""
    return legal_action_mask[:37].any() | legal_action_mask[Action.TSUMOGIRI]


def build_reading_targets(pre_state):
    """Build unbatched ReadingTargets from the current full sanma EnvState.

    Each pon/kan counts as three effective tiles; kita is not a meld. Fifth
    copies in the receiver's own hand/melds/extracted Norths cannot be waits.
    Other opponents' hands and the wall do not determine shape availability.
    Ron income uses Mahjax han/fu and dealer rounding, actual known own aka,
    current public dora/nuki, ippatsu/double-riichi/houtei; no ura, honba,
    kyotaku, pao payment redistribution, or concealed pending kan indicator.
    """
    s = pre_state
    p, rs = s.players, s.round_state
    opponents = (s.current_player.astype(jnp.int32) + jnp.array([1, 2])) % 3
    live = ~(s.terminated | s.truncated | rs.terminated_round)
    discard_now = _discard_action_available(s.legal_action_mask)
    pending_indicator = rs.n_kan_doras.astype(jnp.int32) < p.n_kan.astype(jnp.int32).sum()
    danger_context = live & discard_now & ~pending_indicator & ~rs.kan_declared & ~rs.kita_declared
    houtei = rs.is_haitei | (rs.next_deck_ix < rs.last_deck_ix.astype(jnp.int32))
    # Replacing the hidden ura array with a constant also ensures judge cannot
    # accidentally use it for an established riichi receiver.
    public_rs = rs.replace(
        ura_dora_indicators=jnp.full_like(rs.ura_dora_indicators, -1),
        kan_declared=jnp.bool_(False),
        kita_declared=jnp.bool_(False),
        can_robbing_kan=jnp.bool_(False),
    )
    public_state = s.replace(round_state=public_rs)

    def opponent_labels(seat):
        hand = p.hand_with_red[seat]
        folded = Hand.to_34(hand)
        melds = p.melds[seat]
        # Use actual meld words, not a potentially stale hand-count cache.
        meld_count = jnp.sum(melds != EMPTY_MELD, dtype=jnp.int32)
        effective = jnp.sum(folded, dtype=jnp.int32) + 3 * meld_count
        valid = live & (effective == 13)
        owned_copies = Yaku.flatten(hand, melds, meld_count).astype(jnp.int32)
        owned_copies = owned_copies.at[NORTH_TYPE].add(p.kita[seat].astype(jnp.int32))

        def shape_wait(tile):
            available = owned_copies[tile] < 4
            # Keep base-5 cache digits in 0..4 even for the rejected fifth tile.
            candidate = folded.at[tile].add((folded[tile] < 4).astype(folded.dtype))
            return valid & available & Hand.can_tsumo(candidate)

        waits = jax.vmap(shape_wait)(jnp.arange(TILE_TYPES, dtype=jnp.int32))
        tenpai = waits.any()
        # Authoritative shape waits already determine zero; invalid rows are
        # unknown. Only non-tenpai effective-13 hands need the shanten lookup.
        raw_shanten = jax.lax.cond(valid & ~tenpai, lambda: Shanten.number(folded), lambda: jnp.int32(0))
        # Shape enumeration is authoritative at zero, including the impossible
        # fifth-copy corner case that concealed-only shanten cannot see.
        shanten = jnp.where(tenpai, 0, jnp.maximum(raw_shanten, 1))
        shanten = jnp.where(valid, jnp.minimum(shanten, 3), 0).astype(jnp.int32)
        river = River.decode_tile(p.river[seat])
        river_type = Tile.to_tile_type(river)
        river_furiten = jnp.any((river >= 0) & waits[jnp.clip(river_type, 0, TILE_TYPES - 1)])
        furiten = river_furiten | p.furiten_by_pass[seat]
        legal_context = valid & danger_context & ~furiten

        def candidate_ron(tile):
            # Only a fresh per-tile judgement is valid. The environment's
            # cached fan/fu/has_yaku refer to its previous target/next draw.
            eval_state = public_state.replace(round_state=public_rs.replace(target=tile.astype(jnp.int8)))
            yaku, fan, fu = Yaku.judge(hand, jnp.bool_(True), seat, eval_state)
            legal = yaku.any() | houtei
            extra = (
                (p.ippatsu[seat] & p.riichi[seat]).astype(jnp.int32)
                + p.double_riichi[seat].astype(jnp.int32)
                + houtei.astype(jnp.int32)
            )
            basic = Yaku.score(fan + jnp.where(fu == 0, 0, extra), fu)
            raw = basic * jnp.where(rs.dealer == seat, 6, 4)
            income = ((raw + 99) // 100 * 100).astype(jnp.float32) / POINTS_SCALE
            return legal, jnp.where(legal, income, 0)

        exists = jnp.asarray(CANONICAL_TILE_EXISTS)
        waits = waits & exists
        danger, points = _score_wait_candidates(waits & legal_context, candidate_ron)
        danger = danger & exists
        points = jnp.where(exists, points, 0)

        def canonical(x):
            return x  # already canonical-34

        danger_valid = exists & valid & danger_context
        waits_valid = exists & valid & tenpai
        return (
            tenpai,
            shanten,
            canonical(danger),
            canonical(waits),
            canonical(points),
            valid,
            valid,
            danger_valid,
            waits_valid,
            canonical(danger),
        )

    # Keep the opponent axis scalar too; vmap here would undo the conditional
    # skipping inside each tile judge, even though collection uses lax.map.
    return ReadingTargets(*jax.lax.map(opponent_labels, opponents))
