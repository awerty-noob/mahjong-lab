"""Public sanma observation schema for the V5 Actor/Critic.

One module owns every tensor contract that crosses the simulator/network
boundary: the 255-token observation, the enriched public statistics that the
tokenizer consumes, the privileged inputs of the training-only central Q, and
the exact shape allowlists used to validate a live fingerprint.

Compact 27-space action/observation ids are the action ABI; canonical 34-space
ids appear only inside network tensors. Nothing here reads private opponent
hands, the wall, or next-round randomness.
"""

from collections.abc import Mapping
from functools import lru_cache
from typing import Dict

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax.custom_batching import sequential_vmap

from mahjax.common.constants import NUM_TILE_TYPES, NUM_TILE_TYPES_WITH_RED
from mahjax.common.shanten import Shanten
from mahjax.common.tile import Tile
from mahjax.mahjong_3p.action import Action
from mahjax.mahjong_3p.constants import DORA_ARRAY
from mahjax.mahjong_3p.state import default_state

NUM_PLAYERS = 3
# The simulator numbers north-setting as 87 (88 actions); the deployed Actor keeps
# the historical 78 slot inside 87 actions (network.compact_legal_mask bridges).
NETWORK_ACTION_COUNT = 87
# Self-kan actions occupy ids 30..56 in the 87-action ABI (30 + tile type).
SELF_KAN_OFFSET = 30
SELF_KAN_END = 57
Array = jnp.ndarray


def orthogonal_init(scale: float = jnp.sqrt(2.0)):
    return nn.initializers.orthogonal(scale)


# --- public statistics ---------------------------------------------------------

_REQUIRED_SHAPES = {
    "hand": (27,),
    "discard_shanten": (27, 3),
    "is_discard_node": (),
    "last_draw": (),
    "tiles_seen": (27,),
    "melds": (3, 3, 4),
    "dora_indicators": (5,),
    "action_history": (3, 200),
    "riichi": (3,),
}


@sequential_vmap
def _dense_candidate_ukeire(
    hand: Array,
    shanten: Array,
    valid: Array,
    seen: Array,
    meld_count: Array,
    is_discard_node: Array,
) -> Array:
    """Weighted improvements; skip the cache scan on non-discard nodes.

    A normal vmap of a cache-bearing cond can broadcast the ~70 MiB shanten
    cache per lane. Sequential batching preserves the scalar cond and shares
    that cache, including when callers vmap the public augmentation API.
    """

    def scan():
        eye = jnp.eye(NUM_TILE_TYPES, dtype=jnp.int32)
        reduced = jnp.maximum(hand.astype(jnp.int32)[None, :] - eye, 0)
        drawn = reduced[:, None, :] + eye[None, :, :]
        # Never query the base-5 cache with an impossible fifth copy. Its value
        # is masked below, but a count of five can alias an unrelated cache row.
        safe_drawn = jnp.minimum(drawn, 4)
        after_draw = jax.vmap(jax.vmap(Shanten.number, in_axes=(0, None)), in_axes=(0, None))(safe_drawn, meld_count)
        unseen = jnp.clip(4 - seen.astype(jnp.int32), 0, 4)
        improves = (after_draw < shanten[:, None]) & (reduced < 4)
        weighted = jnp.where(improves, unseen[None, :], 0)
        return jnp.where(valid[:, None], weighted, -1).astype(jnp.int8)

    return jax.lax.cond(
        is_discard_node,
        scan,
        lambda: jnp.full((NUM_TILE_TYPES, NUM_TILE_TYPES), -1, dtype=jnp.int8),
    )


_CANDIDATE_CAPACITY = 14


def _compact_candidates(valid: Array) -> tuple[Array, Array]:
    """Compact the valid compact candidates into a fixed 14-slot batch."""
    return (
        jnp.nonzero(valid, size=_CANDIDATE_CAPACITY, fill_value=0)[0],
        valid.sum(dtype=jnp.int32),
    )


@sequential_vmap
def _candidate_ukeire(
    hand: Array,
    shanten: Array,
    valid: Array,
    seen: Array,
    meld_count: Array,
    is_discard_node: Array,
) -> Array:
    """Compact legal valid rows into one parallel 14-by-27 draw batch.

    Shanten is evaluated in the simulator's canonical 34 space and the improving
    draws are gathered back onto the compact 27 rows. ``meld_count`` is accepted
    for call-site compatibility; the current Shanten ignores it.
    """
    del meld_count
    indices, count = _compact_candidates(valid)
    empty = jnp.full((27, 27), -1, dtype=jnp.int8)

    def sparse():
        compact = jnp.asarray(COMPACT_TO_CANONICAL, jnp.int32)
        # Upstream ``Hand.to_34`` folds a 37-slot red-aware hand; our hand is
        # already the 27 compact type counts, so scatter it onto canonical 34.
        hand34 = jnp.zeros(34, dtype=jnp.int32).at[compact].set(hand.astype(jnp.int32))
        eye34 = jnp.eye(34, dtype=jnp.int32)
        unseen = jnp.clip(4 - seen.astype(jnp.int32), 0, 4)
        reduced = jnp.maximum(hand34[None, :] - eye34[compact[indices]], 0)
        safe_drawn = jnp.minimum(reduced[:, None, :] + eye34[None, :, :], 4)
        after_draw_full = jax.vmap(jax.vmap(Shanten.number))(safe_drawn)
        after_draw = jnp.take(after_draw_full, compact, axis=-1)
        # A draw is only available when this particular tile type still has a
        # free copy; checking only column 0 wrongly suppressed every draw in
        # hands holding a kan of 1m.
        drawable = jnp.take(reduced < 4, compact, axis=-1)
        improves = (after_draw < shanten[indices, None]) & drawable
        weighted = jnp.where(improves, unseen[None, :], 0).astype(jnp.int8)
        destinations = jnp.where(jnp.arange(_CANDIDATE_CAPACITY) < count, indices, 27)
        return empty.at[destinations].set(weighted, mode="drop")

    return jax.lax.cond(
        jnp.asarray(is_discard_node),
        lambda: jax.lax.cond(
            count > 0, lambda: jax.lax.cond(count <= _CANDIDATE_CAPACITY, sparse, sparse), lambda: empty
        ),
        lambda: empty,
    )


def _history_features(obs: Dict) -> Dict:
    history = obs["action_history"]
    player, action, tsumogiri = history.astype(jnp.int32)
    ix = jnp.arange(history.shape[1], dtype=jnp.int32)
    valid = (player >= 0) & (player < NUM_PLAYERS)
    # A legacy saved dict has no overflow bit. A contiguous prefix with spare
    # capacity proves completeness; a full buffer does not. Explicit metadata
    # from observe_v3 can disambiguate an exactly-full, non-overflowed history.
    prefix_length = valid.sum(dtype=jnp.int32)
    contiguous = jnp.all(valid == (ix < prefix_length))
    overflow = jnp.asarray(obs.get("history_overflow", False), dtype=jnp.bool_)
    complete = (
        contiguous
        & ~overflow
        & jnp.asarray(obs.get("public_history_complete", ~valid[-1]), dtype=jnp.bool_)
    )
    discard = valid & (tsumogiri >= 0) & (action >= 0) & (action < NUM_TILE_TYPES_WITH_RED)
    # 3p history records compact local ids (0..26) plus the red-five ids 27..29.
    # Fold those onto the canonical 34 axis before scattering so the compact
    # take below keeps the actual tile type.
    tile = local_tile_to_canonical(action)
    safe_tile = jnp.where(discard, tile, 0)
    seats = jnp.arange(NUM_PLAYERS)[:, None]
    own = (player[None, :] == seats) & discard[None, :]
    counts = jax.vmap(
        lambda mask: jnp.zeros(NUM_TILE_TYPES, dtype=jnp.int16).at[safe_tile].add(mask.astype(jnp.int16))
    )(own)

    riichi_action = valid & (action == Action.RIICHI) & (tsumogiri == -1)
    riichi_index = jnp.min(
        jnp.where((player[None, :] == seats) & riichi_action[None, :], ix[None, :], history.shape[1]),
        axis=1,
    )
    declaration_index = jnp.min(
        jnp.where(own & (ix[None, :] > riichi_index[:, None]), ix[None, :], history.shape[1]),
        axis=1,
    )
    declared = declaration_index < history.shape[1]
    declaration_tile = action[jnp.minimum(declaration_index, history.shape[1] - 1)]
    later = discard[None, :] & (ix[None, :] > declaration_index[:, None]) & declared[:, None]
    post_counts = jax.vmap(
        lambda mask: jnp.zeros(NUM_TILE_TYPES, dtype=jnp.int16).at[safe_tile].add(mask.astype(jnp.int16))
    )(later & own)
    elapsed = later.sum(axis=1, dtype=jnp.int16)

    # Do not promote a freshly discarded tile to genbutsu while ron is still
    # being resolved. Passes are deliberately absent from public history. Only
    # completion of that response window (a following public action, or a new
    # discard node) establishes safety without consulting private response masks.
    last_event = jnp.max(jnp.where(valid, ix, -1))
    pending_discard = (~obs["is_discard_node"]) & discard & (ix == last_event)
    # ``discard_counts`` reports every discard, including a response tile still
    # being claimed.  Genbutsu is stronger: it cannot be promoted until that
    # response window closes.
    pending_own = pending_discard[None, :] & own
    pending_counts = jax.vmap(
        lambda mask: jnp.zeros(NUM_TILE_TYPES, dtype=jnp.int16).at[safe_tile].add(mask.astype(jnp.int16))
    )(pending_own)
    resolved_later = later & ~pending_discard[None, :]
    public_safe = jax.vmap(
        lambda mask: jnp.zeros(NUM_TILE_TYPES, dtype=jnp.int16).at[safe_tile].add(mask.astype(jnp.int16))
    )(resolved_later)
    # Declaration alone is not accepted riichi: it may still be ron-ed. The
    # existing public riichi flag, not riichi_declared, gates permanent safety.
    genbutsu = ((counts - pending_counts) > 0) | (
        (public_safe > 0) & obs["riichi"][:, None] & complete
    )
    known_declaration = declared & complete
    compact = jnp.asarray(COMPACT_TO_CANONICAL, jnp.int32)
    return {
        "public_history_complete": complete,
        "riichi_declaration_tile": jnp.where(known_declaration, declaration_tile, -1).astype(jnp.int8),
        "riichi_declaration_index": jnp.where(known_declaration, declaration_index, -1).astype(jnp.int16),
        "riichi_elapsed": jnp.where(known_declaration, elapsed, -1).astype(jnp.int16),
        "discard_counts": jnp.where(complete, jnp.take(counts, compact, axis=1), -1).astype(jnp.int16),
        "post_riichi_discard_counts": jnp.where(complete, jnp.take(post_counts, compact, axis=1), -1).astype(jnp.int16),
        "genbutsu": jnp.take(genbutsu, compact, axis=1),
    }


def augment_observation(base_obs: Dict, legal_action_mask: Array) -> Dict:
    """Return a new dict containing unchanged v2 fields and explicit v3 fields.

    This API accepts ONE unbatched public dict and its matching (87,) legal mask;
    callers may jit/vmap it. Missing required keys or incompatible static shapes
    raise ValueError rather than inventing public history, masks, or hand data.

    Added fields:
      feature_version: scalar int8 = 3.
      candidate_valid: (27,) bool, held types with an actual legal discard or
        TSUMOGIRI action. Red and ordinary five actions fold onto one type.
      candidate_shanten: (27,) int8, best shape shanten after each legal discard.
      candidate_ukeire: (27,27) int8, improving draws weighted by UNSEEN copies
        (4 - tiles_seen), not wall-truth counts. All legal candidates are scanned,
        not only minimum-shanten candidates. Exhausted improvements have zero.
        The discarded copy remains visible, so it is NOT added back to unseen.
      candidate_ukeire_types/count: (27,) int16, nonzero kinds / weighted sum.
        Invalid/non-discard rows of all candidate values are -1; mask is False.
      dora_count: (27,) int8, revealed indicator multiplicity per tile type;
        excludes red, kita bonuses and ura. Uses sanma 1m/9m and honour cycles.
      riichi_declaration_tile: (3,) int8, red-aware first discard AFTER RIICHI.
      riichi_declaration_index: (3,) int16, its zero-based PUBLIC HISTORY index,
        NOT the RIICHI action index, a river index, or an env step count.
      riichi_elapsed: (3,) int16, number of ALL seats' discards strictly after
        that declaration discard. These three fields are -1 until it exists.
      discard_counts/post_riichi_discard_counts: (3,27) int16, own discard counts,
        with the latter strictly excluding the declaration discard. Called-away
        discards still count; kita/kan/pon are not discard events.
      genbutsu: (3,27) bool, own historic discards plus resolved discards after
        ACCEPTED riichi. The newest unresolved response tile is not promoted.
      public_history_complete: scalar bool. On overflow/incomplete history the
        declaration and count fields are -1; genbutsu retains only provable own
        discards (a conservative lower bound). For legacy dicts with no flag,
        full buffers are conservatively incomplete. All seat axes are relative
        to the observer, just like action_history and riichi in the base dict.

    Unseen counts are public availability estimates, NOT a claim that the tiles
    remain in the live wall. Shanten/ukeire are shape-only, not yaku/furiten tests.
    """
    for key, shape in _REQUIRED_SHAPES.items():
        if key not in base_obs:
            raise ValueError(f"v3 public observation requires field {key!r}")
        if jnp.shape(base_obs[key]) != shape:
            raise ValueError(f"v3 field {key!r} requires unbatched shape {shape}, got {jnp.shape(base_obs[key])}")
    if jnp.shape(legal_action_mask) != (NETWORK_ACTION_COUNT,):
        raise ValueError("v3 legal_action_mask requires unbatched shape (87,)")
    if "public_history_complete" in base_obs and jnp.shape(base_obs["public_history_complete"]) != ():
        raise ValueError("v3 public_history_complete requires scalar shape ()")
    obs = dict(base_obs)
    hand = obs["hand"]
    mask = jnp.asarray(legal_action_mask, dtype=jnp.bool_)
    # Compact slots 0..26 are tile types in the 27-slot ABI; 27/28/29 are the
    # red-five actions, which fold onto 1m (unreachable), 5p and 5s.
    slot_types = jnp.asarray((*range(27), 0, 6, 15), jnp.int32)
    discard_actions = mask[:30]
    legal_types = jnp.zeros(27, dtype=jnp.bool_).at[slot_types].max(discard_actions)
    draw = jnp.asarray(obs["last_draw"], dtype=jnp.int32)
    draw_valid = (draw >= 0) & (draw < 27)
    draw_type = jnp.where(draw_valid, draw, 0)
    legal_types = legal_types.at[draw_type].set(legal_types[draw_type] | (mask[Action.TSUMOGIRI] & draw_valid))
    valid = obs["is_discard_node"] & legal_types & (hand > 0)
    shanten = jnp.where(valid, jnp.min(obs["discard_shanten"], axis=1), -1).astype(jnp.int8)
    meld_count = (obs["melds"][0, 0] > 0).sum(dtype=jnp.int32)
    ukeire = _candidate_ukeire(hand, shanten, valid, obs["tiles_seen"], meld_count, obs["is_discard_node"])
    obs.update(
        {
            "feature_version": jnp.int8(3),
            "candidate_valid": valid,
            "candidate_shanten": shanten,
            "candidate_ukeire": ukeire,
            "candidate_ukeire_types": jnp.where(valid, (ukeire > 0).sum(axis=1), -1).astype(jnp.int16),
            "candidate_ukeire_count": jnp.where(valid, jnp.maximum(ukeire, 0).sum(axis=1), -1).astype(jnp.int16),
        }
    )
    indicators = jnp.asarray(obs["dora_indicators"], dtype=jnp.int32)
    revealed = (indicators >= 0) & (indicators < 27)
    indicator_compact = jnp.where(revealed, indicators, 0)
    dora_compact = jnp.asarray(COMPACT_DORA, jnp.int32)
    obs["dora_count"] = jnp.zeros(27, dtype=jnp.int8).at[dora_compact[indicator_compact]].add(revealed.astype(jnp.int8))
    obs.update(_history_features(obs))
    return obs


# --- tokenizer -----------------------------------------------------------------

NUM_TOKENS = 255
NUM_TILE_TOKENS = 34
NUM_PLAYER_TOKENS = 4
NUM_MELD_TOKENS = 16
NUM_EVENT_TOKENS = 200
TILE_SLICE = slice(0, 34)
PLAYER_SLICE = slice(34, 38)
MELD_SLICE = slice(38, 54)
EVENT_SLICE = slice(54, 254)
HISTORY_SLICE = EVENT_SLICE
GLOBAL_INDEX = 254
GLOBAL_SLICE = slice(254, 255)
COMPACT_TO_CANONICAL = (0, 8, *range(9, 34))
_CANONICAL_TO_COMPACT = tuple(COMPACT_TO_CANONICAL.index(i) if i in COMPACT_TO_CANONICAL else -1 for i in range(34))
# Compact indicator type -> compact dora type (1m/9m cycle plus honours cycle).
COMPACT_DORA = tuple(_CANONICAL_TO_COMPACT[int(DORA_ARRAY[canonical])] for canonical in COMPACT_TO_CANONICAL)
HAND_POSITIONS = COMPACT_TO_CANONICAL
PLAYER_POSITIONS = tuple(range(34, 38))
MELD_POSITIONS = tuple(range(38, 54))
EVENT_POSITIONS = tuple(range(54, 254))
TOKEN_SLICES = {
    "tile": TILE_SLICE,
    "player": PLAYER_SLICE,
    "meld": MELD_SLICE,
    "event": EVENT_SLICE,
    "global": GLOBAL_SLICE,
}
FEATURE_SHAPES = {
    "feature_version": (),
    "candidate_valid": (27,),
    "candidate_shanten": (27,),
    "candidate_ukeire": (27, 27),
    "candidate_ukeire_types": (27,),
    "candidate_ukeire_count": (27,),
    "dora_count": (27,),
    "riichi_declaration_tile": (3,),
    "riichi_declaration_index": (3,),
    "riichi_elapsed": (3,),
    "discard_counts": (3, 27),
    "post_riichi_discard_counts": (3, 27),
    "genbutsu": (3, 27),
    "public_history_complete": (),
}
BASE_SHAPES = {
    "hand": (27,),
    "is_red": (27,),
    "last_draw": (),
    "shanten_count": (),
    "discard_shanten": (27, 3),
    "discard_can_win": (27, 27),
    "furiten": (),
    "is_discard_node": (),
    "n_kita": (3,),
    "kita_tiles": (4,),
    "melds": (3, 3, 4),
    "action_history": (3, 200),
    "scores": (3,),
    "round": (),
    "round_limit": (),
    "honba": (),
    "kyotaku": (),
    "prevalent_wind": (),
    "seat_wind": (),
    "absolute_dealer": (),
    "dora_indicators": (5,),
    "ippatsu": (3,),
    "riichi": (3,),
    "is_hand_concealed": (3,),
    "wall_remaining": (),
    "tiles_seen": (27,),
    "target": (),
    "last_player": (),
}

# Semantic event categories, independent of the overlapping raw action IDs.
EVENT_UNKNOWN, EVENT_DISCARD, EVENT_SELF_KAN, EVENT_RIICHI = 0, 1, 2, 3
EVENT_TSUMO, EVENT_RON, EVENT_PON, EVENT_OPEN_KAN = 4, 5, 6, 7
EVENT_KITA, EVENT_KYUUSHU, EVENT_ROUND_END = 8, 9, 10
NUM_EVENT_KINDS = 11
# Identity colour: 0 unknown, 1 known ordinary, 2 known red.
RED_UNKNOWN, RED_ORDINARY, RED_RED = 0, 1, 2


def _batch(value, base_ndim):
    value = jnp.asarray(value)
    return value[None] if value.ndim == base_ndim else value


def _is_red_local(ids):
    """Red-five test for the compact 3p local ABI (27/28/29), not full 37-space."""
    ids = jnp.asarray(ids, jnp.int32)
    return (ids >= 27) & (ids < 30)


def compact_to_canonical34(types):
    """Compact *types* to canonical34, preserving unknown/out-of-range as -1."""
    types = jnp.asarray(types, jnp.int32)
    safe = jnp.clip(types, 0, 26)
    return jnp.where((types >= 0) & (types < 27), jnp.asarray(COMPACT_TO_CANONICAL)[safe], -1)


def local_tile_to_canonical(ids):
    """Red-aware local IDs to canonical types; reserved red 5m stays unknown.

    The 3p local ABI uses 0..26 for tile types and 27/28/29 for the red fives.
    Slot 27 is deliberately reserved (it has no canonical type in this ABI), while
    28 and 29 denote red 5p and red 5s and fold onto canonical 13 and 22.
    """
    ids = jnp.asarray(ids, jnp.int32)
    canonical = compact_to_canonical34(jnp.clip(ids, 0, 26))
    canonical = jnp.where(ids == 28, 13, canonical)
    canonical = jnp.where(ids == 29, 22, canonical)
    return jnp.where((ids >= 0) & (ids < 30) & (ids != 27), canonical, -1)


def canonicalize_tile_axis(values, *, axis=-1, fill=0):
    """Scatter one compact27 axis into canonical34, retaining caller sentinels."""
    values = jnp.asarray(values)
    moved = jnp.moveaxis(values, axis, -1)
    if moved.shape[-1] != 27:
        raise ValueError("V4 canonicalization requires a compact27 tile axis")
    out = jnp.full(moved.shape[:-1] + (34,), fill, dtype=values.dtype)
    out = out.at[..., jnp.asarray(COMPACT_TO_CANONICAL)].set(moved)
    return jnp.moveaxis(out, -1, axis)


def _validate_observation(obs):
    if "hand" not in obs:
        raise ValueError("V4 observation requires hand")
    batch = jnp.shape(obs["hand"])[:-1]
    if len(batch) > 1:
        raise ValueError("V4 observations support at most one leading batch dimension")
    for key, shape in {**BASE_SHAPES, **FEATURE_SHAPES}.items():
        if key not in obs:
            raise ValueError(f"V4 observation requires {key}; prepare features explicitly")
        if jnp.shape(obs[key]) != batch + shape:
            raise ValueError(f"invalid V4 observation shape for {key}: expected {batch + shape}")
    version = jnp.asarray(obs["feature_version"])
    if not isinstance(version, jax.core.Tracer) and not np.all(np.asarray(version) == 4):
        raise ValueError("V4 observation requires feature_version=4")
    return batch


@lru_cache(maxsize=1)
def _preparers():
    return jax.jit(augment_observation), jax.jit(jax.vmap(augment_observation))


def prepare_observation(observation, *, legal_action_mask=None, history_overflow=None):
    """Prepare raw public observations, or validate already-prepared V4 data.

    Accepts unbatched or one leading batch dimension and never reads state. A
    raw observation requires its matching 87-action legal mask. Live callers may
    supply the PUBLIC history-overflow flag, used to distinguish exactly-full
    from truncated histories. Legacy full buffers stay conservatively unknown.
    V3/partial enriched data is rejected, not silently re-versioned/recomputed.
    The inexpensive schema checks are static under jit; a traced bad version is
    retained and the tokenizer poisons that lane with NaN (fail closed).
    """
    if "hand" not in observation:
        raise ValueError("V4 observation requires hand")
    batch = jnp.shape(observation["hand"])[:-1]
    if len(batch) > 1:
        raise ValueError("V4 observations support at most one leading batch dimension")
    enriched = (set(FEATURE_SHAPES) - {"public_history_complete"}) & set(observation)
    if enriched:
        _validate_observation(observation)
        return observation
    if legal_action_mask is None:
        raise ValueError("V4 enrichment requires the observer's legal_action_mask")
    if jnp.shape(legal_action_mask) != batch + (NETWORK_ACTION_COUNT,):
        raise ValueError("V4 legal_action_mask must match observation batch and 87 actions")
    obs = dict(observation)
    if history_overflow is not None:
        if jnp.shape(history_overflow) != batch:
            raise ValueError("V4 history_overflow must match observation batch")
        obs["public_history_complete"] = ~jnp.asarray(history_overflow, jnp.bool_)
    # Validate raw required public fields too, without evaluating any shanten.
    for key, shape in BASE_SHAPES.items():
        if key not in obs or jnp.shape(obs[key]) != batch + shape:
            raise ValueError(f"invalid or missing V4 public observation field {key}")
    single, batched = _preparers()
    prepared = (batched if batch else single)(obs, legal_action_mask)
    prepared["feature_version"] = jnp.full(batch, 4, jnp.int8)
    return prepared


def decode_public_events(obs):
    """Decode mixed history IDs to public structured fields (always batched).

    Returned tile is canonical34 (-1 unknown); actor/source are relative seats
    (-1 unknown); colour uses unknown/ordinary/red = 0/1/2. Self-kan action IDs
    provide TYPE only, never claimed full red composition or closed/added kind.
    Calls reuse the immediately previous event's public tile/actor only; no
    target is invented for truncated histories or calls without that evidence.
    TSUMO has no public tile in this buffer and therefore remains unknown.
    """
    history = _batch(obs["action_history"], 2).astype(jnp.int32)
    actor, action, tsumogiri = history[:, 0], history[:, 1], history[:, 2]
    # Public history also records KITA (87), which is outside the 87-slot
    # network action space but is still a public event.
    valid = (actor >= 0) & (actor < 3) & (action >= 0) & (action <= Action.KITA)
    # Declined calls/ron are private decisions, never public event tokens.
    valid &= action != Action.PASS
    discard = valid & (tsumogiri >= 0) & (tsumogiri <= 1) & (local_tile_to_canonical(action) >= 0)
    raw = valid & (tsumogiri == -1)
    kan = raw & (action >= SELF_KAN_OFFSET) & (action < SELF_KAN_END)
    kita = raw & (action == Action.KITA)
    kind = jnp.full_like(action, EVENT_UNKNOWN)
    kind = jnp.where(discard, EVENT_DISCARD, kind)
    kind = jnp.where(kan, EVENT_SELF_KAN, kind)
    for raw_id, event_id in (
        (Action.RIICHI, EVENT_RIICHI),
        (Action.TSUMO, EVENT_TSUMO),
        (Action.RON, EVENT_RON),
        (Action.PON, EVENT_PON),
        (Action.PON_RED, EVENT_PON),
        (Action.OPEN_KAN, EVENT_OPEN_KAN),
        (Action.KITA, EVENT_KITA),
        (Action.KYUUSHU, EVENT_KYUUSHU),
        (Action.DUMMY, EVENT_ROUND_END),
    ):
        kind = jnp.where(raw & (action == raw_id), event_id, kind)
    tile = jnp.where(
        discard,
        local_tile_to_canonical(action),
        jnp.where(kan, compact_to_canonical34(action - SELF_KAN_OFFSET), jnp.where(kita, 30, -1)),
    )
    colour = jnp.where(
        discard, 1 + _is_red_local(action).astype(jnp.int32), jnp.where(kita, RED_ORDINARY, RED_UNKNOWN)
    )
    previous_tile = jnp.pad(tile[:, :-1], ((0, 0), (1, 0)), constant_values=-1)
    previous_colour = jnp.pad(colour[:, :-1], ((0, 0), (1, 0)))
    previous_actor = jnp.pad(jnp.where(valid, actor, -1)[:, :-1], ((0, 0), (1, 0)), constant_values=-1)
    previous_discard = jnp.pad(discard[:, :-1], ((0, 0), (1, 0)))
    previous_target = jnp.pad((discard | kan | kita)[:, :-1], ((0, 0), (1, 0)))
    call = (kind == EVENT_PON) | (kind == EVENT_OPEN_KAN)
    ron = kind == EVENT_RON
    proven = (call & previous_discard) | (ron & previous_target)
    proven &= (previous_actor >= 0) & (previous_actor != actor)
    tile = jnp.where(proven, previous_tile, tile)
    colour = jnp.where(proven, previous_colour, colour)
    source = jnp.where(proven, previous_actor, -1)
    # Distinct from called-tile colour: PON_RED can consume a red from hand.
    contains_red = jnp.where(raw & (action == Action.PON_RED), RED_RED, jnp.where(discard | kita, colour, RED_UNKNOWN))
    rank = jnp.cumsum(valid, axis=-1)
    positions = jnp.where(valid, rank - valid.sum(-1, keepdims=True) - 1, 0).astype(jnp.int32)
    return {
        "valid": valid,
        "kind": kind,
        "tile": tile,
        "colour": colour,
        "actor": jnp.where(valid, actor, -1),
        "source": source,
        "tsumogiri": jnp.where(discard, tsumogiri + 1, 0),
        "contains_red": contains_red,
        "positions": positions,
    }


def token_positions(obs):
    """(B,255) int positions: static=0, latest public event=-1, older=-2,..."""
    events = decode_public_events(obs)
    positions = jnp.zeros((events["valid"].shape[0], NUM_TOKENS), jnp.int32)
    return positions.at[:, EVENT_SLICE].set(events["positions"])


class ObservationTokenizer(nn.Module):
    """Dynamic-width tokenizer; V4 parameter names are independent of V2/V3."""

    d_model: int = 128
    ARCHITECTURE = "v4"
    D_MODEL = 128
    NUM_ACTIONS = NETWORK_ACTION_COUNT
    NUM_TOKENS = NUM_TOKENS
    NUM_HAND_TOKENS = 34
    NUM_TILE_TOKENS = NUM_TILE_TOKENS
    NUM_PLAYER_TOKENS = NUM_PLAYER_TOKENS
    NUM_MELD_TOKENS = NUM_MELD_TOKENS
    NUM_HIST_TOKENS = NUM_EVENT_TOKENS
    NUM_EVENT_TOKENS = NUM_EVENT_TOKENS
    NUM_GLOBAL_TOKENS = 1
    TILE_SLICE, PLAYER_SLICE, MELD_SLICE = TILE_SLICE, PLAYER_SLICE, MELD_SLICE
    EVENT_SLICE, HISTORY_SLICE = EVENT_SLICE, HISTORY_SLICE
    GLOBAL_INDEX, GLOBAL_SLICE = GLOBAL_INDEX, GLOBAL_SLICE
    HAND_POSITIONS, PLAYER_POSITIONS = HAND_POSITIONS, PLAYER_POSITIONS
    MELD_POSITIONS, EVENT_POSITIONS = MELD_POSITIONS, EVENT_POSITIONS
    TOKEN_SLICES = TOKEN_SLICES
    CLS_POSITIONS = ()
    token_positions = staticmethod(token_positions)

    def setup(self):
        if self.d_model < 1:
            raise ValueError("V4 d_model must be positive")

        def emb(n):
            return nn.Embed(n, self.d_model, embedding_init=orthogonal_init())

        self.tile_identity = emb(35)  # null then canonical34
        self.suit_identity = emb(5)  # null, man/pin/sou/honour
        self.rank_identity = emb(10)  # null for honours/unknown, then 1..9
        self.honour_identity = emb(8)  # null, E/S/W/N/white/green/red
        self.colour_identity = emb(3)
        self.segment_identity = emb(5)
        self.seat_identity = emb(5)  # null then relative seats
        self.source_identity = emb(5)  # distinct role, not commutative with actor
        self.meld_slot_identity = emb(4)
        self.event_kind_identity = emb(NUM_EVENT_KINDS)
        self.tsumogiri_identity = emb(3)
        self.shanten_identity = emb(16)  # unknown, -1..13 (14 is unavailable)
        for name in (
            "tile_features",
            "candidate_features",
            "public_tile_features",
            "player_features",
            "meld_features",
            "table_features",
        ):
            setattr(self, name, nn.Dense(self.d_model, kernel_init=orthogonal_init()))

    def _canonical_identity(self, canonical):
        canonical = jnp.asarray(canonical, jnp.int32)
        known = canonical >= 0
        numbered = known & (canonical < 27)
        suit = jnp.where(known, canonical // 9 + 1, 0)
        rank = jnp.where(numbered, canonical % 9 + 1, 0)
        honour = jnp.where(known & ~numbered, canonical - 26, 0)
        return (
            self.tile_identity(canonical + 1)
            + self.suit_identity(suit)
            + self.rank_identity(rank)
            + self.honour_identity(honour)
        )

    def _identity(self, ids):
        ids = jnp.asarray(ids, jnp.int32)
        canonical = local_tile_to_canonical(ids)
        colour = jnp.where(canonical >= 0, 1 + _is_red_local(ids).astype(jnp.int32), 0)
        return self._canonical_identity(canonical) + self.colour_identity(colour)

    def __call__(self, obs):
        _validate_observation(obs)
        b = _batch
        hand = b(obs["hand"], 1).astype(jnp.float32)
        B, D = hand.shape[0], self.d_model
        canonical = jnp.asarray(COMPACT_TO_CANONICAL)
        reds = b(obs["is_red"], 1).astype(jnp.int32)
        draw = b(obs["last_draw"], 0).astype(jnp.int32)
        target = b(obs["target"], 0).astype(jnp.int32)
        draw_at = canonical[None] == local_tile_to_canonical(draw)[:, None]
        target_at = canonical[None] == local_tile_to_canonical(target)[:, None]
        waits = b(obs["discard_can_win"], 2).astype(jnp.float32)
        # Full canonical34 wait identities as well as per-element known markers.
        wait_values = canonicalize_tile_axis(jnp.maximum(waits, 0))
        wait_known = canonicalize_tile_axis((waits >= 0).astype(jnp.float32))
        ds = b(obs["discard_shanten"], 2).astype(jnp.int32)
        ds_code = jnp.where((ds >= -1) & (ds < 14), ds + 2, 0)
        local_features = jnp.stack(
            [
                hand / 4.0,
                b(obs["tiles_seen"], 1) / 4.0,
                b(obs["dora_count"], 1) / 5.0,
                draw_at,
                draw_at & _is_red_local(draw)[:, None],
                target_at,
                target_at & _is_red_local(target)[:, None],
            ],
            axis=-1,
        ).astype(jnp.float32)
        tile_tok = (
            self._canonical_identity(canonical)[None]
            + self.colour_identity(1 + reds)
            + self.segment_identity(jnp.int32(0))
            + self.tile_features(jnp.concatenate([local_features, wait_values, wait_known], -1))
            + self.shanten_identity(ds_code).sum(-2)
        )
        valid = b(obs["candidate_valid"], 1).astype(jnp.float32)
        shanten = b(obs["candidate_shanten"], 1)
        ukeire = b(obs["candidate_ukeire"], 2)
        # -1 shanten is a known complete hand, not an unknown sentinel when valid.
        candidate = jnp.concatenate(
            [
                valid[..., None],
                ((shanten + 1) / 9.0)[..., None] * valid[..., None],
                jnp.maximum(b(obs["candidate_ukeire_types"], 1), 0)[..., None] / 34.0 * valid[..., None],
                jnp.maximum(b(obs["candidate_ukeire_count"], 1), 0)[..., None] / 108.0 * valid[..., None],
                canonicalize_tile_axis(jnp.maximum(ukeire, 0)) / 4.0 * valid[..., None],
                canonicalize_tile_axis((ukeire >= 0).astype(jnp.float32)) * valid[..., None],
            ],
            -1,
        )
        tile_tok += self.candidate_features(candidate)
        declaration = b(obs["riichi_declaration_tile"], 1).astype(jnp.int32)
        declaration_at = local_tile_to_canonical(declaration)[..., None] == canonical[None, None]
        public = []
        for key in ("discard_counts", "post_riichi_discard_counts"):
            values = b(obs[key], 2).astype(jnp.float32)
            public.extend(
                [
                    jnp.swapaxes(jnp.maximum(values, 0) / 4.0, 1, 2),
                    jnp.swapaxes((values >= 0).astype(jnp.float32), 1, 2),
                ]
            )
        # False genbutsu with incomplete history is only a conservative lower
        # bound; completeness is separately represented here and on table/seat.
        public.extend(
            [
                jnp.swapaxes(b(obs["genbutsu"], 2), 1, 2),
                jnp.swapaxes(declaration_at, 1, 2),
                jnp.broadcast_to(b(obs["public_history_complete"], 0)[:, None, None], (B, 27, 1)),
            ]
        )
        tile_tok += self.public_tile_features(jnp.concatenate(public, -1).astype(jnp.float32))
        full_tiles = jnp.zeros((B, 34, D), jnp.float32).at[:, canonical].set(tile_tok)
        tile_mask = jnp.zeros((B, 34), jnp.float32).at[:, canonical].set(1)

        index = b(obs["riichi_declaration_index"], 1).astype(jnp.float32)
        elapsed = b(obs["riichi_elapsed"], 1).astype(jnp.float32)
        wind = (b(obs["seat_wind"], 0)[:, None] + jnp.arange(3)[None]) % 3
        player_vec = jnp.stack(
            [
                (b(obs["scores"], 1) + 350.0) / 1750.0,
                b(obs["riichi"], 1),
                b(obs["ippatsu"], 1),
                b(obs["is_hand_concealed"], 1),
                b(obs["n_kita"], 1) / 4.0,
                wind / 3.0,
                wind == 0,
                declaration >= 0,
                jnp.maximum(index, 0) / 200.0,
                index >= 0,
                jnp.maximum(elapsed, 0) / 200.0,
                elapsed >= 0,
            ],
            -1,
        ).astype(jnp.float32)
        player_tok = (
            self.player_features(player_vec)
            + self.seat_identity(jnp.arange(3) + 1)[None]
            + self.segment_identity(jnp.int32(1))
            + self._identity(declaration)
        )
        player_tok = jnp.pad(player_tok, ((0, 0), (0, 1), (0, 0)))
        player_mask = jnp.broadcast_to(jnp.array([1, 1, 1, 0], jnp.float32), (B, 4))

        melds = b(obs["melds"], 3).astype(jnp.int32)
        action, tile, src = melds[:, 0], melds[:, 1], melds[:, 2]
        meld_valid = (action >= 0) & (local_tile_to_canonical(tile) >= 0)
        selfkan = (action >= SELF_KAN_OFFSET) & (action < SELF_KAN_END)
        openkan = action == Action.OPEN_KAN
        pon = (action == Action.PON) | (action == Action.PON_RED)
        kind = jnp.where(selfkan, EVENT_SELF_KAN, jnp.where(openkan, EVENT_OPEN_KAN, jnp.where(pon, EVENT_PON, 0)))
        owners = jnp.arange(3)[None, :, None]
        source = jnp.where((src > 0) & (src < 3), (owners + src) % 3, -1)
        meld_vec = jnp.stack(
            [selfkan, openkan, pon, action == Action.PON_RED, src == 0, (src >= 0) & (src < 3)], -1
        ).astype(jnp.float32)
        meld_tok = (
            self._identity(tile)
            + self.event_kind_identity(kind)
            + self.seat_identity(owners + 1)
            + self.source_identity(source + 1)
            + self.meld_slot_identity(jnp.arange(4))[None, None]
            + self.segment_identity(jnp.int32(2))
            + self.meld_features(meld_vec)
        )
        meld_tok = jnp.pad(meld_tok, ((0, 0), (0, 1), (0, 0), (0, 0))).reshape(B, 16, D)
        meld_mask = jnp.pad(meld_valid, ((0, 0), (0, 1), (0, 0))).reshape(B, 16).astype(jnp.float32)

        events = decode_public_events(obs)
        event_tok = (
            self._canonical_identity(events["tile"])
            + self.colour_identity(events["colour"])
            + self.colour_identity(events["contains_red"])
            + self.event_kind_identity(events["kind"])
            + self.seat_identity(events["actor"] + 1)
            + self.source_identity(events["source"] + 1)
            + self.tsumogiri_identity(events["tsumogiri"])
            + self.segment_identity(jnp.int32(3))
        )
        event_mask = events["valid"].astype(jnp.float32)

        scalar_keys = (
            "round",
            "round_limit",
            "honba",
            "kyotaku",
            "wall_remaining",
            "prevalent_wind",
            "seat_wind",
            "absolute_dealer",
            "shanten_count",
            "furiten",
            "is_discard_node",
            "public_history_complete",
        )
        scales = (10.0, 10.0, 10.0, 10.0, 55.0, 3.0, 3.0, 8.0, 8.0, 1.0, 1.0, 1.0)
        table_vec = [b(obs[k], 0).astype(jnp.float32)[:, None] / scale for k, scale in zip(scalar_keys, scales)]
        table_vec.append(jax.nn.one_hot(b(obs["last_player"], 0) + 1, 4))
        table_tok = (
            self.table_features(jnp.concatenate(table_vec, -1))
            + self._identity(target)
            + self._identity(draw)
            + self.segment_identity(jnp.int32(4))
        )
        # Indicator identities/redness without asserting unrevealed identities;
        # ordering is immaterial. Dora multiplicity is also on canonical tiles.
        for key in ("dora_indicators", "kita_tiles"):
            ids = b(obs[key], 1).astype(jnp.int32)
            table_tok += (self._identity(ids) * (local_tile_to_canonical(ids) >= 0)[..., None]).sum(1)
        tokens = jnp.concatenate([full_tiles, player_tok, meld_tok, event_tok, table_tok[:, None]], 1)
        mask = jnp.concatenate([tile_mask, player_mask, meld_mask, event_mask, jnp.ones((B, 1))], 1)
        tokens *= mask[..., None]
        version_ok = b(obs["feature_version"], 0) == 4
        return jnp.where(version_ok[:, None, None], tokens, jnp.nan), mask


# --- privileged central-Q inputs -----------------------------------------------

ACTOR_OBSERVATION_KEYS = frozenset(BASE_SHAPES) | frozenset(FEATURE_SHAPES) | {"can_win"}
# Explicit reviewed allowlists: new simulator fields are NOT silently admitted.
PLAYER_FIELDS = (
    "hand",
    "hand_with_red",
    "hand_ids",
    "legal_action_mask",
    "can_win",
    "has_yaku",
    "fan",
    "fu",
    "melds",
    "meld_counts",
    "river",
    "discard_counts",
    "riichi",
    "riichi_declared",
    "double_riichi",
    "ippatsu",
    "furiten_by_discard",
    "furiten_by_pass",
    "is_hand_concealed",
    "pon",
    "has_won",
    "n_kan",
    "has_nagashi_mangan",
    "kita",
)
ROUND_FIELDS = (
    "action_history",
    "round_step",
    "history_overflow",
    "round",
    "round_limit",
    "terminated_round",
    "honba",
    "kyotaku",
    "init_wind",
    "seat_wind",
    "dealer",
    "order_points",
    "score",
    "deck",
    "next_deck_ix",
    "last_deck_ix",
    "draw_next",
    "last_draw",
    "last_player",
    "dora_indicators",
    "ura_dora_indicators",
    "is_abortive_draw_normal",
    "dummy_count",
    "is_haitei",
    "target",
    "n_kan_doras",
    "kan_declared",
    "can_after_kan",
    "can_robbing_kan",
    "kita_declared",
)
STATE_FIELDS = (
    "current_player",
    "legal_action_mask",
    "step_count",
    "rewards",
    "pending_rewards",
    "pending_winners",
    "terminated",
    "truncated",
)
POLICY_CONTEXT_SHAPES = {
    "learner_seat": (),
    "policy_ids": (3,),
    "policy_versions": (3,),
    "role_ids": (3,),
}
_DEFAULT = default_state()
STATE_SHAPES = {key: jnp.shape(getattr(_DEFAULT, key)) for key in STATE_FIELDS}
PLAYER_SHAPES = {key: jnp.shape(getattr(_DEFAULT.players, key)) for key in PLAYER_FIELDS}
ROUND_SHAPES = {key: jnp.shape(getattr(_DEFAULT.round_state, key)) for key in ROUND_FIELDS}
del _DEFAULT


def validate_actor_observation(observation):
    """Fail closed on extra keys BEFORE V4 tokenization, including actor-only apply."""
    if not isinstance(observation, Mapping):
        raise ValueError("actor observation must be a dictionary")
    extra = set(observation) - ACTOR_OBSERVATION_KEYS
    if extra:
        raise ValueError(f"forbidden actor observation keys: {sorted(extra)}")
    return observation


def _numeric(value, name):
    value = jnp.asarray(value)
    if value.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be compact numeric data, not RNG or objects")
    return value


def _policy_context(context, batch):
    if context is None:
        context = {}
    if not isinstance(context, Mapping) or set(context) - set(POLICY_CONTEXT_SHAPES):
        raise ValueError("unsupported policy_context fields (RNG and labels are forbidden)")
    out = {}
    for key, shape in POLICY_CONTEXT_SHAPES.items():
        value = _numeric(context.get(key, jnp.full(batch + shape, -1, jnp.int32)), key)
        if value.shape != batch + shape or value.dtype.kind not in "biu":
            raise ValueError(f"policy_context {key} must have integer shape {batch + shape}")
        out[key] = value.astype(jnp.int32)
    return out


def build_privileged_q_state(pre_state, policy_context=None):
    """Capture current sanma state only; native compact dtypes, no one-hot storage.

    Supports one state or one leading batch dimension (and use under jit/vmap).
    Environment rewards here are CURRENT/PENDING state, never future targets.
    """
    batch = jnp.shape(pre_state.current_player)
    if len(batch) > 1:
        raise ValueError("privileged Q state supports at most one leading batch dimension")
    privileged = {key: _numeric(getattr(pre_state, key), key) for key in STATE_FIELDS}
    privileged["players"] = {key: _numeric(getattr(pre_state.players, key), key) for key in PLAYER_FIELDS}
    privileged["round_state"] = {key: _numeric(getattr(pre_state.round_state, key), key) for key in ROUND_FIELDS}
    privileged["policy_context"] = _policy_context(policy_context, batch)
    privileged["state_input_schema"] = jnp.ones(batch, jnp.int8)
    validate_privileged_q_state(privileged)
    return privileged


def validate_privileged_q_state(privileged):
    """Exact numeric schema; refuse labels, RNG and malformed deck/player arrays."""
    expected = set(STATE_FIELDS) | {"players", "round_state", "policy_context", "state_input_schema"}
    if not isinstance(privileged, Mapping) or set(privileged) != expected:
        raise ValueError("privileged Q state requires exact schema-1 fields")
    batch = jnp.shape(privileged["current_player"])
    if len(batch) > 1:
        raise ValueError("privileged Q state supports at most one batch dimension")
    groups = (
        (privileged, STATE_SHAPES),
        (privileged["players"], PLAYER_SHAPES),
        (privileged["round_state"], ROUND_SHAPES),
    )
    for data, shapes in groups:
        if not isinstance(data, Mapping) or (data is not privileged and set(data) != set(shapes)):
            raise ValueError("privileged Q state requires exact current-state field groups")
        for key, shape in shapes.items():
            if _numeric(data[key], key).shape != batch + shape:
                raise ValueError(f"invalid privileged Q state shape for {key}")
    context = privileged["policy_context"]
    if not isinstance(context, Mapping) or set(context) != set(POLICY_CONTEXT_SHAPES):
        raise ValueError("privileged policy_context requires exact schema-1 fields")
    _policy_context(context, batch)
    version = _numeric(privileged["state_input_schema"], "state_input_schema")
    if version.shape != batch:
        raise ValueError("invalid state_input_schema shape")
    if not isinstance(version, jax.core.Tracer) and not np.all(np.asarray(version) == 1):
        raise ValueError("privileged Q state requires state_input_schema=1")
    return batch


def assemble_central_q_input(actor_observation, privileged):
    """Join separately stored public/privileged parts, without duplicating public replay."""
    validate_actor_observation(actor_observation)
    batch = validate_privileged_q_state(privileged)
    observation = prepare_observation(actor_observation)
    if jnp.shape(observation["hand"])[:-1] != batch:
        raise ValueError("public and privileged Q input batches must match")
    return {"observation": dict(observation), "privileged": privileged}


def build_central_q_input(pre_state, actor_observation, policy_context=None):
    """Prepare EXACT V4 candidate/ukeire public features and current hidden state.

    The legal mask and public overflow bookkeeping are used only when the supplied
    actor observation is raw. Prepared actor data is validated, never re-labelled.
    """
    validate_actor_observation(actor_observation)
    mask = jnp.asarray(pre_state.legal_action_mask, jnp.bool_)
    compact_mask = mask[..., :NETWORK_ACTION_COUNT]
    compact_mask = compact_mask.at[..., 78].set(mask[..., Action.KITA])
    observation = prepare_observation(
        actor_observation,
        legal_action_mask=compact_mask,
        history_overflow=pre_state.round_state.history_overflow,
    )
    return assemble_central_q_input(observation, build_privileged_q_state(pre_state, policy_context))
