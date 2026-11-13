"""Simulator to network boundary: compaction, observation adapter, builders.

The simulator speaks canonical 34-tile ids and exposes ``players.kita`` counts;
the deployed Actor/Critic speak the compact 27-tile action ABI. Exactly one
translation happens here, in ``compact_observation``, and it is the only place
that knows both schemas: networks above this module never touch a
``mahjax.mahjong_3p`` state object.

Nothing in this module reads hidden information. The privileged central-Q input
is assembled by ``feature.build_central_q_input`` from an explicit pre-action
state, which is a training-only path.
"""

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mahjax.common.tile import Tile

from .configs import (
    DEFAULT_CRITIC_CONFIG,
    DEFAULT_NETWORK_CONFIG,
    checkpoint_critic_config,
    checkpoint_head_config,
    checkpoint_network_config,
    validate_critic_config,
    validate_head_config,
    validate_network_config,
    validate_reading_config,
)
from .feature import COMPACT_TO_CANONICAL, prepare_observation
from .transformer import make_actor_critic, make_central_q_net

# The simulator numbers north-setting as 87 and keeps chi ids 79..83 unused;
# the deployed Actor keeps the historical 78 slot for north inside 87 actions.
NEXT_ACTION_SIZE = 87
KITA_NETWORK_SLOT = 78
KITA_SIMULATOR_SLOT = 87

NUM_CANONICAL_TILES = 34
NUM_COMPACT_TILES = len(COMPACT_TO_CANONICAL)
# Compact local id of north (canonical 30); every set-aside north carries it.
NORTH_LOCAL_ID = COMPACT_TO_CANONICAL.index(30)

_CANONICAL_TO_COMPACT = np.full(NUM_CANONICAL_TILES, -1, np.int32)
for _compact, _canonical in enumerate(COMPACT_TO_CANONICAL):
    _CANONICAL_TO_COMPACT[_canonical] = _compact

# The simulator has a canonical-34 tile axis plus red ids 34/35/36.  The
# network has a compact-27 tile axis plus red action slots 27/28/29.  These
# tables are deliberately the ONLY knowledge of both schemas.
_RAW_TILE_TO_COMPACT = np.full(Tile.NUM_TILE_TYPE_WITH_RED, -1, np.int32)
for _canonical in range(NUM_CANONICAL_TILES):
    _RAW_TILE_TO_COMPACT[_canonical] = _CANONICAL_TO_COMPACT[_canonical]
_RAW_TILE_TO_COMPACT[34] = 27
_RAW_TILE_TO_COMPACT[35] = 28
_RAW_TILE_TO_COMPACT[36] = 29

_COMPACT_TILE_TO_CANONICAL = np.full(30, -1, np.int32)
for _compact in range(NUM_COMPACT_TILES):
    _COMPACT_TILE_TO_CANONICAL[_compact] = COMPACT_TO_CANONICAL[_compact]
_COMPACT_TILE_TO_CANONICAL[27] = 4
_COMPACT_TILE_TO_CANONICAL[28] = 13
_COMPACT_TILE_TO_CANONICAL[29] = 22

_RAW_ACTION_TO_NETWORK = np.full(88, -1, np.int32)
for _canonical in range(NUM_CANONICAL_TILES):
    _compact = _CANONICAL_TO_COMPACT[_canonical]
    if _compact >= 0:
        _RAW_ACTION_TO_NETWORK[_canonical] = _compact
_RAW_ACTION_TO_NETWORK[34] = 27
_RAW_ACTION_TO_NETWORK[35] = 28
_RAW_ACTION_TO_NETWORK[36] = 29
for _canonical in range(NUM_CANONICAL_TILES):
    _compact = _CANONICAL_TO_COMPACT[_canonical]
    if _compact >= 0:
        _RAW_ACTION_TO_NETWORK[37 + _canonical] = 30 + _compact
for _action in range(71, 87):
    _RAW_ACTION_TO_NETWORK[_action] = _action
# Raw 78 is CHI_L in the simulator; network 78 is KITA.  It is never legal in
# sanma, but must not collide with the north slot.
_RAW_ACTION_TO_NETWORK[78] = -1
_RAW_ACTION_TO_NETWORK[87] = 78

_NETWORK_ACTION_TO_RAW = np.full(87, -1, np.int32)
for _compact in range(NUM_COMPACT_TILES):
    _NETWORK_ACTION_TO_RAW[_compact] = COMPACT_TO_CANONICAL[_compact]
_NETWORK_ACTION_TO_RAW[27] = 34
_NETWORK_ACTION_TO_RAW[28] = 35
_NETWORK_ACTION_TO_RAW[29] = 36
for _compact in range(NUM_COMPACT_TILES):
    _NETWORK_ACTION_TO_RAW[30 + _compact] = 37 + COMPACT_TO_CANONICAL[_compact]
for _action in range(71, 78):
    _NETWORK_ACTION_TO_RAW[_action] = _action
_NETWORK_ACTION_TO_RAW[78] = 87
for _action in range(79, 87):
    _NETWORK_ACTION_TO_RAW[_action] = _action


def compact_tile_axis(values, *, axis: int = -1) -> jax.Array:
    """Select the 27 compact slots of a canonical-34 tile axis."""
    return jnp.take(values, jnp.asarray(COMPACT_TO_CANONICAL, jnp.int32), axis=axis % jnp.ndim(values))


def canonical_tile_to_compact(values) -> jax.Array:
    """Map simulator tile ids (canonical 0..33, red 34..36) to compact ids.

    The returned red ladder is 27/28/29 for red 5m/5p/5s.  This is a scalar
    tile-id mapping; tile axes are converted with :func:`compact_tile_axis`.
    """
    ids = jnp.asarray(values, jnp.int32)
    safe = jnp.clip(ids, 0, Tile.NUM_TILE_TYPE_WITH_RED - 1)
    mapped = jnp.asarray(_RAW_TILE_TO_COMPACT)[safe]
    return jnp.where((ids >= 0) & (ids < Tile.NUM_TILE_TYPE_WITH_RED), mapped, -1).astype(jnp.int32)


def compact_tile_to_canonical(values) -> jax.Array:
    """Inverse scalar mapping from compact observation ids to canonical types."""
    ids = jnp.asarray(values, jnp.int32)
    safe = jnp.clip(ids, 0, 29)
    mapped = jnp.asarray(_COMPACT_TILE_TO_CANONICAL)[safe]
    return jnp.where((ids >= 0) & (ids < 30), mapped, -1).astype(jnp.int32)


def compact_local_ids(values) -> jax.Array:
    """Backward-compatible alias for :func:`canonical_tile_to_compact`."""
    return canonical_tile_to_compact(values)


def canonical_action_to_compact(actions) -> jax.Array:
    """Map simulator action ids (0..87) to the 87-action network ABI."""
    ids = jnp.asarray(actions, jnp.int32)
    safe = jnp.clip(ids, 0, 87)
    mapped = jnp.asarray(_RAW_ACTION_TO_NETWORK)[safe]
    return jnp.where((ids >= 0) & (ids < 88), mapped, -1).astype(jnp.int32)


def compact_action_to_simulator(actions) -> jax.Array:
    """Inverse action mapping; network-only/reserved slots map to -1."""
    ids = jnp.asarray(actions, jnp.int32)
    safe = jnp.clip(ids, 0, 86)
    mapped = jnp.asarray(_NETWORK_ACTION_TO_RAW)[safe]
    return jnp.where((ids >= 0) & (ids < 87), mapped, -1).astype(jnp.int32)


def compact_observation(raw: Mapping[str, Any]) -> dict:
    """Translate one simulator observation into the Actor/Critic schema."""
    observation = dict(raw)
    observation["hand"] = compact_tile_axis(raw["hand"], axis=-1)
    observation["is_red"] = compact_tile_axis(raw["is_red"], axis=-1)
    observation["tiles_seen"] = compact_tile_axis(raw["tiles_seen"], axis=-1)
    # Set-aside norths are public dead tiles but are no longer in any hand, river,
    # or meld array the simulator's tiles_seen walks.
    raw_kita = jnp.asarray(raw["kita"], jnp.int8)
    kita_total = raw_kita.sum(axis=-1, dtype=jnp.int8)
    observation["tiles_seen"] = observation["tiles_seen"].at[..., NORTH_LOCAL_ID].add(kita_total)
    observation["discard_shanten"] = compact_tile_axis(raw["discard_shanten"], axis=-2)
    observation["discard_can_win"] = compact_tile_axis(compact_tile_axis(raw["discard_can_win"], axis=-2), axis=-1)
    observation["dora_indicators"] = canonical_tile_to_compact(raw["dora_indicators"])
    observation["last_draw"] = canonical_tile_to_compact(raw["last_draw"])
    observation["target"] = canonical_tile_to_compact(raw["target"])
    melds = jnp.asarray(raw["melds"], jnp.int32)
    # Melds are (channels, seats, slots); channel 0 is the simulator action and
    # channel 1 the canonical target type.  Index the channel axis explicitly:
    # ``at[..., 0]`` would rewrite slot 0 across every channel.
    melds = melds.at[..., 0, :, :].set(canonical_action_to_compact(melds[..., 0, :, :]))
    melds = melds.at[..., 1, :, :].set(canonical_tile_to_compact(melds[..., 1, :, :]))
    observation["melds"] = melds
    kita = jnp.asarray(raw["kita"], jnp.int8)
    observation["n_kita"] = kita
    # North ids are identical (there is one north), so only the observer's count
    # matters for the (4,) set-aside slot vector.  Keep this batch-safe.
    own_kita = kita[..., 0].astype(jnp.int32)
    observation["kita_tiles"] = jnp.where(jnp.arange(4) < own_kita[..., None], NORTH_LOCAL_ID, -1).astype(jnp.int8)
    observation["absolute_dealer"] = jnp.asarray(raw["seat_wind"] == 0, jnp.int8)
    observation.pop("kita", None)
    observation.pop("can_win", None)
    return observation


def compact_legal_mask(mask) -> jax.Array:
    """Translate an 88-action simulator mask onto the 87-action network ABI.

    Already-compact 87-action masks are passed through, so callers can safely
    use this function at either side of a prepared-observation boundary.
    """
    mask = jnp.asarray(mask, jnp.bool_)
    if mask.shape[-1] == NEXT_ACTION_SIZE:
        return mask
    if mask.shape[-1] != KITA_SIMULATOR_SLOT + 1:
        raise ValueError("legal_action_mask must have 87 or 88 actions")
    mapping = jnp.asarray(_RAW_ACTION_TO_NETWORK)
    valid = mapping >= 0
    safe = jnp.where(valid, mapping, 0)
    compact = jnp.zeros(mask.shape[:-1] + (NEXT_ACTION_SIZE,), jnp.bool_)
    return compact.at[..., safe].max(mask & valid)


def to_simulator_action(action) -> jax.Array:
    """Translate network action ids back to the simulator's 88-action ABI."""
    return compact_action_to_simulator(action)


def tokenize_state(env, state) -> dict:
    """Observe a live state and return the prepared Actor observation."""
    raw = env.observe(state)
    compact = compact_observation(raw)
    return prepare_observation(
        compact,
        legal_action_mask=compact_legal_mask(state.legal_action_mask),
        history_overflow=state.round_state.history_overflow,
    )


class _NetworkObservationEnv:
    """Delegate rules/stepping; change only what ``observe`` returns."""

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def observe(self, state):
        return tokenize_state(self._env, state)


def with_network_observation(env, network_config=None):
    """Return an environment whose observations are ready for the network."""
    if network_config is not None:
        validate_network_config(network_config)
    if isinstance(env, _NetworkObservationEnv):
        return env
    return _NetworkObservationEnv(env)


def observe_for_network(env, state, network_config=None) -> dict:
    """Observe once through the adapter (works with wrapped or raw environments).

    ``network_config`` is accepted for the migrated call sites that carry the
    checkpoint's architecture; the consolidated adapter has only the v5 schema.
    """
    if network_config is not None:
        validate_network_config(network_config)
    return with_network_observation(env).observe(state)


def build_actor(network_config=None, *, head_config=None, reading_config=None):
    """Construct the deployable Actor/Critic for an explicit configuration."""
    config = validate_network_config(dict(DEFAULT_NETWORK_CONFIG) if network_config is None else network_config)
    heads = validate_head_config(head_config)
    reading = validate_reading_config(reading_config)
    return make_actor_critic(
        num_layers=config["num_layers"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        mlp_dim=config["mlp_dim"],
        head_config=heads,
        reading_config=reading,
    )()


def build_central_q(network_config=None, *, head_config=None, critic_config=None, reading_config=None):
    """Construct the training-only central action Q over an identical Actor."""
    config = validate_network_config(dict(DEFAULT_NETWORK_CONFIG) if network_config is None else network_config)
    heads = validate_head_config(head_config)
    critic = validate_critic_config(dict(DEFAULT_CRITIC_CONFIG) if critic_config is None else critic_config)
    if critic["kind"] != "central_action_q":
        raise ValueError("central action Q requires an explicit version-1 critic_config")
    reading = validate_reading_config(reading_config)
    return make_central_q_net(
        num_layers=config["num_layers"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        mlp_dim=config["mlp_dim"],
        head_config=heads,
        critic_config=critic,
        reading_config=reading,
    )()


def build_from_checkpoint(checkpoint: Mapping[str, Any]):
    """Choose the exact backbone and heads from root checkpoint provenance."""
    config = checkpoint_network_config(checkpoint)
    heads = checkpoint_head_config(checkpoint)
    # v4 ordinary model artifacts have no reading head; v5 artifacts do.
    reading = checkpoint["reading_config"] if checkpoint.get("reading_config") is not None else None
    critic = checkpoint_critic_config(checkpoint)
    if critic["kind"] == "central_action_q":
        return build_central_q(config, head_config=heads, critic_config=critic, reading_config=reading)
    return build_actor(config, head_config=heads, reading_config=reading)


def get_network_cls(env_name: str, **config):
    """Legacy zero-arg factory: tests call ``get_network_cls("sanma", **cfg)()``."""
    del env_name
    head = config.pop("head_config", None)
    reading = config.pop("reading_config", None)
    return lambda: build_actor(config or None, head_config=head, reading_config=reading)


def get_central_q_network_cls(env_name: str, **config):
    """Legacy zero-arg factory for the central-Q critic."""
    del env_name
    actor_config = config.pop("actor_config", None)
    network_config = config.pop("network_config", None)
    head = config.pop("head_config", None)
    critic = config.pop("critic_config", None)
    return lambda: build_central_q(
        actor_config or network_config or config or None,
        head_config=head,
        critic_config=critic,
    )


def adapt_network_observation(observation, network_config, config=None, *, legal_action_mask=None, env_name="sanma"):
    """Legacy adapter: upstream observations are already tokenizer-compatible."""
    del network_config, config, legal_action_mask, env_name
    return observation
