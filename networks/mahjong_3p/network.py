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
    checkpoint_reading_config,
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
# The 3p simulator uses three local red ids (27/28/29) on top of its 27
# compact tile types; the full 37-slot ABI uses 34/35/36.  Both fold onto the
# same black type.  Local id 27 is the red 5m, which sanma never deals.
_LOCAL_TO_COMPACT = np.full(Tile.NUM_TILE_TYPE_WITH_RED, -1, np.int32)
for _local in range(27):
    _LOCAL_TO_COMPACT[_local] = _CANONICAL_TO_COMPACT[COMPACT_TO_CANONICAL[_local]]
_LOCAL_TO_COMPACT[28] = _CANONICAL_TO_COMPACT[13]
_LOCAL_TO_COMPACT[29] = _CANONICAL_TO_COMPACT[22]
_LOCAL_TO_COMPACT[34] = _CANONICAL_TO_COMPACT[4]
_LOCAL_TO_COMPACT[35] = _CANONICAL_TO_COMPACT[13]
_LOCAL_TO_COMPACT[36] = _CANONICAL_TO_COMPACT[22]


def compact_tile_axis(values, *, axis: int = -1) -> jax.Array:
    """Select the 27 compact slots of a canonical-34 tile axis."""
    return jnp.take(values, jnp.asarray(COMPACT_TO_CANONICAL, jnp.int32), axis=axis % jnp.ndim(values))


def compact_local_ids(values) -> jax.Array:
    """Map a scalar red-aware tile id (or -1) to a compact local id (or -1)."""
    ids = jnp.asarray(values, jnp.int32)
    safe = jnp.clip(ids, 0, Tile.NUM_TILE_TYPE_WITH_RED - 1)
    mapped = jnp.asarray(_LOCAL_TO_COMPACT)[safe]
    return jnp.where(ids >= 0, mapped, -1).astype(jnp.int32)


def compact_observation(raw: Mapping[str, Any]) -> dict:
    """Translate one simulator observation into the Actor/Critic schema."""
    observation = dict(raw)
    observation["hand"] = compact_tile_axis(raw["hand"], axis=-1)
    observation["is_red"] = compact_tile_axis(raw["is_red"], axis=-1)
    observation["tiles_seen"] = compact_tile_axis(raw["tiles_seen"], axis=-1)
    # Set-aside norths are public dead tiles but are no longer in any hand, river,
    # or meld array the simulator's tiles_seen walks.
    observation["tiles_seen"] = observation["tiles_seen"].at[NORTH_LOCAL_ID].add(
        jnp.asarray(raw["kita"], jnp.int8).sum(dtype=jnp.int8)
    )
    observation["discard_shanten"] = compact_tile_axis(raw["discard_shanten"], axis=-2)
    observation["discard_can_win"] = compact_tile_axis(compact_tile_axis(raw["discard_can_win"], axis=-2), axis=-1)
    observation["dora_indicators"] = compact_local_ids(raw["dora_indicators"])
    observation["last_draw"] = compact_local_ids(raw["last_draw"])
    observation["target"] = compact_local_ids(raw["target"])
    melds = jnp.asarray(raw["melds"], jnp.int32)
    # (channel, seat, slot); channel 1 holds canonical target ids (0xFFFF when empty).
    observation["melds"] = melds.at[1].set(compact_local_ids(melds[1]))
    kita = jnp.asarray(raw["kita"], jnp.int8)
    observation["n_kita"] = kita
    # North ids are identical (there is one north), so only the observer's count
    # matters for the (4,) set-aside slot vector.
    own_kita = kita[0].astype(jnp.int32)
    observation["kita_tiles"] = jnp.where(jnp.arange(4) < own_kita, NORTH_LOCAL_ID, -1).astype(jnp.int8)
    observation["absolute_dealer"] = jnp.asarray(raw["seat_wind"] == 0, jnp.int8)
    observation.pop("kita", None)
    observation.pop("can_win", None)
    return observation


def compact_legal_mask(mask) -> jax.Array:
    """Translate the simulator's 88-action mask onto the 87-action network ABI."""
    mask = jnp.asarray(mask, jnp.bool_)
    compact = mask[..., :NEXT_ACTION_SIZE]
    return compact.at[..., KITA_NETWORK_SLOT].set(mask[..., KITA_SIMULATOR_SLOT])


def to_simulator_action(action) -> jax.Array:
    """Translate a network action id back to the simulator's id space."""
    action = jnp.asarray(action, jnp.int32)
    return jnp.where(action == KITA_NETWORK_SLOT, KITA_SIMULATOR_SLOT, action)


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
    reading = checkpoint_reading_config(checkpoint)
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


def adapt_network_observation(observation, network_config, config=None, *,
                              legal_action_mask=None, env_name="sanma"):
    """Legacy adapter: upstream observations are already tokenizer-compatible."""
    del network_config, config, legal_action_mask, env_name
    return observation
