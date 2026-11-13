"""Golden simulator-to-network mappings used by training and evaluation."""

import jax
import jax.numpy as jnp
import numpy as np

from networks.mahjong_3p.feature import COMPACT_TO_CANONICAL
from networks.mahjong_3p.network import (
    canonical_action_to_compact,
    canonical_tile_to_compact,
    compact_action_to_simulator,
    compact_legal_mask,
    compact_observation,
    compact_tile_to_canonical,
)
from train.configs import default_reward_config, rule_spec


def test_action_mapping_golden():
    # Discards: canonical honors/terminals compact, red ids move to 27..29.
    assert int(canonical_action_to_compact(0)) == 0
    assert int(canonical_action_to_compact(27)) == 20  # East
    assert int(canonical_action_to_compact(30)) == 23  # North
    assert int(canonical_action_to_compact(34)) == 27  # red 5m
    assert int(canonical_action_to_compact(35)) == 28  # red 5p
    assert int(canonical_action_to_compact(36)) == 29  # red 5s
    # Self-kan 37..70 -> 30..56.
    assert int(canonical_action_to_compact(37)) == 30
    assert int(canonical_action_to_compact(70)) == 56
    # Raw 78 is CHI_L, never legal in sanma and not the network KITA slot.
    assert int(canonical_action_to_compact(78)) == -1
    assert int(canonical_action_to_compact(87)) == 78
    for raw in (0, 13, 27, 30, 34, 35, 36, 37, 70, 84, 85, 86, 87):
        compact = int(canonical_action_to_compact(raw))
        assert int(compact_action_to_simulator(compact)) == raw
    for compact in (27, 28, 29, 30, 56, 78, 84, 85, 86):
        raw = int(compact_action_to_simulator(compact))
        assert int(canonical_action_to_compact(raw)) == compact


def test_tile_mapping_golden():
    for canonical in range(34):
        if canonical in COMPACT_TO_CANONICAL:
            compact = int(canonical_tile_to_compact(canonical))
            assert int(compact_tile_to_canonical(compact)) == canonical
    assert int(canonical_tile_to_compact(34)) == 27
    assert int(canonical_tile_to_compact(35)) == 28
    assert int(canonical_tile_to_compact(36)) == 29
    assert int(compact_tile_to_canonical(27)) == 4
    assert int(compact_tile_to_canonical(28)) == 13
    assert int(compact_tile_to_canonical(29)) == 22


def test_meld_mapping_is_channel_axis_not_slot_axis():
    env = rule_spec("sanma").make_env("half", reward_config=default_reward_config("sanma"))
    state = env.init(jax.random.PRNGKey(0))
    raw = dict(env.observe(state))
    raw["melds"] = jnp.full((3, 3, 4), -1, jnp.int32)
    # Slot 0: PON_RED 5s (76), target red 5s (36), source 1.
    # Slot 1: closed self-kan 5p (50), target 5p (13), source 0.
    # Channel 2 (src) and the empty sentinel must remain untouched.
    raw["melds"] = (
        raw["melds"]
        .at[0, 0, 0]
        .set(76)
        .at[1, 0, 0]
        .set(36)
        .at[2, 0, 0]
        .set(1)
        .at[0, 0, 1]
        .set(50)
        .at[1, 0, 1]
        .set(13)
        .at[2, 0, 1]
        .set(0)
    )
    melds = compact_observation(raw)["melds"]
    assert int(melds[0, 0, 0]) == 76
    assert int(melds[1, 0, 0]) == 29
    assert int(melds[2, 0, 0]) == 1
    assert int(melds[0, 0, 1]) == 36
    assert int(melds[1, 0, 1]) == 6
    assert int(melds[2, 0, 1]) == 0
    assert int(melds[0, 1, 1]) == -1 and int(melds[1, 1, 1]) == -1


def test_compact_observation_is_batched_and_adds_kita_to_tiles_seen():
    env = rule_spec("sanma").make_env("half", reward_config=default_reward_config("sanma"))
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    states = jax.vmap(env.init)(keys)
    raws = dict(jax.vmap(env.observe)(states))
    raws["kita"] = jnp.zeros_like(raws["kita"]).at[0, 0].set(2)
    raws["melds"] = jnp.full_like(raws["melds"], -1).at[0, 0, 0, 0].set(76).at[0, 1, 0, 0].set(36)
    obs = compact_observation(raws)
    assert obs["tiles_seen"].shape == (2, 27)
    assert obs["melds"].shape == (2, 3, 3, 4)
    assert int(obs["melds"][0, 0, 0, 0]) == 76
    assert int(obs["melds"][0, 1, 0, 0]) == 29
    assert int(obs["melds"][1, 0, 0, 0]) == -1
    # North is compact 23; the two set-aside tiles are public dead tiles.
    first = jax.tree.map(lambda value: value[0], states)
    second = jax.tree.map(lambda value: value[1], states)
    base0 = int(compact_observation(env.observe(first))["tiles_seen"][23])
    base1 = int(compact_observation(env.observe(second))["tiles_seen"][23])
    assert int(obs["tiles_seen"][0, 23]) == base0 + 2
    assert int(obs["tiles_seen"][1, 23]) == base1
    assert int(obs["n_kita"][0, 0]) == 2 and int(obs["n_kita"][1, 0]) == 0


def test_live_mask_and_observation_use_the_same_abi():
    env = rule_spec("sanma").make_env("half", reward_config=default_reward_config("sanma"))
    state = env.init(jax.random.PRNGKey(0))
    raw = env.observe(state)
    obs = compact_observation(raw)
    mask = compact_legal_mask(state.legal_action_mask)
    assert state.legal_action_mask.shape == (88,)
    assert mask.shape == (87,)
    # Every network-legal action is raw-legal, and every raw-legal action maps
    # to exactly one network-legal slot (raw 78 is never legal).
    network_to_raw = np.asarray([compact_action_to_simulator(i) for i in range(87)])
    raw_legal = np.asarray(state.legal_action_mask)
    for network, raw_action in enumerate(network_to_raw):
        if raw_action >= 0:
            assert bool(mask[network]) == bool(raw_legal[raw_action])
    assert int(obs["last_draw"]) == int(canonical_tile_to_compact(raw["last_draw"]))
    hand = np.asarray(raw["hand"], np.int32)[np.asarray(COMPACT_TO_CANONICAL)]
    np.testing.assert_array_equal(np.asarray(obs["hand"]), hand)
