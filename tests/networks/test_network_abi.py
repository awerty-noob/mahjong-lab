"""Golden simulator-to-network mappings used by training and evaluation."""

import jax
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
