"""V4 tokenizer/schema tests only: no training or optimizer steps."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mahjax.mahjong_3p.action import Action

from networks.mahjong_3p.feature import (
    COMPACT_TO_CANONICAL,
    EVENT_DISCARD,
    EVENT_KITA,
    EVENT_PON,
    EVENT_SELF_KAN,
    EVENT_SLICE,
    FEATURE_SHAPES,
    GLOBAL_INDEX,
    MELD_SLICE,
    PLAYER_SLICE,
    ObservationTokenizer,
    canonicalize_tile_axis,
    compact_to_canonical34,
    decode_public_events,
    local_tile_to_canonical,
    prepare_observation,
    token_positions,
)


def observation():
    obs = {
        "hand": jnp.zeros(27, jnp.int8).at[6].set(2),
        "is_red": jnp.zeros(27, bool).at[6].set(True),
        "last_draw": jnp.int8(28),
        "shanten_count": jnp.int8(2),
        "discard_shanten": jnp.full((27, 3), 14, jnp.int8),
        "discard_can_win": jnp.full((27, 27), -1, jnp.int8),
        "furiten": jnp.bool_(False),
        "is_discard_node": jnp.bool_(False),
        "n_kita": jnp.zeros(3, jnp.int8),
        "kita_tiles": jnp.full(4, -1, jnp.int8),
        "melds": jnp.full((3, 3, 4), -1, jnp.int8),
        "action_history": jnp.full((3, 200), -1, jnp.int8),
        "scores": jnp.full(3, 350, jnp.int32),
        "round": jnp.int8(0),
        "round_limit": jnp.int8(6),
        "honba": jnp.int8(0),
        "kyotaku": jnp.int8(0),
        "prevalent_wind": jnp.int8(0),
        "seat_wind": jnp.int8(0),
        "absolute_dealer": jnp.int8(0),
        "dora_indicators": jnp.full(5, -1, jnp.int8),
        "ippatsu": jnp.zeros(3, bool),
        "riichi": jnp.zeros(3, bool),
        "is_hand_concealed": jnp.ones(3, bool),
        "wall_remaining": jnp.int32(40),
        "tiles_seen": jnp.zeros(27, jnp.int8).at[6].set(2),
        "target": jnp.int8(-1),
        "last_player": jnp.int8(-1),
    }
    for key, shape in FEATURE_SHAPES.items():
        obs[key] = jnp.full(shape, -1, jnp.int16)
    obs.update(
        feature_version=jnp.int8(4),
        candidate_valid=jnp.zeros(27, bool),
        dora_count=jnp.zeros(27, jnp.int8),
        genbutsu=jnp.zeros((3, 27), bool),
        public_history_complete=jnp.bool_(True),
    )
    return obs


@pytest.fixture(scope="module")
def bundle():
    obs = observation()
    model = ObservationTokenizer(d_model=12)
    params = jax.jit(model.init)(jax.random.key(41), obs)
    return obs, model, params


def test_layout_dynamic_width_batch_and_mask(bundle):
    obs, model, params = bundle
    tokens, mask = model.apply(params, obs)
    assert tokens.shape == (1, 255, 12) and mask.shape == (1, 255)
    assert np.isfinite(tokens).all()
    assert int(mask.sum()) == 31  # 27 tiles, three seats, one table.
    assert np.all(np.asarray(mask)[0, 1:8] == 0)
    assert mask[0, PLAYER_SLICE][-1] == 0
    assert np.all(np.asarray(mask)[0, MELD_SLICE] == 0)
    assert np.all(np.asarray(tokens)[np.asarray(mask) == 0] == 0)
    batch = jax.tree.map(lambda x: jnp.stack([x, x]), obs)
    batched_tokens, batched_mask = jax.jit(model.apply)(params, batch)
    assert batched_tokens.shape == (2, 255, 12)
    np.testing.assert_allclose(batched_tokens[0], tokens[0], atol=1e-6)
    np.testing.assert_array_equal(batched_mask[0], mask[0])


def test_compact_red_and_both_ukeire_axes_use_canonical34():
    np.testing.assert_array_equal(compact_to_canonical34(jnp.arange(27)), COMPACT_TO_CANONICAL)
    np.testing.assert_array_equal(
        local_tile_to_canonical(jnp.array([-1, 0, 1, 6, 15, 27, 28, 29, 30])), [-1, 0, 8, 13, 22, 4, 13, 22, -1]
    )
    values = jnp.zeros((27, 27)).at[6, 15].set(4)
    canonical = canonicalize_tile_axis(canonicalize_tile_axis(values), axis=0)
    assert canonical.shape == (34, 34) and canonical[13, 22] == 4
    assert canonical.sum() == 4


def test_mixed_history_semantics_red_source_and_temporal_positions():
    obs = observation()
    history = (
        obs["action_history"]
        .at[:, :7]
        .set(
            jnp.array(
                [
                    [1, 2, 2, 0, 0, 1, 2],
                    [35, Action.PON_RED, 13, 59, Action.KITA, Action.RIICHI, 36],
                    [0, -1, 1, -1, -1, -1, 0],
                ],
                jnp.int8,
            )
        )
    )
    obs["action_history"] = history
    events = decode_public_events(obs)
    np.testing.assert_array_equal(
        events["kind"][0, :5], [EVENT_DISCARD, EVENT_PON, EVENT_DISCARD, EVENT_SELF_KAN, EVENT_KITA]
    )
    np.testing.assert_array_equal(events["tile"][0, :7], [13, 13, 13, 22, 30, -1, 22])
    np.testing.assert_array_equal(events["colour"][0, :7], [2, 2, 1, 0, 1, 0, 2])
    assert events["source"][0, 1] == 1 and events["contains_red"][0, 1] == 2
    positions = token_positions(obs)
    np.testing.assert_array_equal(positions[0, EVENT_SLICE][:7], np.arange(-7, 0))
    assert np.all(np.asarray(positions[0, :54]) == 0) and positions[0, GLOBAL_INDEX] == 0
    assert np.all(np.asarray(positions[0, EVENT_SLICE][7:]) == 0)
    assert not bool(events["valid"][0, 7])


def test_truncated_call_does_not_invent_target():
    obs = observation()
    obs["action_history"] = obs["action_history"].at[:, 0].set(jnp.array([1, Action.PON_RED, -1], jnp.int8))
    event = decode_public_events(obs)
    assert event["tile"][0, 0] == -1 and event["colour"][0, 0] == 0
    assert event["source"][0, 0] == -1 and event["contains_red"][0, 0] == 2


def test_strict_version_shapes_eager_and_jit_fail_closed(bundle):
    obs, model, params = bundle
    assert prepare_observation(obs) is obs
    wrong = dict(obs, feature_version=jnp.int8(3))
    with pytest.raises(ValueError, match="feature_version=4"):
        model.apply(params, wrong)
    with pytest.raises(ValueError, match="feature_version=4"):
        prepare_observation(wrong)
    tokens, _ = jax.jit(model.apply)(params, wrong)
    assert not np.isfinite(tokens).any()
    with pytest.raises(ValueError, match="shape for candidate_ukeire"):
        model.apply(params, dict(obs, candidate_ukeire=jnp.zeros((34, 34))))
    partial = dict(obs)
    partial.pop("candidate_valid")
    with pytest.raises(ValueError, match="requires candidate_valid"):
        prepare_observation(partial)


def test_own_draw_target_and_ukeire_reach_tokens_without_stale_can_win(bundle):
    obs, model, params = bundle
    baseline, _ = model.apply(params, obs)
    changed, _ = model.apply(params, dict(obs, last_draw=jnp.int8(6)))
    assert not np.allclose(baseline[0, 13], changed[0, 13])  # red vs ordinary same type
    changed, _ = model.apply(params, dict(obs, target=jnp.int8(28)))
    assert not np.allclose(baseline[0, 13], changed[0, 13])
    assert not np.allclose(baseline[0, GLOBAL_INDEX], changed[0, GLOBAL_INDEX])
    valid = obs["candidate_valid"].at[6].set(True)
    altered = dict(
        obs,
        candidate_valid=valid,
        candidate_shanten=obs["candidate_shanten"].at[6].set(2),
        candidate_ukeire=obs["candidate_ukeire"].at[6].set(jnp.zeros(27, jnp.int16)).at[6, 15].set(3),
    )
    changed, _ = model.apply(params, altered)
    assert not np.allclose(baseline[0, 13], changed[0, 13])
    irrelevant, _ = model.apply(params, dict(obs, can_win=jnp.ones(27), others_hand=jnp.ones((2, 27))))
    np.testing.assert_array_equal(irrelevant, baseline)


def test_unknown_history_statistics_are_distinct_from_known_zero(bundle):
    obs, model, params = bundle
    unknown, _ = model.apply(params, obs)
    known, _ = model.apply(
        params, dict(obs, discard_counts=jnp.zeros((3, 27)), post_riichi_discard_counts=jnp.zeros((3, 27)))
    )
    assert not np.allclose(unknown[0, 13], known[0, 13])


def test_semantic_geometry_and_role_specific_source_embeddings(bundle):
    _, _, params = bundle
    assert params["params"]["suit_identity"]["embedding"].shape == (5, 12)
    assert params["params"]["rank_identity"]["embedding"].shape == (10, 12)
    assert params["params"]["honour_identity"]["embedding"].shape == (8, 12)
    assert "source_identity" in params["params"]


def test_private_passes_do_not_create_public_tokens_or_age_other_events():
    obs = observation()
    obs["action_history"] = (
        obs["action_history"]
        .at[:, :3]
        .set(
            jnp.array(
                [
                    [0, 1, 2],
                    [6, Action.PASS, 29],
                    [0, -1, 1],
                ],
                jnp.int8,
            )
        )
    )
    events = decode_public_events(obs)
    np.testing.assert_array_equal(events["valid"][0, :3], [True, False, True])
    np.testing.assert_array_equal(events["positions"][0, :3], [-2, 0, -1])


def test_meld_masks_source_and_red_called_identity(bundle):
    obs, model, params = bundle
    melds = obs["melds"].at[:, 1, 2].set(jnp.array([Action.PON_RED, 28, 2], jnp.int8))
    tokens, mask = model.apply(params, dict(obs, melds=melds))
    index = MELD_SLICE.start + 1 * 4 + 2
    assert mask[0, index] == 1 and mask[:, MELD_SLICE].sum() == 1
    assert np.isfinite(tokens[0, index]).all()
    ordinary = melds.at[1, 1, 2].set(6)
    changed, _ = model.apply(params, dict(obs, melds=ordinary))
    assert not np.allclose(tokens[0, index], changed[0, index])
    assert np.all(np.asarray(mask)[0, MELD_SLICE.stop - 4 : MELD_SLICE.stop] == 0)


def test_raw_preparation_is_explicit_preserves_base_and_overflow():
    enriched = observation()
    raw = {key: value for key, value in enriched.items() if key not in FEATURE_SHAPES}
    with pytest.raises(ValueError, match="legal_action_mask"):
        prepare_observation(raw)
    with pytest.raises(ValueError, match="87 actions"):
        prepare_observation(raw, legal_action_mask=jnp.zeros(27, bool))
    prepared = prepare_observation(raw, legal_action_mask=jnp.zeros(87, bool), history_overflow=jnp.bool_(True))
    assert prepared["feature_version"] == 4 and not prepared["public_history_complete"]
    assert np.all(np.asarray(prepared["discard_counts"]) == -1)
    assert np.all(np.asarray(prepared["candidate_ukeire"]) == -1)
    for key in raw:
        np.testing.assert_array_equal(raw[key], prepared[key])
    batch = jax.tree.map(lambda x: jnp.stack([x, x]), raw)
    prepared_batch = jax.jit(lambda o, m: prepare_observation(o, legal_action_mask=m))(batch, jnp.zeros((2, 87), bool))
    np.testing.assert_array_equal(prepared_batch["feature_version"], [4, 4])
