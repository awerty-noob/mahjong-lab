"""CPU-sized public v3 tests: exact candidates, event semantics, no wall peeks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mahjax.common.logic import _append_action_history
from mahjax.common.meld import Meld
from mahjax.common.shanten import Shanten
from mahjax.common.tile import River
from mahjax.mahjong_3p.observation import _observe_dict
from mahjax.mahjong_3p.action import Action
from networks.mahjong_3p.feature import COMPACT_TO_CANONICAL, SELF_KAN_OFFSET, augment_observation
from networks.mahjong_3p.network import compact_legal_mask, compact_observation
from mahjax.mahjong_3p.state import default_state

from networks.mahjong_3p.feature import _candidate_ukeire

BASE = jax.jit(lambda state: compact_observation(_observe_dict(state)))
AUGMENT = jax.jit(augment_observation)
NUMBER = jax.jit(Shanten.number)
HAND = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 15, 0]
V3_KEYS = {
    "feature_version", "candidate_valid", "candidate_shanten", "candidate_ukeire",
    "candidate_ukeire_types", "candidate_ukeire_count", "dora_count",
    "public_history_complete", "riichi_declaration_tile", "riichi_declaration_index",
    "riichi_elapsed", "discard_counts", "post_riichi_discard_counts", "genbutsu",
}


def state_with_hand(tiles=HAND, *, melds=()):
    state = default_state()
    red_hand = jnp.bincount(jnp.asarray(tiles), length=37).astype(jnp.int8)
    tile_ids = jnp.asarray(tiles, jnp.int32)
    compact = jnp.asarray(COMPACT_TO_CANONICAL, jnp.int32)
    canonical = compact[jnp.clip(tile_ids, 0, 26)]
    canonical = jnp.where(tile_ids == 27, 4, canonical)
    canonical = jnp.where(tile_ids == 28, 13, canonical)
    canonical = jnp.where(tile_ids == 29, 22, canonical)
    hand = jnp.bincount(canonical, length=34).astype(jnp.int8)
    mask = jnp.zeros(87, dtype=jnp.bool_).at[:30].set(red_hand[:30] > 0)
    meld_words = state.players.melds
    for i, word in enumerate(melds):
        meld_words = meld_words.at[0, i].set(word)
    return state.replace(
        legal_action_mask=mask,
        players=state.players.replace(
            hand=state.players.hand.at[0].set(hand),
            hand_with_red=state.players.hand_with_red.at[0].set(red_hand),
            melds=meld_words,
            meld_counts=state.players.meld_counts.at[0].set(len(melds)),
        ),
        round_state=state.round_state.replace(last_draw=jnp.int8(tiles[-1])),
    )


def with_history(state, events, *, riichi=(False, False, False)):
    history = jnp.full((3, 200), -1, dtype=jnp.int8)
    if events:
        history = history.at[:, :len(events)].set(jnp.asarray(events, dtype=jnp.int8).T)
    return state.replace(
        players=state.players.replace(riichi=jnp.asarray(riichi)),
        round_state=state.round_state.replace(action_history=history, round_step=jnp.int32(len(events))),
    )


def network_mask(state):
    mask = state.legal_action_mask
    return compact_legal_mask(mask) if mask.shape[-1] == 88 else mask


def augmented(state):
    base = dict(BASE(state))
    # The live observer can pass the public overflow bit explicitly; the V3
    # feature dict accepts it without exposing a new actor-observation key.
    # Without that bit, leave the conservative full-buffer default intact.
    if bool(state.round_state.history_overflow):
        base["public_history_complete"] = jnp.bool_(False)
    return AUGMENT(base, network_mask(state))


def assert_candidate_reference(state, obs):
    hand = np.asarray(obs["hand"], dtype=np.int32)
    unseen = np.clip(4 - np.asarray(obs["tiles_seen"], dtype=np.int32), 0, 4)
    compact = np.asarray(COMPACT_TO_CANONICAL, dtype=np.int32)
    hand34 = np.zeros(34, dtype=np.int32)
    hand34[compact] = hand
    for discard in np.flatnonzero(np.asarray(obs["candidate_valid"])):
        reduced34 = hand34.copy()
        reduced34[compact[discard]] -= 1
        shanten = int(NUMBER(jnp.asarray(reduced34)))
        assert int(obs["candidate_shanten"][discard]) == shanten
        expected = np.zeros(27, dtype=np.int8)
        for draw in range(27):
            drawn34 = reduced34.copy()
            if drawn34[compact[draw]] == 4:
                continue
            drawn34[compact[draw]] += 1
            if int(NUMBER(jnp.asarray(drawn34))) < shanten:
                expected[draw] = unseen[draw]
        np.testing.assert_array_equal(obs["candidate_ukeire"][discard], expected)
        assert int(obs["candidate_ukeire_types"][discard]) == int(np.count_nonzero(expected))
        assert int(obs["candidate_ukeire_count"][discard]) == int(expected.sum())

def test_opt_in_api_preserves_every_v2_key_and_reuses_base():
    state = state_with_hand()
    base = BASE(state)
    before = dict(base)
    obs = augmented(state)
    assert set(obs) == set(base) | V3_KEYS
    assert not V3_KEYS.intersection(base)
    assert int(obs["feature_version"]) == 3
    for key in base:
        np.testing.assert_array_equal(obs[key], base[key], err_msg=key)
        assert base[key] is before[key]
    offline = AUGMENT(base, network_mask(state))
    for key in obs:
        np.testing.assert_array_equal(obs[key], offline[key], err_msg=key)
    assert obs["candidate_ukeire"].shape == (27, 27)
    assert obs["discard_counts"].shape == (3, 27)


def test_every_legal_candidate_including_worse_shanten_has_exact_weighted_ukeire():
    state = state_with_hand()
    obs = augmented(state)
    valid = np.asarray(obs["candidate_valid"])
    shanten = np.asarray(obs["candidate_shanten"])
    assert len(set(shanten[valid].tolist())) > 1
    worse = valid & (shanten > shanten[valid].min())
    assert np.any(np.asarray(obs["candidate_ukeire_count"])[worse] > 0)
    assert_candidate_reference(state, obs)
    assert np.all(np.asarray(obs["candidate_ukeire"])[~valid] == -1)


@pytest.mark.parametrize("tiles,melds", [
    ([0, 0, 1, 1, 2, 2, 10, 10, 11, 11, 19, 19, 20, 21], ()),
    ([0, 1, 2, 10, 11, 19, 20, 21, 22, 23, 24, 25, 26, 0], ()),
    ([2, 3, 4, 5, 6, 7, 11, 12, 15, 15, 0], (Meld.init(Action.PON, 20, 1),)),
    ([2, 3, 4, 5, 6, 7, 11, 12, 15, 15, 0], (Meld.init(SELF_KAN_OFFSET + 20, 20, 0),)),
    ([0, 0, 0, 0, 2, 3, 4, 11, 12, 13, 20, 20, 21, 21], ()),
])
def test_special_shapes_melds_and_four_copy_boundary_match_scalar_reference(tiles, melds):
    state = state_with_hand(tiles, melds=melds)
    assert_candidate_reference(state, augmented(state))


def test_unseen_weighting_exhaustion_and_discarded_copy_stays_visible():
    state = state_with_hand([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 15])
    base = BASE(state)
    obs = AUGMENT(base, state.legal_action_mask)
    # Removing 3s from the completed hand waits on 3s. Its discarded copy is
    # face-up, not a fourth unseen tile returned to the wall.
    assert int(obs["candidate_shanten"][13]) == 0
    assert int(obs["candidate_ukeire"][13, 13]) == 3
    exhausted = dict(base, tiles_seen=base["tiles_seen"].at[13].set(4))
    dead = AUGMENT(exhausted, state.legal_action_mask)
    assert int(dead["candidate_ukeire"][13, 13]) == 0
    expected_loss = int(obs["candidate_ukeire"][13, 13])
    assert int(dead["candidate_ukeire_count"][13]) == int(obs["candidate_ukeire_count"][13]) - expected_loss
    assert int(dead["candidate_ukeire_types"][13]) == int(obs["candidate_ukeire_types"][13]) - 1


def test_red_only_discard_tsumogiri_and_legality_fold_to_compact_type():
    state = state_with_hand([2, 3, 4, 5, 28, 7, 8, 9, 10, 11, 12, 15, 15, 0])
    mask = jnp.zeros(87, dtype=jnp.bool_).at[28].set(True)
    red = augmented(state.replace(legal_action_mask=mask))
    np.testing.assert_array_equal(np.flatnonzero(red["candidate_valid"]), [6])
    normal_state = state_with_hand()
    normal = augmented(normal_state.replace(legal_action_mask=mask.at[28].set(False).at[6].set(True)))
    np.testing.assert_array_equal(red["candidate_ukeire"][6], normal["candidate_ukeire"][6])
    drawn_red = state.replace(
        legal_action_mask=jnp.zeros(87, dtype=jnp.bool_).at[Action.TSUMOGIRI].set(True),
        round_state=state.round_state.replace(last_draw=jnp.int8(28)),
    )
    np.testing.assert_array_equal(np.flatnonzero(augmented(drawn_red)["candidate_valid"]), [6])
    assert int(red["candidate_shanten"][0]) == -1  # held, but not a legal action


@pytest.mark.parametrize("action", [Action.RON, Action.PON, Action.PASS, Action.OPEN_KAN])
def test_response_and_terminal_candidates_are_unknown(action):
    state = state_with_hand().replace(legal_action_mask=jnp.zeros(87, dtype=jnp.bool_).at[action].set(True))
    obs = augmented(state)
    assert not np.any(obs["candidate_valid"])
    for key in ("candidate_shanten", "candidate_ukeire", "candidate_ukeire_types", "candidate_ukeire_count"):
        assert np.all(np.asarray(obs[key]) == -1), key
    terminal = state.replace(terminated=jnp.bool_(True), legal_action_mask=jnp.ones(87, dtype=jnp.bool_))
    assert not np.any(augmented(terminal)["candidate_valid"])


def test_public_indicators_use_sanma_cycles_red_folding_and_multiplicity():
    state = state_with_hand()
    for indicator, expected in enumerate([1, 0, 3, 4, 5, 6, 7, 8, 9, 10, 2, 12, 13, 14, 15, 16, 17, 18, 19, 11, 21, 22, 23, 20, 25, 26, 24]):
        base = dict(BASE(state), dora_indicators=jnp.array([indicator, -1, -1, -1, -1], jnp.int8))
        # Response path avoids a new candidate scan for this indicator-only test.
        base["is_discard_node"] = jnp.bool_(False)
        obs = AUGMENT(base, jnp.zeros(87, dtype=jnp.bool_))
        expected_counts = np.zeros(27, dtype=np.int8)
        expected_counts[expected] = 1
        np.testing.assert_array_equal(obs["dora_count"], expected_counts)
    base = dict(BASE(state), dora_indicators=jnp.array([6, 6, 15, 0, -1], jnp.int8))
    base["is_discard_node"] = jnp.bool_(False)
    counts = AUGMENT(base, jnp.zeros(87, dtype=jnp.bool_))["dora_count"]
    assert int(counts[7]) == 2
    assert int(counts[16]) == 1
    assert int(counts[1]) == 1
    assert int(counts.sum()) == 4


def test_declaration_uses_following_red_discard_not_riichi_action_and_counts_only_discards():
    events = [
        (1, 20, 0), (2, 6, 0), (1, Action.RIICHI, -1),
        (1, 28, 1), (2, Action.PON, -1), (2, 7, 0),
        (0, Action.KITA, -1), (0, 23, 0), (1, 29, 1),
    ]
    state = with_history(state_with_hand(), events, riichi=(False, True, False))
    obs = augmented(state)
    np.testing.assert_array_equal(obs["riichi_declaration_tile"], [-1, 28, -1])
    np.testing.assert_array_equal(obs["riichi_declaration_index"], [-1, 3, -1])
    np.testing.assert_array_equal(obs["riichi_elapsed"], [-1, 3, -1])
    assert int(obs["discard_counts"][1, 6]) == 1
    assert int(obs["discard_counts"][1, 15]) == 1
    assert int(obs["post_riichi_discard_counts"][1, 6]) == 0
    assert int(obs["post_riichi_discard_counts"][1, 15]) == 1
    assert int(obs["discard_counts"][0, 23]) == 1  # KITA itself is not a discard
    assert bool(obs["genbutsu"][1, 20])  # pre-riichi own river
    assert bool(obs["genbutsu"][1, 7])
    assert bool(obs["genbutsu"][1, 6])  # own declaration is safe
    # Old discards by a different seat, before the declaration, are not safe.
    assert not bool(obs["genbutsu"][1, 0])
    rotated = state.replace(current_player=jnp.int8(2))
    rotated_obs = augmented(rotated)
    np.testing.assert_array_equal(rotated_obs["riichi_declaration_tile"], [-1, -1, 28])
    np.testing.assert_array_equal(rotated_obs["discard_counts"], np.asarray(obs["discard_counts"])[[2, 0, 1]])


def test_riichi_action_alone_and_unaccepted_or_ronned_declaration_do_not_add_safety():
    state = with_history(state_with_hand(), [(1, Action.RIICHI, -1)])
    obs = augmented(state)
    assert int(obs["riichi_declaration_tile"][1]) == -1
    assert int(obs["riichi_declaration_index"][1]) == -1
    assert int(obs["riichi_elapsed"][1]) == -1
    events = [(1, Action.RIICHI, -1), (1, 28, 0), (2, 13, 0)]
    unaccepted = augmented(with_history(state, events))
    assert int(unaccepted["riichi_declaration_tile"][1]) == 28
    assert bool(unaccepted["genbutsu"][1, 6])
    assert not bool(unaccepted["genbutsu"][1, 13])
    ronned = augmented(with_history(state, events + [(0, Action.RON, -1)]))
    assert not bool(ronned["genbutsu"][1, 13])


def test_pending_response_tile_is_not_yet_genbutsu_but_post_call_and_next_draw_are():
    events = [(1, Action.RIICHI, -1), (1, 20, 0), (2, 13, 0)]
    state = with_history(state_with_hand(), events, riichi=(False, True, False))
    response = state.replace(legal_action_mask=jnp.zeros(87, dtype=jnp.bool_).at[Action.RON].set(True))
    pending = augmented(response)
    assert not bool(pending["genbutsu"][1, 13])
    # The next discard node proves all response opportunities were resolved.
    assert bool(augmented(state)["genbutsu"][1, 13])
    # A following public call likewise resolves the discard's ron window.
    called = with_history(response, events + [(0, Action.PON, -1)], riichi=(False, True, False))
    assert bool(augmented(called)["genbutsu"][1, 13])


def test_live_history_writer_keeps_riichi_and_tsumogiri_separate_with_red_id():
    state = state_with_hand([2, 3, 4, 5, 28, 7, 8, 9, 10, 11, 12, 15, 15, 0])
    state = state.replace(round_state=state.round_state.replace(last_draw=jnp.int8(28)))
    append = jax.jit(_append_action_history)
    state = append(state, jnp.int32(Action.RIICHI))
    assert int(augmented(state)["riichi_declaration_tile"][0]) == -1
    state = append(state, jnp.int32(Action.TSUMOGIRI))
    obs = augmented(state)
    assert int(obs["riichi_declaration_tile"][0]) == 28
    assert int(obs["riichi_declaration_index"][0]) == 1
    assert int(obs["discard_counts"][0, 6]) == 1


def test_history_overflow_is_explicit_unknown_and_legacy_full_buffers_are_conservative():
    state = with_history(state_with_hand(), [(1, Action.RIICHI, -1), (1, 28, 0)])
    state = state.replace(round_state=state.round_state.replace(history_overflow=jnp.bool_(True)))
    obs = augmented(state)
    assert not bool(obs["public_history_complete"])
    for key in ("riichi_declaration_tile", "riichi_declaration_index", "riichi_elapsed", "discard_counts", "post_riichi_discard_counts"):
        assert np.all(np.asarray(obs[key]) == -1), key
    assert bool(obs["genbutsu"][1, 6])  # provable own-history lower bound
    events = [(i % 3, 20, 0) for i in range(200)]
    full = with_history(state_with_hand(), events)
    assert not bool(augmented(full)["public_history_complete"])


def test_no_opponent_private_hand_wall_ura_or_hidden_response_leak():
    state = with_history(state_with_hand(), [(1, Action.RIICHI, -1), (1, 20, 0)], riichi=(False, True, False))
    baseline = augmented(state)
    changed = state.replace(
        players=state.players.replace(
            hand=state.players.hand.at[1:].set(4),
            hand_with_red=state.players.hand_with_red.at[1:].set(3),
            can_win=state.players.can_win.at[1:].set(True),
            furiten_by_pass=state.players.furiten_by_pass.at[1:].set(True),
        ),
        round_state=state.round_state.replace(
            deck=jnp.full_like(state.round_state.deck, 26),
            ura_dora_indicators=jnp.full_like(state.round_state.ura_dora_indicators, 26),
        ),
    )
    actual = augmented(changed)
    for key in baseline:
        np.testing.assert_array_equal(actual[key], baseline[key], err_msg=key)


def test_public_visibility_accounts_for_called_discards_melds_kita_and_indicators_once():
    state = state_with_hand()
    river = River.add_discard(state.players.river, jnp.int8(35), 1, 0, False, False)
    river = River.add_meld(river, Action.PON_RED, 1, 0, 2)
    melds = state.players.melds.at[2, 0].set(Meld.init(Action.PON_RED, 13, 1))
    state = state.replace(
        players=state.players.replace(
            river=river, melds=melds,
            kita=state.players.kita.at[1].set(1),
        ),
        round_state=state.round_state.replace(dora_indicators=jnp.array([7, -1, -1, -1, -1], jnp.int8)),
    )
    obs = augmented(state)
    assert int(obs["tiles_seen"][6]) == 4  # one own + three pon, not five
    assert int(obs["tiles_seen"][23]) == 1
    assert int(obs["tiles_seen"][7]) == 1
    assert np.all(np.asarray(obs["candidate_ukeire"])[np.asarray(obs["candidate_valid"]), 6] == 0)
    assert_candidate_reference(state, obs)


def test_missing_or_batched_offline_inputs_are_rejected_instead_of_fabricated():
    state = state_with_hand()
    base = BASE(state)
    for key in ("hand", "action_history", "melds", "tiles_seen", "riichi"):
        incomplete = dict(base)
        incomplete.pop(key)
        with pytest.raises(ValueError, match=key):
            augment_observation(incomplete, state.legal_action_mask)
    with pytest.raises(ValueError, match="legal_action_mask"):
        augment_observation(base, jnp.zeros((1, 87), jnp.bool_))
    with pytest.raises(ValueError, match="unbatched shape"):
        augment_observation(dict(base, hand=base["hand"][None, :]), state.legal_action_mask)


def test_jit_vmap_matches_single_lane_and_preserves_conditional_scan():
    state = state_with_hand()
    base = BASE(state)
    response_base = dict(base, is_discard_node=jnp.bool_(False), last_draw=jnp.int8(-1))
    batch = jax.tree.map(lambda a, b: jnp.stack([a, b]), base, response_base)
    masks = jnp.stack([state.legal_action_mask, jnp.zeros(87, jnp.bool_).at[Action.PASS].set(True)])
    obs = jax.jit(jax.vmap(augment_observation))(batch, masks)
    single = AUGMENT(base, state.legal_action_mask)
    for key in single:
        np.testing.assert_array_equal(obs[key][0], single[key], err_msg=key)
    assert np.all(np.asarray(obs["candidate_ukeire"])[1] == -1)
    # Structural guard: cache work remains inside a cond, and batching uses scan
    # rather than a select that broadcasts the entire cache across batch lanes.
    args = (base["hand"], single["candidate_shanten"], single["candidate_valid"], base["tiles_seen"], jnp.int32(0), jnp.bool_(False))
    assert "cond[" in str(jax.make_jaxpr(_candidate_ukeire)(*args))
    batched_args = jax.tree.map(lambda a: jnp.stack([a, a]), args)
    assert "scan[" in str(jax.make_jaxpr(jax.vmap(_candidate_ukeire))(*batched_args))
