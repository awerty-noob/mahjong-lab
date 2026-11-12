"""Tiny real-V4 CPU checks only; no training/checkpoints/full-L allocation."""

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from networks.mahjong_3p.configs import (
    DEFAULT_CENTRAL_Q_CRITIC_CONFIG,
    LEGACY_CRITIC_CONFIG,
    PLAYER_FIELDS,
    ROUND_FIELDS,
    STATE_FIELDS,
    assemble_central_q_input,
    build_central_q_input,
    build_privileged_q_state,
    checkpoint_critic_config,
    validate_critic_config,
)
from networks.mahjong_3p.network import (
    build_actor,
    compact_observation,
    get_central_q_network_cls,
    get_network_cls,
    observe_for_network,
)
from train.configs import default_reward_config, rule_spec

from networks.mahjong_3p.transformer import clone_actor_and_public_encoder, relative_q_to_absolute

TINY = dict(architecture="v5", num_layers=1, d_model=8, num_heads=2, mlp_dim=16)
HEAD = {"version": 1, "hidden_dim": 5}


@pytest.fixture(scope="module")
def bundle():
    env = rule_spec("sanma").make_env(
        "half", reward_config=default_reward_config("sanma")
    )
    state = env.init(jax.random.key(641))
    observation = observe_for_network(env, state, TINY)
    central = build_central_q_input(state, observation)
    model = get_central_q_network_cls(
        "sanma", network_config=TINY, head_config=HEAD,
        critic_config=DEFAULT_CENTRAL_Q_CRITIC_CONFIG)()
    variables = jax.jit(model.init)(jax.random.key(642), observation, central)
    return env, state, observation, central, model, variables


def test_forward_init_roots_batch_jit_and_zero_fresh_q(bundle):
    _, _, obs, central, model, variables = bundle
    logits, q = jax.jit(model.apply)(variables, obs, central)
    assert set(variables["params"]) == {
        "policy_extractor", "policy_mlp", "critic_extractor", "value_critic_mlp"}
    assert set(variables["params"]["critic_extractor"]) == {"public_encoder", "private_encoder", "fusion"}
    assert logits.shape == (1, 87) and q.shape == (1, 3, 87)
    assert np.isfinite(logits).all()
    np.testing.assert_array_equal(q, 0)
    batched_obs = jax.tree.map(lambda x: jnp.stack([x, x]), obs)
    batched_central = jax.tree.map(lambda x: jnp.stack([x, x]), central)
    out_logits, out_q = jax.jit(model.apply)(variables, batched_obs, batched_central)
    assert out_q.shape == (2, 3, 87)
    np.testing.assert_allclose(out_logits, jnp.repeat(logits, 2, axis=0), atol=2e-6)
    actor = {"params": {key: variables["params"][key] for key in ("policy_extractor", "policy_mlp")}}
    np.testing.assert_allclose(model.apply(actor, obs, method=model.get_action_logits), logits, atol=2e-6)
    with pytest.raises(TypeError):
        model.init(jax.random.key(643), obs)


def test_exact_bc_actor_copy_and_safe_public_clone(bundle):
    _, _, obs, central, model, variables = bundle
    bc = get_network_cls("sanma", **TINY, head_config=HEAD)()
    source = jax.jit(bc.init)(jax.random.key(644), obs)
    cloned = clone_actor_and_public_encoder(source, variables)
    expected = bc.apply(source, obs, method=bc.get_action_and_aux)
    actual = model.apply(cloned, obs, method=model.get_action_and_aux)
    for x, y in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_array_equal(x, y)
    for x, y in zip(jax.tree.leaves(cloned["params"]["critic_extractor"]["public_encoder"]),
                    jax.tree.leaves(source["params"]["policy_extractor"])):
        np.testing.assert_array_equal(x, y)
    for root in ("private_encoder", "fusion"):
        for x, y in zip(jax.tree.leaves(cloned["params"]["critic_extractor"][root]),
                        jax.tree.leaves(variables["params"]["critic_extractor"][root])):
            np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(model.apply(cloned, central, method=model.get_action_q), 0)
    bad = copy.deepcopy(source)
    bad["params"]["policy_mlp"]["discard"]["kernel"] = jnp.zeros((8, 1))
    with pytest.raises(ValueError, match="shape/dtype"):
        clone_actor_and_public_encoder(bad, variables)


def test_hidden_hand_deck_prestate_and_actor_privacy(bundle):
    env, state, obs, central, model, variables = bundle
    other = (state.current_player + 1) % 3
    players = state.players.replace(hand=state.players.hand.at[other, 0].add(1))
    deck = state.round_state.deck.at[17].set((state.round_state.deck[17] + 1) % 27)
    changed = state.replace(players=players, round_state=state.round_state.replace(deck=deck))
    central_changed = build_central_q_input(changed, obs)
    np.testing.assert_array_equal(central_changed["observation"]["hand"], obs["hand"])
    assert not np.array_equal(central_changed["privileged"]["players"]["hand"], central["privileged"]["players"]["hand"])
    assert not np.array_equal(central_changed["privileged"]["round_state"]["deck"], central["privileged"]["round_state"]["deck"])
    # Actor method has no privileged argument, and an actor-only variables tree works.
    actor = {"params": {key: variables["params"][key] for key in ("policy_extractor", "policy_mlp")}}
    np.testing.assert_array_equal(model.apply(actor, obs, method=model.get_action_logits),
                                  model.apply(actor, central_changed["observation"], method=model.get_action_logits))
    for key in ("privileged", "others_hand", "deck", "rng", "return", "outcome"):
        injected = dict(obs, **{key: jnp.zeros(())})
        with pytest.raises(ValueError, match="forbidden actor"):
            model.apply(variables, injected, method=model.get_action_logits)
        with pytest.raises(ValueError, match="forbidden actor"):
            build_central_q_input(state, injected)
    # Prepared and raw public paths must reproduce existing V4 candidate/ukeire exactly.
    raw_central = build_central_q_input(state, compact_observation(env.observe(state)))
    assert set(raw_central["observation"]) == set(obs)
    for key in obs:
        np.testing.assert_array_equal(raw_central["observation"][key], obs[key])


def test_compact_current_state_schema_storage_and_reassembly(bundle):
    _, state, obs, central, _, _ = bundle
    hidden = build_privileged_q_state(state)
    assert "observation" not in hidden
    assert set(hidden["players"]) == set(PLAYER_FIELDS)
    assert set(hidden["round_state"]) == set(ROUND_FIELDS)
    assert set(STATE_FIELDS).issubset(hidden)
    assert hidden["round_state"]["deck"].shape == (108,)
    assert hidden["round_state"]["deck"].dtype == jnp.int8
    assert hidden["players"]["hand"].dtype == jnp.int8
    assert hidden["players"]["melds"].dtype == jnp.uint16
    assert hidden["pending_rewards"].dtype == jnp.float32
    assert jax.tree.structure(assemble_central_q_input(obs, hidden)) == jax.tree.structure(central)
    # Jit and vmap builders keep storage numeric and compact.
    built = jax.jit(build_privileged_q_state)(state)
    batched = jax.vmap(build_privileged_q_state)(jax.tree.map(lambda x: jnp.stack([x, x]), state))
    assert batched["players"]["hand"].shape == (2, 3, 34)
    for x, y in zip(jax.tree.leaves(hidden), jax.tree.leaves(built)):
        np.testing.assert_array_equal(x, y)
    for key in ("rng", "seed", "future_wall", "returns", "outcomes"):
        with pytest.raises(ValueError, match="policy_context"):
            build_privileged_q_state(state, {key: jnp.zeros(())})
        bad = dict(hidden, **{key: jnp.zeros(())})
        with pytest.raises(ValueError, match="exact schema"):
            assemble_central_q_input(obs, bad)
    bad = dict(hidden, state_input_schema=jnp.int8(2))
    with pytest.raises(ValueError, match="state_input_schema=1"):
        assemble_central_q_input(obs, bad)


def _nonzero_outputs(variables):
    # At zero-output initialization, only the output projections get gradients.
    # Model a subsequent nonzero projection to test private backpropagation too.
    variables = copy.deepcopy(variables)
    for name in ("discard", "self_kan", "special"):
        layer = variables["params"]["value_critic_mlp"][name]
        layer["kernel"] = jnp.full_like(layer["kernel"], 0.03)
    return variables


def test_new_q_gradients_private_finite_and_actor_independent(bundle):
    _, _, obs, central, model, variables = bundle
    grad = jax.jit(jax.grad(lambda p: model.apply(p, central, method=model.get_action_q).sum()))
    fresh = grad(variables)["params"]
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(fresh))
    assert any(np.any(np.asarray(x) != 0) for x in jax.tree.leaves(fresh["value_critic_mlp"]))
    updated = _nonzero_outputs(variables)
    gradients = grad(updated)["params"]
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(gradients))
    for root in ("policy_extractor", "policy_mlp"):
        assert all(np.all(np.asarray(x) == 0) for x in jax.tree.leaves(gradients[root]))
    for branch in ("public_encoder", "private_encoder", "fusion"):
        assert any(np.any(np.asarray(x) != 0) for x in jax.tree.leaves(gradients["critic_extractor"][branch]))
    hidden_changed = copy.deepcopy(central)
    hidden_changed["privileged"]["round_state"]["deck"] = hidden_changed["privileged"]["round_state"]["deck"].at[0].set(28)
    expected = model.apply(updated, central, method=model.get_action_q)
    actual = model.apply(updated, hidden_changed, method=model.get_action_q)
    assert not np.array_equal(expected, actual)
    assert not np.allclose(np.asarray(expected).sum(axis=1), 0)  # no forced zero-sum


def test_absolute_payoff_seat_rotation_preserves_acting_action_columns():
    relative = jnp.broadcast_to(jnp.arange(3 * 87).reshape(1, 3, 87), (3, 3, 87))
    q = jax.jit(relative_q_to_absolute)(relative, jnp.arange(3))
    for acting in range(3):
        for absolute in range(3):
            np.testing.assert_array_equal(q[acting, absolute], relative[acting, (absolute - acting) % 3])
    with pytest.raises(ValueError, match="shape"):
        relative_q_to_absolute(jnp.zeros((1, 87)), jnp.zeros(1))


def test_q_is_unmasked_including_reserved_actions(bundle):
    _, state, obs, central, model, variables = bundle
    variables = copy.deepcopy(variables)
    for name in ("discard", "self_kan", "special"):
        variables["params"]["value_critic_mlp"][name]["bias"] = jnp.ones_like(
            variables["params"]["value_critic_mlp"][name]["bias"])
    q = model.apply(variables, central, method=model.get_action_q)
    np.testing.assert_array_equal(q, 1)
    assert not bool(state.legal_action_mask[27])
    np.testing.assert_array_equal(q[:, :, 27], 1)


def test_critic_root_schema_strict_and_legacy_behavior_unchanged(bundle):
    _, _, obs, _, _, _ = bundle
    assert checkpoint_critic_config({}) == LEGACY_CRITIC_CONFIG
    assert checkpoint_critic_config({"metadata": {"critic_config": DEFAULT_CENTRAL_Q_CRITIC_CONFIG}}) == LEGACY_CRITIC_CONFIG
    assert checkpoint_critic_config({"critic_config": DEFAULT_CENTRAL_Q_CRITIC_CONFIG}) == DEFAULT_CENTRAL_Q_CRITIC_CONFIG
    for config in (None, {}, {"version": 1}, dict(DEFAULT_CENTRAL_Q_CRITIC_CONFIG, version=True),
                   dict(DEFAULT_CENTRAL_Q_CRITIC_CONFIG, payoff_seats="relative3"),
                   dict(DEFAULT_CENTRAL_Q_CRITIC_CONFIG, action_count=86),
                   dict(DEFAULT_CENTRAL_Q_CRITIC_CONFIG, state_input_schema=True),
                   dict(DEFAULT_CENTRAL_Q_CRITIC_CONFIG, extra=0)):
        with pytest.raises(ValueError, match="critic_config"):
            validate_critic_config(config)
    assert get_central_q_network_cls("sanma", network_config=TINY)().critic_config == DEFAULT_CENTRAL_Q_CRITIC_CONFIG
    with pytest.raises(ValueError, match="v5"):
        get_central_q_network_cls(
            "sanma",
            network_config=dict(TINY, architecture="v3"),
            critic_config=DEFAULT_CENTRAL_Q_CRITIC_CONFIG,
        )()
    legacy = build_actor(TINY, head_config={"version": 0})
    assert legacy.head_config == {"version": 0}
    old = jax.jit(legacy.init)(jax.random.key(645), obs)
    logits, value = jax.jit(legacy.apply)(old, obs)
    assert logits.shape == (1, 87) and value.shape == (1,)
    q = get_central_q_network_cls(
        "sanma", network_config=TINY, head_config=HEAD,
        critic_config=DEFAULT_CENTRAL_Q_CRITIC_CONFIG)()
    assert q.critic_config == DEFAULT_CENTRAL_Q_CRITIC_CONFIG


def test_full_l_shapes_only_no_full_parameter_allocation(bundle):
    _, _, obs, central, _, _ = bundle
    model = get_central_q_network_cls("sanma", critic_config=DEFAULT_CENTRAL_Q_CRITIC_CONFIG)()
    assert (model.num_layers, model.d_model, model.num_heads, model.mlp_dim) == (12, 256, 8, 704)
    shapes = jax.eval_shape(model.init, jax.random.key(646), obs, central)
    for root in (shapes["params"]["policy_extractor"], shapes["params"]["critic_extractor"]["public_encoder"]):
        assert sum(key.startswith("block_") for key in root) == 12
    assert not any(key.startswith("block_") for key in shapes["params"]["critic_extractor"]["private_encoder"])
    outputs = jax.eval_shape(model.apply, shapes, obs, central)
    assert outputs[0].shape == (1, 87) and outputs[1].shape == (1, 3, 87)
