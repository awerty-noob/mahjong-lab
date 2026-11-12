"""V5 five-head Actor contract; tiny CPU fixtures, never training experiments."""

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.errors import ScopeParamNotFoundError

from networks.mahjong_3p.configs import (
    DEFAULT_CENTRAL_Q_CRITIC_CONFIG,
    DEFAULT_HEAD_CONFIG,
    DEFAULT_NETWORK_CONFIG,
    DEFAULT_READING_CONFIG,
    LARGE_NETWORK_CONFIG,
    MEDIUM_NETWORK_CONFIG,
    SMALL_NETWORK_CONFIG,
    build_central_q_input,
    checkpoint_head_config,
    checkpoint_network_config,
    checkpoint_reading_config,
    model_reading_config,
    validate_network_config,
    validate_reading_config,
)
from networks.mahjong_3p.feature import prepare_observation
from networks.mahjong_3p.network import (
    adapt_network_observation,
    build_actor,
    get_central_q_network_cls,
    get_network_cls,
    observe_for_network,
    with_network_observation,
)
from networks.mahjong_3p.transformer import (
    ActorCriticNet as V5ACNet,
)
from networks.mahjong_3p.transformer import (
    ReadingHead as V5ReadingHead,
)
from networks.mahjong_3p.transformer import (
    TransformerEncoder as V4Encoder,
)
from networks.mahjong_3p.transformer import (
    clone_actor_and_public_encoder,
)
from networks.mahjong_3p.transformer import (
    make_actor_critic as make_v5_acnet,
)
from train.configs import default_reward_config, rule_spec


def prepare_network_observation(observation, network_config=None):
    if network_config is not None:
        validate_network_config(network_config)
    return prepare_observation(observation)


def _checkpoint_actor(checkpoint):
    return build_actor(
        checkpoint_network_config(checkpoint),
        head_config=checkpoint_head_config(checkpoint),
        reading_config=checkpoint_reading_config(checkpoint),
    )


TINY = dict(architecture="v5", num_layers=1, d_model=8, num_heads=2, mlp_dim=16)
HEAD = {"version": 1, "hidden_dim": 5}
READING_SHAPES = {
    "tenpai_logits": (2,),
    "shanten_logits": (2, 4),
    "danger_logits": (2, 34),
    "waits_logits": (2, 34),
    "danger_points": (2, 34),
}


@pytest.fixture(scope="module")
def bundle():
    env = rule_spec("sanma").make_env("half", reward_config=default_reward_config("sanma"))
    state = env.init(jax.random.key(751))
    obs = observe_for_network(env, state, TINY)
    model = get_network_cls("sanma", **TINY, head_config=HEAD)()
    variables = jax.jit(model.init)(jax.random.key(752), obs)
    central = build_central_q_input(state, obs)
    q_model = get_central_q_network_cls(
        "sanma", network_config=TINY, head_config=HEAD, critic_config=DEFAULT_CENTRAL_Q_CRITIC_CONFIG
    )()
    q_variables = jax.jit(q_model.init)(jax.random.key(753), obs, central)
    return env, state, obs, model, variables, central, q_model, q_variables


def _assert_same(a, b):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(x, y)


def test_v5_raw_logits_reading_shapes_batch_jit_actor_only_and_roots(bundle):
    _, _, obs, model, variables, _, _, _ = bundle
    assert set(variables["params"]) == {"policy_extractor", "policy_mlp", "critic_extractor", "value_critic_mlp"}
    policy = variables["params"]["policy_mlp"]
    assert "auxiliary" in policy and "reading" in policy
    assert "reading" not in variables["params"]["value_critic_mlp"]
    assert set(policy["reading"]) == {"tenpai", "shanten", "danger", "waits", "danger_points"}
    assert policy["auxiliary"]["hidden"]["kernel"].shape == (8, 5)
    actor = {"params": {k: variables["params"][k] for k in ("policy_extractor", "policy_mlp")}}
    for batch in (1, 2):
        observations = obs if batch == 1 else jax.tree.map(lambda x: jnp.stack([x, x]), obs)
        logits, reading = jax.jit(lambda p, o: model.apply(p, o, method=model.get_action_and_reading))(
            actor, observations
        )
        assert logits.shape == (batch, 87)
        assert set(reading) == set(READING_SHAPES)
        for name, shape in READING_SHAPES.items():
            assert reading[name].shape == (batch,) + shape
            assert np.isfinite(reading[name]).all()
        np.testing.assert_allclose(logits, model.apply(actor, observations, method=model.get_action_logits), atol=2e-6)
    logits, values = model.apply(variables, obs)
    assert logits.shape == (1, 87) and values.shape == (1,)
    # Reserved/illegal action slots remain finite raw logits (not legality-masked).
    assert np.isfinite(logits).all()


def test_reading_uses_contextual_opponents_and_tiles_in_one_trunk_pass(bundle):
    _, _, obs, model, variables, _, _, _ = bundle
    (logits, reading), captured = model.apply(
        variables,
        obs,
        method=model.get_action_and_reading,
        capture_intermediates=lambda module, method: isinstance(module, V4Encoder),
        mutable=["intermediates"],
    )
    features = captured["intermediates"]["policy_extractor"]["__call__"]
    assert len(features) == 1
    features = features[0]
    assert features["opponents"].shape == (1, 2, 8)
    assert features["tiles"].shape == (1, 34, 8)
    assert "critic_extractor" not in captured["intermediates"]
    p = variables["params"]["policy_mlp"]["reading"]
    seats, tiles = features["opponents"], features["tiles"]
    np.testing.assert_allclose(
        reading["tenpai_logits"], (seats @ p["tenpai"]["kernel"] + p["tenpai"]["bias"])[..., 0], atol=1e-6
    )
    np.testing.assert_allclose(
        reading["shanten_logits"], seats @ p["shanten"]["kernel"] + p["shanten"]["bias"], atol=1e-6
    )
    for name, output in (("danger", "danger_logits"), ("waits", "waits_logits"), ("danger_points", "danger_points")):
        head = p[name]
        assert head["seat"]["kernel"].shape == (8, 64)
        assert head["tile"]["kernel"].shape == (8, 64)
        seat = (seats @ head["seat"]["kernel"] + head["seat"]["bias"]).astype(jnp.float32)
        tile = (tiles @ head["tile"]["kernel"] + head["tile"]["bias"]).astype(jnp.float32)
        expected = jnp.einsum("bod,btd->bot", seat, tile) / jnp.sqrt(jnp.float32(64)) + head["bias"]
        np.testing.assert_allclose(reading[output], expected, atol=1e-6)
        assert reading[output].dtype == jnp.float32
    assert logits.shape == (1, 87)


def test_five_heads_share_weights_across_opponents_and_preserve_tile_order():
    features = {
        "opponents": jax.random.normal(jax.random.key(754), (2, 2, 8)),
        "tiles": jax.random.normal(jax.random.key(755), (2, 34, 8)),
    }
    head = V5ReadingHead()
    variables = head.init(jax.random.key(756), features)
    original = head.apply(variables, features)
    swapped = head.apply(variables, dict(features, opponents=features["opponents"][:, ::-1]))
    for name in original:
        np.testing.assert_allclose(swapped[name], original[name][:, ::-1], atol=1e-6)
    tile_permutation = jnp.arange(33, -1, -1)
    permuted = head.apply(variables, dict(features, tiles=features["tiles"][:, tile_permutation]))
    for name in ("danger_logits", "waits_logits", "danger_points"):
        np.testing.assert_allclose(permuted[name], original[name][:, :, tile_permutation], atol=1e-6)
    for name in ("tenpai_logits", "shanten_logits"):
        _assert_same(permuted[name], original[name])


def test_rollout_skips_reading_and_own_aux_and_reading_cannot_change_policy(bundle):
    _, _, obs, model, variables, _, _, _ = bundle
    stripped = copy.deepcopy(variables)
    del stripped["params"]["policy_mlp"]["reading"]
    del stripped["params"]["policy_mlp"]["auxiliary"]
    _assert_same(model.apply(stripped, obs), model.apply(variables, obs))
    with pytest.raises(ScopeParamNotFoundError, match="reading"):
        model.apply(stripped, obs, method=model.get_action_and_reading)
    changed = copy.deepcopy(variables)
    changed["params"]["policy_mlp"]["reading"] = jax.tree.map(
        lambda x: jnp.full_like(x, 2), changed["params"]["policy_mlp"]["reading"]
    )
    _assert_same(model.apply(changed, obs), model.apply(variables, obs))
    _assert_same(
        model.apply(changed, obs, method=model.get_action_and_aux),
        model.apply(variables, obs, method=model.get_action_and_aux),
    )


def test_reading_gradient_reaches_actor_trunk_only_and_all_five_heads(bundle):
    _, _, obs, model, variables, _, _, _ = bundle

    def loss(p):
        _, reading = model.apply(p, obs, method=model.get_action_and_reading)
        return sum(jnp.square(x).sum() for x in reading.values())

    gradients = jax.jit(jax.grad(loss))(variables)["params"]
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(gradients))
    assert any(np.any(np.asarray(x) != 0) for x in jax.tree.leaves(gradients["policy_extractor"]))
    for head in ("tenpai", "shanten", "danger", "waits", "danger_points"):
        assert any(np.any(np.asarray(x) != 0) for x in jax.tree.leaves(gradients["policy_mlp"]["reading"][head]))
    for name, subtree in gradients["policy_mlp"].items():
        if name != "reading":
            assert all(np.all(np.asarray(x) == 0) for x in jax.tree.leaves(subtree))
    for root in ("critic_extractor", "value_critic_mlp"):
        assert all(np.all(np.asarray(x) == 0) for x in jax.tree.leaves(gradients[root]))


@pytest.mark.parametrize("key", ["privileged", "others_hand", "deck", "rng", "return", "outcome"])
def test_scalar_and_q_actor_privacy_allowlist(bundle, key):
    _, _, obs, model, variables, _, q_model, q_variables = bundle
    injected = dict(obs, **{key: jnp.zeros(())})
    for network, params in ((model, variables), (q_model, q_variables)):
        for method in (network.get_action_logits, network.get_action_and_aux, network.get_action_and_reading):
            with pytest.raises(ValueError, match="forbidden actor"):
                network.apply(params, injected, method=method)
    for method in (model.get_value, model.get_value_and_aux, model.get_training_outputs):
        with pytest.raises(ValueError, match="forbidden actor"):
            model.apply(variables, injected, method=method)


def test_central_q_exact_v5_actor_clone_and_no_v4_transfer(bundle):
    _, _, obs, model, variables, central, q_model, q_variables = bundle
    cloned = clone_actor_and_public_encoder(variables, q_variables)
    _assert_same(
        model.apply(variables, obs, method=model.get_action_and_reading),
        q_model.apply(cloned, obs, method=q_model.get_action_and_reading),
    )
    _assert_same(
        model.apply(variables, obs, method=model.get_action_and_aux),
        q_model.apply(cloned, obs, method=q_model.get_action_and_aux),
    )
    _assert_same(cloned["params"]["critic_extractor"]["public_encoder"], variables["params"]["policy_extractor"])
    logits, q = jax.jit(q_model.apply)(cloned, obs, central)
    assert logits.shape == (1, 87) and q.shape == (1, 3, 87)
    np.testing.assert_array_equal(q, 0)
    for root in ("private_encoder", "fusion"):
        _assert_same(cloned["params"]["critic_extractor"][root], q_variables["params"]["critic_extractor"][root])
    legacy = copy.deepcopy(variables)
    del legacy["params"]["policy_mlp"]["reading"]
    with pytest.raises(ValueError, match="tree mismatch"):
        clone_actor_and_public_encoder(legacy, q_variables)


def test_v5_explicit_presets_do_not_change_v4_defaults():
    assert DEFAULT_NETWORK_CONFIG["architecture"] == "v5"
    for preset, dimensions in (
        (SMALL_NETWORK_CONFIG, (4, 128, 4, 352)),
        (MEDIUM_NETWORK_CONFIG, (8, 192, 6, 512)),
        (LARGE_NETWORK_CONFIG, (12, 256, 8, 704)),
    ):
        model = get_network_cls("sanma", **preset)()
        assert (model.num_layers, model.d_model, model.num_heads, model.mlp_dim) == dimensions
        assert model.head_config == DEFAULT_HEAD_CONFIG
        assert model_reading_config(model) == DEFAULT_READING_CONFIG
    for model in (V5ACNet(), make_v5_acnet()(), get_network_cls("sanma", **DEFAULT_NETWORK_CONFIG)()):
        assert (model.num_layers, model.d_model, model.num_heads, model.mlp_dim) == (12, 256, 8, 704)
    assert model_reading_config(get_network_cls("sanma")()) == DEFAULT_READING_CONFIG
    with pytest.raises(ValueError, match="even"):
        validate_network_config(dict(TINY, d_model=6), "sanma")


def test_v5_reading_provenance_is_exact_required_and_root_only():
    checkpoint = {"network_config": TINY, "head_config": HEAD, "reading_config": DEFAULT_READING_CONFIG}
    assert checkpoint_reading_config(checkpoint) == DEFAULT_READING_CONFIG
    assert model_reading_config(_checkpoint_actor(checkpoint)) == DEFAULT_READING_CONFIG
    q_checkpoint = dict(checkpoint, critic_config=DEFAULT_CENTRAL_Q_CRITIC_CONFIG)
    q_model = get_central_q_network_cls(
        "sanma",
        network_config=checkpoint_network_config(q_checkpoint),
        head_config=checkpoint_head_config(q_checkpoint),
        critic_config=q_checkpoint["critic_config"],
    )()
    assert q_model.architecture == "v5" and q_model.critic_config == DEFAULT_CENTRAL_Q_CRITIC_CONFIG
    for bad in (
        {"network_config": TINY, "head_config": HEAD},
        {"network_config": TINY, "head_config": HEAD, "metadata": {"reading_config": DEFAULT_READING_CONFIG}},
        dict(checkpoint, reading_config=None),
    ):
        with pytest.raises(ValueError, match="reading_config"):
            _checkpoint_actor(bad)
    for bad in (
        {},
        {"version": 0},
        {"version": 1},
        dict(DEFAULT_READING_CONFIG, version=True),
        dict(DEFAULT_READING_CONFIG, latent_dim=32),
        dict(DEFAULT_READING_CONFIG, points_scale=8000.0),
        dict(DEFAULT_READING_CONFIG, target_schema=2),
        dict(DEFAULT_READING_CONFIG, extra=0),
    ):
        with pytest.raises(ValueError, match="reading_config"):
            validate_reading_config(bad, "v5")
    with pytest.raises(ValueError, match="reading_config"):
        checkpoint_reading_config({"network_config": TINY})


def test_v5_reuses_public_feature_v4_and_explicit_adapters(bundle):
    env, state, obs, _, _, _, _, _ = bundle
    _assert_same(prepare_network_observation(obs, TINY), obs)
    _assert_same(with_network_observation(env, TINY).observe(state), obs)
    np.testing.assert_array_equal(obs["feature_version"], 4)
    v4 = dict(TINY, architecture="v5")
    _assert_same(adapt_network_observation(obs, TINY, v4), obs)
    _assert_same(adapt_network_observation(obs, v4, TINY), obs)
    v3 = dict(TINY, architecture="v3")
    bridged = adapt_network_observation(obs, TINY, v3)
    _assert_same(bridged, obs)
    _assert_same(adapt_network_observation(bridged, v3, TINY), obs)
    with pytest.raises(ValueError, match="feature_version=4"):
        prepare_network_observation(dict(obs, feature_version=jnp.int8(5)), TINY)
