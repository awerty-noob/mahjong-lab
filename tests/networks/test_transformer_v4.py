"""V4 forward/backward and trainer ABI checks; no optimisation or training."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from networks.mahjong_3p.configs import (
    DEFAULT_HEAD_CONFIG,
    MEDIUM_NETWORK_CONFIG,
    SMALL_NETWORK_CONFIG,
)
from networks.mahjong_3p.network import build_actor, get_network_cls, observe_for_network
from networks.mahjong_3p.transformer import (
    COMPACT_TO_CANONICAL,
    temporal_rope,
)
from networks.mahjong_3p.transformer import (
    ActorCriticNet as V4ACNet,
)
from networks.mahjong_3p.transformer import (
    PolicyHead as V4PolicyHead,
)
from networks.mahjong_3p.transformer import (
    TransformerBlock as V4Block,
)
from networks.mahjong_3p.transformer import (
    TransformerEncoder as V4Encoder,
)
from networks.mahjong_3p.transformer import (
    make_actor_critic as make_v4_acnet,
)
from train.configs import default_reward_config, rule_spec
from train.optimizer import param_group_labels

TINY = dict(architecture="v5", num_layers=1, d_model=8, num_heads=2, mlp_dim=16)


@pytest.fixture(scope="module")
def bundle():
    env = rule_spec("sanma").make_env("half", reward_config=default_reward_config("sanma"))
    state = env.init(jax.random.PRNGKey(410))
    obs = observe_for_network(env, state, TINY)
    model = get_network_cls("sanma", **TINY)()
    params = jax.jit(model.init)(jax.random.PRNGKey(411), obs)
    return env, state, obs, model, params


def test_forward_jit_batch_and_actor_only(bundle):
    _, _, obs, model, params = bundle
    logits, value = jax.jit(model.apply)(params, obs)
    assert logits.shape == (1, 87) and value.shape == (1,)
    assert np.isfinite(logits).all() and np.isfinite(value).all()
    batched = jax.tree.map(lambda x: jnp.stack([x, x]), obs)
    other_logits, other_values = jax.jit(model.apply)(params, batched)
    np.testing.assert_allclose(other_logits, jnp.repeat(logits, 2, axis=0), atol=2e-6)
    np.testing.assert_allclose(other_values, jnp.repeat(value, 2, axis=0), atol=2e-6)
    actor = {"params": {key: params["params"][key] for key in ("policy_extractor", "policy_mlp")}}
    np.testing.assert_allclose(model.apply(actor, obs, method=model.get_action_logits), logits, atol=2e-6)


def test_parameter_groups_and_independent_gradients(bundle):
    _, _, obs, model, params = bundle
    labels = param_group_labels(params)["params"]
    assert set(labels) == {"policy_extractor", "policy_mlp", "critic_extractor", "value_critic_mlp"}
    grads = jax.jit(jax.grad(lambda p: model.apply(p, obs, method=model.get_value).sum()))(params)
    for name in ("policy_extractor", "policy_mlp"):
        assert all(np.all(np.asarray(x) == 0) for x in jax.tree.leaves(grads["params"][name]))
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(grads))
    assert any(np.any(np.asarray(x) != 0) for x in jax.tree.leaves(grads["params"]["critic_extractor"]))


def test_rope_identity_and_relative_translation():
    q = jax.random.normal(jax.random.key(412), (2, 5, 2, 4))
    k = jax.random.normal(jax.random.key(413), q.shape)
    positions = jnp.broadcast_to(jnp.arange(-5, 0), (2, 5))
    np.testing.assert_allclose(temporal_rope(q, jnp.zeros_like(positions)), q)

    def score(shift):
        return jnp.einsum("bthd,bshd->bhts", temporal_rope(q, positions + shift), temporal_rope(k, positions + shift))

    np.testing.assert_allclose(score(0), score(13), atol=2e-6)
    with pytest.raises(ValueError, match="even"):
        temporal_rope(jnp.zeros((1, 1, 1, 3)), jnp.zeros((1, 1)))


def test_padding_does_not_change_valid_attention_outputs():
    block = V4Block(d_model=8, num_heads=2, mlp_dim=16)
    x = jax.random.normal(jax.random.key(414), (1, 5, 8))
    mask = jnp.array([[True, True, False, False, True]])
    positions = jnp.array([[0, -1, 0, 0, 0]])
    params = block.init(jax.random.key(415), x, mask, positions)
    expected = block.apply(params, x, mask, positions)
    changed = jnp.where(mask[..., None], x, 1e4)
    actual = block.apply(params, changed, mask, positions)
    np.testing.assert_allclose(actual, expected, atol=2e-6)
    np.testing.assert_array_equal(actual[:, 2:4], 0)


def test_canonical_and_red_action_gather(bundle):
    _, _, obs, _, _ = bundle
    features = {"summary": jnp.ones((1, 8)), "tiles": jax.random.normal(jax.random.key(416), (1, 34, 8))}
    head = V4PolicyHead(8)
    params = head.init(jax.random.key(417), features, obs)
    logits, state = head.apply(params, features, obs, capture_intermediates=True, mutable=["intermediates"])
    local = state["intermediates"]["local"]["__call__"][0]
    dense = params["params"]["discard"]
    discard = local @ dense["kernel"] + dense["bias"]
    normal = discard[:, jnp.array(COMPACT_TO_CANONICAL), 0]
    red = discard[:, jnp.array([4, 13, 22]), 1]
    np.testing.assert_allclose(logits[:, :30], jnp.concatenate([normal, red], axis=-1), atol=1e-6)
    dense = params["params"]["self_kan"]
    kan = (local @ dense["kernel"] + dense["bias"])[..., 0]
    np.testing.assert_allclose(logits[:, 30:57], kan[:, jnp.array(COMPACT_TO_CANONICAL)], atol=1e-6)


def test_v4_factory_and_direct_defaults_are_m_with_heads():
    for model in (V4ACNet(), make_v4_acnet()(), get_network_cls("sanma")()):
        assert (model.num_layers, model.d_model, model.num_heads, model.mlp_dim) == (12, 256, 8, 704)
        assert model.head_config == DEFAULT_HEAD_CONFIG
    assert make_v4_acnet(head_config={"version": 0})().head_config == {"version": 0}


@pytest.mark.parametrize(
    "config,width,layers,count",
    [
        (None, 256, 12, 22_254_485),
        (MEDIUM_NETWORK_CONFIG, 192, 8, 8_823_893),
    ],
)
def test_default_l_and_explicit_m_abstract_shapes_without_full_compute(bundle, config, width, layers, count):
    _, _, obs, _, _ = bundle
    model = get_network_cls("sanma", **(config or {}))()
    shapes = jax.eval_shape(model.init, jax.random.key(421), obs)
    assert sum(int(np.prod(leaf.shape)) for leaf in jax.tree.leaves(shapes)) == count
    for root in ("policy_mlp", "value_critic_mlp"):
        assert shapes["params"][root]["auxiliary"]["hidden"]["kernel"].shape == (width, 128)
    for root in ("policy_extractor", "critic_extractor"):
        assert sum(name.startswith("block_") for name in shapes["params"][root]) == layers
    outputs = jax.eval_shape(lambda p, o: model.apply(p, o, method=model.get_training_outputs), shapes, obs)
    assert outputs[0].shape == (1, 87) and outputs[1].shape == (1,)
    assert outputs[2]["rank_logits"].shape == (1, 3) and outputs[2]["score_delta"].shape == (1,)


def test_auxiliary_heads_initialized_by_public_call_and_skipped_at_inference(bundle):
    from flax.errors import ScopeParamNotFoundError

    _, _, obs, model, variables = bundle
    assert set(variables["params"]) == {"policy_extractor", "policy_mlp", "critic_extractor", "value_critic_mlp"}
    for root in ("policy_mlp", "value_critic_mlp"):
        aux = variables["params"][root]["auxiliary"]
        assert set(aux) == {"hidden", "rank", "score"}
        assert aux["hidden"]["kernel"].shape == (8, 128)
        assert aux["rank"]["kernel"].shape == (128, 3)
        assert aux["score"]["kernel"].shape == (128, 1)
    # Missing auxiliary leaves cannot be silently created by an inference apply.
    inference_params = {
        "params": {
            root: {k: v for k, v in leaves.items() if k != "auxiliary"} for root, leaves in variables["params"].items()
        }
    }
    actual = model.apply(inference_params, obs)
    expected = model.apply(variables, obs)
    for x, y in zip(actual, expected):
        np.testing.assert_array_equal(x, y)
    with pytest.raises(ScopeParamNotFoundError, match="auxiliary"):
        model.apply(inference_params, obs, method=model.get_value_and_aux)


def test_auxiliary_batch_shapes_actor_only_and_single_critic_forward(bundle):

    _, _, obs, model, variables = bundle
    observations = jax.tree.map(lambda x: jnp.stack([x, x]), obs)
    logits, values = model.apply(variables, observations)
    actor = {"params": {k: variables["params"][k] for k in ("policy_extractor", "policy_mlp")}}
    actor_logits, actor_aux = jax.jit(lambda p, o: model.apply(p, o, method=model.get_action_and_aux))(
        actor, observations
    )
    critic_values, critic_aux = jax.jit(lambda p, o: model.apply(p, o, method=model.get_value_and_aux))(
        variables, observations
    )
    np.testing.assert_allclose(actor_logits, logits, atol=2e-6)
    np.testing.assert_allclose(critic_values, values, atol=2e-6)
    for aux in (actor_aux, critic_aux):
        assert set(aux) == {"rank_logits", "score_delta"}
        assert aux["rank_logits"].shape == (2, 3)
        assert aux["score_delta"].shape == (2,)
        assert all(np.isfinite(x).all() for x in jax.tree.leaves(aux))
    outputs, captured = model.apply(
        variables,
        observations,
        method=model.get_training_outputs,
        capture_intermediates=lambda module, method: isinstance(module, V4Encoder),
        mutable=["intermediates"],
    )
    np.testing.assert_allclose(outputs[0], logits, atol=2e-6)
    np.testing.assert_allclose(outputs[1], values, atol=2e-6)
    for key in critic_aux:
        np.testing.assert_allclose(outputs[2][key], critic_aux[key], atol=2e-6)
    assert len(captured["intermediates"]["critic_extractor"]["__call__"]) == 1
    assert len(captured["intermediates"]["policy_extractor"]["__call__"]) == 1


@pytest.mark.parametrize("side", ["actor", "critic"])
def test_auxiliary_gradients_stay_on_their_own_side(bundle, side):
    _, _, obs, model, variables = bundle
    method = model.get_action_and_aux if side == "actor" else model.get_value_and_aux

    def loss(p):
        _, aux = model.apply(p, obs, method=method)
        return jnp.square(aux["rank_logits"]).sum() + jnp.square(aux["score_delta"]).sum()

    grads = jax.jit(jax.grad(loss))(variables)["params"]
    active = ("policy_extractor", "policy_mlp") if side == "actor" else ("critic_extractor", "value_critic_mlp")
    inactive = set(grads) - set(active)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(grads))
    for root in inactive:
        assert all(np.all(np.asarray(x) == 0) for x in jax.tree.leaves(grads[root]))
    for root in active:
        assert any(np.any(np.asarray(x) != 0) for x in jax.tree.leaves(grads[root]))
    for branch in ("hidden", "rank", "score"):
        assert any(np.any(np.asarray(x) != 0) for x in jax.tree.leaves(grads[active[1]]["auxiliary"][branch]))


def test_legacy_disabled_heads_use_exact_old_tree_and_empty_aux(bundle):
    _, _, obs, _, variables = bundle
    legacy = build_actor(TINY, head_config={"version": 0})
    assert legacy.head_config == {"version": 0}
    initialized = jax.jit(legacy.init)(jax.random.key(419), obs)
    assert all("auxiliary" not in initialized["params"][root] for root in ("policy_mlp", "value_critic_mlp"))
    # Existing pre-head trees are accepted without implicit migration.
    old_tree = {
        "params": {
            root: {k: v for k, v in leaves.items() if k != "auxiliary"} for root, leaves in variables["params"].items()
        }
    }
    for method in (legacy.get_action_and_aux, legacy.get_value_and_aux):
        _, aux = legacy.apply(old_tree, obs, method=method)
        assert aux == {}
    logits, values, aux = legacy.apply(old_tree, obs, method=legacy.get_training_outputs)
    assert logits.shape == (1, 87) and values.shape == (1,) and aux == {}
    assert jax.tree.structure(initialized) == jax.tree.structure(old_tree)


def test_direct_model_init_eager_includes_custom_aux_width(bundle):
    _, _, obs, _, _ = bundle
    model = V4ACNet(num_layers=1, d_model=8, num_heads=2, mlp_dim=16, head_config={"version": 1, "hidden_dim": 5})
    initialized = model.init(jax.random.key(420), obs)
    for root in ("policy_mlp", "value_critic_mlp"):
        assert initialized["params"][root]["auxiliary"]["hidden"]["kernel"].shape == (8, 5)


def test_v4_s_explicit_legacy_preset_forward(bundle):
    _, _, obs, _, _ = bundle
    model = get_network_cls("sanma", **SMALL_NETWORK_CONFIG)()
    assert (model.num_layers, model.d_model, model.num_heads, model.mlp_dim) == (4, 128, 4, 352)
    params = jax.jit(model.init)(jax.random.key(418), obs)
    logits, value = jax.jit(model.apply)(params, obs)
    assert logits.shape == (1, 87) and value.shape == (1,)
    assert np.isfinite(logits).all() and np.isfinite(value).all()
