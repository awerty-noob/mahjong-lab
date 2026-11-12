"""CPU synthetic no-Adam variant tests; no model files, environment or training."""
import json
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.core import freeze

from train.configs import tiny_debug_preset
from train.optimizer import (
    VRPOMuonState,
    make_vrpo_muon,
    scale_by_katago_muon,
    vrpo_optimizer_metadata,
    vrpo_parameter_labels,
)


def parameters():
    return freeze({"params": {
        "policy_extractor": {
            "attention": {"qkv": {"kernel": jnp.ones((8, 24))},
                          "attention_out": {"kernel": jnp.ones((8, 8))}},
            "tokens": {"embedding": jnp.ones((35, 4))},
            "norm": {"scale": jnp.ones(8)}, "scalar": jnp.float32(2.)},
        "policy_mlp": {
            "discard": {"kernel": jnp.ones((8, 2)), "bias": jnp.ones(2)},
            "self_kan": {"kernel": jnp.ones((8, 1))},
            "special": {"kernel": jnp.ones((8, 1))},
            "auxiliary": {"kernel": jnp.ones((8, 35)), "bias": jnp.ones(35)}},
        "value_critic_mlp": {
            "discard": {"kernel": jnp.ones((8, 6))},
            "self_kan": {"kernel": jnp.ones((8, 3))},
            "special": {"kernel": jnp.ones((8, 3))}},
        "critic_extractor": {"deck": {"embedding": jnp.ones((2, 4))}},
    }})


def test_all_heads_embeddings_and_vectors_have_exhaustive_explicit_labels():
    labels = vrpo_parameter_labels(parameters())["params"]
    for head in ("discard", "self_kan", "special"):
        assert labels["policy_mlp"][head]["kernel"] == "muon_decay"
        assert labels["value_critic_mlp"][head]["kernel"] == "muon_decay"
    assert labels["policy_extractor"]["attention"]["qkv"]["kernel"] == "muon_decay"
    assert labels["policy_extractor"]["tokens"]["embedding"] == "muon_no_decay"
    assert labels["critic_extractor"]["deck"]["embedding"] == "muon_no_decay"
    assert labels["policy_extractor"]["norm"]["scale"] == "nesterov_no_decay"
    assert labels["policy_extractor"]["scalar"] == "nesterov_no_decay"
    assert set(jax.tree.leaves(labels["policy_mlp"]["auxiliary"])) == {"frozen"}
    with pytest.raises(ValueError, match="unsupported Muon kernel"):
        vrpo_parameter_labels({"params": {"unknown": {"kernel": jnp.ones((2, 3, 4))}}})


@pytest.mark.parametrize("shape,name", [((8, 1), "kernel"), ((8, 2), "kernel"),
    ((8, 3), "kernel"), ((35, 4), "embedding"), ((2, 4), "embedding"),
    ((8, 24), "kernel")])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_matrix_steps_reuse_exact_ema_nesterov_ns_and_direct_lr(shape, name, dtype):
    cfg = tiny_debug_preset(actor_lr=.004, weight_decay=0., actor_max_grad_norm=1e6,
                             muon_ns_dtype=dtype)
    p = {"params": {"policy_extractor": {"probe": {name: jnp.ones(shape)}}}}
    tx = make_vrpo_muon(p, cfg, "actor")
    state = tx.init(p)
    # The legacy pure projection is only a numerical oracle; no Adam path used.
    projection = scale_by_katago_muon(momentum=cfg.muon_momentum,
        ns_steps=cfg.muon_ns_steps, ns_dtype=dtype)
    expected_state = projection.init(p)
    for step in range(2):
        grads = jax.tree.map(lambda x: jnp.sin(jnp.arange(x.size).reshape(x.shape) + step), p)
        expected, expected_state = projection.update(grads, expected_state)
        actual, state = tx.update(grads, state, p)
        for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            np.testing.assert_array_equal(a, -cfg.actor_lr * b)
    assert isinstance(state, VRPOMuonState) and int(state.count) == 2
    assert state._fields == ("count", "momentum")  # No second moments or Adam slots.


def test_vectors_and_scalar_exact_uncorrected_nesterov_sgd_without_decay():
    cfg = tiny_debug_preset(actor_lr=.004, weight_decay=.9, actor_max_grad_norm=1e6)
    p = {"params": {"policy_extractor": {"norm": {"scale": jnp.ones(3)},
                                         "scalar": jnp.float32(2.)}}}
    grads = jax.tree.map(lambda x: jnp.full_like(x, .2), p)
    tx = make_vrpo_muon(p, cfg, "actor")
    slots = tx.init(p)
    momentum = 0.
    for _ in range(3):
        updates, slots = tx.update(grads, slots, p)
        momentum = .95 * momentum + .05 * .2
        direction = .05 * .2 + .95 * momentum
        for value in jax.tree.leaves(updates):
            np.testing.assert_allclose(value, -.004 * direction, atol=1e-9, rtol=1e-6)


def test_weight_decay_only_projection_kernels_and_auxiliary_excluded_before_clipping():
    cfg = tiny_debug_preset(actor_lr=.004, weight_decay=.1, actor_max_grad_norm=.5)
    p = parameters()
    grads = jax.tree.map(lambda x: jnp.zeros_like(x), p)
    grads = grads.copy({"params": grads["params"].copy({"policy_mlp": grads["params"]["policy_mlp"].copy({
        "auxiliary": jax.tree.map(lambda x: jnp.full_like(x, 1e12), p["params"]["policy_mlp"]["auxiliary"])})})})
    tx = make_vrpo_muon(p, cfg, "actor")
    updates, slots = tx.update(grads, tx.init(p), p)
    labels = vrpo_parameter_labels(p)
    for update, parameter, label in zip(jax.tree.leaves(updates), jax.tree.leaves(p),
                                       jax.tree.leaves(labels), strict=True):
        expected = -.004 * .1 * parameter if label == "muon_decay" else jnp.zeros_like(parameter)
        np.testing.assert_allclose(update, expected, atol=1e-9)
    for leaf in jax.tree.leaves(slots.momentum):
        np.testing.assert_array_equal(leaf, 0.)
    # Huge frozen gradients must not consume the active vector's clipping budget.
    active = grads.copy({"params": grads["params"].copy({"policy_extractor":
        grads["params"]["policy_extractor"].copy({"norm": {"scale": jnp.full((8,), .2)}})})})
    clean = jax.tree.map(lambda g, label: jnp.zeros_like(g) if label == "frozen" else g, active, labels)
    dirty_updates, dirty_slots = tx.update(active, tx.init(p), p)
    clean_updates, clean_slots = tx.update(clean, tx.init(p), p)
    for a, b in zip(jax.tree.leaves((dirty_updates, dirty_slots)),
                    jax.tree.leaves((clean_updates, clean_slots)), strict=True):
        np.testing.assert_array_equal(a, b)
    # Frozen slots remain byte-identical even with deliberately nonzero buffers.
    warm = slots._replace(momentum=jax.tree.map(lambda x: x + .3, slots.momentum))
    _, next_slots = tx.update(grads, warm, p)
    np.testing.assert_array_equal(next_slots.momentum["params"]["policy_mlp"]["auxiliary"]["kernel"],
                                  warm.momentum["params"]["policy_mlp"]["auxiliary"]["kernel"])


def test_phase_clipping_and_rates_are_independent_and_metadata_is_json_only():
    cfg = tiny_debug_preset(actor_lr=.004, critic_lr=.007,
                            actor_max_grad_norm=.5, critic_max_grad_norm=.25)
    p = {"params": {"policy_extractor": {"bias": jnp.ones(2)}}}
    g = jax.tree.map(jnp.ones_like, p)
    actor, critic = (make_vrpo_muon(p, cfg, phase) for phase in ("actor", "critic"))
    a, sa = actor.update(g, actor.init(p), p)
    b, sb = critic.update(g, critic.init(p), p)
    np.testing.assert_allclose(jax.tree.leaves(b)[0] / jax.tree.leaves(a)[0], .007 / .004 * .25 / .5)
    np.testing.assert_allclose(jax.tree.leaves(sb.momentum)[0] / jax.tree.leaves(sa.momentum)[0], .5)
    metadata = json.loads(json.dumps(vrpo_optimizer_metadata(cfg)))
    assert metadata["adam"] is False and metadata["learning_rate_semantics"] == "direct_no_multiplier"
    assert metadata["actor_lr"] == .004 and metadata["critic_lr"] == .007
    assert "momentum_formula" in metadata and "optimizer_slots" in metadata
    with pytest.raises(ValueError, match="no-Adam"):
        make_vrpo_muon(p, SimpleNamespace(optimizer="adamw"), "actor")
    with pytest.raises(ValueError, match="unsupported optimizer"):
        replace(cfg, optimizer="adamw")
