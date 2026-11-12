"""Synthetic CPU boundary roundtrips only: no files, real weights or training."""

import copy
import pickle
from dataclasses import FrozenInstanceError
from typing import NamedTuple

import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from networks.mahjong_3p.configs import (
    DEFAULT_CENTRAL_Q_CRITIC_CONFIG,
    DEFAULT_READING_CONFIG,
    RULESET,
)
from train.checkpoint import (
    VRPO_MODEL_KIND,
    VRPO_RESUME_KIND,
    VRPO_RESUME_SCHEMA_VERSION,
    VRPO_V5_RESUME_SCHEMA_VERSION,
    VRPOResumeResult,
    build_vrpo_model,
    build_vrpo_resume,
    restore_vrpo_resume,
)
from train.configs import VRPOTrainingState, default_reward_config


class Moments(NamedTuple):
    count: object
    momentum: object
    variance: object


def tree_equal(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right)):
        np.testing.assert_array_equal(a, b)


def _algorithm_config():
    from train.configs import READING_PROTOCOL

    return {
        "version": 5,
        "kind": "vrpo_actor_reading_v5",
        "reading_auxiliary": True,
        "reading_aux_coef": 0.01,
        "reading_protocol": {**READING_PROTOCOL, "tasks": list(READING_PROTOCOL["tasks"])},
        "rollout_games": 4,
    }


def make_contract():
    return {
        "network_config": {"architecture": "v5", "num_layers": 1, "d_model": 8,
                           "num_heads": 2, "mlp_dim": 16},
        "head_config": {"version": 0},
        "critic_config": dict(DEFAULT_CENTRAL_Q_CRITIC_CONFIG),
        "reading_config": dict(DEFAULT_READING_CONFIG),
        "algorithm_config": _algorithm_config(),
        "reward_config": default_reward_config("sanma"),
        "optimizer_config": {"actor": {"name": "synthetic_momentum", "lr": .01},
                             "critic": {"name": "synthetic_momentum", "lr": .02}},
        "environment_config": {"env_name": "sanma", "round_mode": "half", "ruleset": RULESET},
        "source_versions": {"implementation_sha256": "a" * 64, "jax": jax.__version__,
                            "flax": "test", "optax": "test"},
        "retention_enabled": True,
    }


@jax.jit
def toy_boundary(state, rng):
    rng, akey, ckey = jax.random.split(rng, 3)
    actor_grad = jax.random.normal(akey, (2, 3))
    critic_grad = jax.random.normal(ckey, (2, 3))
    a = Moments(state.actor_opt_state.count + 1,
                .9 * state.actor_opt_state.momentum + .1 * actor_grad,
                .99 * state.actor_opt_state.variance + .01 * actor_grad ** 2)
    c = Moments(state.critic_opt_state.count + 1,
                .9 * state.critic_opt_state.momentum + .1 * critic_grad,
                .99 * state.critic_opt_state.variance + .01 * critic_grad ** 2)
    p = state.params
    params = {"policy_extractor": p["policy_extractor"] - .01 * a.momentum,
              "policy_mlp": p["policy_mlp"] - .01 * a.momentum,
              "critic_extractor": p["critic_extractor"] - .02 * c.momentum,
              "value_critic_mlp": p["value_critic_mlp"] - .02 * c.momentum}
    return state.replace(params=params, actor_opt_state=a, critic_opt_state=c,
                         actor_steps=state.actor_steps + 1, critic_steps=state.critic_steps + 1,
                         completed_updates=state.completed_updates + 1), rng


@pytest.fixture
def case():
    params = {key: jnp.full((2, 3), float(i + 1)) for i, key in enumerate(
        ("policy_extractor", "policy_mlp", "critic_extractor", "value_critic_mlp"))}
    slots = Moments(jnp.int32(0), jnp.zeros((2, 3)), jnp.zeros((2, 3)))
    fresh = VRPOTrainingState(params=params, actor_opt_state=slots, critic_opt_state=slots)
    state, rng = toy_boundary(fresh, jax.random.PRNGKey(371))
    state, rng = toy_boundary(state, rng)
    contract = make_contract()
    kwargs = dict(contract=contract,
                  progress={"completed_updates": 2, "envsteps": 509, "schedule_iteration": 2},
                  collection_state={"version": 1, "kind": "current_joint_all_seats", "next_game_id": 8},
                  retention_state={"version": 1, "teacher_sha256": "b" * 64,
                                   "reservoir": np.ones((3, 2), np.float32)},
                  lineage={"source_kind": "bc", "source_sha256": "c" * 64})
    payload = build_vrpo_resume(state, rng, **kwargs)
    return {"fresh": fresh, "state": state, "rng": rng, "contract": contract,
            "kwargs": kwargs, "payload": payload}


def restore(case, payload=None, contract=None):
    return restore_vrpo_resume(case["payload"] if payload is None else payload, case["fresh"],
                               contract=case["contract"] if contract is None else contract)


def rejected(case, payload, match=None):
    state = case["fresh"]
    before = jax.device_get(state)
    identities = tuple(getattr(state, key) for key in state.__dataclass_fields__)
    with pytest.raises(ValueError, match=match):
        restore(case, payload)
    tree_equal(before, state)
    assert all(getattr(state, key) is old for key, old in zip(state.__dataclass_fields__, identities))


def test_exact_next_boundary_after_host_only_in_memory_roundtrip(case):
    # Bytes only, never a pickle/checkpoint file. No functions or static tx enter.
    payload = pickle.loads(pickle.dumps(case["payload"], protocol=pickle.HIGHEST_PROTOCOL))
    assert payload["checkpoint_kind"] == VRPO_RESUME_KIND
    assert payload["schema_version"] == VRPO_V5_RESUME_SCHEMA_VERSION == 5
    assert payload["phase"] == "complete"
    assert all(isinstance(x, np.ndarray) for x in jax.tree.leaves(payload["train_state"]))
    assert isinstance(payload["train_state"]["actor_opt_state"], Moments)
    assert isinstance(payload["train_state"]["critic_opt_state"], Moments)
    result = restore(case, payload)
    assert isinstance(result, VRPOResumeResult)
    tree_equal(result.state, case["state"])
    np.testing.assert_array_equal(result.rng, case["rng"])
    assert result.progress == case["kwargs"]["progress"]
    assert result.collection_state == case["kwargs"]["collection_state"]
    assert "pool_state" not in payload and "pool_payload" not in payload
    tree_equal(result.retention_state, case["kwargs"]["retention_state"])
    assert result.lineage == case["kwargs"]["lineage"]
    expected_state, expected_rng = toy_boundary(case["state"], case["rng"])
    actual_state, actual_rng = toy_boundary(result.state, result.rng)
    tree_equal(actual_state, expected_state)
    np.testing.assert_array_equal(actual_rng, expected_rng)
    with pytest.raises(FrozenInstanceError):
        result.rng = None


def test_export_restore_arrays_and_metadata_do_not_alias(case):
    result = restore(case)
    case["payload"]["progress"]["envsteps"] = 0
    case["payload"]["retention_state"]["reservoir"][:] = 19
    next(iter(case["payload"]["train_state"]["params"].values()))[:] = 101
    assert result.progress["envsteps"] == 509
    np.testing.assert_array_equal(result.retention_state["reservoir"], 1)
    np.testing.assert_array_equal(case["kwargs"]["retention_state"]["reservoir"], 1)
    tree_equal(result.state, case["state"])


def test_ordinary_model_envelope_is_explicit_not_resume(case):
    model = build_vrpo_model(case["state"].params, contract=case["contract"], metadata={"critic_trained": False})
    assert model["checkpoint_kind"] == VRPO_MODEL_KIND
    assert model["env_name"] == "sanma" and model["round_mode"] == "half" and model["ruleset"] == RULESET
    for key in ("network_config", "head_config", "critic_config", "algorithm_config", "reward_config"):
        assert model[key] == case["contract"][key]
    assert model["metadata"]["optimizer_state_included"] is False
    assert model["metadata"]["resume_semantics"] == "initialization_only"
    assert "rng" not in model and "train_state" not in model
    assert all(isinstance(x, np.ndarray) for x in jax.tree.leaves(model["params"]))
    with pytest.raises(ValueError):
        restore(case, model)
    for metadata in ({"optimizer_state_included": True}, {"optimizer_state_included": 0},
                     {"resume_semantics": "full_resume"}):
        with pytest.raises(ValueError, match="ordinary vrpo_model"):
            build_vrpo_model(case["state"].params, contract=case["contract"], metadata=metadata)


@pytest.mark.parametrize("key,value", [
    ("schema_version", 1), ("schema_version", 2), ("schema_version", True), ("schema_version", 3.),
    ("checkpoint_kind", "ppo_boundary_resume"), ("checkpoint_kind", "vrpo_model"),
    ("phase", "actor"), ("phase", "critic"), ("phase", "collecting"),
    ("boundary", {"kind": "update_boundary", "active_batch": True}),
    ("boundary", {"kind": "update_boundary", "active_batch": 0}),
])
def test_old_v1_partial_phase_and_nonempty_boundary_rejected(case, key, value):
    payload = copy.deepcopy(case["payload"])
    payload[key] = value
    rejected(case, payload)


@pytest.mark.parametrize("key", ["network_config", "head_config", "critic_config", "algorithm_config",
                                  "reward_config", "optimizer_config", "environment_config", "source_versions"])
def test_contract_mismatch_rejected_even_for_shape_compatible_parameters(case, key):
    payload = copy.deepcopy(case["payload"])
    if key == "network_config":
        payload["config_contract"][key]["num_heads"] = 1
    elif key == "head_config":
        payload["config_contract"][key] = {"version": 1, "hidden_dim": 8}
    elif key == "critic_config":
        payload["config_contract"][key]["state_input_schema"] = 2
    elif key == "reward_config":
        payload["config_contract"][key]["uma_points"] = [0, 0, 0]
    elif key == "environment_config":
        payload["config_contract"][key]["round_mode"] = "east"
    else:
        payload["config_contract"][key]["additional_change"] = "different"
    rejected(case, payload)


@pytest.mark.parametrize("key", sorted(make_contract()))
def test_missing_contract_field_rejected_even_if_both_sides_match(case, key):
    contract = copy.deepcopy(case["contract"])
    del contract[key]
    payload = copy.deepcopy(case["payload"])
    payload["config_contract"] = contract
    with pytest.raises(ValueError):
        restore(case, payload, contract)


@pytest.mark.parametrize("key,value", [
    ("head_config", None), ("head_config", {}), ("critic_config", None),
    ("critic_config", {"version": 0, "kind": "legacy_v"}), ("reward_config", None),
    ("algorithm_config", {}), ("optimizer_config", {}), ("source_versions", {}),
    ("retention_enabled", 1),
])
def test_invalid_explicit_contract_rejected_even_if_both_sides_match(case, key, value):
    contract = copy.deepcopy(case["contract"])
    contract[key] = value
    payload = copy.deepcopy(case["payload"])
    payload["config_contract"] = contract
    with pytest.raises(ValueError):
        restore(case, payload, contract)


@pytest.mark.parametrize("section,key", [("network_config", "num_heads"),
                                         ("head_config", "version"),
                                         ("critic_config", "state_input_schema"),
                                         ("reward_config", "normalization_points")])
def test_partial_inner_schema_rejected_without_fill_defaults(case, section, key):
    contract = copy.deepcopy(case["contract"])
    del contract[section][key]
    payload = copy.deepcopy(case["payload"])
    payload["config_contract"] = contract
    with pytest.raises(ValueError):
        restore(case, payload, contract)


@pytest.mark.parametrize("key", ["params", "actor_opt_state", "critic_opt_state"])
@pytest.mark.parametrize("corruption", ["nan", "shape", "dtype", "structure", "scalar"])
def test_entire_params_both_optimizer_trees_shape_dtype_finite_required(case, key, corruption):
    payload = copy.deepcopy(case["payload"])
    tree = payload["train_state"][key]
    if corruption == "structure":
        payload["train_state"][key] = [tree]
    else:
        leaves, structure = jax.tree.flatten(tree)
        index = next(i for i, x in enumerate(leaves) if x.dtype == np.float32)
        value = leaves[index]
        if corruption == "nan":
            leaves[index] = np.full_like(value, np.nan)
        elif corruption == "shape":
            leaves[index] = np.zeros((77,), value.dtype)
        elif corruption == "dtype":
            leaves[index] = value.astype(np.float64)
        else:
            leaves[index] = 0
        payload["train_state"][key] = jax.tree.unflatten(structure, leaves)
    rejected(case, payload)


@pytest.mark.parametrize("key", ["actor_steps", "critic_steps", "completed_updates"])
@pytest.mark.parametrize("value", [-1, True, 2., np.array([2], np.int32), np.array(2, np.uint32),
                                    np.array(-1, np.int32), np.array(2, np.int64)])
def test_state_counters_are_exact_nonnegative_signed_scalar_arrays(case, key, value):
    payload = copy.deepcopy(case["payload"])
    payload["train_state"][key] = value
    rejected(case, payload)


@pytest.mark.parametrize("key", ["completed_updates", "envsteps", "schedule_iteration"])
@pytest.mark.parametrize("value", [True, -1, 1.5, np.array(2, np.int32), 2**80])
def test_required_progress_counters_strict_host_ints(case, key, value):
    payload = copy.deepcopy(case["payload"])
    payload["progress"][key] = value
    rejected(case, payload)


def test_completed_updates_must_match_progress_at_build_and_restore(case):
    payload = copy.deepcopy(case["payload"])
    payload["progress"]["completed_updates"] = 3
    rejected(case, payload, "must match")
    with pytest.raises(ValueError, match="must match"):
        build_vrpo_resume(case["state"].replace(completed_updates=jnp.int32(0)), case["rng"], **case["kwargs"])


@pytest.mark.parametrize("schedule_iteration", [0, 1, 3])
def test_schedule_iteration_must_track_completed_updates_at_build_and_restore(case, schedule_iteration):
    payload = copy.deepcopy(case["payload"])
    payload["progress"]["schedule_iteration"] = schedule_iteration
    rejected(case, payload, "schedule_iteration must match")
    kwargs = copy.deepcopy(case["kwargs"])
    kwargs["progress"]["schedule_iteration"] = schedule_iteration
    with pytest.raises(ValueError, match="schedule_iteration must match"):
        build_vrpo_resume(case["state"], case["rng"], **kwargs)


def test_fresh_online_boundary_has_zero_completed_updates_and_schedule(case):
    state = case["state"].replace(completed_updates=jnp.int32(0))
    kwargs = copy.deepcopy(case["kwargs"])
    kwargs["progress"].update(completed_updates=0, schedule_iteration=0)
    kwargs["collection_state"]["next_game_id"] = 0
    payload = build_vrpo_resume(state, case["rng"], **kwargs)
    result = restore_vrpo_resume(payload, state, contract=kwargs["contract"])
    assert int(result.state.completed_updates) == result.progress["completed_updates"] == result.progress["schedule_iteration"] == 0


@pytest.mark.parametrize("value", [jax.random.key(2), [1, 2], np.zeros((1, 2), np.uint32),
                                    np.zeros(2, np.int32), np.zeros(3, np.uint32)])
def test_rng_requires_raw_uint32_two_vector(case, value):
    payload = copy.deepcopy(case["payload"])
    payload["rng"] = value
    rejected(case, payload)


def test_retention_disabled_lineage_optional_and_metadata_finite(case):
    kwargs = copy.deepcopy(case["kwargs"])
    kwargs["contract"]["retention_enabled"] = False
    kwargs["retention_state"] = None
    kwargs["lineage"] = None
    payload = build_vrpo_resume(case["state"], case["rng"], **kwargs)
    result = restore(case, payload, kwargs["contract"])
    assert result.retention_state is None and result.lineage == {}
    payload["retention_state"] = {"version": 1}
    with pytest.raises(ValueError, match="disabled retention"):
        restore(case, payload, kwargs["contract"])
    payload = copy.deepcopy(case["payload"])
    payload["retention_state"]["reservoir"][0, 0] = np.inf
    rejected(case, payload, "nonfinite")
    payload = copy.deepcopy(case["payload"])
    payload["lineage"]["static_function"] = lambda: None
    rejected(case, payload, "unsupported data type")


def test_static_dataclass_pytrees_and_unknown_dynamic_fields_rejected(case):
    @flax.struct.dataclass
    class DisguisedState:
        value: object
        static: object = flax.struct.field(pytree_node=False)

    payload = copy.deepcopy(case["payload"])
    payload["train_state"]["actor_opt_state"] = DisguisedState(jnp.ones(2), lambda: None)
    rejected(case, payload, "static objects")
    payload = copy.deepcopy(case["payload"])
    payload["train_state"]["tx"] = lambda: None
    rejected(case, payload, "missing/unknown")
    payload = copy.deepcopy(case["payload"])
    payload["extra_runtime"] = {}
    rejected(case, payload, "missing/unknown")
    for key in ("actor_opt_state", "critic_opt_state"):
        payload = copy.deepcopy(case["payload"])
        payload["train_state"][key] = Moments(np.array(-1, np.int32), np.ones((2, 3), np.float32), np.ones((2, 3), np.float32))
        rejected(case, payload, "negative optimizer counter")


@pytest.mark.parametrize("version", [1, 2])
def test_old_schema_is_explicitly_rejected_before_missing_field_checks(case, version):
    # An old pool checkpoint is not a malformed v3 checkpoint or migration input.
    payload = {"checkpoint_kind": VRPO_RESUME_KIND, "schema_version": version}
    rejected(case, payload, "schema|version|legacy|unsupported")


@pytest.mark.parametrize("bad", [None, {}, {"version": True, "kind": "current_joint_all_seats", "next_game_id": 8},
    {"version": 1, "kind": "current_joint_all_seats", "next_game_id": -1},
    {"version": 1, "kind": "current_joint_all_seats", "next_game_id": 8, "banks": []}])
def test_resume_collection_state_strict_at_build_and_restore(case, bad):
    payload = copy.deepcopy(case["payload"])
    payload["collection_state"] = bad
    rejected(case, payload)
    with pytest.raises(ValueError):
        build_vrpo_resume(case["state"], case["rng"], **dict(case["kwargs"], collection_state=bad))


def test_collection_state_is_detached_on_export_and_restore(case):
    result = restore(case)
    case["payload"]["collection_state"]["next_game_id"] = 99
    assert result.collection_state["next_game_id"] == case["kwargs"]["collection_state"]["next_game_id"] == 8
