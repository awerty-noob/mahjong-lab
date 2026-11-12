"""Strict, array-only PPO update-boundary continuation protocol.

This is deliberately NOT the ordinary ``checkpoint_kind='ppo'`` initialization
format. It contains no TrainState, transformation, apply function, or executable.
The caller must authenticate any pickle before loading it, construct a fresh
network/optimizer from the explicit contract, validate pool/retention internals,
and only then install the returned candidate. No file I/O occurs here.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

SCHEMA_VERSION = 1
CHECKPOINT_KIND = "ppo_boundary_resume"
_NETWORK_FIELDS = {"architecture", "num_layers", "d_model", "num_heads", "mlp_dim"}
_OPTIMIZER_FIELDS = {
    "name",
    "auxiliary_learning_rate",
    "muon_learning_rate",
    "weight_decay",
    "max_grad_norm",
    "muon_momentum",
    "muon_lr_multiplier",
    "muon_ns_steps",
    "muon_ns_dtype",
    "muon_variant",
    "adam_betas",
    "adam_epsilon",
    "parameter_groups",
    "state_restored",
}
_PPO_FIELDS = {
    "env_name",
    "round_mode",
    "rollout_mode",
    "num_envs",
    "num_steps",
    "gamma",
    "gae_lambda",
    "update_epochs",
    "minibatch_size",
    "update_batch_mode",
    "effective_minibatch_size",
    "target_kl",
    "learner_temperature",
    "ent_coef",
    "clip_eps",
    "vf_coef",
    "aux_rank_coef",
    "aux_score_coef",
    "opponent_pool",
    "learner_seats",
}
_CONTRACT_FIELDS = {
    "network_config",
    "head_config",
    "reward_config",
    "optimizer_config",
    "ppo_config",
    "training_profile",
    "retention_enabled",
    "required_schedule_counters",
}
_PAYLOAD_FIELDS = {
    "schema_version",
    "checkpoint_kind",
    "boundary",
    "config_contract",
    "train_state",
    "rng",
    "progress",
    "pool_game_state",
    "opponent_pool_state",
    "retention_state",
    "lineage",
}


@dataclass(frozen=True)
class BoundaryResumeResult:
    """Validated candidate; construction never mutates the supplied prototype."""

    state: object
    rng: object
    progress: dict
    pool_state: object
    pool_payload: object
    retention_state: object
    lineage: dict


def _mapping(value, name, *, required=(), exact=False):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    if not all(isinstance(key, str) and key for key in value):
        raise ValueError(f"{name} keys must be nonempty strings")
    if not set(required) <= set(value) or (exact and set(value) != set(required)):
        raise ValueError(f"{name} missing/unknown fields; required {sorted(required)}")
    return value


def _counter(value, name):
    # In particular reject bool, float, scalar arrays, and silent int coercions.
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a nonnegative integer scalar")
    if not 0 <= int(value) <= np.iinfo(np.int64).max:
        raise ValueError(f"{name} out of range")
    return int(value)


def _copy_metadata(value, name):
    """Whitelisted host data, preserving tuple/NamedTuple containers and arrays."""
    if value is None or type(value) in (str, bool, int, bytes):
        return value
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} contains nonfinite metadata")
        return float(value)
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        _mapping(value, name)
        return {key: _copy_metadata(item, f"{name}.{key}") for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        items = [_copy_metadata(item, f"{name}[{index}]") for index, item in enumerate(value)]
        if isinstance(value, tuple) and hasattr(value, "_fields"):
            return type(value)(*items)
        return tuple(items) if isinstance(value, tuple) else items
    if isinstance(value, (np.ndarray, jax.Array)):
        return _checked_array(value, name).copy()
    raise ValueError(f"{name} contains unsupported data type {type(value).__name__}")


def _contract(value):
    _mapping(value, "config_contract", required=_CONTRACT_FIELDS)
    _mapping(value["network_config"], "network_config", required=_NETWORK_FIELDS, exact=True)
    network = value["network_config"]
    if network["architecture"] not in ("v1", "v2", "v3", "v4"):
        raise ValueError("network_config architecture must be explicit and supported")
    for key in _NETWORK_FIELDS - {"architecture"}:
        if _counter(network[key], f"network_config.{key}") == 0:
            raise ValueError(f"network_config.{key} must be positive")
    heads = _mapping(value["head_config"], "head_config", required={"version"})
    version = _counter(heads["version"], "head_config.version")
    expected = {"version"} if version == 0 else {"version", "hidden_dim"}
    if version not in (0, 1) or set(heads) != expected:
        raise ValueError("head_config requires an explicit supported complete schema")
    if version == 1 and _counter(heads["hidden_dim"], "head_config.hidden_dim") == 0:
        raise ValueError("head_config.hidden_dim must be positive")
    for key, fields in (("optimizer_config", _OPTIMIZER_FIELDS), ("ppo_config", _PPO_FIELDS)):
        _mapping(value[key], key, required=fields)
    _mapping(
        value["reward_config"],
        "reward_config",
        required={"version", "objective", "uma_points", "oka_points", "normalization_points", "score_unit_points"},
        exact=True,
    )
    # These validators never fill absent fields: completeness was checked above.
    from networks.common import validate_head_config, validate_network_config
    from train.rewards import validate_reward_config

    env_name = value["ppo_config"]["env_name"]
    validate_network_config(dict(network), env_name)
    validate_head_config(heads, network["architecture"])
    validate_reward_config(value["reward_config"], env_name)
    if not isinstance(value["training_profile"], str) or not value["training_profile"]:
        raise ValueError("training_profile must be explicit")
    if type(value["retention_enabled"]) is not bool:
        raise ValueError("retention_enabled must be bool")
    names = value["required_schedule_counters"]
    if (
        not isinstance(names, (list, tuple))
        or not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("required_schedule_counters must explicitly name unique counters")
    return _copy_metadata(value, "config_contract")


def _equal_contract(actual, expected, name="config_contract"):
    # Python equality would accept True == 1, and ignore list versus tuple.
    if type(actual) is not type(expected):
        raise ValueError(f"{name} type mismatch")
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f"{name} fields mismatch")
        for key in expected:
            _equal_contract(actual[key], expected[key], f"{name}.{key}")
    elif isinstance(expected, (tuple, list)):
        if len(actual) != len(expected):
            raise ValueError(f"{name} length mismatch")
        for index, (a, b) in enumerate(zip(actual, expected)):
            _equal_contract(a, b, f"{name}[{index}]")
    elif isinstance(expected, np.ndarray):
        if actual.dtype != expected.dtype or actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise ValueError(f"{name} array mismatch")
    elif actual != expected:
        raise ValueError(f"{name} mismatch")


def _checked_array(value, name):
    if not isinstance(value, (jax.Array, np.ndarray, np.generic)):
        raise ValueError(f"{name} must be an array")
    try:
        array = np.asarray(jax.device_get(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be raw numeric array data (typed PRNG keys unsupported)") from error
    if not (jnp.issubdtype(array.dtype, jnp.number) or array.dtype == np.bool_):
        raise ValueError(f"{name} must have a numeric dtype")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite values")
    return array


def _array_tree(value, name, prototype=None):
    leaves, structure = jax.tree.flatten(value)
    if not leaves:
        raise ValueError(f"{name} must contain array leaves")
    if prototype is not None:
        refs, ref_structure = jax.tree.flatten(prototype)
        if structure != ref_structure:
            raise ValueError(f"{name} tree structure mismatch")
    else:
        refs = [None] * len(leaves)
    arrays = []
    for index, (leaf, ref) in enumerate(zip(leaves, refs)):
        label = f"{name} leaf {index}"
        array = _checked_array(leaf, label)
        if ref is not None:
            expected = _checked_array(ref, f"prototype {label}")
            if array.shape != expected.shape or array.dtype != expected.dtype:
                raise ValueError(f"{label} shape/dtype mismatch")
        if name == "opt_state" and jnp.issubdtype(array.dtype, jnp.integer) and (array < 0).any():
            raise ValueError(f"{label} has a negative optimizer counter")
        arrays.append(array.copy())
    # Tree reconstruction retains Optax NamedTuple states (including empty states
    # and MaskedNode), unlike converting through generic dictionaries or lists.
    return jax.tree.unflatten(structure, arrays)


def _step(value, prototype=None):
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        count = _counter(value, "step")
        dtype = np.dtype(jnp.asarray(0).dtype)
        if count > np.iinfo(dtype).max:
            raise ValueError("step out of TrainState range")
        array = np.asarray(count, dtype=dtype)
    else:
        array = _checked_array(value, "step")
    if array.shape != () or not jnp.issubdtype(array.dtype, jnp.signedinteger) or int(array) < 0:
        raise ValueError("step must be a nonnegative scalar signed integer array")
    if prototype is not None and array.dtype != np.dtype(jnp.asarray(prototype).dtype):
        raise ValueError("step dtype mismatch with TrainState convention")
    return array.copy()


def _rng(value):
    array = _checked_array(value, "rng")
    if array.shape != (2,) or array.dtype != np.uint32:
        raise ValueError("rng must be raw uint32[2] PRNGKey data")
    return array.copy()


def _progress(value, contract):
    _mapping(value, "progress", required={"completed_updates", "envsteps", "schedule_counters"})
    result = _copy_metadata(value, "progress")
    for name in ("completed_updates", "envsteps"):
        result[name] = _counter(value[name], f"progress.{name}")
    schedules = _mapping(
        value["schedule_counters"], "schedule_counters", required=contract["required_schedule_counters"], exact=True
    )
    result["schedule_counters"] = {
        key: _counter(counter, f"schedule_counters.{key}") for key, counter in schedules.items()
    }
    for schedule, clock in (("last_completed_update", "completed_updates"), ("last_completed_envsteps", "envsteps")):
        if schedule in schedules and result["schedule_counters"][schedule] != result[clock]:
            raise ValueError(f"schedule_counters.{schedule} must match absolute progress.{clock}")
    return result


def _runtime_metadata(pool_state, pool_payload, retention_state, lineage, contract):
    if pool_state is None or pool_payload is None:
        raise ValueError("pool_game_state and opponent_pool_state are required")
    if contract["retention_enabled"] and retention_state is None:
        raise ValueError("enabled retention requires complete retention_state")
    if not contract["retention_enabled"] and retention_state is not None:
        raise ValueError("disabled retention must have retention_state=None")
    if not _mapping(lineage, "lineage"):
        raise ValueError("lineage provenance must be explicit and nonempty")
    return tuple(
        _copy_metadata(value, name)
        for name, value in (
            ("pool_game_state", pool_state),
            ("opponent_pool_state", pool_payload),
            ("retention_state", retention_state),
            ("lineage", lineage),
        )
    )


def build_boundary_resume(
    train_state, rng, *, config_contract, progress, pool_state, pool_payload, retention_state, lineage
):
    """Export an already committed update boundary as independent host data.

    The caller asserts there is no live batch/collector and that all environment
    games have drained. ``progress`` clocks are absolute, not reset per segment.
    Full pool/retention schema validation remains their managers' responsibility.
    """
    contract = _contract(config_contract)
    saved_progress = _progress(progress, contract)
    pool, opponents, retention, provenance = _runtime_metadata(
        pool_state, pool_payload, retention_state, lineage, contract
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_kind": CHECKPOINT_KIND,
        "boundary": {"kind": "update_boundary", "active_batch": False},
        "config_contract": contract,
        "train_state": {
            "params": _array_tree(train_state.params, "params"),
            "opt_state": _array_tree(train_state.opt_state, "opt_state"),
            "step": _step(train_state.step),
        },
        "rng": _rng(rng),
        "progress": saved_progress,
        "pool_game_state": pool,
        "opponent_pool_state": opponents,
        "retention_state": retention,
        "lineage": provenance,
    }


def restore_boundary_resume(payload, template_state, *, config_contract):
    """Validate everything, then return a candidate using fresh static functions.

    Does not deserialize, mutate a state/manager, or accept historical params-only
    metadata as resume evidence. The caller must validate manager-specific state
    before installing ANY of these candidates in its live runtime.
    """
    _mapping(payload, "boundary resume payload", required=_PAYLOAD_FIELDS, exact=True)
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != SCHEMA_VERSION
        or payload["checkpoint_kind"] != CHECKPOINT_KIND
    ):
        raise ValueError("unsupported boundary resume schema/version/kind")
    boundary = _mapping(payload["boundary"], "boundary", required={"kind", "active_batch"}, exact=True)
    if boundary["kind"] != "update_boundary" or boundary["active_batch"] is not False:
        raise ValueError("resume requires update_boundary with no active batch")
    contract = _contract(config_contract)
    saved_contract = _contract(payload["config_contract"])
    _equal_contract(saved_contract, contract)
    dynamic = _mapping(payload["train_state"], "train_state", required={"params", "opt_state", "step"}, exact=True)
    params = _array_tree(dynamic["params"], "params", template_state.params)
    slots = _array_tree(dynamic["opt_state"], "opt_state", template_state.opt_state)
    step = _step(dynamic["step"], template_state.step)
    rng = _rng(payload["rng"])
    progress = _progress(payload["progress"], contract)
    pool, opponents, retention, lineage = _runtime_metadata(
        payload["pool_game_state"],
        payload["opponent_pool_state"],
        payload["retention_state"],
        payload["lineage"],
        contract,
    )
    # Assignment happens only in this new immutable state, after all checks.
    state = template_state.replace(
        params=jax.tree.map(jnp.asarray, params), opt_state=jax.tree.map(jnp.asarray, slots), step=jnp.asarray(step)
    )
    return BoundaryResumeResult(state, jnp.asarray(rng), progress, pool, opponents, retention, lineage)
