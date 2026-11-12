"""CPU-only runtime contracts; artifacts are synthetic temporary bytes only.

No real checkpoint, rollout, optimizer step, or training process is used here.
Production V4 preset checks below validate schemas only, never allocate real parameters.
"""
import copy
import hashlib
import pickle
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from networks.mahjong_3p.configs import (
    DEFAULT_CENTRAL_Q_CRITIC_CONFIG,
    LARGE_NETWORK_CONFIG,
    MEDIUM_NETWORK_CONFIG,
    RULESET,
    SMALL_NETWORK_CONFIG,
    build_central_q_input,
)
from networks.mahjong_3p.network import get_network_cls, observe_for_network

from train import selfplay as runtime
from train.configs import VRPOConfig, default_reward_config, tiny_debug_preset
from train.update import phase_params

TINY = dict(architecture="v5", num_layers=1, d_model=8, num_heads=2, mlp_dim=16)
HEAD = {"version": 1, "hidden_dim": 5}
SHA = "a" * 64


def options(**overrides):
    values = dict(algorithm=tiny_debug_preset(),
                  bc_path="synthetic-bc.pkl", bc_sha256=SHA,
                  output_dir="synthetic-new-run", run_id="runtime-test")
    return runtime.VRPORunConfig(**(values | overrides))


def envelope(params=None, **overrides):
    values = dict(checkpoint_kind="bc", schema_version=1, env_name="sanma",
                  round_mode="half", ruleset=RULESET, network_config=dict(TINY),
                  head_config=dict(HEAD), params={} if params is None else params)
    return values | overrides


def equal(left, right):
    def leaf_equal(a, b):
        a, b = np.asarray(a), np.asarray(b)
        return np.array_equal(a, b, equal_nan=a.dtype.kind in "fc" and b.dtype.kind in "fc")
    return (jax.tree.structure(left) == jax.tree.structure(right)
            and all(leaf_equal(a, b)
                    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)))


def test_wrong_sha_is_rejected_before_any_pickle_code(tmp_path, monkeypatch):
    path = tmp_path / "synthetic.pkl"
    path.write_bytes(b"not even a pickle; deserialization must not happen")
    called = []
    monkeypatch.setattr(runtime.pickle, "loads", lambda data: called.append(data))
    with pytest.raises(ValueError, match="BEFORE deserialization"):
        runtime.load_verified(path, SHA)
    assert not called


def test_verified_load_hashes_exact_bytes_passed_to_pickle(tmp_path, monkeypatch):
    data = pickle.dumps({"synthetic": np.arange(3, dtype=np.int16)})
    path = tmp_path / "synthetic.pkl"
    path.write_bytes(data)
    calls = []
    sentinel = object()
    def loads(actual):
        calls.append(actual)
        # Even a source path changed after authentication cannot replace the
        # already-read bytes that are actually sent to the deserializer.
        path.write_bytes(b"changed after read")
        return sentinel
    monkeypatch.setattr(runtime.pickle, "loads", loads)
    assert runtime.load_verified(path, hashlib.sha256(data).hexdigest()) is sentinel
    assert calls == [data]


@pytest.mark.parametrize("digest", [None, "", "a" * 63, "A" * 64, "g" * 64, True])
def test_invalid_digest_fails_before_file_read_or_deserialization(digest, tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.pickle, "loads", lambda *a: pytest.fail("unpickle before digest validation"))
    with pytest.raises(ValueError, match="lowercase SHA256"):
        runtime.load_verified(tmp_path / "does-not-exist.pkl", digest)


def test_run_config_explicit_nested_algorithm_roundtrip_and_budget_defaults():
    cfg = options()
    loaded = runtime.VRPORunConfig.from_dict(cfg.to_dict())
    assert loaded == cfg
    assert type(loaded.algorithm) is VRPOConfig
    assert loaded.algorithm.preset == "tiny_debug"
    assert loaded.min_free_bytes == 4 * 1024 ** 3
    assert not hasattr(loaded, "history_capacity")
    assert not hasattr(loaded, "opponent_pool")
    assert not hasattr(loaded, "snapshot_envsteps")
    assert (loaded.algorithm.t_eta, loaded.algorithm.t_alpha) == (2, 2)
    data = cfg.to_dict()
    data["algorithm"]["t_eta"] = 999
    assert cfg.algorithm.t_eta == 2


@pytest.mark.parametrize("data", [None, [], {}, {"algorithm": None}, {"algorithm": []},
    {"algorithm": tiny_debug_preset()}])
def test_run_config_requires_an_explicit_algorithm_mapping(data):
    with pytest.raises(ValueError, match="algorithm mapping"):
        runtime.VRPORunConfig.from_dict(data)


@pytest.mark.parametrize("where", ["outer_unknown", "inner_unknown", "missing_eta", "missing_alpha"])
def test_run_config_rejects_unknown_fields_and_inferred_production_horizons(where):
    data = options().to_dict()
    if where == "outer_unknown":
        data["gpu_magic"] = True
    elif where == "inner_unknown":
        data["algorithm"]["normalize_advantage"] = True
    else:
        del data["algorithm"]["t_eta" if where == "missing_eta" else "t_alpha"]
    with pytest.raises((TypeError, ValueError)):
        runtime.VRPORunConfig.from_dict(data)


@pytest.mark.parametrize("change", [dict(seed=True), dict(seed=-1), dict(seed=2 ** 32),
    dict(seed=1.), dict(history_capacity=True), dict(history_capacity=1), dict(history_capacity=7),
    dict(snapshot_envsteps=True), dict(snapshot_envsteps=0), dict(min_free_bytes=True),
    dict(min_free_bytes=0), dict(min_free_bytes=-1), dict(max_updates=True), dict(max_updates=0),
    dict(max_updates=1.), dict(max_wall_seconds=True), dict(max_wall_seconds=0),
    dict(max_wall_seconds=float("nan")), dict(max_wall_seconds=float("inf")),
    dict(bc_path=""), dict(output_dir=""), dict(run_id=""), dict(bc_sha256="A" * 64),
    dict(algorithm={}), dict(opponent_pool="random:1")])
def test_invalid_run_config_fails_closed(change):
    with pytest.raises((ValueError, TypeError)):
        options(**change)


@pytest.mark.parametrize("name", ["t_eta", "t_alpha"])
@pytest.mark.parametrize("value", [True, 0, -1, 2.])
def test_nested_algorithm_horizons_are_not_coerced(name, value):
    data = options().to_dict()
    data["algorithm"][name] = value
    with pytest.raises(ValueError, match=name):
        runtime.VRPORunConfig.from_dict(data)


def test_checked_parameters_returns_exact_tree_dtype_values_without_input_mutation():
    loaded = {"params": {"x": np.array([[1., 2.]], np.float32), "n": np.array([3], np.int16)}}
    template = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), loaded)
    before = copy.deepcopy(loaded)
    result = runtime.checked_parameters(loaded, template)
    assert equal(result, loaded) and equal(before, loaded)
    assert all(isinstance(x, jax.Array) for x in jax.tree.leaves(result))
    assert result["params"]["n"].dtype == jnp.int16


@pytest.mark.parametrize("bad", ["structure", "shape", "float64", "int", "nan", "inf", "list"])
def test_checked_parameters_rejects_bad_tree_shape_dtype_nonfinite_without_coercion(bad):
    template = {"params": {"x": jax.ShapeDtypeStruct((2,), np.float32)}}
    loaded = {"params": {"x": np.ones(2, np.float32)}}
    if bad == "structure":
        loaded["extra"] = np.zeros(1, np.float32)
    elif bad == "shape":
        loaded["params"]["x"] = np.ones((1, 2), np.float32)
    elif bad == "float64":
        loaded["params"]["x"] = np.ones(2, np.float64)
    elif bad == "int":
        loaded["params"]["x"] = np.ones(2, np.int32)
    elif bad == "list":
        loaded["params"]["x"] = [1., 2.]
    else:
        loaded["params"]["x"][0] = np.nan if bad == "nan" else np.inf
    before = copy.deepcopy(loaded)
    with pytest.raises(ValueError, match="tree mismatch|shape/dtype/finite"):
        runtime.checked_parameters(loaded, template)
    assert equal(before, loaded)


@pytest.mark.parametrize("change", [dict(checkpoint_kind="ppo"), dict(checkpoint_kind="bc_critic"),
    dict(schema_version=True), dict(schema_version=2), dict(env_name="red_mahjong"),
    dict(round_mode="east"), dict(ruleset="unknown"), dict(network_config={}),
    dict(network_config=dict(TINY, architecture="v3")), dict(head_config=None),
    dict(params=None), dict(params={"params": {"policy_extractor": {}}})])
def test_bc_actor_provenance_and_root_failures_happen_before_model_init(change, monkeypatch):
    monkeypatch.setattr(runtime, "get_network_cls", lambda *a, **k: pytest.fail("invalid source reached factory"))
    with pytest.raises(ValueError):
        runtime._bc_actor(envelope(**change), object(), debug=True)


def test_bc_actor_root_metadata_cannot_repair_missing_network_provenance():
    checkpoint = envelope(metadata={"network_config": dict(TINY), "head_config": HEAD})
    del checkpoint["network_config"]
    with pytest.raises(ValueError, match="root network_config"):
        runtime._bc_actor(checkpoint, object(), debug=True)


@pytest.mark.parametrize("network_config", [SMALL_NETWORK_CONFIG, MEDIUM_NETWORK_CONFIG],
                         ids=["v4-s", "v4-m"])
def test_original_s_and_m_are_accepted_by_the_production_guard(network_config, monkeypatch):
    """VRPO may start from any explicit original V4 preset; sizes stay size-agnostic downstream."""
    roots = {"policy_extractor": {"synthetic": np.zeros(1, np.float32)},
             "policy_mlp": {"synthetic": np.ones(1, np.float32)}}
    checkpoint = envelope(params={"params": roots}, network_config=dict(network_config),
                          head_config={"version": 1, "hidden_dim": 128})
    class SchemaModel:
        get_action_logits = object()
        def init(self, *args, **kwargs):
            pytest.fail("production preset init is forbidden in this schema-only test")
    monkeypatch.setattr(runtime, "get_network_cls", lambda env_name, **kwargs: SchemaModel)
    monkeypatch.setattr(runtime.jax, "eval_shape", lambda *args: "schema-only-template")
    monkeypatch.setattr(runtime, "checked_parameters", lambda actor, template: actor)
    actor, network, head = runtime._bc_actor(checkpoint, object())
    assert equal(actor, {"params": roots})
    assert network == network_config
    assert head == {"version": 1, "hidden_dim": 128}


@pytest.mark.parametrize("network_config", [TINY, LARGE_NETWORK_CONFIG,
    dict(architecture="v5", num_layers=2, d_model=24, num_heads=3, mlp_dim=40)])
@pytest.mark.parametrize("head_config", [{"version": 0}, HEAD, {"version": 1, "hidden_dim": 128}])
@pytest.mark.parametrize("debug", [False, True])
def test_bc_acceptance_uses_root_schema_not_preset_or_head_whitelist(network_config, head_config, debug, monkeypatch):
    roots = {"policy_extractor": {"synthetic": np.zeros(1, np.float32)},
             "policy_mlp": {"synthetic": np.ones(1, np.float32)}}
    checkpoint = envelope(params={"params": roots}, network_config=dict(network_config),
                          head_config=dict(head_config))
    seen = {}
    class SchemaModel:
        get_action_logits = object()
        def init(self, *args, **kwargs):
            pytest.fail("production-L init is forbidden in this schema-only test")
    def factory(env_name, **kwargs):
        seen.update(env_name=env_name, **kwargs)
        return SchemaModel
    monkeypatch.setattr(runtime, "get_network_cls", factory)
    monkeypatch.setattr(runtime.jax, "eval_shape", lambda *args: "schema-only-template")
    def checked(actor, template):
        assert template == "schema-only-template"
        return actor
    monkeypatch.setattr(runtime, "checked_parameters", checked)
    actor, network, head = runtime._bc_actor(checkpoint, object(), debug=debug)
    assert equal(actor, {"params": roots})
    assert network == network_config and head == head_config
    assert seen == {"env_name": "sanma", **network, "head_config": head}


@pytest.fixture(scope="module")
def tiny_bc():
    # A fresh tiny public Actor, not loaded weights, and one unstepped state.
    env = runtime.rule_spec("sanma").make_env("half", reward_config=default_reward_config("sanma"))
    state = env.init(jax.random.PRNGKey(51))
    obs = observe_for_network(env, state, TINY)
    actor_model = get_network_cls("sanma", **TINY, head_config=HEAD)()
    actor = jax.jit(lambda key: actor_model.init(key, obs, method=actor_model.get_action_logits))(
        jax.random.PRNGKey(52))
    return env, obs, actor_model, actor


@pytest.mark.parametrize("debug", [False, True])
def test_real_tiny_bc_actor_is_byte_exact_and_legacy_critic_is_ignored(tiny_bc, debug):
    _, obs, _, source = tiny_bc
    checkpoint = envelope(params=copy.deepcopy(source))
    checkpoint["params"]["params"]["critic_extractor"] = {"legacy": np.array([np.nan], np.float32)}
    checkpoint["params"]["params"]["value_critic_mlp"] = {"legacy": np.array([np.inf], np.float32)}
    before = copy.deepcopy(checkpoint)
    actor, network, head = runtime._bc_actor(checkpoint, obs, debug=debug)
    assert equal(actor, source) and equal(checkpoint, before)
    assert network == TINY and head == HEAD
    assert set(actor["params"]) == {"policy_extractor", "policy_mlp"}
    rootless, _, _ = runtime._bc_actor(envelope(params=source["params"]), obs, debug=True)
    assert equal(rootless, actor)


def test_real_tiny_bc_head_zero_is_accepted_without_upgrade_or_debug_exception(tiny_bc):
    _, obs, _, _ = tiny_bc
    head = {"version": 0}
    model = get_network_cls("sanma", **TINY, head_config=head)()
    params = model.init(jax.random.PRNGKey(91), obs, method=model.get_action_logits)
    for debug in (False, True):
        actor, network, actual_head = runtime._bc_actor(envelope(params=params, head_config=head), obs, debug=debug)
        assert equal(actor, params) and network == TINY and actual_head == head
    legacy = envelope(params=params, metadata={"head_config": HEAD})
    del legacy["head_config"]
    actor, _, actual_head = runtime._bc_actor(legacy, obs)
    assert actual_head == {"version": 0} and equal(actor, params)


def test_bc_parameter_shapes_must_match_authenticated_root_even_for_custom_size(tiny_bc):
    _, obs, _, source = tiny_bc
    mismatch = dict(TINY, d_model=16, num_heads=2)
    with pytest.raises(ValueError, match="tree mismatch|shape/dtype"):
        runtime._bc_actor(envelope(params=source, network_config=mismatch), obs)


def test_real_tiny_initialization_copies_only_actor_public_encoder_and_fresh_q(tiny_bc):
    _, obs, actor_model, source = tiny_bc
    checkpoint = envelope(params=copy.deepcopy(source))
    checkpoint["params"]["params"]["critic_extractor"] = {"never_copy": np.full(3, np.nan, np.float32)}
    checkpoint["params"]["params"]["value_critic_mlp"] = {"never_copy": np.full(3, 123., np.float32)}
    before = copy.deepcopy(checkpoint)
    env, model, state, rng, actor, network, head = runtime.initialize_bc(checkpoint, options())
    assert equal(before, checkpoint) and equal(actor, source)
    assert network == TINY and head == HEAD
    assert equal(phase_params(state.params, "actor"), source)
    assert equal(state.params["params"]["critic_extractor"]["public_encoder"],
                 source["params"]["policy_extractor"])
    for name in ("private_encoder", "fusion"):
        assert all(np.isfinite(x).all() for x in jax.tree.leaves(
            state.params["params"]["critic_extractor"][name]))
    assert int(state.completed_updates) == int(state.actor_steps) == int(state.critic_steps) == 0
    assert rng.shape == (2,)
    initial = env.init(jax.random.PRNGKey(53))
    observation = observe_for_network(env, initial, TINY)
    central = build_central_q_input(initial, observation)
    q = model.apply(state.params, central, method=model.get_action_q)
    np.testing.assert_array_equal(q, np.zeros((1, 3, 87), np.float32))
    expected = actor_model.apply(source, obs, method=actor_model.get_action_logits)
    actual = model.apply(phase_params(state.params, "actor"), obs, method=model.get_action_logits)
    np.testing.assert_array_equal(actual, expected)


def q_envelope(actor, **overrides):
    return envelope(params=actor, checkpoint_kind="vrpo_model",
                    critic_config=dict(DEFAULT_CENTRAL_Q_CRITIC_CONFIG),
                    algorithm_config=tiny_debug_preset().algorithm_config()) | overrides


@pytest.mark.parametrize("kind", ["vrpo_model", "vrpo_q_calibrated"])
def test_inference_extracts_only_public_actor_never_builds_or_reads_q(tiny_bc, kind, monkeypatch):
    _, obs, original_model, source = tiny_bc
    params = copy.deepcopy(source)
    params["params"]["critic_extractor"] = object()  # opaque Q is not examined
    params["params"]["value_critic_mlp"] = {"not_actor": np.array([np.nan])}
    monkeypatch.setattr(runtime, "get_central_q_network_cls", lambda *a, **k: pytest.fail("inference built Q"))
    model, actor = runtime.load_vrpo_actor(q_envelope(params, checkpoint_kind=kind), obs)
    assert equal(actor, source)
    assert set(actor["params"]) == {"policy_extractor", "policy_mlp"}
    expected = original_model.apply(source, obs, method=original_model.get_action_logits)
    actual = model.apply(actor, obs, method=model.get_action_logits)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("change", [dict(checkpoint_kind="ppo"), dict(checkpoint_kind="vrpo_boundary_resume"),
    dict(schema_version=True), dict(schema_version=2), dict(head_config=None),
    dict(critic_config={"version": 0, "kind": "legacy_v"}),
    dict(critic_config=dict(DEFAULT_CENTRAL_Q_CRITIC_CONFIG, payoff_seats="relative3")),
    dict(algorithm_config={}), dict(algorithm_config={"version": True, "kind": "vrpo_qboost_chance_residual_v1"}),
    dict(env_name="red_mahjong"), dict(network_config={}),
    dict(network_config=dict(TINY, architecture="v3"))])
def test_inference_rejects_wrong_kind_environment_or_schema_before_model_allocation(change, monkeypatch):
    monkeypatch.setattr(runtime, "get_network_cls", lambda *a, **k: pytest.fail("invalid inference allocated model"))
    with pytest.raises(ValueError):
        runtime.load_vrpo_actor(q_envelope({}, **change), object())


def test_inference_missing_critic_is_legacy_v_not_repaired_from_metadata(monkeypatch):
    checkpoint = q_envelope({}, metadata={"critic_config": DEFAULT_CENTRAL_Q_CRITIC_CONFIG})
    del checkpoint["critic_config"]
    monkeypatch.setattr(runtime, "get_network_cls", lambda *a, **k: pytest.fail("implicit Q migration"))
    with pytest.raises(ValueError, match="explicit Actor and Q schemas"):
        runtime.load_vrpo_actor(checkpoint, object())


@pytest.mark.parametrize("completed", [0, 1, 2])
def test_collection_boundary_requires_exact_absolute_game_clock(completed):
    config = options()
    state = {"version": 1, "kind": "current_joint_all_seats",
             "next_game_id": completed * config.algorithm.rollout_games}
    before = copy.deepcopy(state)
    actual = runtime.checked_vrpo_collection_state(state, config, completed_updates=completed)
    assert actual == state == before and actual is not state
    for delta in (-1, 1):
        with pytest.raises(ValueError):
            runtime.checked_vrpo_collection_state(dict(state, next_game_id=state["next_game_id"] + delta),
                                                  config, completed_updates=completed)


@pytest.mark.parametrize("completed", [True, -1, 1., None])
def test_collection_boundary_requires_exact_nonnegative_update_integer(completed):
    from train.selfplay import initial_collection_state
    with pytest.raises(ValueError):
        runtime.checked_vrpo_collection_state(initial_collection_state(), options(), completed_updates=completed)


def test_source_versions_is_local_code_and_package_identity_only():
    versions = runtime.source_versions()
    assert set(versions) == {"packages", "implementation_sha256"}
    assert set(versions["packages"]) == {"jax", "jaxlib", "flax", "optax"}
    assert all(isinstance(value, str) and value for value in versions["packages"].values())
    sources = versions["implementation_sha256"]
    assert "train/selfplay.py" in sources and "networks/mahjong_3p/network.py" in sources
    root = Path(runtime.__file__).resolve().parents[1]
    assert sources["train/selfplay.py"] == hashlib.sha256((root / "train/selfplay.py").read_bytes()).hexdigest()
    assert all(name.endswith(".py") and name.split("/")[0] in ("train", "networks", "mahjax")
               for name in sources)


def test_runtime_contract_records_independent_phases_explicit_source_and_no_training(monkeypatch):
    versions = {"packages": {"jax": "synthetic"}, "implementation_sha256": {"code.py": SHA}}
    monkeypatch.setattr(runtime, "source_versions", lambda: versions)
    cfg = options()
    contract = runtime.make_contract(cfg, TINY, HEAD)
    assert contract["source_versions"] == versions
    assert contract["network_config"] == TINY and contract["head_config"] == HEAD
    assert contract["critic_config"] == DEFAULT_CENTRAL_Q_CRITIC_CONFIG
    assert contract["reward_config"] == default_reward_config("sanma")
    assert contract["environment_config"] == {"env_name": "sanma", "round_mode": "half", "ruleset": RULESET}
    assert contract["algorithm_config"]["kind"] == cfg.algorithm.kind
    assert contract["algorithm_config"]["collection_profile"]["bc_sha256"] == SHA
    assert contract["algorithm_config"]["collection_profile"] == {
        "kind": "current_joint_all_seats", "seed": cfg.seed, "bc_sha256": SHA}
    assert "muon_lr_multiplier" not in contract["optimizer_config"]
    assert contract["optimizer_config"]["independent_phases"] is True
    assert contract["optimizer_config"]["actor_lr"] == cfg.algorithm.actor_lr
    assert contract["optimizer_config"]["critic_lr"] == cfg.algorithm.critic_lr
    assert contract["retention_enabled"] is False
    retained = runtime.make_contract(options(algorithm=replace(cfg.algorithm, bc_reference_coef=.001)), TINY, HEAD)
    assert retained["retention_enabled"] is True


def test_write_new_synthetic_pickle_returns_sha_and_never_overwrites(tmp_path):
    path = tmp_path / "synthetic-new.pkl"
    payload = {"synthetic": np.arange(5, dtype=np.float32)}
    digest = runtime.write_new_pickle(path, payload, min_free_bytes=1)
    data = path.read_bytes()
    assert digest == hashlib.sha256(data).hexdigest()
    loaded = runtime.load_verified(path, digest)
    np.testing.assert_array_equal(loaded["synthetic"], payload["synthetic"])
    with pytest.raises(FileExistsError):
        runtime.write_new_pickle(path, {"replacement": True}, min_free_bytes=1)
    assert path.read_bytes() == data


def test_disk_reserve_checked_before_creation_includes_serialized_size(tmp_path, monkeypatch):
    path = tmp_path / "never-created.pkl"
    payload = {"synthetic": np.arange(10, dtype=np.float32)}
    size = len(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    monkeypatch.setattr(runtime.shutil, "disk_usage", lambda parent: SimpleNamespace(free=size + 99))
    with pytest.raises(OSError, match="disk reserve"):
        runtime.write_new_pickle(path, payload, min_free_bytes=100)
    assert not path.exists()
    monkeypatch.setattr(runtime.shutil, "disk_usage", lambda parent: SimpleNamespace(free=size + 100))
    runtime.write_new_pickle(path, payload, min_free_bytes=100)
    assert len(path.read_bytes()) == size


def test_serialization_and_fsync_failure_do_not_replace_any_existing_artifact(tmp_path, monkeypatch):
    path = tmp_path / "synthetic-existing.pkl"
    path.write_bytes(b"existing external bytes")
    original = path.read_bytes()
    with monkeypatch.context() as patch:
        def fail_dump(*args, **kwargs):
            raise OSError("injected serialization failure")
        patch.setattr(runtime.pickle, "dumps", fail_dump)
        with pytest.raises(OSError, match="injected"):
            runtime.write_new_pickle(path, {}, min_free_bytes=1)
    assert path.read_bytes() == original
    import os
    monkeypatch.setattr(os, "fsync", lambda *a: (_ for _ in ()).throw(OSError("injected fsync failure")))
    fresh = tmp_path / "synthetic-partial.pkl"
    with pytest.raises(OSError, match="injected fsync"):
        runtime.write_new_pickle(fresh, {"synthetic": True}, min_free_bytes=1)
    # An unacknowledged standalone write is not automatically adopted/replaced.
    assert path.read_bytes() == original
    with pytest.raises(FileExistsError):
        runtime.write_new_pickle(fresh, {"retry": True}, min_free_bytes=1)


def test_existing_symlink_is_never_followed_for_artifact_write(tmp_path):
    target = tmp_path / "unowned.pkl"
    target.write_bytes(b"unowned")
    link = tmp_path / "synthetic-symlink.pkl"
    link.symlink_to(target)
    with pytest.raises(FileExistsError):
        runtime.write_new_pickle(link, {"synthetic": True}, min_free_bytes=1)
    assert link.is_symlink() and target.read_bytes() == b"unowned"


@pytest.mark.parametrize("kind", ["vrpo_model", "vrpo_q_calibrated"])
def test_ordinary_v1_actor_inference_remains_explicitly_supported(tiny_bc, kind):
    _, obs, _, source = tiny_bc
    checkpoint = q_envelope(source, checkpoint_kind=kind,
        algorithm_config={"version": 1, "kind": "vrpo_qboost_chance_residual_v1"})
    _, actor = runtime.load_vrpo_actor(checkpoint, obs)
    assert equal(actor, source)


@pytest.mark.parametrize("version,kind", [(1, "vrpo_qboost_chance_residual_v2"),
    (2, "vrpo_qboost_chance_residual_v1"), (3, "vrpo_qboost_chance_residual_v3")])
def test_inference_algorithm_allowlist_does_not_guess_compatible_pairs(version, kind, monkeypatch):
    monkeypatch.setattr(runtime, "get_network_cls", lambda *a, **k: pytest.fail("unknown algorithm allocated Actor"))
    with pytest.raises(ValueError):
        runtime.load_vrpo_actor(q_envelope({}, algorithm_config={"version": version, "kind": kind}), object())


@pytest.mark.parametrize("field,value", [("opponent_pool", "current:1"),
    ("history_capacity", 6), ("snapshot_envsteps", 500000), ("learner_seats", "all")])
def test_removed_population_configuration_is_not_accepted_as_an_alias(field, value):
    data = options().to_dict()
    data[field] = value
    with pytest.raises((TypeError, ValueError)):
        runtime.VRPORunConfig.from_dict(data)
