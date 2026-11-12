"""CPU-only toy original-BC calibration tests; no collector/checkpoint loading."""

from dataclasses import FrozenInstanceError, replace

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from train.calibration import (
    CalibrationFailure,
    CalibrationResult,
    _mc_targets,
    actor_parameter_fingerprint,
    build_calibrated_model,
    calibrate_q,
)
from train.configs import RawQBatch, RawQTransition, tiny_debug_preset
from train.update import VRPOUpdater, create_training_state, damping, phase_params


class TinyQ(nn.Module):
    """Real four-root Flax network; privileged input is Q-only."""

    def setup(self):
        self.policy_extractor = nn.Dense(2)
        self.policy_mlp = nn.Dense(87)
        self.critic_extractor = nn.Dense(1, use_bias=False, kernel_init=nn.initializers.ones)
        self.value_critic_mlp = nn.Dense(3 * 87, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros)

    def get_action_logits(self, obs):
        return self.policy_mlp(self.policy_extractor(obs["x"]))

    def get_action_q(self, inputs):
        x = inputs["observation"]["x"] + inputs["privileged"]["x"]
        return self.value_critic_mlp(self.critic_extractor(x)).reshape((-1, 3, 87))

    def __call__(self, inputs):
        return self.get_action_logits(inputs["observation"]), self.get_action_q(inputs)


def config(**overrides):
    values = dict(
        optimizer="muon",
        critic_lr=0.05,
        weight_decay=0.0,
        calibration_train_games=3,
        calibration_heldout_games=2,
        calibration_max_epochs=8,
        critic_microbatch_size=8,
        inference_microbatch_size=8,
    )
    return tiny_debug_preset(**(values | overrides))


def batch(lengths=(2, 3, 1), payoff=(0.6, -0.3, 0.15), padding=2):
    lengths = np.asarray(lengths, np.int32)
    count = int(lengths.sum())
    n = count + padding
    offsets = np.cumsum(lengths) - lengths
    done = np.zeros(n, bool)
    done[offsets + lengths - 1] = True
    rewards = np.zeros((n, 3), np.float32)
    rewards[offsets + lengths - 1] = payoff
    players = np.arange(n, dtype=np.int32) % 3
    legal = np.zeros((n, 87), bool)
    legal[:, 0] = True
    legal[::2, 1] = True
    probs = legal.astype(np.float32) / legal.sum(-1, keepdims=True)
    executed = np.arange(n) < count
    raw = RawQTransition(
        {"x": jnp.ones((n, 1))},
        {"x": jnp.zeros((n, 1))},
        jnp.zeros(n, jnp.int32),
        jnp.asarray(legal),
        jnp.asarray(probs),
        jnp.asarray(rewards),
        jnp.asarray(players),
        jnp.asarray(done),
        jnp.asarray(legal.sum(-1) > 1),
        jnp.asarray(players == 0),
        jnp.asarray(executed),
        jnp.asarray(players),
    )
    return RawQBatch(raw, jnp.asarray(offsets), jnp.asarray(lengths), jnp.zeros(len(lengths), jnp.int32))


def fixture(cfg=None):
    cfg = config() if cfg is None else cfg
    net = TinyQ()
    train, heldout = batch(), batch((1, 4))
    inputs = {"observation": train.transitions.observation, "privileged": train.transitions.critic_observation}
    params = net.init(jax.random.key(7), inputs)
    state, _, _ = create_training_state(params, cfg)
    return net, cfg, state, train, heldout


def run(parts, **kwargs):
    net, cfg, state, train, heldout = parts
    return calibrate_q(
        net,
        cfg,
        state,
        jax.random.key(11),
        train,
        heldout,
        train_game_ids=kwargs.pop("train_game_ids", ("train0", "train1", "train2")),
        heldout_game_ids=kwargs.pop("heldout_game_ids", ("held0", "held1")),
        **kwargs,
    )


def equal(a, b):
    return jax.tree.structure(a) == jax.tree.structure(b) and all(
        np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True)
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True)
    )


@pytest.fixture(scope="module")
def calibrated():
    parts = fixture()
    before = jax.tree.map(lambda x: np.asarray(x).copy(), parts[2])
    result = run(parts)
    return parts, before, result


def test_fits_finite_mc_heldout_all_raw_seats_and_preserves_actor(calibrated):
    parts, before, result = calibrated
    report = result.report
    assert report["accepted"]
    assert report["best_heldout_mse"] < report["zero_baseline_mse"]
    assert report["best_heldout_mse"] < report["initial_critic_mse"]
    assert report["actor_sha256_before"] == report["actor_sha256_after"]
    assert report["train_raw_rows"] == 6 and report["heldout_raw_rows"] == 5
    assert report["regression"] == "sampled_action_only"
    assert report["gamma"] == report["trace_lambda"] == 1.0
    assert equal(before, parts[2])  # input params, both slots AND counters unchanged
    assert equal(phase_params(before.params, "actor"), phase_params(result.state.params, "actor"))
    assert np.isfinite(report["epoch_heldout_mse"]).all()
    assert report["best_heldout_mse"] == min(report["epoch_heldout_mse"])
    assert report["best_epoch"] == 1 + np.argmin(report["epoch_heldout_mse"])
    with pytest.raises(FrozenInstanceError):
        result.rng = jax.random.key(0)


def test_calibration_slots_discarded_and_first_online_clock_is_zero(calibrated):
    parts, _, result = calibrated
    fresh, _, _ = create_training_state(result.state.params, parts[1])
    assert equal(result.state, fresh)
    assert int(result.state.actor_steps) == int(result.state.critic_steps) == 0
    assert int(result.state.completed_updates) == 0
    assert result.report["calibration_critic_steps"] == 16
    assert result.report["fresh_optimizer_slots"]
    assert result.report["optimizer_lineage"] == "fresh_optimizer_slots"
    first = damping(parts[1], result.state.completed_updates + 1)
    assert float(first.actor_multiplier) == float(first.critic_multiplier) == 1.0


def test_nonzero_input_slots_are_not_used_or_mutated():
    net, cfg, state, train, heldout = fixture(config(calibration_max_epochs=1))
    state = state.replace(
        actor_opt_state=jax.tree.map(lambda x: x + jnp.ones_like(x), state.actor_opt_state),
        critic_opt_state=jax.tree.map(lambda x: x + jnp.ones_like(x), state.critic_opt_state),
        actor_steps=jnp.int32(9),
        critic_steps=jnp.int32(13),
        completed_updates=jnp.int32(4),
    )
    before = jax.tree.map(lambda x: np.asarray(x).copy(), state)
    result = run((net, cfg, state, train, heldout))
    fresh, _, _ = create_training_state(result.state.params, cfg)
    assert equal(before, state) and equal(result.state, fresh)
    assert int(result.state.completed_updates) == 0


def test_mc_targets_are_true_reverse_rewards_not_learner_or_decision_compaction():
    raw_batch = batch((3,), payoff=(3.0, -1.0, 2.0))
    raw = raw_batch.transitions
    rewards = raw.reward.at[0].set(jnp.array([1.0, 2.0, -3.0])).at[1].set(jnp.array([2.0, -2.0, 1.0]))
    raw_batch = raw_batch._replace(transitions=raw._replace(reward=rewards))
    targets = _mc_targets(raw_batch)
    np.testing.assert_allclose(targets[:3], [[6.0, -1.0, 0.0], [5.0, -3.0, 3.0], [3.0, -1.0, 2.0]])
    np.testing.assert_array_equal(targets[3:], 0.0)
    assert not bool(raw.is_learner[1]) and not bool(raw.is_decision[1])


def test_no_fabricated_targets_for_other_86_actions(calibrated):
    parts, before, result = calibrated
    net, _, _, train, _ = parts
    inputs = {"observation": train.transitions.observation, "privileged": train.transitions.critic_observation}
    initial = net.apply(phase_params(before.params, "critic"), inputs, method=net.get_action_q)
    after = net.apply(phase_params(result.state.params, "critic"), inputs, method=net.get_action_q)
    np.testing.assert_array_equal(after[..., 1:], initial[..., 1:])
    assert np.max(np.abs(np.asarray(after[..., 0]))) > 0.1


def test_only_whole_games_enter_gradients_and_no_actor_update(monkeypatch):
    parts = fixture(config(calibration_max_epochs=1))
    original = VRPOUpdater.__init__
    seen, seen_denominators = [], []

    def instrument(self, net, cfg):
        original(self, net, cfg)
        grad = self.critic_grad

        def wrapped(params, data, denominator):
            # Each target is the same MC utility on EVERY executed raw row,
            # including forced and foreign rows; weights count complete games.
            weight = np.asarray(data[-1])
            seen.append(int(weight.sum()))
            seen_denominators.append(float(denominator))
            assert float(denominator) in (6.0, 12.0)  # 6 * one/two complete games, NOT raw rows
            np.testing.assert_allclose(
                np.asarray(data[2])[weight.astype(bool)], np.broadcast_to([0.6, -0.3, 0.15], (int(weight.sum()), 3))
            )
            assert data[-1].shape[0] % 8 == 0
            return grad(params, data, denominator)

        wrapped.clear_cache = grad.clear_cache
        self.critic_grad = wrapped

        def forbidden(*args, **kwargs):
            pytest.fail("Actor inference/gradient/optimizer advancement is forbidden")

        for name in ("actor_predict", "actor_grad", "actor_opt_step"):
            original_fn = getattr(self, name)

            def blocked(*args, **kwargs):
                return forbidden(*args, **kwargs)

            blocked.clear_cache = original_fn.clear_cache
            setattr(self, name, blocked)

    monkeypatch.setattr(VRPOUpdater, "__init__", instrument)
    monkeypatch.setattr(VRPOUpdater, "update", lambda *a, **k: pytest.fail("Actor training is forbidden"))
    run(parts)
    assert sum(seen) == 6
    assert sorted(seen) in ([1, 5], [2, 4], [3, 3])
    # _parts pads final fixed-width arenas: the one real game must divide by6,
    # not12 merely because a second zero-length game slot exists.
    assert sorted(seen_denominators) == [6.0, 12.0]


@pytest.mark.parametrize(
    "ids", [None, ("a", "b"), ("a", "a", "b"), ("a", "", "c"), ("a", None, "c"), ("a", True, "c"), ("a", [], "c")]
)
def test_missing_invalid_or_duplicate_whole_game_ids_fail(ids):
    with pytest.raises(CalibrationFailure, match="game ID"):
        run(fixture(), train_game_ids=ids)


def test_overlapping_partition_ids_rejected_before_training():
    with pytest.raises(CalibrationFailure, match="overlap"):
        run(fixture(), heldout_game_ids=("train1", "other"))


@pytest.mark.parametrize("bad", ["missing_game", "zero_length", "truncated", "early_terminal", "bad_offset"])
def test_missing_truncated_or_row_partitioned_games_fail(bad):
    net, cfg, state, train, heldout = fixture()
    if bad == "missing_game":
        train = train._replace(
            game_lengths=train.game_lengths[:2],
            game_offsets=train.game_offsets[:2],
            learner_seats=train.learner_seats[:2],
        )
    elif bad == "zero_length":
        train = train._replace(game_lengths=train.game_lengths.at[1].set(0))
    elif bad == "truncated":
        train = train._replace(transitions=train.transitions._replace(done=train.transitions.done.at[4].set(False)))
    elif bad == "early_terminal":
        train = train._replace(transitions=train.transitions._replace(done=train.transitions.done.at[0].set(True)))
    else:
        train = train._replace(game_offsets=train.game_offsets.at[1].add(1))
    with pytest.raises(CalibrationFailure, match="complete games|truncated"):
        run((net, cfg, state, train, heldout))


@pytest.mark.parametrize("where", ["reward", "params", "prediction", "gradient", "slots"])
def test_nonfinite_data_or_candidate_fail_without_input_mutation(where, monkeypatch):
    net, cfg, state, train, heldout = fixture()
    if where == "reward":
        train = train._replace(
            transitions=train.transitions._replace(reward=train.transitions.reward.at[1, 2].set(jnp.nan))
        )
    elif where == "params":
        state = state.replace(params=jax.tree.map(lambda x: x * jnp.nan, state.params))
    elif where == "prediction":

        class NonfiniteQ(TinyQ):
            def get_action_q(self, inputs):
                return super().get_action_q(inputs) * jnp.nan

        net = NonfiniteQ()
    elif where == "gradient":
        from train import update as vrpo_update

        monkeypatch.setattr(vrpo_update, "_critic_loss", lambda *a: jnp.float32(jnp.nan))
    else:
        from train import calibration as vrpo_calibration

        original = vrpo_calibration.create_training_state

        def bad_slots(params, config):
            fresh, actor, critic = original(params, config)
            return (
                fresh.replace(
                    critic_opt_state=jax.tree.map(
                        lambda x: jnp.full_like(x, jnp.nan) if jnp.issubdtype(x.dtype, jnp.floating) else x,
                        fresh.critic_opt_state,
                    )
                ),
                actor,
                critic,
            )

        monkeypatch.setattr(vrpo_calibration, "create_training_state", bad_slots)
    before = jax.tree.map(lambda x: np.asarray(x).copy(), state)
    with pytest.raises(CalibrationFailure):
        run((net, cfg, state, train, heldout))
    assert equal(before, state)


def test_zero_baseline_cannot_be_beaten_even_at_min_relative_zero():
    net, cfg, state, train, _ = fixture()
    heldout = batch((1, 4), payoff=(0.0, 0.0, 0.0))
    with pytest.raises(CalibrationFailure) as failure:
        run((net, cfg, state, train, heldout))
    assert failure.value.report["zero_baseline_mse"] == 0.0
    assert not failure.value.report["beats_zero_baseline"]


def test_perfect_initial_critic_cannot_pass_even_if_better_than_zero():
    net, cfg, state, train, heldout = fixture()
    params = jax.tree.map(lambda x: x, state.params)
    bias = np.zeros((3, 87), np.float32)
    bias[:, 0] = [0.6, -0.3, 0.15]
    params["params"]["value_critic_mlp"]["bias"] = jnp.asarray(bias.reshape(-1))
    state = state.replace(params=params)
    with pytest.raises(CalibrationFailure) as failure:
        run((net, cfg, state, train, heldout))
    report = failure.value.report
    assert report["initial_critic_mse"] == 0.0
    assert not report["beats_initial_critic"]
    assert report["epochs_completed"] == cfg.calibration_max_epochs


def test_no_improvement_is_failure_with_strict_zero_threshold():
    net, cfg, state, _, heldout = fixture()
    train = batch(payoff=(0.0, 0.0, 0.0))
    with pytest.raises(CalibrationFailure) as failure:
        run((net, cfg, state, train, heldout))
    assert failure.value.report["best_epoch"] == 0
    assert failure.value.report["best_heldout_mse"] == failure.value.report["initial_critic_mse"]


def test_unvalidated_config_rejected_before_any_calibration():
    parts = list(fixture())
    parts[1] = {"calibration_max_epochs": 8}
    with pytest.raises(CalibrationFailure, match="validated VRPOConfig"):
        run(parts)


def test_relative_improvement_and_memory_config_fail_closed():
    parts = fixture(config(calibration_max_epochs=1, calibration_min_relative_improvement=0.99))
    with pytest.raises(CalibrationFailure, match="strictly beat"):
        run(parts)
    parts = fixture(config(max_bytes=100))
    with pytest.raises(CalibrationFailure, match="memory admission"):
        run(parts)


@pytest.mark.parametrize(
    "override",
    [
        dict(calibration_train_games=0),
        dict(calibration_heldout_games=-1),
        dict(calibration_max_epochs=0),
        dict(calibration_min_relative_improvement=-0.1),
        dict(calibration_min_relative_improvement=1.0),
        dict(calibration_min_relative_improvement=float("nan")),
    ],
)
def test_bad_calibration_config(override):
    with pytest.raises(ValueError):
        replace(config(), **override)


def test_artifact_is_not_resume_requires_bc_sha_and_accepted_result(calibrated, monkeypatch):
    from train import checkpoint as vrpo_checkpoint

    _, _, result = calibrated
    seen = {}

    def builder(params, *, contract, metadata):
        seen.update(params=params, contract=contract, metadata=metadata)
        return {"checkpoint_kind": "vrpo_model", "schema_version": 1, "metadata": metadata, "params": params}

    monkeypatch.setattr(vrpo_checkpoint, "build_vrpo_model", builder)
    metadata = {"bc_source_sha256": "a" * 64, "source_identity": "original_bc"}
    model = build_calibrated_model(result, contract={"toy": True}, metadata=metadata)
    assert model["checkpoint_kind"] == "vrpo_q_calibrated"
    assert model["critic_trained"] is True and model["schema_version"] != 2
    assert "train_state" not in model and "rng" not in model
    assert model["metadata"]["calibration_report"] == result.report
    assert model["metadata"]["bc_source_sha256"] == "a" * 64
    assert metadata == {"bc_source_sha256": "a" * 64, "source_identity": "original_bc"}
    for bad in ({}, {"bc_source_sha256": "bad"}, {"bc_source_sha256": "A" * 64}):
        with pytest.raises(ValueError, match="bc_source_sha256"):
            build_calibrated_model(result, contract={}, metadata=bad)
    failed = CalibrationResult(result.state, result.rng, {"accepted": False})
    with pytest.raises(ValueError, match="accepted"):
        build_calibrated_model(failed, contract={}, metadata=metadata)
    advanced = CalibrationResult(result.state.replace(completed_updates=jnp.int32(1)), result.rng, result.report)
    with pytest.raises(ValueError, match="fresh online counters"):
        build_calibrated_model(advanced, contract={}, metadata=metadata)


def test_actor_fingerprint_ignores_critic_but_detects_actor_bytes(calibrated):
    _, _, result = calibrated
    params = jax.tree.map(lambda x: x, result.state.params)
    before = actor_parameter_fingerprint(params)
    params["params"]["value_critic_mlp"]["bias"] += 1.0
    assert actor_parameter_fingerprint(params) == before
    params["params"]["policy_mlp"]["bias"] += 1.0
    assert actor_parameter_fingerprint(params) != before
