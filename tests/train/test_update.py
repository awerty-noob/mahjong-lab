"""CPU-only four-root toy integration: no environment training or real weights."""

import pickle
from dataclasses import replace

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.core import FrozenDict, freeze

import train.update as update_module
from networks.mahjong_3p.configs import DEFAULT_CENTRAL_Q_CRITIC_CONFIG, DEFAULT_READING_CONFIG, RULESET
from train.checkpoint import build_vrpo_resume, restore_vrpo_resume
from train.configs import (
    READING_ALGORITHM_KIND,
    RawQBatch,
    RawQTransition,
    default_reward_config,
    tiny_debug_preset,
)
from train.optimizer import VRPOMuonState, vrpo_optimizer_metadata
from train.update import (
    VRPOUpdater,
    _actor_loss,
    _critic_loss,
    create_training_state,
    damping,
    finite_tree,
    masked_law,
    phase_params,
    replace_phase,
)


class TinyCentral(nn.Module):
    """Real Flax modules with the exact Actor/Q ownership and two call methods."""

    def setup(self):
        self.policy_extractor = nn.Dense(8)
        self.policy_mlp = nn.Dense(87)
        self.critic_extractor = nn.Dense(8)
        self.value_critic_mlp = nn.Dense(3 * 87)

    def get_action_logits(self, observation):
        return self.policy_mlp(nn.tanh(self.policy_extractor(observation["x"])))

    def get_action_q(self, central):
        features = jnp.concatenate((central["observation"]["x"], central["privileged"]["hidden"]), axis=-1)
        hidden = nn.tanh(self.critic_extractor(features))
        return self.value_critic_mlp(hidden).reshape((-1, 3, 87))

    def __call__(self, observation, central):
        return self.get_action_logits(observation), self.get_action_q(central)


def tree_equal(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right)):
        assert np.asarray(a).dtype == np.asarray(b).dtype
        assert np.asarray(a).shape == np.asarray(b).shape
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()


def tree_close(left, right, **kwargs):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right)):
        np.testing.assert_allclose(a, b, **kwargs)


def changed(left, right):
    return any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right)))


def make_batch(net, params, games=4):
    """Every game retains learner decisions, learner forced and foreign raw rows.

    Unequal lengths, nonzero intermediate rewards and asymmetric chance payoffs
    make dropping/normalizing raw rows observably wrong. No trace oracle is
    duplicated here: that independent suite already covers lambda endpoints.
    """
    lengths = np.array([4 + g % 2 for g in range(games)], np.int32)
    offsets = np.cumsum(lengths) - lengths
    n = int(lengths.sum())
    mask = np.zeros((n, 87), bool)
    actions = np.zeros(n, np.int32)
    seats = np.arange(games, dtype=np.int32) % 3
    players, learner, done = np.zeros(n, np.int32), np.zeros(n, bool), np.zeros(n, bool)
    features = np.zeros((n, 6), np.float32)
    random = np.random.default_rng(173)  # Fixture only, never stored in Q features.
    rewards = random.normal(0.0, 0.04, (n, 3)).astype(np.float32)
    for g, (start, length) in enumerate(zip(offsets, lengths)):
        for t in range(length):
            row = int(start + t)
            own = t in (0, 2, length - 1)
            forced = t == 2 or (length == 5 and t == 3)
            players[row] = seats[g] if own else (seats[g] + 1) % 3
            learner[row] = own
            if forced:
                action = 0 if own else 2
                mask[row, action] = True
            else:
                mask[row, : 2 if own else 3] = True
                action = (g + t) % (2 if own else 3)
            actions[row] = action
            features[row] = [1.0, 0.15 * g, 0.2 * t - 0.3, (row + 1) / n, 0.25 * np.sin(g + t), 0.5 * np.cos(g - t)]
        done[start + length - 1] = True
        # Deliberately not constrained to zero-sum payoff channels.
        rewards[start + length - 1] += np.roll(np.array([1.5, -0.4, 0.8], np.float32), g % 3)
    obs = {"x": jnp.asarray(features), "row_id": jnp.arange(n, dtype=jnp.int32)}
    hidden = jnp.asarray(np.stack((np.sin(features[:, 2]), np.cos(features[:, 3])), axis=-1))
    logits = net.apply(params, obs, method=net.get_action_logits)
    probs, _ = masked_law(logits, jnp.asarray(mask))
    probs = np.asarray(probs).copy()
    for row in np.where(~learner)[0]:
        probs[row] = 0.0
        if mask[row].sum() == 1:
            probs[row, actions[row]] = 1.0
        else:
            probs[row, :3] = [0.2, 0.3, 0.5]
    raw = RawQTransition(
        observation=obs,
        critic_observation={"hidden": hidden},
        action=jnp.asarray(actions),
        action_mask=jnp.asarray(mask),
        behavior_probs=jnp.asarray(probs),
        reward=jnp.asarray(rewards),
        current_player=jnp.asarray(players),
        done=jnp.asarray(done),
        is_decision=jnp.asarray(mask.sum(axis=-1) > 1),
        is_learner=jnp.asarray(learner),
        executed=jnp.ones(n, bool),
        policy_id=jnp.asarray(np.where(learner, 0, 9), jnp.int32),
    )
    return RawQBatch(raw, jnp.asarray(offsets), jnp.asarray(lengths), jnp.asarray(seats))


@pytest.fixture(scope="module")
def case():
    cfg = tiny_debug_preset(
        optimizer="muon",
        rollout_games=4,
        num_envs=2,
        trajectory_minibatches=2,
        actor_epochs=2,
        critic_epochs=1,
        actor_lr=0.003,
        critic_lr=0.004,
        weight_decay=0.1,
        actor_microbatch_size=2,
        critic_microbatch_size=3,
        inference_microbatch_size=2,
        time_bucket=2,
        target_kl=10.0,
        muon_ns_steps=1,
        muon_ns_dtype="float32",
    )
    net = TinyCentral()
    obs = {"x": jnp.ones((2, 6)), "row_id": jnp.arange(2)}
    central = {"observation": obs, "privileged": {"hidden": jnp.ones((2, 2))}}
    params = freeze(net.init(jax.random.PRNGKey(18), obs, central))
    state, _, _ = create_training_state(params, cfg)
    updater = VRPOUpdater(net, cfg)
    result = dict(
        config=cfg,
        network=net,
        state=state,
        updater=updater,
        batch=make_batch(net, params),
        rng=jax.random.PRNGKey(942),
    )
    yield result
    updater.close()


def test_actual_both_phase_update_preserves_prior_and_uses_whole_game_budgets(case):
    state, updater, cfg = case["state"], case["updater"], case["config"]
    before = jax.device_get(state)
    result = updater.update(state, case["rng"], case["batch"])
    assert isinstance(result.state.params, FrozenDict)
    tree_equal(state, before)
    assert bool(finite_tree(result.state))
    assert changed(phase_params(state.params, "actor"), phase_params(result.state.params, "actor"))
    assert changed(phase_params(state.params, "critic"), phase_params(result.state.params, "critic"))
    assert changed(state.actor_opt_state, result.state.actor_opt_state)
    assert changed(state.critic_opt_state, result.state.critic_opt_state)
    assert result.metrics["actor_steps"] == cfg.actor_epochs * cfg.trajectory_minibatches == 4
    assert result.metrics["critic_steps"] == cfg.critic_epochs * cfg.trajectory_minibatches == 2
    assert result.metrics["rejected_epochs"] == 0 and int(result.state.completed_updates) == 1
    for key in ("actor_opt_state", "critic_opt_state"):
        counters = [x for x in jax.tree.leaves(getattr(result.state, key)) if np.issubdtype(x.dtype, np.integer)]
        assert counters and all(int(x) > 0 for x in counters)
        if cfg.optimizer == "muon":
            # Every active matrix now uses Muon, independent of head/axis width.
            nodes = jax.tree.leaves(getattr(result.state, key), is_leaf=lambda x: isinstance(x, VRPOMuonState))
            buffers = [
                leaf for node in nodes if isinstance(node, VRPOMuonState) for leaf in jax.tree.leaves(node.momentum)
            ]
            assert buffers and any(np.any(np.asarray(x) != 0) for x in buffers)
    schedule = damping(cfg, 1)
    assert result.metrics["actor_lr"] == pytest.approx(cfg.actor_lr * float(schedule.actor_multiplier))
    assert result.metrics["critic_lr"] == pytest.approx(cfg.critic_lr * float(schedule.critic_multiplier))


@pytest.mark.parametrize("phase", ["actor", "critic"])
def test_nonzero_slots_decay_and_counters_of_inactive_phase_are_byte_frozen(case, phase):
    updater, state = case["updater"], case["state"]
    # Warm BOTH slot trees so freezing a zero/unused optimizer cannot fake success.
    for own in ("actor", "critic"):
        group = phase_params(state.params, own)
        grads = jax.tree.map(lambda x: jnp.ones_like(x) * 0.07, group)
        step = updater.actor_opt_step if own == "actor" else updater.critic_opt_step
        updated, slots = step(state.params, grads, getattr(state, f"{own}_opt_state"), jnp.float32(1))
        state = state.replace(params=replace_phase(state.params, updated, own), **{f"{own}_opt_state": slots})
    state = state.replace(actor_steps=jnp.int32(7), critic_steps=jnp.int32(11), completed_updates=jnp.int32(3))
    inactive = "critic" if phase == "actor" else "actor"
    step = updater.actor_opt_step if phase == "actor" else updater.critic_opt_step
    # Zero new gradients still have active momentum AND nonzero weight decay.
    zeros = jax.tree.map(jnp.zeros_like, phase_params(state.params, phase))
    updated, slots = step(state.params, zeros, getattr(state, f"{phase}_opt_state"), jnp.float32(0.6))
    candidate = state.replace(
        params=replace_phase(state.params, updated, phase),
        **{f"{phase}_opt_state": slots, f"{phase}_steps": getattr(state, f"{phase}_steps") + 1},
    )
    tree_equal(phase_params(state.params, inactive), phase_params(candidate.params, inactive))
    tree_equal(getattr(state, f"{inactive}_opt_state"), getattr(candidate, f"{inactive}_opt_state"))
    tree_equal(getattr(state, f"{inactive}_steps"), getattr(candidate, f"{inactive}_steps"))
    tree_equal(state.completed_updates, candidate.completed_updates)
    assert changed(phase_params(state.params, phase), updated)
    assert changed(getattr(state, f"{phase}_opt_state"), slots)


def test_current_policy_full_future_trace_rebuilt_before_every_actor_step(case, monkeypatch):
    updater, net, cfg = case["updater"], case["network"], case["config"]
    q_old = updater.old_q(case["state"], case["batch"])
    old_q_host = np.asarray(q_old)
    calls, label_calls, advantages_by_row = [], [], {}
    original = updater._current_labels
    original_qboost = update_module.qboost_labels
    current_params = [None]

    def tracked_labels(params, batch, q):
        current_params[0] = params
        calls.append(jax.device_get(phase_params(params, "actor")))
        result = original(params, batch, q)
        raw = batch.transitions
        assert int(raw.executed.sum()) > int(result.actor_mask.sum())
        assert bool(jnp.any(raw.executed & ~raw.is_learner))
        assert bool(jnp.any(raw.executed & ~raw.is_decision))
        for row, advantage, executed in zip(
            np.asarray(raw.observation["row_id"]), np.asarray(result.advantages), np.asarray(raw.executed)
        ):
            if executed:
                advantages_by_row.setdefault(int(row), []).append(advantage.copy())
        return result

    def tracked_qboost(raw, q, **kwargs):
        if "probabilities" in kwargs:
            law = kwargs["probabilities"]
            expected, _ = masked_law(
                net.apply(phase_params(current_params[0], "actor"), raw.observation, method=net.get_action_logits),
                raw.action_mask,
                cfg.temperature,
            )
            expected = jnp.where(raw.is_learner[:, None], expected, raw.behavior_probs)
            np.testing.assert_allclose(law, expected, rtol=1e-6)
            real = np.asarray(raw.executed)
            rows = np.asarray(raw.observation["row_id"])[real]
            np.testing.assert_array_equal(np.asarray(q)[real], old_q_host[rows])
            label_calls.append(1)
        return original_qboost(raw, q, **kwargs)

    # This test counts TRAINING trace rebuilds, not the final diagnostics pass.
    monkeypatch.setattr(updater, "_diagnostics", lambda *args: {})
    monkeypatch.setattr(updater, "_current_labels", tracked_labels)
    monkeypatch.setattr(update_module, "qboost_labels", tracked_qboost)
    result = updater.update(case["state"], case["rng"], case["batch"], q_old=q_old)
    assert len(calls) == len(label_calls) == result.metrics["actor_steps"] == 4
    assert all(changed(a, b) for a, b in zip(calls, calls[1:]))
    # Same complete game reappears in epoch two: fixed rollout Q, NEW future A.
    assert any(len(values) == 2 and not np.array_equal(values[0], values[1]) for values in advantages_by_row.values())


def actor_data(case, rows=4):
    raw = case["batch"].transitions
    indices = jnp.array([0, 3, 4, 8])[:rows]
    obs = jax.tree.map(lambda x: x[indices], raw.observation)
    mask = raw.action_mask[indices]
    probs, _ = masked_law(
        case["network"].apply(case["state"].params, obs, method=case["network"].get_action_logits), mask
    )
    return (
        obs,
        mask,
        raw.action[indices],
        probs,
        jnp.array([1.0, 3.0, -2.0, 5.0])[:rows],
        jnp.array([1.0, 1.0, 1.0, 0.0])[:rows],
        jnp.ones((rows, 87)) / 87,
    )


def test_advantage_stop_gradient_matches_score_gradient_and_per_game_not_row_reduction(case):
    net = case["network"]
    params = phase_params(case["state"].params, "actor")
    data = actor_data(case)

    def objective(p):
        return _actor_loss(p, net, data, jnp.float32(2), temperature=1.0, clip=0.2, alpha=0.0, bc_coef=0.0)

    value, actual = jax.value_and_grad(objective)(params)
    # At pi=mu, unclipped ratios are one: SUM learner rows / NUMBER games.
    assert float(value) == pytest.approx(-float(jnp.sum(data[4] * data[5])) / 2, abs=1e-6)
    assert float(value) != pytest.approx(-float(jnp.sum(data[4] * data[5])) / 3, abs=1e-6)

    def score(p):
        _, logp = masked_law(net.apply(p, data[0], method=net.get_action_logits), data[1])
        selected = jnp.take_along_axis(logp, data[2][:, None], axis=-1)[:, 0]
        return -jnp.sum(data[5] * jax.lax.stop_gradient(data[4]) * selected) / 2

    tree_close(actual, jax.grad(score)(params), atol=2e-6, rtol=2e-5)
    advantage_grad = jax.grad(
        lambda a: _actor_loss(
            params, net, data[:4] + (a,) + data[5:], 2.0, temperature=1.0, clip=0.2, alpha=0.0, bc_coef=0.0
        )
    )(data[4])
    np.testing.assert_array_equal(advantage_grad, 0.0)

    # Even an advantage expression depending on Actor weights is treated as fixed.
    def dynamic_advantage(p):
        baseline = jnp.mean(p["params"]["policy_mlp"]["kernel"])
        return _actor_loss(
            p,
            net,
            data[:4] + (data[4] + baseline - jax.lax.stop_gradient(baseline),) + data[5:],
            2.0,
            temperature=1.0,
            clip=0.2,
            alpha=0.0,
            bc_coef=0.0,
        )

    tree_close(actual, jax.grad(dynamic_advantage)(params), atol=2e-6, rtol=2e-5)
    micro_value, micro_grad = case["updater"].actor_grad(params, data, jnp.float32(2), jnp.float32(0.2), jnp.float32(0))
    np.testing.assert_allclose(micro_value, value, atol=1e-6)
    tree_close(micro_grad, actual, atol=2e-6, rtol=2e-5)


def test_tempo_anchor_mixes_surrogate_with_kullback_anchor(case):
    net, params, data = case["network"], phase_params(case["state"].params, "actor"), actor_data(case)
    kw = dict(game_count=jnp.float32(2), temperature=1.0, clip=0.2, alpha=0.3, bc_coef=0.0)
    plain = _actor_loss(params, net, data, **kw)
    unit = _actor_loss(params, net, data, tempo_keep=jnp.float32(1.0), **kw)
    assert float(unit) == pytest.approx(float(plain), abs=1e-6)
    # At mu == pi the ratio is one and the anchor KL is zero, so only the
    # surrogate shrinks: loss = sum(weight * (K*(-adv) + alpha*uniform)) / games.
    _, logp = masked_law(net.apply(params, data[0], method=net.get_action_logits), data[1])
    count = jnp.maximum(jnp.sum(data[1], axis=-1), 1)
    uniform = jnp.sum(data[3] * (logp + jnp.log(count[:, None])), axis=-1)
    expected = jnp.sum(data[5] * (0.5 * (-jax.lax.stop_gradient(data[4])) + 0.3 * uniform)) / 2
    half = _actor_loss(params, net, data, tempo_keep=jnp.float32(0.5), **kw)
    assert float(half) == pytest.approx(float(expected), abs=1e-6)
    zero = _actor_loss(params, net, data, tempo_keep=jnp.float32(0.0), **kw)
    expected_zero = jnp.sum(data[5] * (0.3 * uniform)) / 2
    assert float(zero) == pytest.approx(float(expected_zero), abs=1e-6)
    grad = jax.grad(lambda p: _actor_loss(p, net, data, tempo_keep=jnp.float32(0.5), **kw))(params)
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(grad))


def test_critic_only_sampled_action_three_seats_explicit_denominator_and_target_stop_gradient(case):
    raw, net = case["batch"].transitions, case["network"]
    inputs = {
        "observation": jax.tree.map(lambda x: x[:3], raw.observation),
        "privileged": jax.tree.map(lambda x: x[:3], raw.critic_observation),
    }
    action, target, weight = jnp.ones(3, jnp.int32), jnp.array([[1.0, 2.0, 3.0]] * 3), jnp.array([1.0, 1.0, 0.0])
    critic = phase_params(case["state"].params, "critic")
    data = (inputs, action, target, weight)
    value, grad = jax.value_and_grad(lambda p: _critic_loss(p, net, data, 6.0))(critic)
    q = net.apply(critic, inputs, method=net.get_action_q)
    expected = jnp.sum(weight[:, None] * (q[:, :, 1] - target) ** 2) / 6
    np.testing.assert_allclose(value, expected, atol=1e-6)
    output_bias_grad = np.asarray(grad["params"]["value_critic_mlp"]["bias"]).reshape(3, 87)
    assert np.any(output_bias_grad[:, 1] != 0)
    np.testing.assert_array_equal(np.delete(output_bias_grad, 1, axis=1), 0.0)
    target_grad = jax.grad(lambda t: _critic_loss(critic, net, (inputs, action, t, weight), 6.0))(target)
    np.testing.assert_array_equal(target_grad, 0.0)


def test_exact_legal_kl_and_late_epoch_rollback_all_actor_slots_critic_commits(case, monkeypatch):
    updater, raw, net = case["updater"], case["batch"].transitions, case["network"]
    actor = phase_params(case["state"].params, "actor")
    shifted = jax.tree.map(lambda x: x + 0.15, actor)
    full = replace_phase(case["state"].params, shifted, "actor")
    _, logp = masked_law(net.apply(shifted, raw.observation, method=net.get_action_logits), raw.action_mask)
    old = raw.behavior_probs
    row_kl = jnp.sum(old * (jnp.log(jnp.maximum(old, 1e-30)) - logp), axis=-1)
    valid = raw.executed & raw.is_learner & raw.is_decision
    expected = jnp.sum(jnp.where(valid, row_kl, 0.0)) / valid.sum()
    np.testing.assert_allclose(updater._kl(full, case["batch"]), expected, rtol=1e-6, atol=1e-7)
    assert float(expected) > 0.0
    # First mini commits locally; second violates KL, rolling the WHOLE epoch back.
    calls = []

    def late_rejection(params, batch):
        calls.append(1)
        return jnp.float32(0.0 if len(calls) == 1 else case["config"].target_kl * 2)

    monkeypatch.setattr(updater, "_kl", late_rejection)
    state = case["state"]
    for own in ("actor", "critic"):
        grads = jax.tree.map(lambda x: jnp.full_like(x, 0.05), phase_params(state.params, own))
        step = updater.actor_opt_step if own == "actor" else updater.critic_opt_step
        updated, slots = step(state.params, grads, getattr(state, f"{own}_opt_state"), jnp.float32(1))
        state = state.replace(params=replace_phase(state.params, updated, own), **{f"{own}_opt_state": slots})
    state = state.replace(actor_steps=jnp.int32(7), critic_steps=jnp.int32(11))
    result = updater.update(state, case["rng"], case["batch"])
    assert len(calls) == 2 and result.metrics["rejected_epochs"] == 1
    assert result.metrics["actor_steps"] == 0 and int(result.state.actor_steps) == 7
    tree_equal(phase_params(state.params, "actor"), phase_params(result.state.params, "actor"))
    tree_equal(state.actor_opt_state, result.state.actor_opt_state)
    assert changed(phase_params(state.params, "critic"), phase_params(result.state.params, "critic"))
    assert changed(state.critic_opt_state, result.state.critic_opt_state)
    assert int(result.state.critic_steps) == 13 and int(result.state.completed_updates) == 1


@pytest.mark.parametrize("phase", ["actor", "critic"])
def test_nonfinite_any_phase_rejects_whole_iteration_prior_unchanged(case, monkeypatch, phase):
    updater, state, rng = case["updater"], case["state"], case["rng"]
    before, rng_before = jax.device_get(state), np.asarray(rng).copy()
    prior_params, prior_actor, prior_critic = state.params, state.actor_opt_state, state.critic_opt_state
    name = f"{phase}_grad"
    original, calls = getattr(updater, name), []

    def corrupt(*args):
        loss, grads = original(*args)
        calls.append(1)
        leaves, structure = jax.tree.flatten(grads)
        leaves[0] = jnp.full_like(leaves[0], jnp.nan)
        return loss, jax.tree.unflatten(structure, leaves)

    monkeypatch.setattr(updater, name, corrupt)
    with pytest.raises(FloatingPointError, match="nonfinite"):
        updater.update(state, rng, case["batch"])
    assert calls
    tree_equal(state, before)
    np.testing.assert_array_equal(rng, rng_before)
    assert (
        state.params is prior_params and state.actor_opt_state is prior_actor and state.critic_opt_state is prior_critic
    )


def test_schema5_resume_exact_next_actual_toy_optimizer_boundary(case):
    updater, cfg, net = case["updater"], case["config"], case["network"]
    first = updater.update(case["state"], case["rng"], case["batch"])
    contract = {
        "network_config": {"architecture": "v5", "num_layers": 1, "d_model": 8, "num_heads": 2, "mlp_dim": 8},
        "head_config": {"version": 0},
        "critic_config": dict(DEFAULT_CENTRAL_Q_CRITIC_CONFIG),
        "reading_config": dict(DEFAULT_READING_CONFIG),
        "algorithm_config": replace(
            cfg, version=5, kind=READING_ALGORITHM_KIND, reading_auxiliary=True
        ).algorithm_config(),
        "reward_config": default_reward_config("sanma"),
        "optimizer_config": vrpo_optimizer_metadata(cfg),
        "environment_config": {"env_name": "sanma", "round_mode": "half", "ruleset": RULESET},
        "source_versions": {"implementation": "synthetic-test-v1", "jax": jax.__version__},
        "retention_enabled": False,
    }
    payload = build_vrpo_resume(
        first.state,
        first.rng,
        contract=contract,
        progress={"completed_updates": 1, "envsteps": 18, "schedule_iteration": 1},
        collection_state={"version": 1, "kind": "current_joint_all_seats", "next_game_id": cfg.rollout_games},
    )
    # In-memory synthetic bytes only, no checkpoint/model files.
    payload = pickle.loads(pickle.dumps(payload))
    resumed = restore_vrpo_resume(payload, case["state"], contract=contract)
    next_batch = make_batch(net, first.state.params)
    expected = updater.update(first.state, first.rng, next_batch)
    actual = updater.update(resumed.state, resumed.rng, next_batch)
    tree_equal(expected.state, actual.state)
    np.testing.assert_array_equal(expected.rng, actual.rng)
    assert expected.metrics == actual.metrics and int(actual.state.completed_updates) == 2


def test_trajectory_count_bc_reference_and_byte_admission_fail_before_update(case):
    cfg, net, state, batch = case["config"], case["network"], case["state"], case["batch"]
    two = make_batch(net, state.params, games=2)
    with pytest.raises(ValueError, match="configured number"):
        case["updater"].update(state, case["rng"], two)
    missing = VRPOUpdater(net, replace(cfg, bc_reference_coef=0.1))
    with pytest.raises(ValueError, match="explicit frozen BC Actor"):
        missing.update(state, case["rng"], batch)
    missing.close()
    limited = VRPOUpdater(net, replace(cfg, max_bytes=1))
    with pytest.raises(MemoryError, match="byte budget"):
        limited.update(state, case["rng"], batch)
    limited.close()


def test_final_semantic_diagnostics_use_committed_policy_and_scalar_reductions(case):
    updater, cfg = case["updater"], case["config"]
    result = updater.update(case["state"], case["rng"], case["batch"])
    metrics, raw = result.metrics, case["batch"].transitions
    assert metrics["raw_rows"] == int(raw.executed.sum())
    assert metrics["actor_rows"] == int((raw.executed & raw.is_learner & raw.is_decision).sum())
    assert metrics["rollout_games"] == cfg.rollout_games
    assert metrics["actor_final_loss"] == pytest.approx(
        sum(
            metrics[key]
            for key in ("actor_final_surrogate_loss", "actor_final_uniform_loss", "actor_final_reference_loss")
        )
    )
    assert metrics["actor_final_kl"] == pytest.approx(float(updater._kl(result.state.params, case["batch"])), abs=2e-7)
    assert all(np.isscalar(value) and np.isfinite(value) for value in metrics.values())
    for name in ("q_taken", "q_target", "advantage"):
        assert metrics[name + "_absmax"] >= metrics[name + "_rms"] >= 0.0
    assert not any("cache" in key or "misses" in key for key in metrics)


def test_flat_critic_training_denominator_counts_games_not_raw_rows(case, monkeypatch):
    updater, cfg = case["updater"], case["config"]
    original, seen = updater.critic_grad, []

    def tracked(params, data, denominator):
        seen.append(float(denominator))
        assert float(denominator) == 6 * cfg.games_per_minibatch
        return original(params, data, denominator)

    monkeypatch.setattr(updater, "critic_grad", tracked)
    updater.update(case["state"], case["rng"], case["batch"])
    assert len(seen) == cfg.critic_epochs * cfg.trajectory_minibatches
