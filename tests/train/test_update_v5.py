"""Tiny V5 Actor-reading transaction oracles; CPU tests, not training runs."""

import gc
import weakref
from dataclasses import replace

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.core import freeze, unfreeze

from tests.train.test_update import changed, make_batch, tree_close, tree_equal
from tests.train.test_update_paged import forced, paged, unpage

from train.auxhead import TASK_NAMES, ReadingTargets, reading_target_counts
from train.configs import READING_ALGORITHM_KIND, tiny_debug_preset
from train.update import (
    PagedVRPOUpdater,
    VRPOUpdater,
    _actor_loss,
    create_training_state,
    damping,
    masked_law,
    phase_params,
    public_q_input,
    qboost_labels,
    replace_phase,
)


class ReadingPolicy(nn.Module):
    @nn.compact
    def __call__(self, features, *, with_reading=False):
        logits = nn.Dense(87, name="action")(features)
        if self.is_initializing():
            nn.Dense(4, name="auxiliary")(features)
        if with_reading or self.is_initializing():
            output = nn.Dense(214, name="reading")(features)
            reading = dict(
                tenpai_logits=output[:, :2],
                shanten_logits=output[:, 2:10].reshape((-1, 2, 4)),
                danger_logits=output[:, 10:78].reshape((-1, 2, 34)),
                waits_logits=output[:, 78:146].reshape((-1, 2, 34)),
                danger_points=output[:, 146:].reshape((-1, 2, 34)),
            )
        return (logits, reading) if with_reading else logits


class ReadingCentral(nn.Module):
    """Four actual Flax roots, shared Actor encoder, separate frozen own aux."""

    def setup(self):
        self.policy_extractor = nn.Dense(5)
        self.policy_mlp = ReadingPolicy()
        self.critic_extractor = nn.Dense(5)
        self.value_critic_mlp = nn.Dense(3 * 87)

    def _features(self, observation):
        # Any attempt to inject a target or privileged field into Actor fails.
        assert set(observation) == {"x", "row_id"}
        return nn.tanh(self.policy_extractor(observation["x"]))

    def get_action_logits(self, observation):
        return self.policy_mlp(self._features(observation))

    def get_action_and_reading(self, observation):
        return self.policy_mlp(self._features(observation), with_reading=True)

    def get_action_q(self, central):
        assert set(central) == {"observation", "privileged"}
        x = jnp.concatenate((central["observation"]["x"], central["privileged"]["hidden"]), -1)
        return self.value_critic_mlp(nn.tanh(self.critic_extractor(x))).reshape((-1, 3, 87))

    def __call__(self, observation, central):
        return self.get_action_logits(observation), self.get_action_q(central)


def targets_for_rows(n):
    """Sparse unequal counts, unknown payloads and legal positive-only point cells."""
    row = np.arange(n)[:, None]
    seat = np.arange(2)[None, :]
    seat_mask = (row + seat) % 3 != 1
    shape = (n, 2, 34)
    exists = np.ones(34, bool)
    exists[1:8] = False
    tiles = np.arange(34)[None, None, :]
    tile_mask = np.broadcast_to(exists, shape) & ((row[..., None] + tiles) % 4 != 0)
    waits = np.broadcast_to((tiles == 8) | (tiles == 9), shape).copy()
    danger = np.broadcast_to(tiles == 8, shape).copy()
    point_mask = tile_mask & danger
    points = np.where(point_mask, 0.2 + 0.03 * row[..., None], np.nan).astype(np.float32)
    points[:, :, 1:8] = 0
    return ReadingTargets(
        jnp.ones((n, 2), bool),
        jnp.asarray(np.where(seat_mask, (row + seat) % 4, 99), jnp.int32),
        jnp.asarray(danger),
        jnp.asarray(waits),
        jnp.asarray(points),
        jnp.asarray(seat_mask),
        jnp.asarray(seat_mask & ((row + seat) % 2 == 0)),
        jnp.asarray(tile_mask),
        jnp.asarray(tile_mask & seat_mask[..., None]),
        jnp.asarray(point_mask),
    )


def setup(**overrides):
    cfg = tiny_debug_preset(
        **(
            dict(
                version=5,
                kind=READING_ALGORITHM_KIND,
                reading_auxiliary=True,
                reading_aux_coef=0.01,
                execution_block_steps=2,
                storage_page_rows=6,
                actor_microbatch_size=2,
                critic_microbatch_size=2,
                inference_microbatch_size=2,
                actor_epochs=1,
                critic_epochs=1,
                actor_lr=0.001,
                critic_lr=0.002,
                weight_decay=0.05,
                target_kl=10.0,
                time_bucket=2,
                muon_ns_steps=1,
                muon_ns_dtype="float32",
            )
            | overrides
        )
    )
    net = ReadingCentral()
    obs = {"x": jnp.ones((2, 6)), "row_id": jnp.arange(2)}
    central = {"observation": obs, "privileged": {"hidden": jnp.ones((2, 2))}}
    params = freeze(net.init(jax.random.key(551), obs, central))
    state, _, _ = create_training_state(params, cfg)
    batch = make_batch(net, params)
    raw = batch.transitions
    law, _ = masked_law(net.apply(params, raw.observation, method=net.get_action_logits), raw.action_mask)
    raw = raw._replace(
        behavior_probs=law,
        is_learner=jnp.ones_like(raw.is_learner),
        policy_id=jnp.zeros_like(raw.policy_id),
        reading_targets=targets_for_rows(raw.executed.shape[0]),
    )
    batch = batch._replace(transitions=raw, learner_seats=jnp.full_like(batch.learner_seats, -1))
    return net, cfg, state, batch


def reading_reference(predictions, targets, active):
    """Independent five numerators/counts (no production loss/count helper)."""
    numerators, counts = [], []
    for name in TASK_NAMES:
        valid = getattr(targets, name + "_valid")
        mask = valid & active.reshape(active.shape + (1,) * (valid.ndim - 1))
        y = jnp.where(mask, getattr(targets, name), 0)
        key = name if name == "danger_points" else name + "_logits"
        p = jnp.where(mask[..., None] if name == "shanten" else mask, predictions[key], 0.0)
        if name == "shanten":
            loss = -jnp.take_along_axis(jax.nn.log_softmax(p), y[..., None], -1)[..., 0]
        elif name == "danger_points":
            loss = (p - y) ** 2
        else:
            loss = jax.nn.softplus(p) - y * p
        numerators.append(jnp.sum(jnp.where(mask, loss, 0.0)))
        counts.append(jnp.sum(mask, dtype=jnp.float32))
    return jnp.stack(numerators), jnp.stack(counts)


def joint_reference(actor, net, data, games, cfg, schedule):
    obs, mask, action, old, advantage, weight, _, targets = data
    logits, prediction = net.apply(actor, obs, method=net.get_action_and_reading)
    legal_logits = jnp.where(mask, logits / cfg.temperature, -1e30)
    logp = jax.nn.log_softmax(legal_logits)
    law = jnp.where(mask, jnp.exp(logp), 0.0)
    taken = jnp.take_along_axis(logp, action[:, None], -1)[:, 0]
    prior = jnp.take_along_axis(old, action[:, None], -1)[:, 0]
    ratio = jnp.exp(taken) / prior
    advantage = jax.lax.stop_gradient(advantage)
    surrogate = jnp.minimum(ratio * advantage, jnp.clip(ratio, 1 - schedule.clip, 1 + schedule.clip) * advantage)
    uniform = jnp.sum(jnp.where(mask, law * (logp + jnp.log(mask.sum(-1)[:, None])), 0.0), -1)
    sums, counts = reading_reference(prediction, targets, weight > 0)
    reading = jnp.sum(sums / jnp.maximum(counts, 1))
    return jnp.sum(weight * (-surrogate + schedule.uniform_alpha * uniform)) / games + cfg.reading_aux_coef * reading


def actor_data(net, cfg, params, batch, q_old):
    raw = batch.transitions
    logits = net.apply(phase_params(params, "actor"), raw.observation, method=net.get_action_logits)
    law, _ = masked_law(logits, raw.action_mask, cfg.temperature)
    labels = qboost_labels(
        raw,
        q_old,
        probabilities=law,
        gamma=cfg.gamma,
        trace_lambda=cfg.trace_lambda,
        game_offsets=batch.game_offsets,
        game_lengths=batch.game_lengths,
    )
    own = jnp.take_along_axis(labels.advantages, raw.current_player[:, None], -1)[:, 0]
    return (
        raw.observation,
        raw.action_mask,
        raw.action,
        raw.behavior_probs,
        own,
        labels.actor_mask.astype(jnp.float32),
        jnp.ones_like(law) / 87,
        raw.reading_targets,
    )


def test_joint_loss_gradient_matches_independent_reference_and_micro_partition():
    net, cfg, state, batch = setup()
    raw = batch.transitions
    q_old = net.apply(phase_params(state.params, "critic"), public_q_input(raw), method=net.get_action_q)
    # Recompute full traces under a NEW policy, keeping Q-old/behavior fixed.
    shifted = unfreeze(state.params)
    shifted["params"]["policy_mlp"]["action"]["bias"] += jnp.linspace(-0.1, 0.1, 87)
    shifted["params"]["policy_extractor"]["bias"] += 0.03
    params = freeze(shifted)
    data = actor_data(net, cfg, params, batch, q_old)
    old_data = actor_data(net, cfg, state.params, batch, q_old)
    assert changed(data[4], old_data[4])
    actor, schedule = phase_params(params, "actor"), damping(cfg, 1)
    expected, gradient = jax.value_and_grad(lambda p: joint_reference(p, net, data, 4.0, cfg, schedule))(actor)
    counts = reading_target_counts(data[7], data[5] > 0)
    actual, direct_grad = jax.value_and_grad(
        lambda p: _actor_loss(
            p, net, data, 4.0, cfg.temperature, schedule.clip, schedule.uniform_alpha, 0.0, cfg.reading_aux_coef, counts
        )
    )(actor)
    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-6)
    tree_close(direct_grad, gradient, atol=3e-6, rtol=3e-5)
    assert changed(
        gradient["params"]["policy_mlp"]["reading"],
        jax.tree.map(jnp.zeros_like, gradient["params"]["policy_mlp"]["reading"]),
    )
    tree_equal(
        gradient["params"]["policy_mlp"]["auxiliary"],
        jax.tree.map(jnp.zeros_like, gradient["params"]["policy_mlp"]["auxiliary"]),
    )
    # Explicit invalid borrowed rows make every partition sum exactly once.
    n = data[5].shape[0]
    padded = jax.tree.map(lambda x: jnp.concatenate((x, jnp.repeat(x[:1], 24 - n, axis=0))), data)
    padded = padded[:5] + (padded[5].at[n:].set(0.0),) + padded[6:]
    for micro in (1, 2, 3, 4):
        updater = VRPOUpdater(net, replace(cfg, actor_microbatch_size=micro, storage_page_rows=12))
        try:
            loss, grads = updater.actor_grad(
                actor, padded, jnp.float32(4), schedule.clip, schedule.uniform_alpha, counts
            )
            np.testing.assert_allclose(loss, expected, atol=3e-6, rtol=3e-6)
            tree_close(grads, gradient, atol=3e-6, rtol=3e-5)
        finally:
            updater.close()
    stopped = jax.grad(
        lambda a: _actor_loss(
            actor,
            net,
            data[:4] + (a,) + data[5:],
            4.0,
            cfg.temperature,
            schedule.clip,
            schedule.uniform_alpha,
            0.0,
            cfg.reading_aux_coef,
            counts,
        )
    )(data[4])
    np.testing.assert_array_equal(stopped, 0.0)


@pytest.mark.parametrize("interleave", [False, True])
def test_cross_page_minibatch_uses_global_counts_and_current_policy(interleave):
    net, cfg, state, flat = setup()
    pages = paged(flat, cfg.page_rows, interleave=interleave)
    updater = PagedVRPOUpdater(net, cfg)
    try:
        q_old = updater.old_q(state, pages)
        q_full = jnp.asarray(unpage(pages, q_old))
        actor = phase_params(state.params, "actor")
        changed_actor = jax.tree.map(lambda x: x + 0.02, actor)
        params = replace_phase(state.params, changed_actor, "actor")
        schedule = damping(cfg, 1)
        original_data = actor_data(net, cfg, params, flat, q_full)
        row_games = np.repeat(np.arange(4), np.asarray(flat.game_lengths))
        for indices in (np.arange(4), np.array([0, 2])):
            row_selected = jnp.asarray(np.isin(row_games, indices))
            data = original_data[:5] + (original_data[5] * row_selected,) + original_data[6:]
            expected, grad = jax.value_and_grad(
                lambda p: joint_reference(p, net, data, float(len(indices)), cfg, schedule)
            )(changed_actor)
            value, actual, count = updater.actor_minibatch(params, pages, q_old, indices, schedule)
            np.testing.assert_allclose(value, expected, atol=3e-6, rtol=3e-6)
            tree_close(actual, grad, atol=3e-6, rtol=3e-5)
            assert int(count) == int(data[5].sum())
            selected = jnp.asarray(np.isin(np.arange(4), indices))
            global_counts = sum(
                (updater.reading_counts(page, selected) for page in pages.pages), jnp.zeros(5, jnp.float32)
            )
            np.testing.assert_array_equal(global_counts, reading_target_counts(data[7], data[5] > 0))
    finally:
        updater.close()


def assert_committed_reading(metrics, net, params, flat, coef):
    raw = flat.transitions
    _, predictions = net.apply(phase_params(params, "actor"), raw.observation, method=net.get_action_and_reading)
    sums, counts = reading_reference(predictions, raw.reading_targets, raw.executed & raw.is_learner & raw.is_decision)
    losses = np.asarray(sums) / np.maximum(np.asarray(counts), 1)
    for name, value, count in zip(TASK_NAMES, losses, np.asarray(counts)):
        assert metrics[f"reading_{name}_loss"] == pytest.approx(float(value), abs=3e-6)
        assert metrics[f"reading_{name}_valid_cells"] == count
    assert metrics["actor_final_reading_loss"] == pytest.approx(coef * losses.sum(), abs=3e-6)
    assert metrics["actor_final_loss"] == pytest.approx(
        sum(
            metrics[k]
            for k in (
                "actor_final_surrogate_loss",
                "actor_final_uniform_loss",
                "actor_final_reference_loss",
                "actor_final_reading_loss",
            )
        ),
        abs=3e-6,
    )


def test_flat_paged_update_reading_encoder_train_own_aux_frozen_and_critic_independent():
    net, cfg, state, flat = setup()
    momentum = unfreeze(state.actor_opt_state.momentum)
    momentum["params"]["policy_mlp"]["auxiliary"] = jax.tree.map(
        lambda x: jnp.full_like(x, 0.11), momentum["params"]["policy_mlp"]["auxiliary"]
    )
    state = state.replace(actor_opt_state=state.actor_opt_state._replace(momentum=freeze(momentum)))
    updaters = [VRPOUpdater(net, cfg), VRPOUpdater(net, cfg), VRPOUpdater(net, replace(cfg, reading_aux_coef=0.07))]
    try:
        rng = jax.random.key(552)
        a = updaters[0].update(state, rng, flat)
        b = updaters[1].update(state, rng, paged(flat, cfg.page_rows))
        c = updaters[2].update(state, rng, flat)
        tree_close(a.state, b.state, atol=4e-6, rtol=4e-5)
        for key in a.metrics:
            assert b.metrics[key] == pytest.approx(a.metrics[key], abs=4e-6, rel=4e-5)
        for root in ("policy_extractor", "policy_mlp"):
            assert changed(state.params["params"][root], a.state.params["params"][root])
        assert changed(
            state.params["params"]["policy_mlp"]["reading"], a.state.params["params"]["policy_mlp"]["reading"]
        )
        tree_equal(
            state.params["params"]["policy_mlp"]["auxiliary"], a.state.params["params"]["policy_mlp"]["auxiliary"]
        )
        tree_equal(
            state.actor_opt_state.momentum["params"]["policy_mlp"]["auxiliary"],
            a.state.actor_opt_state.momentum["params"]["policy_mlp"]["auxiliary"],
        )
        # A different reading coefficient changes Actor, never sampled-action Q supervision.
        assert changed(phase_params(a.state.params, "actor"), phase_params(c.state.params, "actor"))
        tree_equal(phase_params(a.state.params, "critic"), phase_params(c.state.params, "critic"))
        tree_equal(a.state.critic_opt_state, c.state.critic_opt_state)
        assert_committed_reading(a.metrics, net, a.state.params, flat, cfg.reading_aux_coef)
        assert_committed_reading(b.metrics, net, b.state.params, flat, cfg.reading_aux_coef)
    finally:
        for updater in updaters:
            updater.close()


@pytest.mark.parametrize("layout", ["flat", "paged"])
def test_late_epoch_rejection_restores_reading_historical_slots_and_committed_diagnostics(layout, monkeypatch):
    net, cfg, state, flat = setup()
    updater = VRPOUpdater(net, cfg) if layout == "flat" else PagedVRPOUpdater(net, cfg)
    try:
        # Warm actual historical momentum, including the otherwise frozen own aux.
        momentum = jax.tree.map(lambda x: jnp.full_like(x, 0.07), state.actor_opt_state.momentum)
        state = state.replace(
            actor_opt_state=state.actor_opt_state._replace(count=jnp.int32(9), momentum=momentum),
            actor_steps=jnp.int32(9),
            critic_steps=jnp.int32(4),
        )
        before = jax.device_get(state)
        calls = []

        def gate(*args):
            calls.append(1)
            return jnp.float32(0.0 if len(calls) == 1 else cfg.target_kl * 2)

        monkeypatch.setattr(updater, "_kl", gate)
        batch = flat if layout == "flat" else paged(flat, cfg.page_rows)
        result = updater.update(state, jax.random.key(553), batch)
        assert len(calls) == 2 and result.metrics["rejected_epochs"] == 1
        assert result.metrics["actor_attempts"] == 2 and result.metrics["actor_steps"] == 0
        tree_equal(phase_params(state.params, "actor"), phase_params(result.state.params, "actor"))
        tree_equal(state.actor_opt_state, result.state.actor_opt_state)
        tree_equal(state.actor_steps, result.state.actor_steps)
        assert result.metrics["critic_steps"] == 2
        assert changed(state.critic_opt_state, result.state.critic_opt_state)
        assert changed(phase_params(state.params, "critic"), phase_params(result.state.params, "critic"))
        assert_committed_reading(result.metrics, net, state.params, flat, cfg.reading_aux_coef)
        tree_equal(state, before)
    finally:
        updater.close()


@pytest.mark.parametrize("layout", ["flat", "paged"])
def test_forced_only_does_not_train_reading_or_advance_actor(layout):
    net, cfg, state, flat = setup()
    flat = forced(flat)
    batch = flat if layout == "flat" else paged(flat, cfg.page_rows)
    updater = VRPOUpdater(net, cfg) if layout == "flat" else PagedVRPOUpdater(net, cfg)
    try:
        result = updater.update(state, jax.random.key(554), batch)
        tree_equal(phase_params(state.params, "actor"), phase_params(result.state.params, "actor"))
        tree_equal(state.actor_opt_state, result.state.actor_opt_state)
        assert result.metrics["actor_steps"] == 0 and result.metrics["critic_steps"] == 2
        assert result.metrics["actor_final_reading_loss"] == 0.0
        for task in TASK_NAMES:
            assert result.metrics[f"reading_{task}_valid_cells"] == 0.0
    finally:
        updater.close()


@pytest.mark.parametrize("layout", ["flat", "paged"])
@pytest.mark.parametrize("problem", ["missing", "bad_schema", "nonfinite"])
def test_reading_target_failure_retains_whole_prior_boundary(layout, problem):
    net, cfg, state, flat = setup()
    raw, t = flat.transitions, flat.transitions.reading_targets
    if problem == "missing":
        t = None
    elif problem == "bad_schema":
        t = t._replace(shanten=t.shanten.astype(jnp.float32))
    else:
        # Row1/seat0/tile8 is an active point label, not masked unknown.
        t = t._replace(danger_points=t.danger_points.at[1, 0, 8].set(jnp.nan))
    flat = flat._replace(transitions=raw._replace(reading_targets=t))
    batch = flat if layout == "flat" else paged(flat, cfg.page_rows)
    updater = VRPOUpdater(net, cfg) if layout == "flat" else PagedVRPOUpdater(net, cfg)
    before = jax.device_get(state)
    try:
        with pytest.raises((ValueError, FloatingPointError)):
            updater.update(state, jax.random.key(555), batch)
        tree_equal(state, before)
    finally:
        updater.close()


def test_flat_committed_diagnostics_do_not_repeat_borrowed_padding_labels():
    net, cfg, state, flat = setup(actor_microbatch_size=4, storage_page_rows=12)
    # make_batch has18 rows; diagnostic micro4 needs two borrowed placeholders.
    assert flat.transitions.executed.shape[0] == 18
    updater = VRPOUpdater(net, cfg)
    try:
        result = updater.update(state, jax.random.key(556), flat)
        assert_committed_reading(result.metrics, net, result.state.params, flat, cfg.reading_aux_coef)
    finally:
        updater.close()


@pytest.mark.parametrize("layout", ["flat", "paged"])
def test_reading_sidecars_are_charged_before_update(layout):
    net, cfg, _, flat = setup()
    batch = flat if layout == "flat" else paged(flat, cfg.page_rows)
    updater_cls = VRPOUpdater if layout == "flat" else PagedVRPOUpdater
    updater = updater_cls(net, cfg)
    try:
        charge = updater.admit(batch)
    finally:
        updater.close()
    limited = updater_cls(net, replace(cfg, max_bytes=charge - 1))
    try:
        with pytest.raises(MemoryError, match="byte budget"):
            limited.admit(batch)
        assert all(info["misses"] == 0 for info in limited.cache_statistics().values())
    finally:
        limited.close()


@pytest.mark.parametrize("layout", ["flat", "paged"])
def test_new_reading_kernels_close_cache_and_owner_releases_without_cyclic_gc(layout):
    net, cfg, state, flat = setup()
    updater = VRPOUpdater(net, cfg) if layout == "flat" else PagedVRPOUpdater(net, cfg)
    raw = flat.transitions
    prior_gc = gc.isenabled()
    gc.disable()
    try:
        actor = phase_params(state.params, "actor")
        updater.reading_diagnostic_sums(actor, raw)
        assert updater.reading_diagnostic_sums.cache_info().misses == 1
        if layout == "paged":
            page = paged(flat, cfg.page_rows).pages[0]
            updater.reading_counts(page, jnp.ones(4, bool))
            assert updater.reading_counts.cache_info().misses == 1
        kernels = [updater.reading_diagnostic_sums]
        if layout == "paged":
            kernels.append(updater.reading_counts)
        reference = weakref.ref(updater)
        updater.close()
        assert all(kernel.cache_info().size == 0 for kernel in kernels)
        del updater
        assert reference() is None
    finally:
        if prior_gc:
            gc.enable()
