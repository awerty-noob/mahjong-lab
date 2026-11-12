"""CPU-only page/flat numerical oracles and post-warmup zero-retrace proofs."""
from collections import Counter
from dataclasses import replace
from functools import wraps

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.train.test_update import TinyCentral, make_batch, tree_close, tree_equal

from train.configs import PagedRawQBatch, RawQBatch, RawQPage, tiny_debug_preset
from train.update import (
    PagedVRPOUpdater,
    VRPOUpdater,
    create_training_state,
    page_resident_bytes,
    page_scratch_bytes,
    phase_params,
    public_q_input,
    qboost_labels,
)


def paged(flat, page_rows, *, interleave=True):
    """Test-only repack. Production must NEVER flatten/transfer feature payloads."""
    lengths = np.asarray(flat.game_lengths)
    offsets = np.asarray(flat.game_offsets)
    order = ([(g, t) for t in range(int(lengths.max())) for g in range(len(lengths)) if t < lengths[g]]
             if interleave else [(g, t) for g in range(len(lengths)) for t in range(lengths[g])])
    pages, directory = [], []
    for start in range(0, len(order), page_rows):
        part = order[start:start + page_rows]
        count = len(part)
        valid = jnp.arange(page_rows) < count
        idx = jnp.asarray([int(offsets[g] + t) for g, t in part] + [0] * (page_rows - count))
        borrowed = jax.tree.map(lambda x: x[idx], flat.transitions)
        raw = jax.tree.map(lambda x: jnp.where(valid.reshape((page_rows,) + (1,) * (x.ndim - 1)),
                                                x, jnp.zeros_like(x)), borrowed)
        raw = raw._replace(observation=borrowed.observation,
                           critic_observation=borrowed.critic_observation)
        ids = jnp.asarray([g for g, _ in part] + [-1] * (page_rows - count), jnp.int32)
        steps = jnp.asarray([t for _, t in part] + [-1] * (page_rows - count), jnp.int32)
        pages.append(RawQPage(raw, ids, steps))
        directory.append(tuple(sorted(set(g for g, _ in part))))
    actors = int(np.asarray(jnp.sum(flat.transitions.executed & flat.transitions.is_learner
                                     & flat.transitions.is_decision)))
    resident = sum(1024 + sum(int(x.nbytes) + 256 for x in jax.tree.leaves(p)) for p in pages)
    resident += 4096 + int(lengths.size) * 32 + sum(256 + 4 * len(ids) for ids in directory)
    return PagedRawQBatch(tuple(pages), flat.game_lengths, flat.learner_seats, tuple(directory),
                         page_rows, int(lengths.sum()), actors, resident)


def unpage(batch, values):
    lengths = np.asarray(batch.game_lengths)
    offsets = np.cumsum(lengths) - lengths
    trailing = values[0].shape[1:]
    result = np.zeros((int(lengths.sum()),) + trailing, np.asarray(values[0]).dtype)
    for page, value in zip(batch.pages, values, strict=True):
        ids, steps, real = map(np.asarray, (page.game_ids, page.game_steps, page.transitions.executed))
        result[offsets[ids[real]] + steps[real]] = np.asarray(value)[real]
    return result


def extend(flat, extra):
    """A long outlier game, not a max_game_steps allocation-sized fixture."""
    lengths = np.asarray(flat.game_lengths).copy()
    offsets = np.asarray(flat.game_offsets)
    idx = []
    for game, (start, length) in enumerate(zip(offsets, lengths)):
        idx.extend([int(start)] * (extra if game == 0 else 0))
        idx.extend(range(int(start), int(start + length)))
    lengths[0] += extra
    raw = jax.tree.map(lambda x: x[jnp.asarray(idx)], flat.transitions)
    return RawQBatch(raw, jnp.asarray(np.cumsum(lengths) - lengths), jnp.asarray(lengths), flat.learner_seats)


def forced(flat):
    raw = flat.transitions
    law = jax.nn.one_hot(raw.action, 87, dtype=jnp.float32)
    return flat._replace(transitions=raw._replace(action_mask=law.astype(jnp.bool_),
                                                  behavior_probs=law,
                                                  is_decision=jnp.zeros_like(raw.is_decision)))


def setup(**overrides):
    cfg = tiny_debug_preset(**(dict(optimizer="muon", execution_block_steps=2, storage_page_rows=6,
                                    actor_microbatch_size=2, critic_microbatch_size=3,
                                    inference_microbatch_size=2, actor_epochs=2, critic_epochs=1,
                                    actor_lr=.003, critic_lr=.004, weight_decay=.1,
                                    target_kl=10., time_bucket=2) | overrides))
    net = TinyCentral()
    obs = {"x": jnp.ones((2, 6)), "row_id": jnp.arange(2)}
    central = {"observation": obs, "privileged": {"hidden": jnp.ones((2, 2))}}
    params = net.init(jax.random.key(18), obs, central)
    state, _, _ = create_training_state(params, cfg)
    flat = make_batch(net, params)
    return net, cfg, state, flat


@pytest.mark.parametrize("interleave", [False, True])
@pytest.mark.parametrize("optimizer", ["muon"])
def test_paged_full_update_matches_flat_and_preserves_boundary(interleave, optimizer):
    net, cfg, state, flat = setup(optimizer=optimizer, muon_ns_steps=1, muon_ns_dtype="float32")
    batch = paged(flat, cfg.page_rows, interleave=interleave)
    a, b = VRPOUpdater(net, cfg), VRPOUpdater(net, cfg)
    before = jax.device_get(state)
    try:
        expected = a.update(state, jax.random.key(942), flat)
        actual = b.update(state, jax.random.key(942), batch)
        tree_close(actual.state, expected.state, atol=3e-6, rtol=3e-5)
        tree_equal(jax.random.key_data(actual.rng), jax.random.key_data(expected.rng))
        assert actual.metrics.keys() == expected.metrics.keys()
        for key in actual.metrics:
            assert actual.metrics[key] == pytest.approx(expected.metrics[key], abs=3e-6, rel=3e-5)
        tree_equal(before, state)
        assert actual.metrics["actor_steps"] == 4 and actual.metrics["critic_steps"] == 2
        assert b._paged is not None
    finally:
        a.close()
        b.close()


def test_page_q_old_and_behavior_targets_equal_full_raw_trace():
    net, cfg, state, flat = setup()
    batch = paged(extend(flat, 17), cfg.page_rows)
    updater = PagedVRPOUpdater(net, cfg)
    extended = extend(flat, 17)
    try:
        q = updater.old_q(state, batch)
        expected_q = net.apply(phase_params(state.params, "critic"), public_q_input(extended.transitions),
                               method=net.get_action_q)
        np.testing.assert_allclose(unpage(batch, q), expected_q, atol=2e-6, rtol=2e-6)
        labels = qboost_labels(extended.transitions, expected_q, game_offsets=extended.game_offsets,
                               game_lengths=extended.game_lengths, trace_lambda=cfg.trace_lambda)
        targets = updater.targets(batch, q)
        np.testing.assert_allclose(unpage(batch, targets), labels.targets, atol=3e-6, rtol=3e-6)
        assert len(q) > 3 and batch.raw_count == int(extended.game_lengths.sum())
    finally:
        updater.close()


def test_current_trace_recomputed_for_each_actor_step_and_q_old_frozen(monkeypatch):
    net, cfg, state, flat = setup()
    batch = paged(flat, cfg.page_rows)
    updater = PagedVRPOUpdater(net, cfg)
    q = updater.old_q(state, batch)
    before_q = tuple(np.asarray(x).copy() for x in q)
    calls, current_laws = [], []
    original = updater.actor_minibatch
    def tracked(params, *args, **kwargs):
        calls.append(jax.device_get(phase_params(params, "actor")))
        return original(params, *args, **kwargs)
    monkeypatch.setattr(updater, "actor_minibatch", tracked)
    original_law = updater.current_law
    def law(params, page, selected):
        value = original_law(params, page, selected)
        current_laws.append(np.asarray(value))
        # Foreign acting laws are byte-frozen, not replaced by learner policy.
        frozen = np.asarray(page.transitions.executed
                            & (~page.transitions.is_learner | ~page.transitions.is_decision))
        np.testing.assert_array_equal(np.asarray(value)[frozen],
                                      np.asarray(page.transitions.behavior_probs)[frozen])
        return value
    monkeypatch.setattr(updater, "current_law", law)
    try:
        result = updater.update(state, jax.random.key(942), batch, q_old=q)
        assert len(calls) == result.metrics["actor_steps"] == 4
        assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(calls[0]),
                                                           jax.tree.leaves(calls[1])))
        assert current_laws
        for old, new in zip(before_q, q, strict=True):
            np.testing.assert_array_equal(old, new)
    finally:
        updater.close()


def test_epoch_rollback_restores_all_actor_slots_and_critic_still_updates():
    net, cfg, state, flat = setup(target_kl=1e-12)
    updater = PagedVRPOUpdater(net, cfg)
    try:
        result = updater.update(state, jax.random.key(942), paged(flat, cfg.page_rows))
        assert result.metrics["rejected_epochs"] == 1 and result.metrics["actor_steps"] == 0
        tree_equal(phase_params(state.params, "actor"), phase_params(result.state.params, "actor"))
        tree_equal(state.actor_opt_state, result.state.actor_opt_state)
        assert result.metrics["critic_steps"] == 2
    finally:
        updater.close()


def test_no_actor_rows_do_not_decay_or_advance_actor():
    net, cfg, state, flat = setup()
    updater = PagedVRPOUpdater(net, cfg)
    try:
        result = updater.update(state, jax.random.key(942), paged(forced(flat), cfg.page_rows))
        tree_equal(phase_params(state.params, "actor"), phase_params(result.state.params, "actor"))
        tree_equal(state.actor_opt_state, result.state.actor_opt_state)
        assert result.metrics["actor_steps"] == 0 and result.metrics["critic_steps"] == 2
    finally:
        updater.close()


def test_admission_and_nonfinite_failure_leave_prior_boundary_unchanged():
    net, cfg, state, flat = setup()
    batch = paged(flat, cfg.page_rows)
    before = jax.device_get(state)
    too_small = PagedVRPOUpdater(net, replace(cfg, max_bytes=1))
    try:
        with pytest.raises(MemoryError):
            too_small.update(state, jax.random.key(1), batch)
        assert all(x["misses"] == 0 for x in too_small.cache_statistics().values())
    finally:
        too_small.close()
    updater = PagedVRPOUpdater(net, cfg)
    try:
        q = updater.old_q(state, batch)
        malformed = (q[0].astype(jnp.int32),) + q[1:]
        with pytest.raises(ValueError, match="exact float32"):
            updater.update(state, jax.random.key(1), batch, q_old=malformed)
        q = (q[0].at[0, 0, 0].set(jnp.nan),) + q[1:]
        with pytest.raises(FloatingPointError):
            updater.update(state, jax.random.key(1), batch, q_old=q)
        tree_equal(state, before)
        assert updater.actor_opt_step.cache_info().misses == 0
    finally:
        updater.close()


def test_layout_rejected_before_certificate_jit_and_understated_ledger_cannot_bypass(monkeypatch):
    net, cfg, state, flat = setup()
    batch = paged(flat, cfg.page_rows)
    def forbidden(*args, **kwargs):
        pytest.fail("undeclared P/G must fail before certificate compilation")
    with monkeypatch.context() as local:
        local.setattr("train.update.validate_paged_batch", forbidden)
        updater = PagedVRPOUpdater(net, cfg)
        try:
            with pytest.raises(ValueError, match="page rows"):
                updater.admit(replace(batch, page_rows=cfg.page_rows * 2))
            with pytest.raises(ValueError, match="complete-game count"):
                updater.admit(replace(batch, game_lengths=batch.game_lengths[:-1]))
        finally:
            updater.close()
    fake = replace(batch, resident_bytes=0)
    actual = page_resident_bytes(fake)
    cache = len(batch.pages) * batch.page_rows * (3 * 87 * 4 + 3 * 4)
    updater = PagedVRPOUpdater(net, replace(cfg, max_bytes=actual + cache + page_scratch_bytes(batch) - 1))
    try:
        assert actual > fake.resident_bytes
        with pytest.raises((ValueError, MemoryError)):
            updater.admit(fake)
    finally:
        updater.close()


def test_multiple_legal_actions_cannot_be_marked_forced():
    net, cfg, state, flat = setup()
    raw = flat.transitions
    row = int(np.where(np.asarray(raw.executed & raw.is_learner & ~raw.is_decision))[0][0])
    raw = raw._replace(action_mask=raw.action_mask.at[row, 1].set(True))
    batch = paged(flat._replace(transitions=raw), cfg.page_rows)
    updater = PagedVRPOUpdater(net, cfg)
    try:
        with pytest.raises(ValueError):
            updater.update(state, jax.random.key(1), batch)
        assert updater.actor_opt_step.cache_info().misses == 0
    finally:
        updater.close()


def trace_counter(monkeypatch):
    """Count Python executions while jax.jit traces, not merely host LRU entries."""
    counts = Counter()
    original = jax.jit
    def tracked(fun=None, **kwargs):
        if fun is None:
            return lambda fn: tracked(fn, **kwargs)
        @wraps(fun)
        def traced(*args, **kw):
            counts[(fun.__module__, fun.__qualname__)] += 1
            return fun(*args, **kw)
        return original(traced, **kwargs)
    monkeypatch.setattr(jax, "jit", tracked)
    return counts


def test_more_than_eight_runtime_shapes_have_zero_new_misses_evictions_or_traces(monkeypatch):
    net, cfg, state, flat = setup(actor_epochs=1, bc_reference_coef=.03)
    # All materialization is BEFORE tracing instrumentation. Runtime totals,
    # maximum game length, Actor counts and page tuple lengths vary, P/G do not.
    batches = [paged(extend(flat, extra), cfg.page_rows) for extra in range(10)]
    batches += [paged(forced(extend(flat, 13)), cfg.page_rows)]
    traces = trace_counter(monkeypatch)
    updater = VRPOUpdater(net, cfg)
    try:
        reference = phase_params(state.params, "actor")
        first = updater.update(state, jax.random.key(1), batches[0], reference_actor=reference)
        warm = updater.cache_statistics()
        trace_warm = counts = traces.copy()
        assert counts
        for name, info in warm.items():
            if name != "squared_error":
                assert info["misses"] == 1, (name, info)
            assert info["evictions"] == 0
        for batch in batches[1:]:
            result = updater.update(first.state, first.rng, batch, reference_actor=reference)
            assert result.metrics["critic_steps"] == cfg.trajectory_minibatches
        after = updater.cache_statistics()
        assert traces == trace_warm, (trace_warm, traces)
        for name in warm:
            assert after[name]["misses"] == warm[name]["misses"], name
            assert after[name]["evictions"] == warm[name]["evictions"] == 0, name
    finally:
        updater.close()


def test_same_own_boundary_replay_is_exact():
    net, cfg, state, flat = setup()
    batch = paged(flat, cfg.page_rows)
    updater = PagedVRPOUpdater(net, cfg)
    try:
        first = updater.update(state, jax.random.key(7), batch)
        left = updater.update(first.state, first.rng, batch)
        right = updater.update(first.state, first.rng, batch)
        tree_equal(left.state, right.state)
        tree_equal(jax.random.key_data(left.rng), jax.random.key_data(right.rng))
        assert left.metrics == right.metrics
    finally:
        updater.close()


def test_allseat_joint_update_matches_flat_and_uses_every_action_seats_advantage():
    net, cfg, state, flat = setup(actor_epochs=1, muon_ns_steps=1, muon_ns_dtype="float32")
    raw = flat.transitions
    law = jax.nn.softmax(jnp.where(raw.action_mask,
        net.apply(state.params, raw.observation, method=net.get_action_logits), -1e30), -1)
    raw = raw._replace(is_learner=jnp.ones_like(raw.is_learner), policy_id=jnp.zeros_like(raw.policy_id),
                       behavior_probs=law)
    flat = flat._replace(transitions=raw, learner_seats=jnp.full_like(flat.learner_seats, -1))
    a, b = VRPOUpdater(net, cfg), PagedVRPOUpdater(net, cfg)
    try:
        expected = a.update(state, jax.random.key(3), flat)
        actual = b.update(state, jax.random.key(3), paged(flat, cfg.page_rows))
        tree_close(actual.state, expected.state, atol=3e-6, rtol=3e-5)
        assert actual.metrics["actor_rows"] == int((raw.executed & raw.is_decision).sum())
        for key in expected.metrics:
            assert actual.metrics[key] == pytest.approx(expected.metrics[key], abs=3e-6, rel=3e-5)
    finally:
        a.close()
        b.close()


def test_critic_paged_training_is_half_channel_mean_sumrows_meangames():
    net, cfg, state, flat = setup()
    batch = paged(flat, cfg.page_rows)
    updater = PagedVRPOUpdater(net, cfg)
    try:
        q = updater.old_q(state, batch)
        targets = updater.targets(batch, q)
        indices = np.array([0, 1], np.int32)
        loss, _ = updater.critic_minibatch(state.params, batch, targets, indices)
        old = unpage(batch, q)
        target = unpage(batch, targets)
        selected_rows = int(np.asarray(flat.game_lengths)[indices].sum())
        taken = np.take_along_axis(old[:selected_rows], np.asarray(flat.transitions.action[:selected_rows])[:, None, None], -1)[..., 0]
        expected = np.sum((taken - target[:selected_rows]) ** 2) / (6 * len(indices))
        assert float(loss) == pytest.approx(float(expected), rel=2e-6)
        old_row_mean = np.sum((taken - target[:selected_rows]) ** 2) / (3 * selected_rows)
        assert float(loss) != pytest.approx(float(old_row_mean), rel=1e-3)
    finally:
        updater.close()
