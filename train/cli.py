"""Separate chance-residual VRPO entry point. No work occurs on import/--help.

Examples/configuration are documented separately. These commands are opt-in:
  python -m train.cli calibrate --config CONFIG.json
  python -m train.cli train --config CONFIG.json --artifact CALIBRATED.pkl --sha256 SHA
  python -m train.cli resume --config CONFIG.json --artifact BOUNDARY.pkl --sha256 SHA
"""

import argparse
import json
from pathlib import Path


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate configuration key: " + key)
        result[key] = value
    return result


def _prepare(options, source_config):
    from train.selfplay import initialize_bc, load_verified, make_contract

    bc = load_verified(options.bc_path, options.bc_sha256)
    env, model, state, rng, actor, network, heads = initialize_bc(bc, options)
    contract = make_contract(options, network, heads)
    return env, model, state, rng, actor, contract


def _new_output(options, source_config, *, mode, artifact=None, sha256=None):
    import os

    output = Path(options.output_dir).absolute()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "mode": mode,
        "config": options.to_dict(),
        "config_source": source_config,
        "source_artifact": artifact,
        "source_sha256": sha256,
        "execution_scope": "explicit_command_not_automatic",
    }
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    return output


def run_calibration(options, source_config):
    import jax

    from train.calibration import CalibrationFailure, build_calibrated_model, calibrate_q
    from train.selfplay import CompleteGameCollector, diagnostic_batch, write_new_pickle

    env, model, state, rng, actor, contract = _prepare(options, source_config)
    output = _new_output(options, source_config, mode="calibrate")
    collector = CompleteGameCollector(env, model, options)
    cfg = options.algorithm
    # Domain separation prevents replaying warmup/held-out worlds in online update1.
    rng = jax.random.fold_in(rng, 0x56435131)
    from train.selfplay import initial_collection_state

    collection_state = initial_collection_state()
    try:
        train, rng, collection_state, train_info = collector.collect(
            state, rng, collection_state, num_games=cfg.calibration_train_games, purpose="calibration"
        )
        from train.configs import PagedRawQBatch

        train_bytes = (
            train.resident_bytes + len(train.pages) * train.page_rows * 3 * 4
            if isinstance(train, PagedRawQBatch)
            else sum(x.size * x.dtype.itemsize for x in jax.tree.leaves(train))
        )
        heldout, rng, collection_state, heldout_info = collector.collect(
            state,
            rng,
            collection_state,
            num_games=cfg.calibration_heldout_games,
            reserved_bytes=train_bytes,
            purpose="calibration",
        )
        result = calibrate_q(
            model,
            cfg,
            state,
            rng,
            train,
            heldout,
            train_game_ids=["calibration:" + x for x in train_info["game_ids"]],
            heldout_game_ids=["calibration:" + x for x in heldout_info["game_ids"]],
        )
        payload = build_calibrated_model(
            result,
            contract=contract,
            metadata={"bc_source_sha256": options.bc_sha256, "rng_domain": "fold_in_0x56435131_separate_from_online"},
        )
        digest = write_new_pickle(output / "calibrated.pkl", payload, min_free_bytes=options.min_free_bytes)
        report = result.report | {"artifact_sha256": digest}
        (output / "calibration_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return report
    except BaseException as error:
        from train.selfplay import RawCollectionFailure

        if isinstance(error, CalibrationFailure):
            (output / "calibration_failed.json").write_text(json.dumps(error.report, indent=2, allow_nan=False) + "\n")
        diagnostic = {
            "checkpoint_kind": "vrpo_calibration_diagnostic",
            "resume_semantics": "NOT_RESUMABLE",
            "error_type": type(error).__name__,
            "reason": str(error),
            "bc_source_sha256": options.bc_sha256,
            "evidence": error.evidence if isinstance(error, RawCollectionFailure) else None,
            "train_batch": diagnostic_batch(locals().get("train")),
            "heldout_batch": diagnostic_batch(locals().get("heldout")),
            "frozen_original_params": state.params,
        }
        try:
            write_new_pickle(
                output / "calibration_diagnostic.pkl", jax.device_get(diagnostic), min_free_bytes=options.min_free_bytes
            )
        except Exception:
            pass
        raise
    finally:
        collector.close()


def _checked_calibrated(payload, template, contract, options, actor):
    import math

    from train.calibration import actor_parameter_fingerprint
    from train.selfplay import actor_sha256, checked_parameters
    from train.update import create_training_state, phase_params

    if (
        not isinstance(payload, dict)
        or payload.get("checkpoint_kind") != "vrpo_q_calibrated"
        or type(payload.get("schema_version")) is not int
        or payload["schema_version"] != 1
        or payload.get("critic_trained") is not True
    ):
        raise ValueError("online Actor is blocked until explicit accepted Q calibration")
    from train.checkpoint import validate_vrpo_contract
    from train.continuation import _equal_contract

    contract = validate_vrpo_contract(contract)
    fields = ("network_config", "head_config", "critic_config", "algorithm_config", "reward_config", "source_versions")
    if contract["network_config"]["architecture"] == "v5":
        fields += ("reading_config", "optimizer_config")
    for name in fields:
        _equal_contract(payload.get(name), contract[name], "calibrated artifact contract mismatch: " + name)
    for name, expected in contract["environment_config"].items():
        if payload.get(name) != expected:
            raise ValueError("calibrated environment mismatch")
    metadata = payload.get("metadata", {})
    report = metadata.get("calibration_report", {})
    if (
        metadata.get("bc_source_sha256") != options.bc_sha256
        or metadata.get("critic_trained") is not True
        or metadata.get("fresh_optimizer_slots") is not True
        or metadata.get("optimizer_state_included") is not False
        or metadata.get("resume_semantics") != "initialization_only"
        or report.get("accepted") is not True
        or report.get("beats_zero_baseline") is not True
        or report.get("beats_initial_critic") is not True
    ):
        raise ValueError("missing accepted original-BC calibration lineage")
    scores = [report.get(name) for name in ("best_heldout_mse", "zero_baseline_mse", "initial_critic_mse")]
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in scores):
        raise ValueError("invalid calibration acceptance metrics")
    relative = options.algorithm.calibration_min_relative_improvement
    if not (scores[0] < min(scores[1:]) and all(scores[0] < baseline * (1.0 - relative) for baseline in scores[1:])):
        raise ValueError("calibration did not beat zero and initial Critic")
    params = checked_parameters(payload.get("params"), template.params)
    if actor_sha256(phase_params(params, "actor")) != actor_sha256(actor):
        raise ValueError("calibrated Actor differs from original BC")
    digest = actor_parameter_fingerprint(params)
    if report.get("actor_sha256_before") != digest or report.get("actor_sha256_after") != digest:
        raise ValueError("calibration Actor fingerprint mismatch")
    return create_training_state(params, options.algorithm)[0]


def _reference_state(options, actor):
    if not options.algorithm.bc_reference_coef:
        return None
    from train.selfplay import actor_sha256

    return {
        "kind": "fixed_bc_reference_v1",
        "bc_source_sha256": options.bc_sha256,
        "actor_sha256": actor_sha256(actor),
        "coef": options.algorithm.bc_reference_coef,
    }


def run_online(options, source_config, *, mode, artifact, sha256):
    import signal
    import time

    import jax
    import numpy as np

    from train.checkpoint import (
        CheckpointStore,
        build_vrpo_model,
        build_vrpo_resume,
        restore_vrpo_resume,
    )
    from train.selfplay import (
        CompleteGameCollector,
        RawCollectionFailure,
        checked_vrpo_collection_state,
        diagnostic_batch,
        load_verified,
        write_new_pickle,
    )
    from train.update import VRPOUpdater

    if mode not in ("train", "resume"):
        raise ValueError("explicit train or resume mode required")
    if options.max_updates is None and options.max_wall_seconds is None:
        raise ValueError("online run requires an explicit update or wall-time limit")
    env, model, state, rng, actor, contract = _prepare(options, source_config)
    payload = load_verified(artifact, sha256)
    from train.selfplay import initial_collection_state

    collection_state = initial_collection_state()
    progress = {"completed_updates": 0, "envsteps": 0, "schedule_iteration": 0}
    reference_state = _reference_state(options, actor)
    lineage = {
        "bc_source_sha256": options.bc_sha256,
        "origin": "accepted_bc_q_calibration",
        "calibrated_source_sha256": sha256,
        "fresh_optimizer_slots": True,
    }
    if mode == "train":
        state = _checked_calibrated(payload, state, contract, options, actor)
    else:
        candidate = restore_vrpo_resume(payload, state, contract=contract)
        collection_state = checked_vrpo_collection_state(
            candidate.collection_state, options, completed_updates=candidate.progress["completed_updates"]
        )
        if candidate.retention_state != reference_state:
            raise ValueError("fixed BC reference identity/config mismatch")
        origin = candidate.lineage
        source = origin.get("calibrated_source_sha256")
        if (
            set(origin) != {"bc_source_sha256", "origin", "calibrated_source_sha256", "fresh_optimizer_slots"}
            or origin.get("bc_source_sha256") != options.bc_sha256
            or origin.get("origin") != "accepted_bc_q_calibration"
            or origin.get("fresh_optimizer_slots") is not True
            or not isinstance(source, str)
            or len(source) != 64
            or any(x not in "0123456789abcdef" for x in source)
        ):
            raise ValueError("boundary lacks accepted original-BC calibration lineage")
        state, rng, progress, lineage = candidate.state, candidate.rng, candidate.progress, candidate.lineage
    del payload
    output = _new_output(options, source_config, mode=mode, artifact=artifact, sha256=sha256)
    collector = CompleteGameCollector(env, model, options)
    updater = VRPOUpdater(model, options.algorithm)
    store = None
    iteration_context = None
    from train.health import ActorProgressGuard

    actor_progress = ActorProgressGuard(options.actor_stall_limit)
    actor_stalled = False
    previous_handlers = {}
    stop = [False]
    started = time.monotonic()

    def stopped():
        return stop[0] or (
            options.max_wall_seconds is not None and time.monotonic() - started >= options.max_wall_seconds
        )

    def publish(candidate, key, counters, games):
        games = checked_vrpo_collection_state(games, options, completed_updates=counters["completed_updates"])
        model_payload = build_vrpo_model(
            candidate.params,
            contract=contract,
            metadata={"update": counters["completed_updates"], "envsteps": counters["envsteps"], "lineage": lineage},
        )
        resume_payload = build_vrpo_resume(
            candidate,
            key,
            contract=contract,
            progress=counters,
            collection_state=games,
            retention_state=reference_state,
            lineage=lineage,
        )
        return store.publish(model_payload, resume_payload, counters["completed_updates"], counters["envsteps"])

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, lambda *_: stop.__setitem__(0, True))
        store = CheckpointStore.create(
            output / "checkpoints",
            options.run_id,
            keep_last=2,
            archive_interval=25,
            min_free_bytes=options.min_free_bytes,
            config_source=json.dumps(
                {
                    "source_config": source_config,
                    "effective": options.to_dict(),
                    "mode": mode,
                    "artifact": artifact,
                    "sha256": sha256,
                },
                sort_keys=True,
            ),
        )
        publish(state, rng, progress, collection_state)  # initial online boundary, AFTER accepted calibration
        with (output / "metrics.jsonl").open("x", encoding="utf-8") as metrics:
            while (
                not actor_stalled
                and not stopped()
                and (options.max_updates is None or progress["completed_updates"] < options.max_updates)
            ):
                iteration_context = {
                    "source_complete_progress": dict(progress),
                    "frozen_params": state.params,
                    "master_rng": rng,
                    "collection_state": collection_state,
                }
                batch, next_rng, next_collection, info = collector.collect(
                    state, rng, collection_state, stop_requested=stopped
                )
                iteration_context["raw_batch"] = diagnostic_batch(batch)
                result = updater.update(
                    state, next_rng, batch, reference_actor=actor if reference_state is not None else None
                )
                counters = {
                    "completed_updates": int(np.asarray(result.state.completed_updates)),
                    "envsteps": progress["envsteps"] + info["raw_rows"],
                    "schedule_iteration": int(np.asarray(result.state.completed_updates)),
                }
                publish(result.state, result.rng, counters, next_collection)
                # Observe actual committed counters, never candidate attempts or
                # a post-rollback zero KL. The boundary is durable before stop.
                actor_stalled = actor_progress.observe(
                    actor_rows=int(info["actor_rows"]),
                    committed_steps=int(np.asarray(result.state.actor_steps)) - int(np.asarray(state.actor_steps)),
                )
                state, rng, progress, collection_state = result.state, result.rng, counters, next_collection
                from train.selfplay import qboost_page

                # Operational cold/warm cache health is intentionally NOT part of
                # semantic update metrics or a resumable dynamic-state payload.
                execution = {
                    "layout": options.algorithm.algorithm_config()["execution_layout"],
                    "collector": getattr(collector, "cache_statistics", lambda: {})(),
                    "updater": getattr(updater, "cache_statistics", lambda: {})(),
                    "trace": qboost_page.cache_info()._asdict(),
                    "actor_stall_updates": actor_progress.consecutive,
                }
                row = dict(
                    progress=progress,
                    collection=info,
                    update=result.metrics,
                    execution=execution,
                    wall_seconds=time.monotonic() - started,
                )
                metrics.write(json.dumps(row, allow_nan=False) + "\n")
                metrics.flush()
                del batch, result, info, next_rng, next_collection
                iteration_context = None
        status = {
            "status": "actor_stalled" if actor_stalled else "stopped" if stopped() else "complete",
            "progress": progress,
            "resume_scope": "last_complete_update_boundary",
        }
        if actor_stalled:
            status.update(
                reason="consecutive_eligible_batches_without_committed_actor_steps",
                actor_stall_updates=actor_progress.consecutive,
                actor_stall_limit=actor_progress.limit,
                automatic_retry=False,
            )
        (output / "exit_status.json").write_text(json.dumps(status, indent=2) + "\n")
        return status
    except BaseException as error:
        diagnostic = {
            "checkpoint_kind": "vrpo_diagnostic",
            "resume_semantics": "NOT_RESUMABLE",
            "error_type": type(error).__name__,
            "reason": str(error),
            "prior_complete_progress": progress,
            "evidence": error.evidence if isinstance(error, RawCollectionFailure) else None,
            "iteration": iteration_context,
        }
        # Diagnostic failure must not hide the original or damage last complete pair.
        try:
            write_new_pickle(
                output / "diagnostic.pkl", jax.device_get(diagnostic), min_free_bytes=options.min_free_bytes
            )
        except Exception:
            pass
        if isinstance(error, InterruptedError) or (
            isinstance(error, RawCollectionFailure) and str(error) == "stop during collection; not a resume boundary"
        ):
            status = {
                "status": "stopped",
                "progress": progress,
                "resume_scope": "last_complete_update_boundary",
                "discarded_partial_rollout": True,
            }
            (output / "exit_status.json").write_text(json.dumps(status, indent=2) + "\n")
            return status
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        collector.close()
        updater.close()
        if store is not None:
            store.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Independent chance-residual VRPO (not legacy PPO)")
    parser.add_argument("mode", choices=("calibrate", "train", "resume", "profile"))
    parser.add_argument("--config", required=True, help="explicit JSON run/algorithm configuration")
    parser.add_argument("--artifact", help="accepted calibrated model or exact resume boundary")
    parser.add_argument("--sha256", help="exact artifact SHA256 before deserialization")
    parser.add_argument("--profile-max-seconds", type=int, help="profile-only wall budget, 1..1800")
    args = parser.parse_args(argv)
    if args.mode in ("train", "resume") and (not args.artifact or not args.sha256):
        parser.error("train/resume require --artifact and --sha256")
    if args.mode in ("calibrate", "profile") and (args.artifact or args.sha256):
        parser.error("calibrate/profile initialize from config's ORIGINAL BC, not a trained artifact")
    if args.profile_max_seconds is not None and (args.mode != "profile" or not 1 <= args.profile_max_seconds <= 1800):
        parser.error("--profile-max-seconds requires profile mode and must be 1..1800")
    source = Path(args.config).read_text(encoding="utf-8")
    from train.selfplay import VRPORunConfig

    options = VRPORunConfig.from_dict(json.loads(source, object_pairs_hook=_json_object))
    if args.mode == "profile":
        from train.vrpo_profile import run_profile

        report = run_profile(options, source, max_seconds=args.profile_max_seconds or 1800)
    else:
        report = (
            run_calibration(options, source)
            if args.mode == "calibrate"
            else run_online(options, source, mode=args.mode, artifact=args.artifact, sha256=args.sha256)
        )
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    if report.get("status") == "actor_stalled":
        raise SystemExit(2)
    return report


if __name__ == "__main__":
    main()
