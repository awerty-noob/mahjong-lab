import json

import pytest

from train import cli
from train.configs import tiny_debug_preset
from train.selfplay import VRPORunConfig

SHA = "a" * 64


def _options(**overrides):
    values = dict(
        algorithm=tiny_debug_preset(),
        bc_path="synthetic-bc.pkl",
        bc_sha256=SHA,
        output_dir="synthetic-new-run",
        run_id="cli-test",
    )
    return VRPORunConfig(**(values | overrides))


def test_metrics_output_path_defaults_inside_run_output(tmp_path):
    output = tmp_path / "run"
    assert cli._metrics_output_path(_options(), output) == output / "metrics.jsonl"


def test_metrics_output_path_creates_custom_parent(tmp_path):
    custom = tmp_path / "tensorboard" / "run-1" / "metrics.jsonl"
    path = cli._metrics_output_path(_options(metrics_path=str(custom)), tmp_path / "run")
    assert path == custom
    assert path.parent.is_dir()


def test_metrics_output_path_resolves_relative_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = cli._metrics_output_path(_options(metrics_path="tb/metrics.jsonl"), tmp_path / "run")
    assert path == tmp_path / "tb" / "metrics.jsonl"


def test_tensorboard_root_defaults_to_run_output(tmp_path):
    assert cli._tensorboard_root(_options(), tmp_path / "run") == tmp_path / "run" / "tensorboard"


def test_tensorboard_root_honors_custom_absolute_path(tmp_path):
    custom = tmp_path / "tf-logs" / "run-1"
    assert cli._tensorboard_root(_options(tensorboard_dir=str(custom)), tmp_path / "run") == custom


def test_tensorboard_root_resolves_relative_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cli._tensorboard_root(_options(tensorboard_dir="tf-logs"), tmp_path / "run") == tmp_path / "tf-logs"


def test_calibration_algorithm_identity_allows_online_hyperparameters():
    base = {
        "version": 5,
        "kind": "vrpo_actor_reading_v5",
        "optimizer": "muon",
        "gamma": 1.0,
        "trace_lambda": 0.95,
        "reading_auxiliary": True,
        "actor_lr": 2.5e-05,
        "rollout_games": 8,
        "execution_layout": {"version": 2, "kind": "paged_interleaved_games", "page_rows": 64},
    }
    online = base | {
        "actor_lr": 4e-4,
        "rollout_games": 512,
        "tempo_anchor": False,
        "execution_layout": {"version": 2, "kind": "paged_interleaved_games", "page_rows": 2048},
    }
    assert cli._calibration_algorithm_identity(base) == cli._calibration_algorithm_identity(online)
    assert cli._calibration_algorithm_identity(base | {"version": 2}) != cli._calibration_algorithm_identity(online)
    changed_layout = base | {"execution_layout": {"version": 2, "kind": "flat_reference", "page_rows": 64}}
    assert cli._calibration_algorithm_identity(base) != cli._calibration_algorithm_identity(changed_layout)


def test_calibration_report_requires_one_complete_final_epoch_run():
    scores = [0.17, 0.26, 0.26]
    report = {
        "best_epoch": 2,
        "epochs_completed": 2,
        "max_epochs": 2,
        "epoch_heldout_mse": [0.20, 0.17],
    }
    cli._check_calibration_report(report, scores, 0.0)
    with pytest.raises(ValueError, match="final-epoch"):
        cli._check_calibration_report(report | {"best_epoch": 1}, scores, 0.0)
    with pytest.raises(ValueError, match="history"):
        cli._check_calibration_report(report | {"epoch_heldout_mse": [0.20]}, scores, 0.0)


def test_cli_metrics_path_overrides_config(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(_options().to_dict()), encoding="utf-8")
    captured = {}

    def fake_run_online(options, source_config, *, mode, artifact, sha256):
        captured["options"] = options
        return {"status": "complete", "progress": {}}

    monkeypatch.setattr(cli, "run_online", fake_run_online)
    custom = tmp_path / "tb" / "metrics.jsonl"
    assert cli.main(
        [
            "train",
            "--config",
            str(config),
            "--artifact",
            "calibrated.pkl",
            "--sha256",
            SHA,
            "--metrics-path",
            str(custom),
        ]
    ) == {"status": "complete", "progress": {}}
    assert captured["options"].metrics_path == str(custom)


def test_cli_rejects_metrics_path_for_calibration(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(_options().to_dict()), encoding="utf-8")
    with pytest.raises(SystemExit):
        cli.main(["calibrate", "--config", str(config), "--metrics-path", "metrics.jsonl"])
