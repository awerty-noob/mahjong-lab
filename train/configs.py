"""Run configuration, reward rules, and the environment factory."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any, Literal, NamedTuple, Optional

import flax.struct
import jax
import jax.numpy as jnp
import numpy as np

import mahjax

# --- from vrpo_types.py ---


@flax.struct.dataclass
class VRPOTrainingState:
    """Dynamic state only; transformations/network are reconstructed, not pickled."""

    params: Any
    actor_opt_state: Any
    critic_opt_state: Any
    actor_steps: Any = flax.struct.field(default_factory=lambda: jnp.int32(0))
    critic_steps: Any = flax.struct.field(default_factory=lambda: jnp.int32(0))
    completed_updates: Any = flax.struct.field(default_factory=lambda: jnp.int32(0))


class RawQTransition(NamedTuple):
    observation: Any  # Allowlisted, prepared V4 Actor observation.
    critic_observation: Any  # Compact PRIVILEGED part only (no duplicate public obs).
    action: Any
    action_mask: Any
    behavior_probs: Any  # Actual acting policy at collection, including foreign.
    reward: Any  # [..., 3], absolute-seat utility / 32000.
    current_player: Any  # Absolute acting seat.
    done: Any  # True terminal, never a truncation.
    is_decision: Any  # More than one legal action.
    is_learner: Any
    executed: Any  # Padding and frozen lanes are false.
    policy_id: Any
    reading_targets: Any = None  # V5 training-only oracle sidecar; NEVER a feature.


class RawQBatch(NamedTuple):
    transitions: RawQTransition  # Flat padded arena, complete games in order.
    game_offsets: Any  # [G], first flat row of each complete game.
    game_lengths: Any  # [G], executed raw rows, excluding padding.
    learner_seats: Any  # [G], absolute payoff seat for the Actor objective.


class QBoostLabels(NamedTuple):
    advantages: Any  # [..., 3], RAW normalized-utility units, not standardized.
    targets: Any  # [..., 3], sampled-action Q regression targets.
    values: Any  # [..., 3], exact legal policy expectation of fixed Q.
    q_taken: Any  # [..., 3].
    actor_mask: Any  # [...], executed & learner & decision.
    critic_mask: Any  # [...], every executed row, including foreign/forced.


class RawQPage(NamedTuple):
    """One fixed-size device page; game metadata is NEVER a network feature."""

    transitions: RawQTransition
    game_ids: Any  # [P], padding=-1; real ids in [0,G).
    game_steps: Any  # [P], chronological per-game raw clock; padding=-1.


class PageTraceCarry(NamedTuple):
    next_value: Any  # [G,3], absolute payoff seats.
    next_z: Any  # [G,3], independent per-game reverse recurrence.


@dataclass(frozen=True)
class PagedRawQBatch:
    """HOST owner only: never register/submit its variable page tuple to JIT.

    Every executed game's subsequence is chronological; games may interleave.
    Only the final page may be partial. Fixed-game metadata is allowed on host.
    """

    pages: tuple
    game_lengths: Any
    learner_seats: Any
    page_game_ids: tuple  # Host tuples of game ids present on each page.
    page_rows: int
    raw_count: int
    actor_count: int
    resident_bytes: int

    @property
    def num_games(self):
        return self.game_lengths.shape[0]

    def iter_pages(self, *, reverse=False, game_indices=None):
        chosen = None if game_indices is None else frozenset(int(x) for x in game_indices)
        indices = range(len(self.pages) - 1, -1, -1) if reverse else range(len(self.pages))
        for index in indices:
            if chosen is None or any(game in chosen for game in self.page_game_ids[index]):
                yield index, self.pages[index]

    def to_diagnostic(self):
        return dict(
            kind="paged_raw_q_batch",
            storage_version=2,
            pages=self.pages,
            game_lengths=self.game_lengths,
            learner_seats=self.learner_seats,
            page_game_ids=self.page_game_ids,
            page_rows=self.page_rows,
            raw_count=self.raw_count,
            actor_count=self.actor_count,
            resident_bytes=self.resident_bytes,
        )


# --- from rewards.py ---


def default_reward_config(env_name):
    """Return fresh sanma utility provenance; other rules keep legacy behavior."""
    if env_name != "sanma":
        return None
    return {
        "version": 1,
        "objective": "score_delta_plus_uma",
        "uma_points": [20000, 0, -20000],
        "oka_points": 0,
        "normalization_points": 32000,
        "score_unit_points": 100,
    }


def _point_value(value, field):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"reward_config {field} must be a finite number of raw points")
    if not np.isfinite(value) or value % 100 != 0:
        raise ValueError(f"reward_config {field} must be finite and divisible by 100 raw points")
    return int(value)


def validate_reward_config(config, env_name):
    """Validate a complete version-1 schema, with no implicit field upgrades.

    ``None`` selects the new sanma default. Explicit all-zero UMA is supported
    for historical checkpoints/tests; it is not the default for a new run.
    """
    default = default_reward_config(env_name)
    if config is None:
        return default
    if default is None:
        raise ValueError(f"reward_config is not supported for {env_name!r}")
    if not isinstance(config, Mapping) or set(config) != set(default):
        raise ValueError("reward_config must contain exactly the complete version-1 fields")
    version = config["version"]
    if isinstance(version, (bool, np.bool_)) or not isinstance(version, (int, np.integer)) or version != 1:
        raise ValueError("reward_config version must be 1")
    if config["objective"] != default["objective"]:
        raise ValueError("reward_config objective must be 'score_delta_plus_uma'")
    uma = config["uma_points"]
    if not isinstance(uma, (list, tuple, np.ndarray)) or np.ndim(uma) != 1 or len(uma) != 3:
        raise ValueError("reward_config uma_points must contain three raw-point values")
    uma = [_point_value(value, "uma_points") for value in uma]
    if any(first < second for first, second in zip(uma, uma[1:])):
        raise ValueError("reward_config uma_points must be monotone nonincreasing by rank")
    if sum(uma) != 0:
        raise ValueError("reward_config uma_points must sum to zero for the no-oka protocol")
    result = {"version": 1, "objective": default["objective"], "uma_points": uma}
    for field in ("oka_points", "normalization_points", "score_unit_points"):
        value = _point_value(config[field], field)
        if value != default[field]:
            raise ValueError(f"reward_config {field} must be {default[field]}")
        result[field] = value
    return result


def checkpoint_reward_config(checkpoint, env_name):
    """Read ONLY root reward provenance; missing sanma root means legacy zero UMA.

    Nested metadata/utility aliases cannot repair or override the root. Explicit
    ``None`` is invalid for sanma checkpoints, unlike a new-run configuration.
    """
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a root dictionary")
    if "reward_config" not in checkpoint:
        config = default_reward_config(env_name)
        if config is not None:
            config["uma_points"] = [0, 0, 0]
        return config
    config = checkpoint["reward_config"]
    if env_name == "sanma" and config is None:
        raise ValueError("checkpoint root reward_config must be an explicit dictionary")
    return validate_reward_config(config, env_name)


def settlement_components(final_scores, init_wind, order_points):
    """Recover raw terminal scores and per-seat UMA without double settlement.

    ``final_scores[..., N]`` already includes environment UMA and any leftover
    riichi sticks. ``init_wind`` and rank-ordered ``order_points`` have last
    dimension N and may broadcast over leading batch axes. Returned ranks are
    zero-based *seat* arrays; all other outputs remain in environment score units
    (hundreds of raw points). Monotone UMA preserves the original rank order,
    with ties resolved by initial wind exactly as in the environment.

    Shapes/dtypes are checked during tracing; value checks are eager-only. JIT
    callers must supply finite scores, permutation winds, and monotone UMA.
    """
    scores, winds, points = map(jnp.asarray, (final_scores, init_wind, order_points))
    if scores.ndim < 1 or scores.shape[-1] < 1:
        raise ValueError("final_scores must have shape [..., num_players]")
    for name, value in (("final_scores", scores), ("init_wind", winds), ("order_points", points)):
        if not jnp.issubdtype(value.dtype, jnp.number) or jnp.issubdtype(value.dtype, jnp.complexfloating):
            raise ValueError(f"{name} must contain real numeric values")
        if value.ndim < 1 or value.shape[-1] != scores.shape[-1]:
            raise ValueError(f"{name} must have the same final player axis as final_scores")
    if not jnp.issubdtype(winds.dtype, jnp.integer):
        raise ValueError("init_wind must contain integer seat winds")
    try:
        winds = jnp.broadcast_to(winds, scores.shape)
        points = jnp.broadcast_to(points, scores.shape)
    except ValueError as error:
        raise ValueError("init_wind and order_points must broadcast to final_scores") from error
    for name, value in (("final_scores", scores), ("init_wind", winds), ("order_points", points)):
        if isinstance(value, jax.core.Tracer):
            continue
        host = np.asarray(value)
        if not np.all(np.isfinite(host)):
            raise ValueError(f"{name} must contain finite values")
        if name == "init_wind" and not np.all(np.sort(host, axis=-1) == np.arange(scores.shape[-1])):
            raise ValueError("init_wind must be a permutation of zero-based seat winds")
        if name == "order_points" and np.any(host[..., :-1] < host[..., 1:]):
            raise ValueError("order_points must be monotone nonincreasing by rank")
    order = jnp.lexsort((winds, -scores), axis=-1)
    ranks = jnp.argsort(order, axis=-1).astype(jnp.int32)
    uma = jnp.take_along_axis(points, ranks, axis=-1)
    return {"ranks": ranks, "raw_scores": scores - uma, "uma": uma, "settled_scores": scores}


# --- from rules.py ---


@dataclass(frozen=True)
class RuleSpec:
    name: str
    num_players: int
    num_actions: int
    ruleset: str
    full_game_round_mode: Optional[str]  # None: step-mode rollouts only
    default_architecture: str

    def make_env(
        self,
        round_mode: str,
        observe_type: str = "dict",
        reward_config=None,
        round_end: str = "auto",
    ):
        if round_end not in ("auto", "confirm"):
            raise ValueError(f"round_end must be one of ('auto', 'confirm'), got: {round_end}")
        if self.name == "sanma":
            config = validate_reward_config(reward_config, self.name)
            # Configuration is raw points; the environment settles in hundreds.
            order_points = [value // config["score_unit_points"] for value in config["uma_points"]]
            return mahjax.make(
                "red_3p",
                round_mode=round_mode,
                observe_type=observe_type,
                order_points=order_points,
                round_end=round_end,
            )
        return mahjax.make(
            ENV_IDS.get(self.name, self.name),
            round_mode=round_mode,
            observe_type=observe_type,
            round_end=round_end,
        )

    def validate_full_game(self, round_mode: str) -> None:
        if self.full_game_round_mode is None:
            raise ValueError(f"full_game rollout is not supported for {self.name}")
        if round_mode != self.full_game_round_mode:
            raise ValueError(
                f"full_game rollout for {self.name} requires round_mode={self.full_game_round_mode}, got {round_mode}"
            )


RULE_SPECS = {
    "sanma": RuleSpec("sanma", 3, 87, "riichienv-standard-sanma", "half", "v3"),
    "red_mahjong": RuleSpec("red_mahjong", 4, 87, "riichienv-standard", "half", "v3"),
    # No red fives: 79 actions (34 discard + 34 kan + 11 special).
    "no_red_mahjong": RuleSpec("no_red_mahjong", 4, 79, "riichienv-standard", "half", "v1"),
}


def rule_spec(env_name: str) -> RuleSpec:
    try:
        return RULE_SPECS[env_name]
    except KeyError:
        raise ValueError(f"Unsupported env_name: {env_name!r}; supported: {sorted(RULE_SPECS)}") from None


# --- from vrpo_config.py ---

ALGORITHM_KIND = "vrpo_qboost_chance_residual_v2"
READING_ALGORITHM_KIND = "vrpo_actor_reading_v5"
READING_PROTOCOL = dict(
    version=1,
    target_schema=1,
    tasks=["tenpai", "shanten", "danger", "waits", "danger_points"],
    rows="owned_executed_multioption",
    reduction="sum_independent_task_cell_means",
    phase="actor_only_shared_encoder",
    inputs="own_and_public_only",
    labels="preaction_current_state_oracle",
    points_scale=8000,
    danger_context="legal_discard_no_pending_hidden_dora",
    tree_sampling=False,
)


@dataclass(frozen=True, kw_only=True)
class VRPOConfig:
    t_eta: int
    t_alpha: int
    version: int = 2
    kind: str = ALGORITHM_KIND
    preset: Literal["explicit", "tiny_debug", "paper_reference"] = "explicit"
    gamma: float = 1.0
    trace_lambda: float = 0.95
    actor_lr: float = 4e-4
    critic_lr: float = 4e-4
    clip0: float = 0.02
    uniform_alpha0: float = 0.1
    actor_epochs: int = 4
    critic_epochs: int = 4
    trajectory_minibatches: int = 4
    actor_microbatch_size: int = 64
    critic_microbatch_size: int = 32
    inference_microbatch_size: int = 64
    rollout_games: int = 2048
    num_envs: int = 32
    execution_block_steps: int = 64
    max_game_steps: int = 20000
    time_bucket: int = 128  # Legacy flat-reference API only; never sizes production pages.
    storage_page_rows: int | None = None
    max_bytes: int = 8 * 1024**3
    temperature: float = 1.0
    actor_max_grad_norm: float = 0.5
    critic_max_grad_norm: float = 0.5
    target_kl: float = 0.01
    optimizer: Literal["muon"] = "muon"
    weight_decay: float = 0.01
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    muon_ns_dtype: Literal["bfloat16", "float32"] = "bfloat16"
    bc_reference_coef: float = 0.0
    # TEMPO filter (Epsilon-Prism `learn/Teacher.java`): the Actor is pulled back
    # toward the iteration that collected the batch while the surrogate shrinks.
    tempo_anchor: bool = False
    tempo_filter_gain: float = 0.9
    tempo_filter_tau: float = 4.0
    tempo_filter_power: float = 2.0 / 3.0
    tempo_lr_floor: float = 0.25
    reading_auxiliary: bool = False
    reading_aux_coef: float = 0.01
    calibration_train_games: int = 256
    calibration_heldout_games: int = 128
    calibration_max_epochs: int = 64
    calibration_min_relative_improvement: float = 0.01

    def __post_init__(self):
        if type(self.reading_auxiliary) is not bool:
            raise ValueError("reading_auxiliary must be bool")
        expected = (5, READING_ALGORITHM_KIND) if self.reading_auxiliary else (2, ALGORITHM_KIND)
        if type(self.version) is not int or (self.version, self.kind) != expected:
            raise ValueError("unsupported VRPO algorithm schema; V5 reading requires explicit version=5")
        integer_fields = (
            "t_eta",
            "t_alpha",
            "actor_epochs",
            "critic_epochs",
            "trajectory_minibatches",
            "actor_microbatch_size",
            "critic_microbatch_size",
            "inference_microbatch_size",
            "rollout_games",
            "num_envs",
            "execution_block_steps",
            "max_game_steps",
            "time_bucket",
            "max_bytes",
            "muon_ns_steps",
            "calibration_train_games",
            "calibration_heldout_games",
            "calibration_max_epochs",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be an explicit positive integer")
        for name in (
            "gamma",
            "trace_lambda",
            "actor_lr",
            "critic_lr",
            "clip0",
            "uniform_alpha0",
            "temperature",
            "actor_max_grad_norm",
            "critic_max_grad_norm",
            "target_kl",
            "weight_decay",
            "muon_momentum",
            "bc_reference_coef",
            "reading_aux_coef",
            "calibration_min_relative_improvement",
            "tempo_filter_gain",
            "tempo_filter_tau",
            "tempo_filter_power",
            "tempo_lr_floor",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.gamma != 1.0 or not 0 <= self.trace_lambda <= 1:
            raise ValueError("VRPO v2 requires gamma=1 and trace_lambda in [0,1]")
        if self.actor_lr <= 0 or self.critic_lr <= 0 or not 0 < self.clip0 < 1:
            raise ValueError("positive phase rates and clip0 in (0,1) required")
        if (
            self.uniform_alpha0 < 0
            or self.bc_reference_coef < 0
            or self.weight_decay < 0
            or self.temperature <= 0
            or self.target_kl <= 0
            or self.actor_max_grad_norm <= 0
            or self.critic_max_grad_norm <= 0
            or not 0 <= self.muon_momentum < 1
            or not 0 <= self.calibration_min_relative_improvement < 1
        ):
            raise ValueError("invalid regularization, optimizer, or calibration options")
        if type(self.tempo_anchor) is not bool:
            raise ValueError("tempo_anchor must be bool")
        if (
            not 0 <= self.tempo_filter_gain <= 1
            or self.tempo_filter_tau <= 0
            or self.tempo_filter_power <= 0
            or self.tempo_lr_floor <= 0
        ):
            raise ValueError("invalid TEMPO filter schedule")
        if self.reading_aux_coef < 0 or (self.reading_auxiliary and self.reading_aux_coef <= 0):
            raise ValueError("enabled V5 reading requires a positive auxiliary coefficient")
        if self.optimizer != "muon" or self.muon_ns_dtype not in ("bfloat16", "float32"):
            raise ValueError("unsupported optimizer")
        if self.preset not in ("explicit", "tiny_debug", "paper_reference"):
            raise ValueError("unsupported preset")
        if self.rollout_games % self.num_envs or self.rollout_games % self.trajectory_minibatches:
            raise ValueError("whole games must divide into cohorts and trajectory minibatches")
        if self.storage_page_rows is not None and (
            type(self.storage_page_rows) is not int or self.storage_page_rows <= 0
        ):
            raise ValueError("storage_page_rows must be a positive integer or None")
        if self.page_rows < self.num_envs * self.execution_block_steps or self.page_rows % self.page_alignment:
            raise ValueError("page rows must fit a whole execution block and align all microbatches")

    @property
    def page_alignment(self):
        return math.lcm(self.actor_microbatch_size, self.critic_microbatch_size, self.inference_microbatch_size)

    @property
    def page_rows(self):
        if self.storage_page_rows is not None:
            return self.storage_page_rows
        rows = self.num_envs * self.execution_block_steps
        return ((rows + self.page_alignment - 1) // self.page_alignment) * self.page_alignment

    @property
    def games_per_minibatch(self):
        return self.rollout_games // self.trajectory_minibatches

    def algorithm_config(self):
        return {
            **asdict(self),
            "trace_clock": "raw_environment_transition",
            "payoff_seats": "absolute3",
            "training_seats": "all",
            "actor_reduction": "sum_rows_mean_games",
            "critic_reduction": "half_sum_rows_mean_games",
            "payoff_channel_reduction": "mean_absolute3",
            "critic_targets": "sampled_action_only",
            "advantage_normalization": False,
            "bc_reference_divergence": "KL(fixed_BC||policy)",
            "execution_layout": {"version": 2, "kind": "paged_interleaved_games", "page_rows": self.page_rows},
            "critic_auxiliary": False,
            "replay": False,
            **(
                {"reading_protocol": {**READING_PROTOCOL, "tasks": list(READING_PROTOCOL["tasks"])}}
                if self.reading_auxiliary
                else {}
            ),
        }


def tiny_debug_preset(**overrides):
    values = dict(
        t_eta=2,
        t_alpha=2,
        preset="tiny_debug",
        rollout_games=4,
        num_envs=2,
        trajectory_minibatches=2,
        actor_epochs=1,
        critic_epochs=1,
        actor_microbatch_size=2,
        critic_microbatch_size=2,
        inference_microbatch_size=2,
        execution_block_steps=8,
        time_bucket=8,
        calibration_train_games=4,
        calibration_heldout_games=2,
    )
    return VRPOConfig(**(values | overrides))


def paper_reference_preset(*, t_eta, t_alpha, actor_lr, critic_lr, clip0, uniform_alpha0, **overrides):
    """Reference equations/budgets; all transferable horizons/rates are explicit."""
    return VRPOConfig(
        **(
            dict(
                t_eta=t_eta,
                t_alpha=t_alpha,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                clip0=clip0,
                uniform_alpha0=uniform_alpha0,
                preset="paper_reference",
            )
            | overrides
        )
    )


# Upstream registers the three-player rule sets as ``red_3p``/``no_red_3p``.
ENV_IDS = {"sanma": "red_3p", "red_3p": "red_3p", "no_red_3p": "no_red_3p"}
