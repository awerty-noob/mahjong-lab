"""Muon/Nesterov optimizer construction."""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

ACTOR_PARAM_ROOTS = ("policy_extractor", "policy_mlp")
CRITIC_PARAM_ROOTS = ("critic_extractor", "value_critic_mlp")
PARAM_GROUPS = ("actor", "critic")


def param_group_labels(params):
    """Static per-leaf actor/critic labels for the trainer's two-group optimizer."""
    roots = params["params"] if isinstance(params, dict) and "params" in params else params
    if set(roots) != set(ACTOR_PARAM_ROOTS) | set(CRITIC_PARAM_ROOTS):
        raise ValueError(f"unexpected actor/critic parameter roots: {sorted(roots)}")
    return {
        "params": {
            name: jax.tree.map(
                lambda _, group=name: "actor" if group in ACTOR_PARAM_ROOTS else "critic",
                subtree,
            )
            for name, subtree in roots.items()
        }
    }


# --- from optimizers.py ---


class MuonState(NamedTuple):
    momentum: optax.Updates


def newton_schulz5(matrix, *, steps=5, dtype="bfloat16"):
    """Reference quintic approximate orthogonalization, NOT an exact polar SVD."""
    if matrix.ndim != 2 or min(matrix.shape) == 0:
        raise ValueError("Muon orthogonalization requires a nonempty matrix")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("muon_ns_steps must be a positive integer")
    if dtype not in ("float32", "bfloat16"):
        raise ValueError("muon_ns_dtype must be float32 or bfloat16")
    x = matrix.astype(jnp.dtype(dtype))
    transpose = x.shape[0] > x.shape[1]
    if transpose:
        x = x.T
    norm = jnp.linalg.norm(x.astype(jnp.float32)).astype(x.dtype)
    x = x / (norm + jnp.asarray(1e-7, x.dtype))

    def iteration(_, x):
        a = x @ x.T
        # Preserve reference left association: BF16 rounds c*A before matmul.
        b = -4.7750 * a + (2.0315 * a) @ a
        return 3.4445 * x + b @ x

    x = jax.lax.fori_loop(0, steps, iteration, x)
    return x.T if transpose else x


def _path_names(path):
    return tuple(str(getattr(key, "key", getattr(key, "name", key))) for key in path)


def _matrix_view(array, path):
    """Flax kernels -> PyTorch-style [out,in], retaining an inverse reshape."""
    if array.ndim == 2:
        return array.T, lambda x: x.T
    if array.ndim == 3 and path[-2] in ("query", "key", "value"):
        # Flax DenseGeneral: [in, heads, head_dim].
        return array.reshape(array.shape[0], -1).T, lambda x: x.T.reshape(array.shape)
    if array.ndim == 3 and path[-2] == "out":
        # Flax attention output: [heads, head_dim, out].
        return array.reshape(-1, array.shape[-1]).T, lambda x: x.T.reshape(array.shape)
    if array.ndim == 4:
        # Flax Conv [height,width,in,out], unlike the reference's [out,in,h,w].
        moved = jnp.moveaxis(array, -1, 0)
        return moved.reshape(moved.shape[0], -1), lambda x: jnp.moveaxis(x.reshape(moved.shape), 0, -1)
    raise ValueError(f"unsupported Muon kernel: {'/'.join(path)} {array.shape}")


def _is_output_head(path):
    names = path[1:] if path[0] == "params" else path
    # KataGo excludes ENTIRE policy/value heads, including hidden projections.
    return names[0] in ("policy_mlp", "value_critic_mlp")


def muon_parameter_labels(params):
    """Static, exhaustive Muon/AdamW labels; embeddings/norms/biases are not Muon.

    Hidden Dense/attention/Conv kernels with both matrix axes >4 use Muon.
    Entire output heads and small kernels use AdamW. Biases, norm scales and embedding
    tables have no decay. Unknown tensor layouts conservatively use AdamW.
    """

    def label(path, value):
        names = _path_names(path)
        if names[-1] in ("bias", "scale", "embedding") or value.ndim < 2:
            return "adamw_no_decay"
        supported = value.ndim in (2, 4) or (value.ndim == 3 and names[-2] in ("query", "key", "value", "out"))
        if names[-1] == "kernel" and supported and not _is_output_head(names):
            matrix, _ = _matrix_view(value, names)
            original_shape_ok = value.ndim != 4 or sum(size > 4 for size in value.shape) > 1
            if min(matrix.shape) > 4 and original_shape_ok:
                return "muon"
        return "adamw"

    return jax.tree_util.tree_map_with_path(label, params)


def scale_by_katago_muon(*, momentum=0.95, ns_steps=5, ns_dtype="bfloat16"):
    """Uncorrected EMA + Nesterov + quintic NS; only apply to labelled kernels."""

    def init(params):
        return MuonState(jax.tree.map(lambda p: jnp.zeros(p.shape, jnp.float32), params))

    def update(updates, state, params=None):
        del params
        buffers = jax.tree.map(
            lambda g, m: momentum * m + (1 - momentum) * g.astype(jnp.float32), updates, state.momentum
        )

        def project(path, g, m):
            direction = (1 - momentum) * g.astype(jnp.float32) + momentum * m
            matrix, restore = _matrix_view(direction, _path_names(path))
            orthogonal = newton_schulz5(matrix, steps=ns_steps, dtype=ns_dtype)
            # Reference scales by ORIGINAL dimensions, not flattened aspect ratio.
            # DenseGeneral's logical original parameter is a 2D linear map.
            original_max = max(matrix.shape) if g.ndim == 3 else max(g.shape)
            scaled = orthogonal * jnp.asarray(math.sqrt(original_max), orthogonal.dtype)
            return restore(scaled).astype(g.dtype)

        return jax.tree_util.tree_map_with_path(project, updates, buffers), MuonState(buffers)

    return optax.GradientTransformation(init, update)


def validate_optimizer_options(
    *,
    lr,
    max_grad_norm,
    optimizer="muon",
    weight_decay=1e-4,
    muon_lr_multiplier=8.0,
    muon_momentum=0.95,
    muon_ns_steps=5,
    muon_ns_dtype="bfloat16",
):
    if optimizer not in ("muon", "adamw"):
        raise ValueError("optimizer must be muon or adamw")
    for name, value, zero_ok in (
        ("lr", lr, True),
        ("max_grad_norm", max_grad_norm, False),
        ("weight_decay", weight_decay, True),
        ("muon_lr_multiplier", muon_lr_multiplier, False),
    ):
        if isinstance(value, bool) or not math.isfinite(value) or value < 0 or (not zero_ok and value == 0):
            raise ValueError(f"{name} must be finite and {'nonnegative' if zero_ok else 'positive'}")
    if isinstance(muon_momentum, bool) or not math.isfinite(muon_momentum) or not 0 <= muon_momentum < 1:
        raise ValueError("muon_momentum must be finite and in [0, 1)")
    if isinstance(muon_ns_steps, bool) or not isinstance(muon_ns_steps, int) or muon_ns_steps <= 0:
        raise ValueError("muon_ns_steps must be a positive integer")
    if muon_ns_dtype not in ("bfloat16", "float32"):
        raise ValueError("muon_ns_dtype must be float32 or bfloat16")


def make_optimizer(
    params,
    *,
    lr,
    max_grad_norm,
    optimizer="muon",
    weight_decay=1e-4,
    muon_lr_multiplier=8.0,
    muon_momentum=0.95,
    muon_ns_steps=5,
    muon_ns_dtype="bfloat16",
):
    """Global clipping -> actor/critic partition -> independent hybrid states.

    Keep the outer (clip_state, PartitionState{actor,critic}) ABI: KL rejection
    must restore ALL actor Muon/Adam slots but commit the critic's slots. Legacy
    AdamW is explicit, numerically identical to the former optimizer by default.
    """
    validate_optimizer_options(
        lr=lr,
        max_grad_norm=max_grad_norm,
        optimizer=optimizer,
        weight_decay=weight_decay,
        muon_lr_multiplier=muon_lr_multiplier,
        muon_momentum=muon_momentum,
        muon_ns_steps=muon_ns_steps,
        muon_ns_dtype=muon_ns_dtype,
    )
    labels = muon_parameter_labels(params) if optimizer == "muon" else None

    def group_optimizer():
        if optimizer == "adamw":
            return optax.adamw(lr, eps=1e-5, weight_decay=weight_decay)
        return optax.multi_transform(
            {
                "muon": optax.chain(
                    scale_by_katago_muon(momentum=muon_momentum, ns_steps=muon_ns_steps, ns_dtype=muon_ns_dtype),
                    optax.add_decayed_weights(weight_decay),
                    optax.scale(-lr * muon_lr_multiplier),
                ),
                "adamw": optax.adamw(lr, b1=muon_momentum, b2=0.995, eps=1e-6, weight_decay=weight_decay),
                "adamw_no_decay": optax.adamw(lr, b1=muon_momentum, b2=0.995, eps=1e-6, weight_decay=0),
            },
            labels,
        )

    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.multi_transform({name: group_optimizer() for name in PARAM_GROUPS}, param_group_labels(params)),
    )


def optimizer_metadata(config):
    """Record algorithm provenance, without implying that slots are checkpointed."""
    muon = config.optimizer == "muon"
    return {
        "name": config.optimizer,
        "auxiliary_learning_rate": config.lr,
        "muon_learning_rate": config.lr * config.muon_lr_multiplier if muon else None,
        "weight_decay": config.weight_decay,
        "max_grad_norm": config.max_grad_norm,
        "muon_momentum": config.muon_momentum if muon else None,
        "muon_lr_multiplier": config.muon_lr_multiplier if muon else None,
        "muon_ns_steps": config.muon_ns_steps if muon else None,
        "muon_ns_dtype": config.muon_ns_dtype if muon else None,
        "muon_variant": "katago_ema_nesterov_sqrt_max_original_dim" if muon else None,
        "adam_betas": [config.muon_momentum, 0.995] if muon else [0.9, 0.999],
        "adam_epsilon": 1e-6 if muon else 1e-5,
        "parameter_groups": "actor_critic_independent",
        "state_restored": False,
    }


# --- from vrpo_muon.py ---


class VRPOMuonState(NamedTuple):
    count: object
    momentum: object


def vrpo_parameter_labels(params):
    """Exhaustive static labels; unsupported higher-rank layouts fail closed."""

    def label(path, value):
        names = _path_names(path)
        roots = names[1:] if names[0] == "params" else names
        if roots[:2] == ("policy_mlp", "auxiliary"):
            return "frozen"
        if value.ndim < 2:
            return "nesterov_no_decay"
        # Validate layouts now, not at the first real optimizer step. Embeddings
        # are genuine 2D matrices here, including their small vocabulary axes.
        _matrix_view(value, names)
        return "muon_decay" if names[-1] == "kernel" else "muon_no_decay"

    return jax.tree_util.tree_map_with_path(label, params)


def make_vrpo_muon(params, config, phase):
    """Phase-local clipping, direct LR, one momentum buffer and no Adam slots."""
    if config.optimizer != "muon":
        raise ValueError("VRPO v2 only supports its explicit no-Adam Muon optimizer")
    if phase not in ("actor", "critic"):
        raise ValueError("unknown VRPO optimizer phase")
    labels = vrpo_parameter_labels(params)
    lr = config.actor_lr if phase == "actor" else config.critic_lr
    clip = config.actor_max_grad_norm if phase == "actor" else config.critic_max_grad_norm
    beta = config.muon_momentum

    def init(parameters):
        return VRPOMuonState(jnp.int32(0), jax.tree.map(lambda p: jnp.zeros(p.shape, jnp.float32), parameters))

    def update(grads, state, parameters=None):
        if parameters is None:
            raise ValueError("VRPO Muon requires parameters for explicit decoupled decay")
        # Frozen auxiliaries are excluded BEFORE clipping as well as from momentum
        # and decay; even synthetic nonzero gradients cannot perturb the phase.
        active = jax.tree.map(lambda g, label: jnp.zeros_like(g) if label == "frozen" else g, grads, labels)
        norm = optax.global_norm(active)
        clipped = jax.tree.map(lambda g: g * jnp.minimum(1.0, clip / (norm + 1e-12)), active)
        momentum = jax.tree.map(
            lambda g, m, label: m if label == "frozen" else beta * m + (1.0 - beta) * g.astype(jnp.float32),
            clipped,
            state.momentum,
            labels,
        )

        def direction(path, g, m, parameter, label):
            if label == "frozen":
                return jnp.zeros_like(g)
            u = (1.0 - beta) * g.astype(jnp.float32) + beta * m
            if label.startswith("muon_"):
                matrix, restore = _matrix_view(u, _path_names(path))
                projected = newton_schulz5(matrix, steps=config.muon_ns_steps, dtype=config.muon_ns_dtype)
                original_max = max(matrix.shape) if g.ndim == 3 else max(g.shape)
                scaled = projected * jnp.asarray(math.sqrt(original_max), projected.dtype)
                u = restore(scaled).astype(jnp.float32)
            if label == "muon_decay":
                u = u + config.weight_decay * parameter.astype(jnp.float32)
            return (-lr * u).astype(g.dtype)

        updates = jax.tree_util.tree_map_with_path(direction, clipped, momentum, parameters, labels)
        return updates, VRPOMuonState(optax.safe_int32_increment(state.count), momentum)

    return optax.GradientTransformation(init, update)


def vrpo_optimizer_metadata(config):
    """Exact numerical provenance; no claim that parameter exports include slots."""
    return dict(
        version=1,
        name="muon",
        variant="vrpo_muon_nesterov_no_adam_v1",
        independent_phases=True,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        learning_rate_semantics="direct_no_multiplier",
        actor_max_grad_norm=config.actor_max_grad_norm,
        critic_max_grad_norm=config.critic_max_grad_norm,
        weight_decay=config.weight_decay,
        decay="kernel_matrices_only",
        matrix_parameters="all_including_embeddings_and_narrow_output_heads",
        nonmatrix_update="uncorrected_ema_nesterov_sgd_no_decay",
        frozen_parameters=["policy_mlp/auxiliary"],
        muon_momentum=config.muon_momentum,
        muon_ns_steps=config.muon_ns_steps,
        muon_ns_dtype=config.muon_ns_dtype,
        muon_ns_epsilon=1e-7,
        muon_ns_coefficients=[3.4445, -4.7750, 2.0315],
        matrix_scaling="sqrt_max_logical_original_shape",
        momentum_formula="m=beta*m+(1-beta)*g;u=(1-beta)*g+beta*m",
        update_formula="p-=phase_multiplier*phase_lr*(project(u)+wd*p_if_kernel)",
        clipping="global_norm_per_phase_before_momentum_excluding_frozen",
        optimizer_slots="one_fp32_momentum_tree_and_int32_step_count_per_phase",
        adam=False,
    )
