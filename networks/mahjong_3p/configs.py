"""Frozen configuration and checkpoint-provenance schemas for the V5 sanma stack.

Every network in this package is the same architecture; the presets only choose
width. Head, reading and critic schemas are explicit and exact: a checkpoint
either declares a complete, supported schema or it is rejected. Two strings are
*data contracts* rather than code versions and are therefore kept verbatim:
``ARCHITECTURE`` (checkpoint envelope) and ``FEATURE_VERSION`` (observation
schema). They let existing weights be read; nothing else is inferred.
"""

from collections.abc import Mapping
from typing import Any

ARCHITECTURE = "v5"
FEATURE_VERSION = 4
RULESET = "riichienv-standard-sanma"

SMALL_NETWORK_CONFIG = {
    "architecture": ARCHITECTURE,
    "num_layers": 4,
    "d_model": 128,
    "num_heads": 4,
    "mlp_dim": 352,
}
MEDIUM_NETWORK_CONFIG = {
    "architecture": ARCHITECTURE,
    "num_layers": 8,
    "d_model": 192,
    "num_heads": 6,
    "mlp_dim": 512,
}
LARGE_NETWORK_CONFIG = {
    "architecture": ARCHITECTURE,
    "num_layers": 12,
    "d_model": 256,
    "num_heads": 8,
    "mlp_dim": 704,
}
NETWORK_PRESETS = {
    "small": SMALL_NETWORK_CONFIG,
    "medium": MEDIUM_NETWORK_CONFIG,
    "large": LARGE_NETWORK_CONFIG,
}
DEFAULT_NETWORK_CONFIG = dict(LARGE_NETWORK_CONFIG)

# Own final-rank and remaining-score auxiliary head of the Critic trunk.
DEFAULT_HEAD_CONFIG = {"version": 1, "hidden_dim": 128}
# Actor-only opponent-reading head: two relative opponents, canonical 34 tiles.
DEFAULT_READING_CONFIG = {
    "version": 1,
    "kind": "opponent_reading",
    "target_schema": 1,
    "opponents": 2,
    "tile_count": 34,
    "latent_dim": 64,
    "points_scale": 8000,
}
DISABLED_HEAD_CONFIG = {"version": 0}
# Training-only central action Q over 87 actions with absolute payoff seats.
DEFAULT_CRITIC_CONFIG = {
    "version": 1,
    "kind": "central_action_q",
    "state_input_schema": 1,
    "action_count": 87,
    "payoff_seats": "absolute3",
}

DEFAULT_CENTRAL_Q_CRITIC_CONFIG = DEFAULT_CRITIC_CONFIG
LEGACY_CRITIC_CONFIG = {"version": 0, "kind": "legacy_v"}


def network_config(name: str = "large") -> dict:
    """Return a fresh preset copy; unknown names raise."""
    try:
        return dict(NETWORK_PRESETS[name])
    except KeyError:
        raise ValueError(f"unknown network preset {name!r}") from None


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"network {field_name} must be a positive integer")
    return value


def validate_network_config(*args: Any, **kwargs: Any) -> dict:
    config = _single_config(args, kwargs)
    """Accept exactly the five backbone fields of this architecture."""
    if not isinstance(config, Mapping) or set(config) != set(DEFAULT_NETWORK_CONFIG):
        raise ValueError("network_config must contain exactly the backbone fields")
    result = dict(config)
    if result["architecture"] != ARCHITECTURE:
        raise ValueError(f"network architecture must be {ARCHITECTURE!r}")
    for key in ("num_layers", "d_model", "num_heads", "mlp_dim"):
        _positive_int(result[key], key)
    if result["d_model"] % result["num_heads"]:
        raise ValueError("network d_model must be divisible by num_heads")
    if (result["d_model"] // result["num_heads"]) % 2:
        raise ValueError("attention head dimension must be even for RoPE")
    return result


def validate_head_config(*args: Any, **kwargs: Any) -> dict:
    config = _single_config(args, kwargs)
    """Validate the independent own-auxiliary head schema."""
    if config is None:
        return dict(DEFAULT_HEAD_CONFIG)
    if not isinstance(config, Mapping):
        raise ValueError("head_config must be a dictionary")
    version = config.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version not in (0, 1):
        raise ValueError("head_config version must be 0 or 1")
    expected = {"version"} if version == 0 else {"version", "hidden_dim"}
    if set(config) != expected:
        raise ValueError("head_config must contain exactly the fields for its version")
    if version == 1:
        _positive_int(config["hidden_dim"], "head_config hidden_dim")
    return dict(config)


def validate_reading_config(*args: Any, **kwargs: Any) -> dict:
    config = _single_config(args, kwargs)
    """The Actor reading branch has one exact supported schema."""
    if config is None:
        return dict(DEFAULT_READING_CONFIG)
    if not isinstance(config, Mapping) or set(config) != set(DEFAULT_READING_CONFIG):
        raise ValueError("reading_config must contain exactly the supported fields")
    for key, value in DEFAULT_READING_CONFIG.items():
        actual = config[key]
        if type(actual) is not type(value) or actual != value:
            raise ValueError(f"unsupported reading_config {key}")
    return dict(config)


def validate_critic_config(*args: Any, **kwargs: Any) -> dict:
    config = _single_config(args, kwargs)
    """Validate an exact critic schema without inferring it from parameters."""
    if not isinstance(config, Mapping):
        raise ValueError("critic_config must be an explicit dictionary")
    version = config.get("version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("critic_config version must be an integer")
    expected = LEGACY_CRITIC_CONFIG if version == 0 else DEFAULT_CRITIC_CONFIG
    if version not in (0, 1) or set(config) != set(expected):
        raise ValueError("critic_config must contain exactly the fields for its version")
    for key, value in expected.items():
        actual = config[key]
        if type(actual) is not type(value) or actual != value:
            raise ValueError(f"unsupported critic_config {key}")
    return dict(config)


def _single_config(args, kwargs):
    """Accept both ``(config)`` and the legacy ``(kind, config)`` call shape."""
    if kwargs:
        return dict(kwargs)
    if len(args) == 1:
        return args[0]
    if len(args) == 2 and isinstance(args[1], str):
        return args[0]
    if len(args) == 2 and isinstance(args[0], str):
        return args[1]
    raise TypeError(f"expected one config mapping, got {len(args)} positional arguments")


def checkpoint_network_config(checkpoint: Any) -> dict:
    """Read the root backbone provenance; missing fields are an error, not a default."""
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a root dictionary")
    if "network_config" not in checkpoint or checkpoint["network_config"] is None:
        raise ValueError("checkpoint root network_config must be an explicit dictionary")
    return validate_network_config(checkpoint["network_config"])


def checkpoint_head_config(checkpoint: Any) -> dict:
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a root dictionary")
    if "head_config" not in checkpoint or checkpoint["head_config"] is None:
        raise ValueError("checkpoint root head_config must be an explicit dictionary")
    return validate_head_config(checkpoint["head_config"])


def checkpoint_reading_config(checkpoint: Any) -> dict:
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a root dictionary")
    if "reading_config" not in checkpoint or checkpoint["reading_config"] is None:
        raise ValueError("checkpoint root reading_config must be an explicit dictionary")
    return validate_reading_config(checkpoint["reading_config"])


def checkpoint_critic_config(checkpoint: Any) -> dict:
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a root dictionary")
    if "critic_config" not in checkpoint:
        return dict(LEGACY_CRITIC_CONFIG)
    return validate_critic_config(checkpoint["critic_config"])


def model_head_config(model: Any) -> dict:
    """Return serializable head provenance for a live model."""
    return validate_head_config(getattr(model, "head_config", None))


def model_reading_config(model: Any) -> dict:
    """Return serializable reading provenance for a live model."""
    return validate_reading_config(getattr(model, "reading_config", None))


_FEATURE_EXPORTS = {"PLAYER_FIELDS", "ROUND_FIELDS", "STATE_FIELDS",
                    "assemble_central_q_input", "build_central_q_input",
                    "build_privileged_q_state"}
_NETWORK_EXPORTS = {"build_actor", "build_central_q", "get_network_cls",
                    "get_central_q_network_cls", "adapt_network_observation"}


def __getattr__(name):
    if name in _FEATURE_EXPORTS:
        from . import feature

        return getattr(feature, name)
    if name in _NETWORK_EXPORTS:
        from . import network

        return getattr(network, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
