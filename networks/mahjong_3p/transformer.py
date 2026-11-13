"""Actor/Critic transformer, reading heads, and the training-only central Q.

Four parameter roots are fixed: ``policy_extractor``, ``policy_mlp``,
``critic_extractor`` and ``value_critic_mlp``. Actor and Critic never share
parameters. Public tokens are the only observation input; the central Q is a
training oracle that sees privileged state through ``feature`` helpers.

Reading is a supervised Actor-only sibling branch: it never feeds the action
policy, and its parameters live under ``policy_mlp/reading``.
"""

from collections.abc import Mapping
from dataclasses import field

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core import FrozenDict, freeze, unfreeze

from .configs import (
    ARCHITECTURE,
    DEFAULT_CRITIC_CONFIG,
    DEFAULT_HEAD_CONFIG,
    DEFAULT_READING_CONFIG,
    validate_critic_config,
    validate_head_config,
    validate_reading_config,
)
from .feature import (
    COMPACT_TO_CANONICAL,
    PLAYER_FIELDS,
    ROUND_FIELDS,
    STATE_FIELDS,
    ObservationTokenizer,
    assemble_central_q_input,
    local_tile_to_canonical,
    raw_tile_to_canonical,
    token_positions,
    validate_actor_observation,
)

# --- actor/critic backbone -----------------------------------------------------


def temporal_rope(x, positions, theta=10000.0):
    """Rotate Q/K [B,T,H,Dh] by public-event age; current tokens have age 0."""
    dim = x.shape[-1]
    if dim % 2:
        raise ValueError("attention head dimension must be even for RoPE")
    freq = theta ** (-jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
    angle = positions[..., None, None].astype(jnp.float32) * freq
    cosine, sine = jnp.cos(angle), jnp.sin(angle)
    even, odd = x[..., 0::2], x[..., 1::2]
    return jnp.stack((even * cosine - odd * sine, even * sine + odd * cosine), axis=-1).reshape(x.shape)


def masked_mean(x, mask):
    weights = mask.astype(x.dtype)[..., None]
    return (x * weights).sum(1) / jnp.maximum(weights.sum(1), 1)


class SwiGLU(nn.Module):
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        up = nn.Dense(self.hidden_dim, use_bias=False, name="up")(x)
        gate = nn.Dense(self.hidden_dim, use_bias=False, name="gate")(x)
        return nn.Dense(self.output_dim, use_bias=False, name="down")(nn.silu(up) * gate)


class TransformerBlock(nn.Module):
    d_model: int
    num_heads: int
    mlp_dim: int

    @nn.compact
    def __call__(self, x, mask, positions):
        head_dim = self.d_model // self.num_heads
        y = nn.RMSNorm(epsilon=1e-6, name="attention_norm")(x)
        qkv = nn.Dense(3 * self.d_model, use_bias=False, name="qkv")(y)
        q, k, v = [part.reshape(*x.shape[:2], self.num_heads, head_dim) for part in jnp.split(qkv, 3, axis=-1)]
        q, k = temporal_rope(q, positions), temporal_rope(k, positions)
        scores = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(float(head_dim))
        scores = jnp.where(mask[:, None, None, :], scores, -jnp.inf)
        weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(v.dtype)
        attended = jnp.einsum("bhts,bshd->bthd", weights, v).reshape(x.shape)
        x = x + nn.Dense(self.d_model, use_bias=False, name="attention_out")(attended)
        y = nn.RMSNorm(epsilon=1e-6, name="ffn_norm")(x)
        x = x + SwiGLU(self.mlp_dim, self.d_model, name="ffn")(y)
        return x * mask[..., None]


class TransformerEncoder(nn.Module):
    num_layers: int = 12
    d_model: int = 256
    num_heads: int = 8
    mlp_dim: int = 704

    @nn.compact
    def __call__(self, obs):
        spec = ObservationTokenizer
        x, mask = spec(d_model=self.d_model)(obs)
        mask = mask.astype(jnp.bool_)
        positions = token_positions(obs)
        for index in range(self.num_layers):
            x = TransformerBlock(self.d_model, self.num_heads, self.mlp_dim, name=f"block_{index}")(x, mask, positions)
        x = nn.RMSNorm(epsilon=1e-6, name="final_norm")(x) * mask[..., None]
        tiles, tile_mask = x[:, :34], mask[:, :34]
        tile_max = jnp.max(jnp.where(tile_mask[..., None], tiles, -jnp.inf), axis=1)
        summary = jnp.concatenate(
            [
                x[:, 254],
                x[:, 34],
                masked_mean(tiles, tile_mask),
                tile_max,
                masked_mean(x[:, 54:254], mask[:, 54:254]),
            ],
            axis=-1,
        )
        summary = nn.silu(nn.Dense(self.d_model, name="summary")(summary))
        # Contextual relative opponents (next, previous) share this public trunk.
        return {"summary": summary, "tiles": tiles, "opponents": x[:, 35:37]}


def _selected_tile(tiles, raw_ids):
    """Gather red-aware compact observation IDs, with a zero null identity."""
    ids = jnp.asarray(raw_ids, jnp.int32).reshape(-1)
    canonical = local_tile_to_canonical(ids)
    selected = tiles[jnp.arange(tiles.shape[0]), jnp.maximum(canonical, 0)]
    return jnp.where((canonical >= 0)[:, None], selected, 0.0)


class AuxiliaryHead(nn.Module):
    """Own final rank and remaining-score predictions from one side's summary."""

    hidden_dim: int = 128

    @nn.compact
    def __call__(self, summary):
        hidden = nn.silu(nn.Dense(self.hidden_dim, name="hidden")(summary))
        return {
            "rank_logits": nn.Dense(3, name="rank")(hidden),
            "score_delta": nn.Dense(1, name="score")(hidden)[..., 0],
        }


class PolicyHead(nn.Module):
    d_model: int = 128
    auxiliary_hidden_dim: int | None = 128

    @nn.compact
    def __call__(self, features, obs, *, with_aux=False):
        tiles, summary = features["tiles"], features["summary"]
        context = jnp.broadcast_to(summary[:, None], tiles.shape)
        local = SwiGLU(2 * self.d_model, self.d_model, name="local")(jnp.concatenate([tiles, context], axis=-1))
        discard = nn.Dense(2, kernel_init=nn.initializers.orthogonal(0.01), name="discard")(local)
        compact = jnp.asarray(COMPACT_TO_CANONICAL)
        # Slot 27 is reserved red 5m, permanently illegal in sanma. Keep the ABI.
        discard = jnp.concatenate([discard[:, compact, 0], discard[:, jnp.array([4, 13, 22]), 1]], axis=-1)
        kan = nn.Dense(1, kernel_init=nn.initializers.orthogonal(0.01), name="self_kan")(local)
        kan = kan[:, compact, 0]

        # Tail slots 57..86 include unused slots. Legality is NOT inferred here.
        actions = jnp.arange(57, 87)
        action_emb = nn.Embed(87, self.d_model, name="action")(actions)[None]
        target = _selected_tile(tiles, obs["target"])
        draw = _selected_tile(tiles, obs["last_draw"])
        target_action = ((actions >= 74) & (actions <= 77)).astype(tiles.dtype)
        draw_action = ((actions == 71) | (actions == 73)).astype(tiles.dtype)
        north_action = (actions == 78).astype(tiles.dtype)
        relevant = (
            target[:, None] * target_action[None, :, None]
            + draw[:, None] * draw_action[None, :, None]
            + tiles[:, 30:31] * north_action[None, :, None]
        )
        context = jnp.broadcast_to(summary[:, None], relevant.shape)
        action_emb = jnp.broadcast_to(action_emb, relevant.shape)
        special = SwiGLU(2 * self.d_model, self.d_model, name="special_hidden")(
            jnp.concatenate([context, relevant, action_emb], axis=-1)
        )
        special = nn.Dense(1, kernel_init=nn.initializers.orthogonal(0.01), name="special")(special)[..., 0]
        logits = jnp.concatenate([discard, kan, special], axis=-1)
        aux = {}
        # Regular init creates the complete tree, while inference skips aux work.
        if self.auxiliary_hidden_dim is not None and (with_aux or self.is_initializing()):
            aux = AuxiliaryHead(self.auxiliary_hidden_dim, name="auxiliary")(summary)
        return (logits, aux) if with_aux else logits


class ValueHead(nn.Module):
    d_model: int = 128
    auxiliary_hidden_dim: int | None = 128

    @nn.compact
    def __call__(self, features, *, with_aux=False):
        x = SwiGLU(2 * self.d_model, self.d_model, name="hidden")(features["summary"])
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0), name="value")(x)[..., 0]
        aux = {}
        if self.auxiliary_hidden_dim is not None and (with_aux or self.is_initializing()):
            aux = AuxiliaryHead(self.auxiliary_hidden_dim, name="auxiliary")(features["summary"])
        return (value, aux) if with_aux else value


# --- reading heads -------------------------------------------------------------


class BilinearHead(nn.Module):
    """One task's shared-across-opponents seat/tile projections and scalar bias."""

    latent_dim: int = 64

    @nn.compact
    def __call__(self, opponents, tiles):
        seat = nn.Dense(self.latent_dim, name="seat")(opponents).astype(jnp.float32)
        tile = nn.Dense(self.latent_dim, name="tile")(tiles).astype(jnp.float32)
        bias = self.param("bias", nn.initializers.zeros_init(), ())
        return jnp.einsum("bod,btd->bot", seat, tile) / jnp.sqrt(jnp.asarray(self.latent_dim, jnp.float32)) + bias


class ReadingHead(nn.Module):
    """No seat-specific parameters, stop-gradient, legality mask, or oracle data."""

    latent_dim: int = 64

    @nn.compact
    def __call__(self, features):
        opponents, tiles = features["opponents"], features["tiles"]
        return {
            "tenpai_logits": nn.Dense(1, name="tenpai")(opponents)[..., 0],
            "shanten_logits": nn.Dense(4, name="shanten")(opponents),
            "danger_logits": BilinearHead(self.latent_dim, name="danger")(opponents, tiles),
            "waits_logits": BilinearHead(self.latent_dim, name="waits")(opponents, tiles),
            "danger_points": BilinearHead(self.latent_dim, name="danger_points")(opponents, tiles),
        }


class ReadingPolicyHead(PolicyHead):
    """Keep all action/own-aux paths, with a separate sibling reading subtree."""

    @nn.compact
    def __call__(self, features, obs, *, with_aux=False, with_reading=False):
        outputs = super().__call__(features, obs, with_aux=with_aux)
        reading = {}
        if with_reading or self.is_initializing():
            reading = ReadingHead(name="reading")(features)
        if with_reading:
            if with_aux:
                raise ValueError("request own auxiliary and opponent reading separately")
            return outputs, reading
        return outputs


class ActorCriticNet(nn.Module):
    """Deployable Actor and Critic trunks with an Actor-only reading sibling."""

    architecture: str = ARCHITECTURE
    num_layers: int = 12
    d_model: int = 256
    num_heads: int = 8
    mlp_dim: int = 704
    head_config: dict = field(default_factory=lambda: dict(DEFAULT_HEAD_CONFIG))
    reading_config: dict = field(default_factory=lambda: dict(DEFAULT_READING_CONFIG))

    def setup(self):
        if self.architecture != ARCHITECTURE:
            raise ValueError(f"ActorCriticNet requires architecture={ARCHITECTURE!r}")
        heads = validate_head_config(self.head_config)
        validate_reading_config(self.reading_config)
        if min(self.num_layers, self.d_model, self.num_heads, self.mlp_dim) < 1:
            raise ValueError("network dimensions must be positive")
        if self.d_model % self.num_heads or (self.d_model // self.num_heads) % 2:
            raise ValueError("d_model must divide num_heads with an even head dimension")
        config = dict(
            num_layers=self.num_layers,
            d_model=self.d_model,
            num_heads=self.num_heads,
            mlp_dim=self.mlp_dim,
        )
        self.policy_extractor = TransformerEncoder(**config)
        self.critic_extractor = TransformerEncoder(**config)
        aux_width = heads["hidden_dim"] if heads["version"] else None
        self.policy_mlp = ReadingPolicyHead(self.d_model, auxiliary_hidden_dim=aux_width)
        self.value_critic_mlp = ValueHead(self.d_model, auxiliary_hidden_dim=aux_width)

    def __call__(self, obs):
        return self.get_action_logits(obs), self.get_value(obs)

    def get_action_logits(self, obs):
        validate_actor_observation(obs)
        return self.policy_mlp(self.policy_extractor(obs), obs)

    def get_value(self, obs):
        validate_actor_observation(obs)
        return self.value_critic_mlp(self.critic_extractor(obs))

    def get_action_and_aux(self, obs):
        validate_actor_observation(obs)
        return self.policy_mlp(self.policy_extractor(obs), obs, with_aux=True)

    def get_value_and_aux(self, obs):
        validate_actor_observation(obs)
        return self.value_critic_mlp(self.critic_extractor(obs), with_aux=True)

    def get_action_and_reading(self, obs):
        """One Actor pass returns raw [B, 87] action logits and the five heads."""
        validate_actor_observation(obs)
        return self.policy_mlp(self.policy_extractor(obs), obs, with_reading=True)

    def get_training_outputs(self, obs):
        logits = self.get_action_logits(obs)
        values, critic_aux = self.get_value_and_aux(obs)
        return logits, values, critic_aux


def make_actor_critic(*, num_layers=12, d_model=256, num_heads=8, mlp_dim=704, head_config=None, reading_config=None):
    """Bind an explicit, checkpoint-recordable configuration to a zero-arg class."""
    layers, width, heads, hidden = num_layers, d_model, num_heads, mlp_dim
    bound_head = validate_head_config(head_config)
    bound_reading = validate_reading_config(reading_config)

    class BoundActorCriticNet(ActorCriticNet):
        num_layers: int = layers
        d_model: int = width
        num_heads: int = heads
        mlp_dim: int = hidden
        head_config: dict = field(default_factory=lambda: dict(bound_head))
        reading_config: dict = field(default_factory=lambda: dict(bound_reading))

    return BoundActorCriticNet


# --- central action Q ----------------------------------------------------------


def relative_q_to_absolute(q, current_player):
    """[B, relative payoff seat, action] -> [B, absolute payoff seat, action]."""
    if q.ndim != 3 or q.shape[1:] != (3, 87):
        raise ValueError("relative Q must have shape [batch,3,87]")
    seat = jnp.asarray(current_player, jnp.int32).reshape(-1)
    if seat.shape[0] != q.shape[0]:
        raise ValueError("Q/current-player batches must match")
    relative = (jnp.arange(3)[None, :] - seat[:, None]) % 3
    return jnp.take_along_axis(q, relative[..., None], axis=1)


def _batched(value, batch):
    return jnp.asarray(value)[None] if not batch else jnp.asarray(value)


def _scaled_flat(value):
    # Current state includes bitpacked uint16 melds/rivers and raw point caches.
    # A bounded-growth numeric transform keeps them finite without clipping data.
    value = jnp.asarray(value, jnp.float32)
    value = jnp.sign(value) * jnp.log1p(jnp.abs(value))
    return value.reshape(value.shape[0], -1)


class CompactPrivateEncoder(nn.Module):
    d_model: int

    @nn.compact
    def __call__(self, privileged):
        batch = jnp.shape(privileged["current_player"])
        seat = _batched(privileged["current_player"], batch).astype(jnp.int32)
        relative = (jnp.arange(3)[None] + seat[:, None]) % 3

        def rotated(value):
            value = _batched(value, batch)
            index = relative.reshape(relative.shape + (1,) * (value.ndim - 2))
            return jnp.take_along_axis(value, index, axis=1)

        # Counts remain compact in replay. Canonical34 expansion happens ONLY in
        # this micro-forward; red counts are also in the numeric state below.
        # The simulator already stores canonical-34 counts.
        hand_features = _scaled_flat(rotated(privileged["players"]["hand"]))
        deck = _batched(privileged["round_state"]["deck"], batch)
        canonical = raw_tile_to_canonical(deck)
        red = (deck >= 34) & (deck <= 36)
        ids = jnp.where(canonical >= 0, canonical + 1, 0)
        deck_features = nn.Embed(35, 4, name="deck_tile")(ids)
        deck_features += nn.Embed(2, 4, name="deck_red")(red.astype(jnp.int32))
        # Flatten preserves every wall POSITION including the dead wall and its
        # currently generated ura indicators; no bag-of-tiles information loss.
        numeric = [hand_features, deck_features.reshape(deck.shape[0], -1)]
        for key in PLAYER_FIELDS:
            if key != "hand":
                numeric.append(_scaled_flat(rotated(privileged["players"][key])))
        for key in ROUND_FIELDS:
            if key == "deck":
                continue
            value = privileged["round_state"][key]
            if key in ("init_wind", "seat_wind", "order_points", "score"):
                value = rotated(value)
            else:
                value = _batched(value, batch)
            numeric.append(_scaled_flat(value))
        for key in STATE_FIELDS:
            value = privileged[key]
            value = (
                rotated(value) if key in ("rewards", "pending_rewards", "pending_winners") else _batched(value, batch)
            )
            numeric.append(_scaled_flat(value))
        for key in sorted(privileged["policy_context"]):
            numeric.append(_scaled_flat(_batched(privileged["policy_context"][key], batch)))
        x = jnp.concatenate(numeric, axis=-1)
        x = nn.silu(nn.Dense(2 * self.d_model, name="input")(x))
        return nn.Dense(self.d_model, name="output")(x)


class CentralQExtractor(nn.Module):
    num_layers: int
    d_model: int
    num_heads: int
    mlp_dim: int

    @nn.compact
    def __call__(self, central_input):
        observation, privileged = central_input["observation"], central_input["privileged"]
        public = TransformerEncoder(self.num_layers, self.d_model, self.num_heads, self.mlp_dim, name="public_encoder")(
            observation
        )
        hidden = CompactPrivateEncoder(self.d_model, name="private_encoder")(privileged)
        summary = nn.silu(nn.Dense(self.d_model, name="fusion")(jnp.concatenate([public["summary"], hidden], axis=-1)))
        return {"summary": summary, "tiles": public["tiles"]}


class CentralQHead(nn.Module):
    d_model: int

    @nn.compact
    def __call__(self, features, observation):
        tiles, summary = features["tiles"], features["summary"]
        context = jnp.broadcast_to(summary[:, None], tiles.shape)
        local = SwiGLU(2 * self.d_model, self.d_model, name="local")(jnp.concatenate([tiles, context], axis=-1))
        # Fresh named output projections are EXACTLY zero, not scalar-V copies.
        output = dict(kernel_init=nn.initializers.zeros_init(), bias_init=nn.initializers.zeros_init())
        discard = nn.Dense(6, name="discard", **output)(local).reshape(*local.shape[:2], 3, 2)
        compact = jnp.asarray(COMPACT_TO_CANONICAL)
        discard = jnp.concatenate([discard[..., 0][:, compact], discard[..., 1][:, jnp.array([4, 13, 22])]], axis=1)
        kan = nn.Dense(3, name="self_kan", **output)(local)[:, compact]
        actions = jnp.arange(57, 87)
        action_emb = nn.Embed(87, self.d_model, name="action")(actions)[None]
        target = _selected_tile(tiles, observation["target"])
        draw = _selected_tile(tiles, observation["last_draw"])
        relevant = (
            target[:, None] * ((actions >= 74) & (actions <= 77))[None, :, None]
            + draw[:, None] * ((actions == 71) | (actions == 73))[None, :, None]
            + tiles[:, 30:31] * (actions == 78)[None, :, None]
        )
        context = jnp.broadcast_to(summary[:, None], relevant.shape)
        special = SwiGLU(2 * self.d_model, self.d_model, name="special_hidden")(
            jnp.concatenate([context, relevant, jnp.broadcast_to(action_emb, relevant.shape)], axis=-1)
        )
        special = nn.Dense(3, name="special", **output)(special)
        return jnp.concatenate([discard, kan, special], axis=1).transpose(0, 2, 1)


class CentralQNet(nn.Module):
    architecture: str = ARCHITECTURE
    num_layers: int = 12
    d_model: int = 256
    num_heads: int = 8
    mlp_dim: int = 704
    head_config: dict = field(default_factory=lambda: dict(DEFAULT_HEAD_CONFIG))
    critic_config: dict = field(default_factory=lambda: dict(DEFAULT_CRITIC_CONFIG))
    reading_config: dict | None = None

    def setup(self):
        if self.architecture != ARCHITECTURE:
            raise ValueError("central action Q requires the current architecture")
        validate_reading_config(self.reading_config)
        if min(self.num_layers, self.d_model, self.num_heads, self.mlp_dim) < 1:
            raise ValueError("central action Q network dimensions must be positive")
        if self.d_model % self.num_heads or (self.d_model // self.num_heads) % 2:
            raise ValueError("d_model must divide num_heads with even head dimension")
        if validate_critic_config(self.critic_config)["kind"] != "central_action_q":
            raise ValueError("central action Q requires version-1 critic_config")
        heads = validate_head_config(self.head_config)
        aux_width = heads["hidden_dim"] if heads["version"] else None
        policy_head = ReadingPolicyHead if self.reading_config is not None else PolicyHead
        # Exact modules, names AND paths of the same-architecture BC Actor.
        self.policy_extractor = TransformerEncoder(self.num_layers, self.d_model, self.num_heads, self.mlp_dim)
        self.policy_mlp = policy_head(self.d_model, auxiliary_hidden_dim=aux_width)
        self.critic_extractor = CentralQExtractor(self.num_layers, self.d_model, self.num_heads, self.mlp_dim)
        self.value_critic_mlp = CentralQHead(self.d_model)

    def __call__(self, actor_observation, central_input):
        """Explicit two-input initialization creates all four parameter roots."""
        return self.get_action_logits(actor_observation), self.get_action_q(central_input)

    def get_action_logits(self, actor_observation):
        validate_actor_observation(actor_observation)
        return self.policy_mlp(self.policy_extractor(actor_observation), actor_observation)

    def get_action_and_aux(self, actor_observation):
        validate_actor_observation(actor_observation)
        return self.policy_mlp(self.policy_extractor(actor_observation), actor_observation, with_aux=True)

    def get_action_and_reading(self, actor_observation):
        if self.architecture != ARCHITECTURE:
            raise ValueError("opponent reading requires the current architecture")
        validate_actor_observation(actor_observation)
        return self.policy_mlp(self.policy_extractor(actor_observation), actor_observation, with_reading=True)

    def get_action_q(self, central_input):
        if not isinstance(central_input, Mapping) or set(central_input) != {"observation", "privileged"}:
            raise ValueError("central input requires exactly observation and privileged")
        central_input = assemble_central_q_input(central_input["observation"], central_input["privileged"])
        features = self.critic_extractor(central_input)
        relative = self.value_critic_mlp(features, central_input["observation"])
        privileged = central_input["privileged"]
        q = relative_q_to_absolute(relative, privileged["current_player"])
        # Traced stale schema or current seat must fail closed.
        valid = (
            (jnp.asarray(privileged["state_input_schema"]).reshape(-1) == 1)
            & (jnp.asarray(privileged["current_player"]).reshape(-1) >= 0)
            & (jnp.asarray(privileged["current_player"]).reshape(-1) < 3)
        )
        return jnp.where(valid[:, None, None], q, jnp.nan)


def make_central_q_net(
    *, num_layers=12, d_model=256, num_heads=8, mlp_dim=704, head_config=None, critic_config=None, reading_config=None
):
    """Bind explicit same-architecture Actor, independent reading and Q schemas."""
    layers, width, heads, hidden = num_layers, d_model, num_heads, mlp_dim
    head = validate_head_config(head_config)
    reading = validate_reading_config(reading_config)
    critic = validate_critic_config(critic_config)
    if critic["kind"] != "central_action_q":
        raise ValueError("central action Q requires explicit version-1 critic_config")

    class SanmaCentralQNet(CentralQNet):
        architecture: str = ARCHITECTURE
        reading_config: dict = field(default_factory=lambda: dict(reading))
        num_layers: int = layers
        d_model: int = width
        num_heads: int = heads
        mlp_dim: int = hidden
        head_config: dict = field(default_factory=lambda: dict(head))
        critic_config: dict = field(default_factory=lambda: dict(critic))

    return SanmaCentralQNet


def clone_actor_and_public_encoder(source_variables, target_variables):
    """Explicit path/shape/dtype clone, never broadcasts scalar V into Q.

    Caller must first validate SAME source/target network and Actor head schemas.
    Private branch, fusion, action embeddings and all Q projections remain fresh.
    """
    source, target = unfreeze(source_variables), unfreeze(target_variables)
    if "params" not in source or "params" not in target:
        raise ValueError("clone requires complete variables with params root")
    paths = (
        (("policy_extractor",), ("policy_extractor",)),
        (("policy_mlp",), ("policy_mlp",)),
        (("policy_extractor",), ("critic_extractor", "public_encoder")),
    )
    for source_path, target_path in paths:
        a, b = source["params"], target["params"]
        for key in source_path:
            if key not in a:
                raise ValueError(f"missing source clone path {source_path}")
            a = a[key]
        for key in target_path:
            if key not in b:
                raise ValueError(f"missing target clone path {target_path}")
            b = b[key]
        if jax.tree.structure(a) != jax.tree.structure(b):
            raise ValueError(f"clone tree mismatch at {target_path}")
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
            if x.shape != y.shape or x.dtype != y.dtype:
                raise ValueError(f"clone shape/dtype mismatch at {target_path}")
        parent = target["params"]
        for key in target_path[:-1]:
            parent = parent[key]
        parent[target_path[-1]] = jax.tree.map(lambda x: jnp.array(x), a)
    return freeze(target) if isinstance(target_variables, FrozenDict) else target
