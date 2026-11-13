# AGENTS.md

本文件是 `mahjong-lab` 仓库内的 agent 工作约定。外层工作区 `/home/ralph/Repos/mahjong-lab/AGENTS.md` 仍然适用；本文件只补充本仓库的架构、ABI 和验证要求。

## 1. 仓库定位

本仓库以上游 MahJax 0.3.3 为基线，并迁入了三人麻将的 V5 VRPO 训练栈：

- `mahjax/`：环境、规则、状态和原始观测。三人麻将入口是 `mahjax.mahjong_3p`。
- `networks/`：网络侧代码。`networks.mahjong_3p` 包含特征化、actor/critic、Transformer 和 Central-Q。
- `train/`：BC 初始化、自对弈收集、Q calibration、VRPO 更新、checkpoint、在线评测和 CLI。
- `tests/networks/`、`tests/train/`：迁移后的契约测试。

不要把 `networks/` 或 `train/` 当成上游 MahJax 的稳定公共 API；它们仍在快速演进，修改时必须同时跑对应测试。

## 2. 开工前检查

```bash
pwd
git status --short
git branch --show-current
```

要求：

- 确认当前目录是本仓库根目录，不要把外层工作区当成 Git 根。
- 不要 reset、checkout 或覆盖未提交的改动；先判断改动是否属于当前任务。
- 本地默认使用 `.venv`，优先 CPU 跑测试：

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q
```

- 未经明确授权，不在本机或服务器启动长时间训练、大规模 self-play 或 GPU 实验。
- 服务器训练必须先确认现有任务状态；不要假定上一次训练仍在运行。

## 3. 环境与常用命令

```bash
# 快速检查本次改动最相关的测试
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q tests/networks tests/train

# 全量测试（较慢，当前约 9-10 分钟）
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q

# 只检查本次改动的格式/静态问题
.venv/bin/python -m ruff check mahjax networks train tests
.venv/bin/python -m ruff format --check mahjax networks train tests
```

`Makefile` 的 `format`/`check` 只覆盖上游 `mahjax/`，不覆盖 `networks/` 和 `train/`。不要因为 `make check` 通过就认为新增代码也通过了 lint。

## 4. 目录速查

| 路径 | 职责 |
|---|---|
| `mahjax/mahjong_3p/action.py` | 三人麻将动作；模拟器使用 88 个动作，`KITA=87` |
| `mahjax/mahjong_3p/observation.py` | `_observe_dict`，原始观测 |
| `mahjax/common/hand.py` | `Hand.to_34`，红五折叠 |
| `networks/mahjong_3p/feature.py` | `prepare_observation`、V3 公共特征、tokenizer、compact/canonical 映射 |
| `networks/mahjong_3p/network.py` | **模拟器 ↔ 网络唯一转换层**：`canonical_tile_to_compact`、`compact_tile_to_canonical`、`canonical_action_to_compact`、`compact_action_to_simulator`；`compact_observation`、`compact_legal_mask`、`to_simulator_action`、`observe_for_network`、`build_from_checkpoint` |
| `networks/mahjong_3p/transformer.py` | `ActorCriticNet`、`CentralQNet`、`TransformerEncoder`、`ReadingHead` |
| `train/selfplay.py` | BC 加载、collector、execution block、checkpoint artifact 写出 |
| `train/calibration.py` | Q calibration、接受门槛、actor fingerprint |
| `train/update.py` | VRPO/Paged VRPO 更新 |
| `train/checkpoint.py` | contract 校验、model/resume checkpoint |
| `train/eval.py` | online 双向 greedy 评测、baseline 加载、TensorBoard 标量写出 |
| `train/cli.py` | `calibrate` / `train` / `resume` 编排入口 |
| `train/health.py` | actor 停滞保护 |
| `data/` | 本地模型和实验数据；被 `.gitignore` 忽略 |

## 5. 必须遵守的 ABI

### 5.1 原始模拟器侧

- 三人麻将模拟器的动作空间是 **88**，其中 `Action.KITA = 87`。
- 原始观测使用 canonical-34 牌轴；标量牌 id（如 `last_draw`、`target`、`dora_indicators`、meld target、deck）使用 canonical `0..33` 加红五 `34/35/36`，不是 30 槽的 local red 布局。
- `state.players.hand` 是 34 计数；`hand_with_red` 使用 37 槽的全量 red id（红五为 34/35/36）。
- 30 槽的 `27/28/29` red 布局只属于网络侧的 compact action/tile ABI（见 5.3），不要拿它解释原始观测标量。

### 5.2 网络侧

- Actor/Critic 使用 **compact-27** 牌轴和 **87** 个动作。
- `KITA` 在网络动作中的位置是 **78**，模拟器侧是 **87**。
- 转换只允许发生在边界层：
  - 观测：`compact_observation(raw)`；
  - mask：`compact_legal_mask(env_mask)`；
  - 动作：`to_simulator_action(network_action)`。
- 标量牌 id 与动作 id 的逐项映射由 `network.py` 的显式函数完成：`canonical_tile_to_compact`、`compact_tile_to_canonical`、`canonical_action_to_compact`、`compact_action_to_simulator`。`compact_observation`/`compact_legal_mask` 只调用这些函数，不在输入函数里内联另一套映射。
- 不要在 `networks/` 或 `train/` 的其他位置重新实现映射，也不要把 88 动作 mask 直接传给网络。
- `build_central_q_input(pre_state, actor_observation)` 要求传入 **已 compact/prepared 的 actor observation**，不是原始 `_observe_dict` 输出。

### 5.3 红五映射

```text
local 28 (red 5p) -> canonical 13 -> compact 6
local 29 (red 5s) -> canonical 22 -> compact 15
raw local 27 (red 5m) -> invalid (-1)
full red 34/35/36 -> compact 27/28/29
compact 27 (red 5m) -> canonical 4
```

任何新增的观测字段、meld 处理或 action 统计都必须复用这套映射；直接用 `Tile.is_tile_red` 或 canonical-34 索引处理三人麻将本地 id 会产生静默错位。

### 5.4 特征与 reading target

- `augment_observation` 是 V3 公共特征接口，输入必须是 compact-27 观测和 `(87,)` mask。
- `history_overflow` 不应作为额外观测键传给 `validate_actor_observation`；只在特征构造时使用。
- `build_reading_targets` 必须把不存在的 2m–8m canonical 槽位标为 invalid：
  `danger_valid = exists & valid & danger_context`，`waits_valid = exists & valid & tenpai`。
  否则 execution block 会被 `validate_reading_targets` 整体拒绝。

## 6. V5 / VRPO 契约

- 当前生产架构是 `v5`；网络、特征和 checkpoint 只维护当前 ABI，不增加兼容分支。
- v5 骨架也支持非 reading 的旧算法 schema：此时 contract 不包含 `reading_config`，算法为
  `{"version": 2, "kind": "vrpo_qboost_chance_residual_v2", "reading_auxiliary": False}`。
  只有 `reading_auxiliary=True` 才要求 `reading_config` 和 reading head。
- 标准 reading algorithm 配置：

```python
{"version": 5, "kind": "vrpo_actor_reading_v5", "reading_auxiliary": True}
```

- Calibration 接受条件很严格：held-out MSE 必须严格优于 zero baseline 和 initial Critic，并满足最小相对提升；同时要求：
  - `checkpoint_kind == "vrpo_q_calibrated"`；
  - `critic_trained is True`；
  - fresh optimizer slots / no optimizer state；
  - calibration 前后 actor fingerprint 一致。
- 不要为了通过测试放宽这些门槛；如果数据量太小导致没有严格提升，应增加数据或训练，而不是改 acceptance 语义。
- `actor_lr=4e-4` 在在线第一阶段会触发 `actor_candidate_kl > 0.01`，导致 actor epoch 被拒绝。生产验证过的配置使用 `actor_lr=2.5e-5`、`critic_lr=4e-4`；不要在没有 KL 证据的情况下把 actor LR 调大。
- `train/cli.py` 的 `profile` 模式尚未实现；当前可用并且已经端到端验证的是 `calibrate`、`train`、`resume`。

## 7. CLI 与训练产物

入口是 `train.cli`：

```bash
.venv/bin/python -m train.cli calibrate --config CONFIG.json
.venv/bin/python -m train.cli train --config CONFIG.json \
  --artifact CALIBRATED.pkl --sha256 SHA256
.venv/bin/python -m train.cli resume --config CONFIG.json \
  --artifact BOUNDARY.pkl --sha256 SHA256
```

注意：

- 输出目录由 config 里的 `output_dir` 指定，CLI 没有 `--output-dir` 参数；该目录必须是尚不存在的目录，`_new_output` 使用 `mkdir(exist_ok=False)`。
- online metrics 默认写到 `<output_dir>/metrics.jsonl`；可用 config 的 `metrics_path` 或 CLI 的 `--metrics-path` 指定其他文件。相对路径按进程 cwd 解析，父目录会自动创建；目标文件已存在时拒绝覆盖。
- online 运行的 config 必须显式设置 `max_updates` 或 `max_wall_seconds`，否则会拒绝启动；CLI 没有对应的 `--max-updates` / `--max-wall-seconds` 参数。
- 每完成 `eval_interval` 个 update，`train/eval.py` 会在 online 循环里对 baseline 做双向 greedy 评测：两个方向各 `nums_eval` 局，共 `2*nums_eval` 局。双方都必须用 `argmax` 选动作，不能采样；评测环境使用 `round_end="confirm"`，否则会跳过中间的和了/流局事件。
- 评测结果写入该 update 的 `metrics.jsonl` 行的 `eval` 字段；`eval.learner` 和 `eval.baseline` 各含同一组指标。装了 `tensorboard` 时还会写 TensorBoard run root 下的 `learner` 和 `baseline` 两个子目录，两边使用相同 tag，因此在 TensorBoard 中同一指标显示为两条曲线。run root 默认是 `<output_dir>/tensorboard`，可用 config 的 `tensorboard_dir` 覆盖；相对路径按进程 cwd 解析。
- 装了 `tensorboard` 时，每个 completed update 还会把该行的训练字段写入 learner run 的 `train/*`：嵌套字段用 `/` 连接（如 `train/progress/envsteps`、`train/update/actor_steps`、`train/execution/cache/hits`），标量 `wall_seconds` 写成 `train/wall_seconds`；`eval` 子树不会重复写成 `train/eval/*`。
- eval 相关 config 字段：`eval_interval`（默认 5）、`baseline_model`（默认本次 online 起始的 BC actor；也可给 checkpoint 路径）、`baseline_sha256`（给路径时必填）、`nums_eval`（默认 30）；TensorBoard 相关字段是 `tensorboard_dir`（默认 `<output_dir>/tensorboard`）。`baseline_model` 的 SHA 与 `bc_sha256` 一样在反序列化前校验。
- 每个完整 update 才产生可 resume 的 boundary；中断的 rollout 不是 resume 点。
- 训练产物、日志和 `data/` 不应提交。模型文件提交前先确认 `.gitignore` 和仓库策略。
- checkpoint 加载前校验 SHA-256；不要直接 `pickle.load` 未经校验的外部文件。
- 报告训练结果时，必须区分：单元测试通过、真实环境 smoke 通过、完整训练收敛。三者不能互相替代。

## 8. 测试与修改原则

### 8.1 提交前的强制检查

**代码、测试或配置修改在提交前，必须执行以下三项，缺一不可：**

```bash
# 1. 格式化
.venv/bin/python -m ruff format mahjax networks train tests

# 2. 静态检查
.venv/bin/python -m ruff check mahjax networks train tests

# 3. 测试
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q
```

要求：

- 三项都必须实际执行，不能只跑其中一项或用“改动很小”为理由跳过。
- 这些检查在 `git commit` 前针对待提交的最终状态执行即可；日常编辑过程中不需要每改一次就重复全量检查。
- 如果 `ruff format` 改动了文件，必须重新运行 `ruff check` 和测试。
- 全量测试较慢时可以先跑相关子集定位问题，但交付前仍必须跑完整测试。
- 报告中写明三项命令的实际结果；如果有失败，不能声称任务完成。
- **纯文档修改（`AGENTS.md`、`README.md`、`docs/` 等）不需要运行测试或 ruff。** 但如果文档修改涉及命令、路径、API 或数值，必须与当前代码核对。

### 8.2 修改原则

- 先读相关实现和测试，再做最小改动；不要为了让测试变绿而删除契约断言。
- 若测试反映的是目标契约、而迁移代码确实有 bug，优先修代码；若测试引用当前不存在的 API，则更新测试到当前 ABI。
- 修改 `feature.py`、`network.py`、`selfplay.py`、`auxhead.py` 后，提交前至少跑：

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q \
  tests/networks/test_feature.py \
  tests/networks/test_feature_public.py \
  tests/networks/test_transformer_central_q.py \
  tests/train/test_selfplay.py \
  tests/train/test_calibration.py
```

- 修改公开行为或 ABI 后，在提交前跑全量测试。最近一次全量基线是 `818 passed, 1 skipped, 21 subtests passed`。
- 测试中的 JAX/NumPy 数值断言优先用 `jnp.allclose` / 明确 tolerance，不要依赖精确浮点相等。

## 9. 当前已知限制

- `data/calibrated.pkl` 是本地校准产物，不进入 Git；重新生成后应校验 SHA 和 `load_verified` 的 contract。
- 仓库文档 `docs/` 仍然主要是上游 MahJax 的环境文档，`networks/` 与 `train/` 的说明以代码和测试为准。

## 10. 交付时报告

完成修改后，至少说明：

1. 改了哪些文件、为什么；
2. 跑了哪些测试，结果是什么；
3. 是否在真实环境/服务器上验证过，以及具体命令和产物路径；
4. 还有哪些未验证项或已知限制。
