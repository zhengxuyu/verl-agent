# GiGPO 历史轨迹复用改动说明（中文）

本文档总结了为 GiGPO 增加“多轮训练历史轨迹复用”所做的改动，并保证参数更新阶段仍然是 on-policy。

## 1. 目标

你希望实现的核心流程是：

1. 训练集有固定 query（例如 100 条）。
2. 每步训练取一个 query（如 `batch_size=1`），并 rollout `n=8` 条轨迹形成 group。
3. 这 8 条轨迹的完整信息要保存为 memory。
4. 下一轮训练遇到同一个 query 时，把这些历史轨迹加入当前 group，作为参考信号。
5. 但最终梯度更新仍只用当前 policy 新生成的样本（保持 on-policy）。

现在代码已经按这个目标实现。

## 2. 变更文件

1. `agent_system/multi_turn_rollout/rollout_loop.py`
2. `verl/trainer/config/ppo_trainer.yaml`
3. `verl/trainer/ppo/ray_trainer.py`

## 3. 关键改动

### 3.1 稳定任务标识（跨轮对齐）

文件：`agent_system/multi_turn_rollout/rollout_loop.py`

- 新增 `task_uid`，格式：`{data_source}::{index}`。
- `index` 使用数据集中的稳定索引（`extra_info.index`），不再使用 batch 内局部 `item`。

作用：

- 不同训练轮次可以稳定地把“同一个 query”的历史轨迹匹配回来。

### 3.2 新增配置项

文件：`verl/trainer/config/ppo_trainer.yaml`

新增：

- `algorithm.gigpo.history_replay.enable`
- `algorithm.gigpo.history_replay.save_path`
- `algorithm.gigpo.history_replay.load_path`
- `algorithm.gigpo.history_replay.max_trajs_per_task`
- `algorithm.gigpo.history_replay.max_saved_trajs_per_task`

说明：

- `max_trajs_per_task`：每次训练时，每个任务最多回放多少条历史轨迹。
  - `<=0` 表示回放该任务下所有已保存轨迹（默认 `-1`）。
- `max_saved_trajs_per_task`：每个任务最多保存多少条历史轨迹（跨轮累计）。
  - `<=0` 表示不设上限（默认 `-1`）。

### 3.3 训练器中的 history replay 管线

文件：`verl/trainer/ppo/ray_trainer.py`

新增方法：

- `_init_history_replay`
- `_rebuild_history_replay_index`
- `_collect_history_replay_from_batch`
- `_merge_history_replay_into_batch`
- `_prepare_onpolicy_update_batch`
- `_persist_history_replay`

行为如下：

1. 启动加载：
   - 优先从 `load_path` 加载。
   - 若 `load_path` 为空，则自动尝试从 `save_path` 加载。

2. 当前步保存：
   - 对当前 batch 中每个 `task_uid`，会把该任务下“当前步生成的所有 traj（例如 8 条）”都保存。
   - 会按 `max_saved_trajs_per_task` 做可选上限控制。

3. 训练时并入历史：
   - 按 `task_uid` 找到历史 traj 并拼到当前 batch。
   - 将历史样本的 `uid` 改成当前任务组的 `uid`，使 GiGPO 分组统计一致。
   - 标记 `is_history_replay=True`。

4. 保持 on-policy 更新：
   - reward/adv 先在“当前 + 历史”扩展组上计算。
   - 在 `update_critic` / `update_actor` 前，剔除 `is_history_replay=True` 的样本。
   - 最终只用当前 policy rollout 样本更新参数。

5. 落盘：
   - checkpoint 保存时落盘。
   - 训练结束时再落盘一次。

## 4. 当前语义（与你目标对齐）

以 `batch_size=1, rollout_n=8` 为例：

1. 当前 query rollout 得到 8 条新轨迹。
2. 这 8 条都会被保存到本地 history（默认不设上限）。
3. 下一轮再次遇到同一 query 时，会把该 query 的历史轨迹加入当前 group（默认全部加入）。
4. reward/adv 用扩展组计算；梯度更新仅使用当前 8 条新轨迹。

## 5. 推荐配置

```bash
algorithm.gigpo.history_replay.enable=True
algorithm.gigpo.history_replay.save_path=/path/to/history.pkl
algorithm.gigpo.history_replay.load_path=/path/to/history.pkl
algorithm.gigpo.history_replay.max_trajs_per_task=-1
algorithm.gigpo.history_replay.max_saved_trajs_per_task=-1
```

备注：

- 如果不写 `load_path`，代码会自动尝试从 `save_path` 读取历史。
- 多轮训练（第 2/3/4 轮）使用同一个 `save_path` 即可持续累积。

## 6. 可观测日志指标

启用后可在日志看到：

- `history_replay/collected_tasks`
- `history_replay/collected_trajs`
- `history_replay/collected_rows`
- `history_replay/stored_tasks`
- `history_replay/replayed_trajs`
- `history_replay/replayed_rows`
- `history_replay/filtered_for_update_rows`
- `history_replay/update_batch_rows`

可用于确认：

- 当前步是否把所有 rollout 轨迹保存了；
- 下一轮是否正确回放；
- 更新前是否已过滤历史样本。
