# Phase 5: 2000 Steps 延长验证

> 验证 Phase 1-3 的最优配置在 4x 训练预算下是否仍然成立。
> 核心问题：Muon 的 27.6% 优势是早期收敛假象，还是真实改善？

## 约束

- **只修改** `trainer/train_pretrain.py` 的 CLI 参数，不改代码
- 每次实验固定 **2000 steps**（4x baseline）
- 用 **val_loss** 作为主指标
- 结果追加到 `results_phase5.tsv`
- 每轮实验判定：keep / discard / crash

## 实验计划（6 个实验，预计 ~50 分钟）

### 5.1 对照组（必跑）

| # | 配置 | 目的 |
|---|------|------|
| P5-1 | AdamW baseline: lr=5e-4, accum=8 | 2000 步基线 |
| P5-2 | AdamW 最优: lr=5e-4, accum=4 | Phase 1 最优的 AdamW 是否持续领先 |
| P5-3 | **Muon 最优**: muon_lr=0.02, accum=4 | Phase 3 最优是否持续领先 |

### 5.2 Muon 长训练调优（基于 P5-3 结果决定）

| # | 配置 | 假设 |
|---|------|------|
| P5-4 | Muon + cosine decay | 500 步时衰减过早，2000 步可能受益 |
| P5-5 | Muon + warmup=100 | 500 步时 warmup 浪费步数，2000 步占比仅 5%，可能有帮助 |
| P5-6 | Muon lr=0.015 | 更保守的 LR，看是否在长训练中更稳定 |

### 5.3 额外实验（如果时间允许）

| # | 配置 | 假设 |
|---|------|------|
| P5-7 | Muon + accum=2 | 500 步时 Muon+accum=2 稳定但慢（5.84），2000 步更多更新是否追上 |
| P5-8 | Muon 最优 → 5000 steps | 最终验证，看收敛曲线是否趋于平稳 |

## 执行规则

```
for each experiment:
  1. 运行 2000 steps（预计 ~8 分钟/次）
  2. 记录 val_loss, 时间
  3. 判定 keep/discard/crash
  4. 追加到 results_phase5.tsv
  5. P5-1/2/3 必须全跑；P5-4/5/6 基于结果动态决定
```

## 关键观察点

- **gap 是否收窄**：Muon vs AdamW 的差距从 27.6% 变成多少？
  - 如果仍 >15% → Muon 确实更优，推荐用于 full pretrain
  - 如果收窄到 <5% → 仅是早期收敛优势，full pretrain 意义不大
  - 如果反转 → AdamW 长训练更稳，Muon 仅适合快速搜索
- **cosine decay**：500 步时过早衰减，2000 步是否开始受益？
- **accum=4 vs accum=8**：500 步时 accum=4 胜，2000 步（500 次更新 vs 250 次）差距变化？

## 复现命令模板

```bash
# P5-1: AdamW baseline, 2000 steps
cd minimind/trainer
PYTORCH_ENABLE_MPS_FALLBACK=1 python train_pretrain.py \
  --device mps --dtype bfloat16 --epochs 1 \
  --batch_size 8 --accumulation_steps 8 \
  --learning_rate 5e-4 --optimizer adamw \
  --max_steps 2000 --num_workers 0 --log_interval 200

# P5-2: AdamW 最优, 2000 steps
PYTORCH_ENABLE_MPS_FALLBACK=1 python train_pretrain.py \
  --device mps --dtype bfloat16 --epochs 1 \
  --batch_size 8 --accumulation_steps 4 \
  --learning_rate 5e-4 --optimizer adamw \
  --max_steps 2000 --num_workers 0 --log_interval 200

# P5-3: Muon 最优, 2000 steps
PYTORCH_ENABLE_MPS_FALLBACK=1 python train_pretrain.py \
  --device mps --dtype bfloat16 --epochs 1 \
  --batch_size 8 --accumulation_steps 4 \
  --optimizer muon --muon_lr 0.02 --learning_rate 3e-4 \
  --max_steps 2000 --num_workers 0 --log_interval 200

# P5-4: Muon + cosine decay, 2000 steps
# （需要在代码中启用 cosine schedule for muon_lr — 参考实验 17 的实现）
PYTORCH_ENABLE_MPS_FALLBACK=1 python train_pretrain.py \
  --device mps --dtype bfloat16 --epochs 1 \
  --batch_size 8 --accumulation_steps 4 \
  --optimizer muon --muon_lr 0.02 --learning_rate 3e-4 \
  --max_steps 2000 --num_workers 0 --log_interval 200

# P5-5: Muon + warmup=100, 2000 steps
PYTORCH_ENABLE_MPS_FALLBACK=1 python train_pretrain.py \
  --device mps --dtype bfloat16 --epochs 1 \
  --batch_size 8 --accumulation_steps 4 \
  --optimizer muon --muon_lr 0.02 --learning_rate 3e-4 \
  --max_steps 2000 --num_workers 0 --log_interval 200

# P5-6: Muon lr=0.015, 2000 steps
PYTORCH_ENABLE_MPS_FALLBACK=1 python train_pretrain.py \
  --device mps --dtype bfloat16 --epochs 1 \
  --batch_size 8 --accumulation_steps 4 \
  --optimizer muon --muon_lr 0.015 --learning_rate 3e-4 \
  --max_steps 2000 --num_workers 0 --log_interval 200
```

## 结果记录模板

```tsv
experiment	val_loss	time_min	status	description
P5-1	???	~8	???	AdamW baseline 2000steps
P5-2	???	~8	???	AdamW accum=4 2000steps
P5-3	???	~8	???	Muon lr=0.02 accum=4 2000steps
P5-4	???	~8	???	Muon cosine decay 2000steps
P5-5	???	~8	???	Muon warmup=100 2000steps
P5-6	???	~8	???	Muon lr=0.015 2000steps
```

## 完成后的决策

基于 Phase 5 结果，决定 full pretrain 策略：

- **如果 Muon 优势持续** → 用 Muon 跑完整 pretrain → 进入 SFT 阶段
- **如果 gap 收窄** → 考虑 Muon warmup + cosine decay 组合，或回归 AdamW
- **如果反转** → 用 AdamW accum=4 跑完整 pretrain

---

*Phase 5 of minimind-autoresearch | 2026-03-26*
