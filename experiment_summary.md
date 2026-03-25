# MiniMind AutoResearch 实验总结

> **日期**：2026-03-25
> **硬件**：M5 MacBook Air, MPS backend
> **模型**：MiniMind 25.83M params (512D × 8L × 8H, GQA, SwiGLU)
> **数据**：pretrain_hq.jsonl (141万样本)
> **每次实验**：500 steps, bfloat16

---

## 最优配置

```
优化器: Muon (矩阵权重) + AdamW (embedding/norm)
Muon lr: 0.02 (固定，不衰减)
AdamW lr: 3e-4
batch_size: 8
accumulation_steps: 4 (effective batch = 32)
grad_clip: 1.0
dtype: bfloat16
```

**val_loss = 4.9293**，比 baseline (6.8054) **降低 27.6%**

---

## 完整实验结果

| # | val_loss | 时间 | 状态 | 配置 |
|---|---------|------|------|------|
| 1 | **6.8054** | 2.5m | baseline | lr=5e-4 bs=8 accum=8 AdamW |
| 2 | 6.7879 | 2.5m | keep | lr=1e-3 (微小改善 -0.3%) |
| 3 | 6.9517 | 2.5m | discard | lr=2e-4 (收敛太慢) |
| 4 | **6.4719** | 2.7m | keep | **accum=4** (-4.9%) |
| 5 | 6.9793 | 2.7m | discard | accum=16 (更差) |
| 6 | 8.3703 | 2.7m | discard | accum=2 (AdamW发散) |
| 7 | 6.7162 | 2.5m | discard | lr=1e-3+accum=4 |
| 8 | 6.7055 | 2.5m | discard | warmup=50 (浪费步数) |
| 9 | 6.6480 | 2.3m | discard | beta2=0.95 |
| 10 | 6.9069 | 2.4m | discard | weight_decay=0.1 |
| 11 | **4.9293** | 2.1m | **最优** | **Muon lr=0.02 + accum=4** |
| 12 | 5.4113 | 2.2m | discard | Muon lr=0.01 |
| 13 | 5.1717 | 2.1m | discard | Muon lr=0.05 |
| 14 | 5.4173 | 2.2m | discard | Muon+accum=8 |
| 15 | 5.8426 | 3.0m | discard | Muon+accum=2 (没发散但慢) |
| 16 | 5.8282 | 2.6m | discard | Muon lr=0.03 |
| 17 | 5.2770 | 2.3m | discard | Muon+cosine LR decay |

---

## 关键发现

### 1. Muon 优化器是最大的改进来源

- **AdamW 最优**: val_loss = 6.4719 (accum=4)
- **Muon 最优**: val_loss = 4.9293 (accum=4, lr=0.02)
- Muon 比最优 AdamW 配置再低 **23.8%**
- 在 26M 中文小模型 + MPS 上，Muon 效果极其显著

### 2. 累积步数 (accumulation_steps) 是 AdamW 下第二大杠杆

500 步固定预算下：
- accum=4 → 125 次 optimizer update → **最优**
- accum=8 → 62 次 optimizer update → baseline
- accum=16 → 31 次 optimizer update → 更差
- accum=2 → 250 次 optimizer update → AdamW 发散，Muon 勉强稳定

**核心洞察**：短训练中，更多 optimizer 更新比更大 batch 更重要。

### 3. 其他超参调优效果甚微

| 超参 | 结论 |
|------|------|
| Learning rate | lr=1e-3 仅比 5e-4 好 0.3%，accum=4 贡献 4.9% |
| Warmup | 500 步中 warmup=50 浪费 10% 的训练步数，有害 |
| AdamW betas | beta2=0.95 (nanochat推荐) 反而比默认 0.999 差 |
| Weight decay | 0.1 比 0.01 差，短训练不需要强正则化 |
| LR schedule | Muon 用固定 LR 比 cosine 衰减好 (还在快速学习阶段) |

### 4. Muon 实验结论

- **有效性**：极其有效，27.6% 改善
- **稳定性**：比 AdamW 更稳定 — AdamW 在 accum=2 发散 (8.37)，Muon 仍能收敛 (5.84)
- **最优 LR**：0.02（0.01 太慢，0.03 和 0.05 太快）
- **LR 衰减**：不需要，500 步还在快速学习阶段
- **参数分配**：25.82M Muon + 0.01M AdamW（几乎全部用 Muon）

### 5. M5 Mac MPS 注意事项

- `PYTORCH_ENABLE_MPS_FALLBACK=1` 是必需的
- `torch.amp.autocast('mps', dtype=bfloat16)` 工作正常
- `GradScaler` 在 MPS 上应禁用（仅 CUDA float16 需要）
- `num_workers=0` 最稳定
- `torch.compile` 在 MPS 上不完全支持，建议关闭
- 每次 500 步实验耗时 ~2-3 分钟，非常快

---

## 复现命令

```bash
# 最优配置
cd minimind/trainer
PYTORCH_ENABLE_MPS_FALLBACK=1 python train_pretrain.py \
  --device mps \
  --dtype bfloat16 \
  --epochs 1 \
  --batch_size 8 \
  --accumulation_steps 4 \
  --optimizer muon \
  --muon_lr 0.02 \
  --learning_rate 3e-4 \
  --max_steps 500 \
  --num_workers 0 \
  --log_interval 100
```

---

## 实验方法论

- 总计 17 个实验，约 40 分钟完成
- 每次实验固定 500 steps，用 val_loss 作为主指标
- 验证集：随机抽取 1% 数据 (14131 样本)
- 只修改 `trainer/train_pretrain.py` 一个文件
- Muon 优化器代码内嵌（~45 行），无需额外依赖
- 基于实验结果动态调整搜索方向，非网格搜索

---

*由 Claude Code (autoresearch 模式) 自动完成*
