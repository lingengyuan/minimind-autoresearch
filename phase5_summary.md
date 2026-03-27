# Phase 5 实验总结：2000 步延长验证

> **日期**：2026-03-27
> **硬件**：M5 MacBook Air, MPS backend
> **模型**：MiniMind 25.83M params
> **目的**：验证 Phase 1-3 的 Muon 优势在 4x 训练预算下是否持续

---

## 结果

| 实验 | val_loss | 时间 | 配置 | 结论 |
|------|---------|------|------|------|
| P5-1 | 7.3272 | 7.2m | AdamW baseline accum=8 | cosine 衰减过度 |
| P5-2 | 8.0267 | 8.3m | AdamW accum=4 | 更差，LR 衰减 + 更多更新 = 灾难 |
| **P5-3** | **3.9268** | 9.7m | **Muon lr=0.02 accum=4** | **🥇 最优，已低于 4.0 目标** |
| P5-4 | 4.1127 | 10.2m | Muon + cosine decay | cosine 仍无益 (-4.7%) |
| P5-5 | 7.7751 | 9.0m | Muon + warmup=100 | warmup 严重有害 |
| P5-6 | 4.0806 | 9.4m | Muon lr=0.015 | 略逊于 0.02 (-3.9%) |

---

## 关键发现

### 1. Muon 优势从 27.6% 扩大到 46.4%

| 训练长度 | Muon 最优 | AdamW 最优 | gap |
|---------|----------|----------|-----|
| 500 步 | 4.93 | 6.47 | -23.8% |
| 2000 步 | **3.93** | **7.33** | **-46.4%** |

gap 不仅没收窄，反而**翻倍**。Muon 的优势随训练步数增加而增大。

### 2. AdamW 在 2000 步时反而变差

这是最出乎意料的发现。AdamW 在 2000 步时的 val_loss（7.33）比 500 步时（6.81）**更差**。

原因：`get_lr` 的 cosine schedule 以 `max_steps` 为总步数，2000 步时 LR 从 5e-4 衰减到 5e-5。后半段 LR 太低，模型几乎停止学习，甚至出现退化。

> AdamW 的 loss 轨迹：step 800 时 loss=6.27（比500步baseline好），但 step 2000 时 loss=7.24（反弹了）。

Muon 不受影响，因为 Muon LR 是固定的（0.02），不走 cosine schedule。

### 3. Warmup 对 Muon 是灾难性的

P5-5 (Muon + warmup=100) 的 val_loss 7.78 是所有实验中第二差的。

**假说**：warmup 只影响 AdamW 控制的 embedding/norm 层（0.01M 参数）。前 100 步 embedding LR 极低（3e-6 → 3e-4），embedding 几乎不更新。但 Muon 同时在全速更新 25.82M 权重矩阵。

结果：Muon 在"冻结的"embedding 上学了 25 次更新的权重。当 warmup 结束、embedding 开始更新时，之前学到的权重全部失配。模型无法恢复。

### 4. Cosine decay 对 Muon 仍然无益

P5-4 (Muon + cosine) val_loss 4.11，比固定 LR 的 3.93 差 4.7%。和 500 步时的结论一致：2000 步仍在快速学习阶段，衰减过早。

### 5. lr=0.02 仍是 Muon 的最优 LR

P5-6 (lr=0.015) val_loss 4.08，比 lr=0.02 差 3.9%。在 500 步和 2000 步下，0.02 都是最优值。

---

## 对 Phase 6 Full Pretrain 的决策

**结论：Phase 6 使用 Muon，gap > 15% 远超阈值。**

推荐配置：
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python train_pretrain.py \
  --device mps --dtype bfloat16 --epochs 1 \
  --batch_size 8 --accumulation_steps 4 \
  --optimizer muon --muon_lr 0.02 --learning_rate 3e-4 \
  --num_workers 0 --log_interval 500 --save_interval 5000
```

**重要注意**：full pretrain 时 `max_steps` 不设（或设为 -1），让 `total_iters = epochs * iters`。这样 cosine schedule 会以完整 epoch 为周期衰减 AdamW LR，避免 Phase 5 中 AdamW 过度衰减的问题。

---

## 公平性说明

AdamW 的 2000 步结果受到了 cosine schedule 的不公平惩罚（LR 衰减到 0.1x）。如果用固定 LR 跑 AdamW，结果应该会好于 7.33。但即使在 step 800（LR 还较高时），AdamW 的 training loss (6.27) 仍然远差于 Muon (4.29)。Muon 的优势是真实的。

---

*Phase 5 of minimind-autoresearch | 6 个实验，~56 分钟 | 2026-03-27*
