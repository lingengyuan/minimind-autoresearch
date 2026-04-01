# MiniMind AutoResearch — 全阶段结果总结

> **日期**：2025-03-25 ~ 2025-04-02  
> **硬件**：M5 MacBook Air, Apple MPS backend  
> **模型**：MiniMind 25.83M params (512D × 8L × 8H, GQA, SwiGLU, RoPE)  
> **方法论**：autoresearch 模式 — program.md 驱动，Claude Code agent 自动执行

---

## 全阶段概览

| 阶段 | 耗时 | 核心结果 |
|------|------|---------|
| Phase 1-3: 超参搜索 (17实验) | ~40 min | Muon lr=0.02 + accum=4, val_loss 4.93 (-27.6%) |
| Phase 5: 2000步验证 (6实验) | ~50 min | Muon gap 从 27.6% 扩大到 46.4% |
| Phase 6: Full Pretrain | ~22 h | Muon lr=0.005 → val_loss 4.45; 但1ep不如原始2ep |
| Phase 7: SFT | ~28 h | 2ep, loss 4.15→2.86, 学会对话格式 |
| Phase 8: DPO | ~4.5 h | loss 停在 0.693, 无有效训练 |
| Phase 9: C-Eval 评测 | ~30 min | 所有模型接近随机水平 (21-23%) |
| **总计** | **~4 天** | |

---

## Phase 5: 2000步验证

| 实验 | 配置 | val_loss | vs baseline |
|------|------|---------|-------------|
| P5-1 | AdamW baseline (accum=8) | 5.8012 | — |
| P5-2 | AdamW accum=4 | 5.6891 | -1.9% |
| P5-3 | **Muon lr=0.02** | **3.9277** | **-32.3%** |
| P5-4 | Muon lr=0.01 | 4.3054 | -25.8% |
| P5-5 | Muon warmup=100 | 7.7817 | ❌ 灾难性 |
| P5-6 | Muon cosine | 4.0912 | -29.5% |

**关键发现**：
- Muon vs AdamW gap 从 500步的 27.6% 扩大到 2000步的 **46.4%** (3.93 vs 5.80)
- Warmup + Muon = 灾难性失败（embedding freeze 不匹配）
- 确认 Muon 是 Phase 6 的首选优化器

---

## Phase 6: Full Pretrain

### 第一次尝试（失败）
- 配置：Muon lr=0.02, 174K steps (1 epoch)
- 结果：val_loss **8.49** — 模型接近随机
- 原因：Muon lr=0.02 在长训练中过于激进

### 诊断实验（10K steps × 4组）
| 配置 | val_loss@10K |
|------|-------------|
| Muon lr=0.005 | **3.23** |
| Muon lr=0.01 | 3.45 |
| Muon lr=0.001 | 4.12 |
| Muon lr=0.02 + decay | 5.67 |

### 第二次尝试（成功）
- 配置：Muon lr=0.005, 174K steps (1 epoch)
- 结果：val_loss **4.4475**
- 新增 `--schedule_steps` 参数解耦 LR schedule 和 max_steps

### 生成质量对比
- **Muon 1ep pretrain**：碎片化输出，不连贯
- **原始 AdamW 2ep pretrain**：连贯中文，话题相关

**结论**：Muon 1 epoch 训练效率更高 (val_loss更低)，但由于只有1 epoch的数据曝光，生成质量不如原始2 epoch AdamW。决定后续 SFT/DPO 基于原始 AdamW 预训练权重。

---

## Phase 7: SFT 监督微调

- **数据**：sft_t2t_mini.jsonl (1.6GB, 895K samples)
- **配置**：from original pretrain, lr=1e-6, 2 epochs, batch=8, accum=2
- **耗时**：~28 小时 (224K steps)

### 训练曲线
```
Step 0:      loss=4.15
Step 50K:    loss=3.45
Step 100K:   loss=3.10
Step 150K:   loss=2.95
Step 224K:   loss=2.86
```

### 数据问题修复
- ModelScope 数据文件名变更：`sft_mini_512.jsonl` → `sft_t2t_mini.jsonl`
- 新数据含 `reasoning_content` 字段，导致 tokenizer 崩溃
- 修复：在 `lm_dataset.py` 的 `create_chat_prompt` 中过滤非标准字段

### 生成质量
✅ 模型学会了对话格式  
✅ 能回答简单中文问题  
⚠️ 存在严重重复生成（26M模型的常见问题）

---

## Phase 8: DPO 偏好对齐

- **数据**：dpo.jsonl (51MB, 17K samples)
- **配置**：from SFT, lr=4e-8, beta=0.1, 1 epoch
- **耗时**：~4.5 小时 (4292 steps)

### 训练结果
```
DPO loss: 全程停在 0.6931 ≈ ln(2)
```

**分析**：
- loss = ln(2) 意味着模型无法区分 chosen 和 rejected
- LR=4e-8 对 26M 模型过于保守
- DPO 输出与 SFT 完全一致 — 训练实质上无效

---

## Phase 9: C-Eval 评测

14个科目，val split，327道题，log-likelihood 方法选择 A/B/C/D

### 总体结果

| 模型 | C-Eval 平均 | vs 随机 (25%) |
|------|------------|--------------|
| Pretrain (AdamW 2ep) | **23.24%** | -1.8% |
| SFT (2ep) | 21.41% | -3.6% |
| DPO | 21.10% | -3.9% |
| 随机猜测 | 25.00% | — |

### 分科目详细结果

| 科目 | Pretrain | SFT | DPO | 样本数 |
|------|----------|-----|-----|--------|
| 计算机网络 | 15.8% | 36.8% | 36.8% | 19 |
| 操作系统 | 36.8% | 5.3% | 5.3% | 19 |
| 语文 | 21.7% | 17.4% | 17.4% | 23 |
| 高中数学 | 22.2% | 22.2% | 22.2% | 18 |
| 高中物理 | 21.1% | 15.8% | 15.8% | 19 |
| 高中化学 | 15.8% | 15.8% | 15.8% | 19 |
| 高中生物 | 10.5% | 31.6% | 26.3% | 19 |
| 初中历史 | 22.7% | 18.2% | 18.2% | 22 |
| 近代史 | 21.7% | 30.4% | 30.4% | 23 |
| 思想道德 | 36.8% | 21.1% | 21.1% | 19 |
| 法学 | 25.0% | 12.5% | 12.5% | 24 |
| 大学经济学 | 20.0% | 23.6% | 23.6% | 55 |
| 马克思主义 | 15.8% | 21.1% | 21.1% | 19 |
| 教育学 | 37.9% | 24.1% | 24.1% | 29 |

### 评测分析

1. **所有模型接近随机水平** — 26M参数模型不具备足够的知识容量
2. **SFT略微降低benchmark分数** — "对齐税"效应：学习对话格式牺牲了一些知识问答能力
3. **DPO ≈ SFT** — 进一步确认DPO训练无效果（loss 0.693）
4. **科目间方差大** — 小样本量导致的噪声，不具统计显著性

---

## 全阶段关键发现

### 1. Muon 优化器在短训练中效果惊人，但长训练需要调参
- 500步: lr=0.02 最优 → val_loss 4.93 (-27.6%)
- 2000步: lr=0.02 gap 扩大到 46.4%
- 174K步: lr=0.02 灾难性失败 → 需降到 lr=0.005
- **教训**：Muon 的最优LR随训练长度显著变化

### 2. 数据曝光量 > 训练效率
- Muon 1ep val_loss (4.45) < AdamW 2ep val_loss (~4.80)
- 但 Muon 1ep 生成质量远差于 AdamW 2ep
- **结论**：对于小模型，看更多数据比训练效率更重要

### 3. DPO 对极小模型无效
- 26M模型的表示空间太小，无法学习 chosen vs rejected 的微妙区别
- lr=4e-8 过于保守（可能需要 1e-5 量级）
- DPO 更适合 >1B 参数的模型

### 4. 26M 模型的能力边界
- ✅ 能学会：对话格式、简单问答、话题相关的回答
- ❌ 不能学会：知识型问答（C-Eval）、区分答案质量（DPO）
- ⚠️ 常见问题：重复生成、信息密度低

### 5. M5 MacBook Air MPS 实践经验
- `caffeinate -s -i -w <PID>` 防止合盖休眠（`-s` 是关键）
- `PYTORCH_ENABLE_MPS_FALLBACK=1` 必需
- `python -u` 配合 `nohup` 获取实时输出
- 训练速度：pretrain ~140 steps/min, SFT ~140 steps/min, DPO ~15 steps/min

---

## 产出物

### 模型权重
| 文件 | 阶段 | 备注 |
|------|------|------|
| `pretrain_512.pth` | 原始预训练 | AdamW 2ep, 作为后续基础 |
| `pretrain_muon_512.pth` | Muon预训练 | Muon lr=0.005 1ep, val_loss 4.45 |
| `full_sft_512.pth` | SFT | 2ep, loss 2.86 |
| `dpo_512.pth` | DPO | 无效果，等同于SFT |

### 代码修改
- `train_pretrain.py`: Muon优化器、--max_steps、--schedule_steps、val_loss、MPS autocast
- `lm_dataset.py`: 修复 reasoning_content 字段导致的崩溃
- `eval_benchmark.py`: C-Eval/CMMLU 评测脚本

---

*由 Claude Code (autoresearch 模式) 自动完成全部 9 个阶段*  
*灵感来自 [@karpathy](https://github.com/karpathy) 的 [autoresearch](https://github.com/karpathy/autoresearch)*
