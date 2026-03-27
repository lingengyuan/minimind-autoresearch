# MiniMind AutoResearch — 全阶段训练计划

> 基于 Phase 1-3 的超参搜索成果，规划从验证到完整训练的全链路  
> 硬件：M5 MacBook Air, Apple MPS  
> 模型：MiniMind 25.83M (512D × 8L × 8H, GQA, SwiGLU, RoPE)  
> 方法论：autoresearch — program.md 驱动，AI agent 自动执行实验

---

## 目录

1. [当前进度](#当前进度)
2. [Phase 5 — 2000 步延长验证](#phase-5--2000-步延长验证)
3. [Phase 6 — Full Pretrain](#phase-6--full-pretrain)
4. [Phase 7 — SFT 监督微调](#phase-7--sft-监督微调)
5. [Phase 8 — DPO 偏好对齐](#phase-8--dpo-偏好对齐)
6. [Phase 9 — 评测](#phase-9--评测)
7. [可选扩展](#可选扩展)
8. [时间与算力估算](#时间与算力估算)
9. [决策树](#决策树)

---

## 当前进度

| Phase | 状态 | 核心成果 |
|-------|------|---------|
| Phase 1: 基础超参 | ✅ 完成 | accum=4 比 accum=8 好 4.9% |
| Phase 2: 学习率调度 | ✅ 完成 | warmup/cosine 在 500 步内无益 |
| Phase 3: 优化器 | ✅ 完成 | Muon lr=0.02 + accum=4 → val_loss 4.93, -27.6% |
| Phase 4: MPS 特定 | ✅ 完成 | bfloat16 + num_workers=0 最稳 |
| Phase 5: 延长验证 | ⏳ 待执行 | 验证 Muon 优势是否在 2000 步持续 |
| Phase 6: Full Pretrain | 📋 已规划 | — |
| Phase 7: SFT | 📋 已规划 | — |
| Phase 8: DPO | 📋 已规划 | — |
| Phase 9: 评测 | 📋 已规划 | — |

---

## Phase 5 — 2000 步延长验证

> 详见 `phase5_program.md`，此处仅列决策逻辑。

### 核心实验（3 个必跑）

| 实验 | 配置 | 目的 |
|------|------|------|
| P5-1 | AdamW baseline, accum=8, 2000 steps | 基线 |
| P5-2 | AdamW accum=4, 2000 steps | AdamW 最优 |
| P5-3 | Muon lr=0.02, accum=4, 2000 steps | Phase 3 最优 |

### 决策分支

```
P5 结果 → Muon vs AdamW gap
  ├─ gap > 15%  → Phase 6 用 Muon（大概率）
  ├─ gap 5-15%  → Phase 6 跑 Muon + cosine decay，再比一次
  └─ gap < 5% 或反转 → Phase 6 用 AdamW accum=4
```

### 预计耗时：~50 分钟（6 个实验 × ~8 分钟）

---

## Phase 6 — Full Pretrain

### 6.1 目标

在 MPS 上完成 pretrain_hq.jsonl（141 万样本）的完整预训练，产出可用于 SFT 的基座权重。

### 6.2 数据

```
dataset/pretrain_hq.jsonl  — 1.6GB, 141 万样本
```

### 6.3 配置方案（基于 Phase 5 决策）

**方案 A：Muon 主力（Phase 5 gap > 15%）**

```bash
cd minimind/trainer
PYTORCH_ENABLE_MPS_FALLBACK=1 python train_pretrain.py \
  --device mps --dtype bfloat16 \
  --epochs 1 \
  --batch_size 8 --accumulation_steps 4 \
  --optimizer muon --muon_lr 0.02 --learning_rate 3e-4 \
  --num_workers 0 --log_interval 500 --save_interval 5000 \
  --use_wandb --wandb_project "minimind-autoresearch"
```

**方案 B：AdamW 稳健（Phase 5 gap < 5%）**

```bash
cd minimind/trainer
PYTORCH_ENABLE_MPS_FALLBACK=1 python train_pretrain.py \
  --device mps --dtype bfloat16 \
  --epochs 1 \
  --batch_size 8 --accumulation_steps 4 \
  --learning_rate 5e-4 --optimizer adamw \
  --num_workers 0 --log_interval 500 --save_interval 5000 \
  --use_wandb --wandb_project "minimind-autoresearch"
```

### 6.4 预计耗时

MPS 上 500 步 ≈ 2.5 分钟。pretrain_hq 有 141 万样本，batch_size=8 → ~176,000 步。

```
预估时间 ≈ 176000 / 500 × 2.5 分钟 ≈ 14.7 小时
```

> 如果需要 2 epoch（MiniMind 官方推荐），则约 30 小时。可以考虑拆成多次，利用 `--from_resume 1` 断点续训。

### 6.5 验证标准

- val_loss 应降至 < 4.0（参考 MiniMind2 官方结果）
- 用 `eval_llm.py` 做简单生成测试，看输出是否连贯中文

### 6.6 可选的 autoresearch 子实验

如果 full pretrain 过程中 loss 下降停滞，可以用 autoresearch 模式快速测试：

| 实验 | 改动 | 目的 |
|------|------|------|
| P6-A | 中途切换 Muon lr 0.02 → 0.01 | 后期是否需要降 LR |
| P6-B | 中途启用 cosine decay | 后半程是否需要衰减 |
| P6-C | 增加 weight_decay 到 0.05 | 长训练是否需要更强正则 |

---

## Phase 7 — SFT 监督微调

### 7.1 目标

在 pretrain 基座上进行指令微调，让模型学会 question → answer 的对话模式。

### 7.2 数据选择

MiniMind 提供多级 SFT 数据，按 MPS 算力从低到高：

| 数据集 | 大小 | max_seq_len | 适用场景 |
|--------|------|-------------|---------|
| `sft_mini_512.jsonl` ✨ | 1.2GB | 512 | **Zero 快速版，推荐先用** |
| `sft_512.jsonl` | 7.5GB | 512 | 完整单轮 |
| `sft_1024.jsonl` | 5.6GB | 1024 | 中等长度 |
| `sft_2048.jsonl` | 9GB | 2048 | 长对话 |

**推荐路径**：先用 `sft_mini_512.jsonl` 跑通流程（MPS 上约 2 小时），确认可用后再考虑更大数据集。

### 7.3 SFT 阶段的 autoresearch

SFT 的超参空间与 pretrain 不同，值得单独搜索：

```
Phase 7 autoresearch program:
- 只修改 trainer/train_full_sft.py
- 每次实验固定 500 steps
- 用 val_loss 作为指标

实验顺序：
1. Baseline: 官方默认 (lr=1e-5, epochs=6, max_seq_len=512)
2. 学习率: 5e-5, 1e-4, 5e-6
3. Muon for SFT: muon_lr=0.005, 0.01, 0.02
   （SFT 通常用更小 LR，Muon 最优 LR 可能也要调低）
4. Epoch 数: 1, 3, 6, 10
5. 最优组合验证
```

### 7.4 关键问题

- **Muon 是否适用于 SFT？** Pretrain 阶段有效不代表 SFT 也有效。SFT 数据量小、学习率低，Muon 的 Newton-Schulz 正交化在微调时可能过于激进。需要实验验证。
- **过拟合风险**：sft_mini_512 只有约 10 万样本，epoch 过多会过拟合。观察 val_loss 上升的拐点。
- **MPS 上 SFT 的 LR schedule**：官方用 1e-5 到 1e-6 的动态 LR，6 个 epoch。MPS 上可能需要调整。

### 7.5 SFT 验证

```bash
# 生成测试
python eval_llm.py --load_from model --weight full_sft
```

测试对话质量：
- 能否回答简单中文问题
- 回答是否遵循指令格式
- 是否存在重复生成 / 乱码

---

## Phase 8 — DPO 偏好对齐

### 8.1 目标

在 SFT 模型基础上，用 DPO 对齐人类偏好，提升回答质量。

### 8.2 数据

```
dataset/dpo.jsonl — 909MB, ~8 万条
格式: { "chosen": "...", "rejected": "..." }
```

### 8.3 DPO 配置

```bash
cd minimind/trainer
PYTORCH_ENABLE_MPS_FALLBACK=1 python train_dpo.py \
  --device mps --dtype bfloat16 \
  --from_weight full_sft \
  --epochs 1 \
  --batch_size 4 --learning_rate 5e-6 \
  --num_workers 0
```

### 8.4 DPO 阶段的 autoresearch

```
Phase 8 autoresearch program:
- 只修改 trainer/train_dpo.py
- 每次实验固定 500 steps
- 指标: val_loss + 生成质量人工判断

实验顺序:
1. Baseline: 官方默认
2. DPO beta: 0.1, 0.2, 0.5（控制偏好强度）
3. Learning rate: 1e-6, 5e-6, 1e-5
4. Muon for DPO: 需要谨慎测试，DPO 对优化器敏感
```

### 8.5 注意事项

- DPO 的 reject/chosen 是离线数据，与 MiniMind 原始输出分布有差异
- 官方观察到 DPO 后模型倾向于说更多「礼貌但不相关」的话，牺牲信息准确性
- 需要在「对齐度」和「能力保持」之间取舍

---

## Phase 9 — 评测

### 9.1 基准测试

MiniMind 支持的评测：

| 评测集 | 类型 | 命令 |
|--------|------|------|
| C-Eval | 中文综合 | `python eval_ceval.py --weight <weight>` |
| C-MMLU | 中文多任务 | `python eval_cmmlu.py --weight <weight>` |
| OpenBookQA | 英文常识 | `python eval_obqa.py --weight <weight>` |

### 9.2 评测对比矩阵

每个阶段产出的模型都应跑一次评测，形成对比：

| 模型 | C-Eval | C-MMLU | 对话质量 | 备注 |
|------|--------|--------|---------|------|
| pretrain (Muon) | ? | ? | — | 只有语言建模能力 |
| pretrain (AdamW) | ? | ? | — | 对照组 |
| sft_mini | ? | ? | 基础对话 | Zero 版本 |
| sft_full | ? | ? | 较好对话 | 如果跑了完整 SFT |
| dpo | ? | ? | 对齐偏好 | 最终模型 |

### 9.3 对话质量测试集

准备 20 个固定测试 prompt，覆盖：

```
- 事实问答（5 个）: "中国的首都是哪里？"
- 常识推理（5 个）: "为什么天空是蓝色的？"
- 指令遵循（5 个）: "用三句话介绍自己"
- 创意写作（3 个）: "写一首关于秋天的短诗"
- 自我认知（2 个）: "你是谁？" "你和 OpenAI 是什么关系？"
```

每个阶段用相同 prompt 生成，对比变化。

---

## 可选扩展

以下阶段在核心链路完成后按兴趣选择：

### RLAIF (PPO / GRPO / CISPO)

```
数据: rlaif.jsonl
脚本: train_rlaif.py
难度: ★★★★ — 在线 RL 在 MPS 上可能不稳定
建议: 仅在 DPO 效果不满意时尝试
```

### LoRA 微调（领域适配）

```
数据: lora_medical.jsonl (34MB) 或自定义数据
脚本: train_lora.py
用途: 在 SFT 模型上低成本适配特定领域
优势: 只训练约 1% 参数，MPS 上极快
```

### Tool Use / Agentic RL

```
数据: agent_rl.jsonl, agent_rl_math.jsonl
脚本: train_agent.py
前提: SFT 模型需要具备基础 Tool Call 能力
```

### MoE 版本

```
参数: --use_moe 1
模型: ~198M params (A64M active)
注意: MPS 内存可能不够，建议仅在 CUDA 上尝试
```

### Reason 模型（思维链）

```
数据: r1_mix_1024.jsonl (340MB)
方式: SFT 蒸馏（从 DeepSeek-R1 蒸馏的数据）
特性: 支持 <think> 标签的自适应思考
```

---

## 时间与算力估算

基于 M5 MacBook Air MPS，所有估算以 1 epoch 计：

| 阶段 | 数据量 | 预计步数 | 预计时间 | 备注 |
|------|--------|---------|---------|------|
| Phase 5: 延长验证 | — | 6×2000 | **~50 分钟** | 6 个实验 |
| Phase 6: Full Pretrain | 141 万样本 | ~176K | **~15 小时** | 可断点续训 |
| Phase 7: SFT (mini) | sft_mini_512 | ~15K | **~2 小时** | Zero 快速版 |
| Phase 7: SFT (full) | sft_512 | ~94K | **~13 小时** | 完整版 |
| Phase 8: DPO | 8 万条 | ~20K | **~3 小时** | — |
| Phase 9: 评测 | — | — | **~30 分钟** | 推理为主 |
| **总计（快速路径）** | | | **~21 小时** | P5→P6→SFT mini→DPO→Eval |
| **总计（完整路径）** | | | **~32 小时** | P5→P6→SFT full→DPO→Eval |

### 执行建议

```
Day 1 晚上：跑 Phase 5（50 分钟，可看着跑）
Day 1 睡前：启动 Phase 6 Full Pretrain（--from_resume 1，过夜跑）
Day 2 白天：检查 Pretrain 结果，启动 SFT（2 小时）
Day 2 下午：SFT autoresearch 超参搜索（1 小时）
Day 2 晚上：最优 SFT 配置跑完整（2 小时）
Day 3 上午：DPO（3 小时）
Day 3 下午：评测 + 对话测试 + 整理报告
```

---

## 决策树

```
                        Phase 5: 2000 步验证
                               │
              ┌────────────────┼────────────────┐
              │                │                │
         gap > 15%        gap 5-15%        gap < 5%
         Muon 持续         Muon 微弱         Muon 无效
              │                │                │
              ▼                ▼                ▼
        Phase 6:          Phase 6:          Phase 6:
       Muon pretrain    Muon+cosine试       AdamW pretrain
              │           再跑一次               │
              │                │                │
              ▼                ▼                ▼
              └────────────────┼────────────────┘
                               │
                        Full Pretrain 完成
                               │
                        eval_llm.py 生成测试
                               │
                    ┌──────────┼──────────┐
                    │          │          │
                 输出连贯   输出尚可   输出乱码
                    │          │          │
                    ▼          ▼          ▼
                Phase 7     再跑 1      检查数据
                SFT        epoch        /代码问题
                    │
                    ▼
             SFT autoresearch
            (500 步快速搜索)
                    │
              ┌─────┼─────┐
              │           │
          Muon 有效    Muon 无效
           for SFT      for SFT
              │           │
              ▼           ▼
           Muon SFT   AdamW SFT
              │           │
              └─────┬─────┘
                    │
              Full SFT 完成
                    │
               Phase 8: DPO
                    │
              DPO autoresearch
             (beta/lr 快速搜索)
                    │
              Full DPO 完成
                    │
               Phase 9: 评测
                    │
              ┌─────┼─────┐
              │           │
          效果满意    效果不满意
              │           │
              ▼           ▼
           完成！     可选扩展:
           发布       RLAIF / LoRA
           模型       / 更多数据
```

---

## Repo 结构规划

```
minimind-autoresearch/
├── README.md                     # 项目总览
├── program.md                    # Phase 1-4 执行指令
├── phase5_program.md             # Phase 5 执行指令
├── phase7_sft_program.md         # Phase 7 SFT autoresearch（待创建）
├── phase8_dpo_program.md         # Phase 8 DPO autoresearch（待创建）
├── experiment_summary.md         # Phase 1-4 实验报告
├── phase5_summary.md             # Phase 5 实验报告（待创建）
├── phase7_sft_summary.md         # Phase 7 实验报告（待创建）
├── full_pipeline_plan.md         # ← 本文件
├── results.tsv                   # Phase 1-4 原始数据
├── results_phase5.tsv            # Phase 5 原始数据（待创建）
├── results_sft.tsv               # Phase 7 原始数据（待创建）
├── results_dpo.tsv               # Phase 8 原始数据（待创建）
├── eval_results/                 # 评测结果（待创建）
│   ├── ceval_pretrain.json
│   ├── ceval_sft.json
│   └── ceval_dpo.json
├── train_pretrain.py             # 修改后的 pretrain 脚本
├── patches/
│   └── train_pretrain.patch
└── test_prompts.txt              # 固定测试 prompt 集（待创建）
```

---

*计划版本：v1.0 | 2026-03-26*  
*基于 Phase 1-4 成果（17 个实验，Muon -27.6%）制定*
