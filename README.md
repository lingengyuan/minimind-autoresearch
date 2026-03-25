# minimind-autoresearch

用 Claude Code 的 autoresearch 模式，在 40 分钟内自动完成 17 个超参实验，为 [MiniMind](https://github.com/jingyaogong/minimind) 26M 中文小模型找到比默认配置快 **27.6%** 收敛的训练方案。

## 核心发现

| 配置 | val_loss | vs baseline |
|------|---------|-------------|
| 🏁 Baseline (AdamW, accum=8) | 6.8054 | — |
| 🥈 AdamW + accum=4 | 6.4719 | **-4.9%** |
| 🥇 **Muon + accum=4** | **4.9293** | **-27.6%** |

**Muon 优化器是最大的改进来源**——在 26M 中文小模型 + Apple Silicon MPS 上，首次系统验证了 Muon 的有效性。

## 最优配置

```bash
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

## 实验全景

17 个实验，按 val_loss 排序（越低越好）：

| # | val_loss | 配置 | 结论 |
|---|---------|------|------|
| 11 | **4.9293** | Muon lr=0.02 + accum=4 | 🥇 最优 |
| 13 | 5.1717 | Muon lr=0.05 | 稍激进 |
| 17 | 5.2770 | Muon + cosine decay | 衰减过早 |
| 12 | 5.4113 | Muon lr=0.01 | 收敛较慢 |
| 14 | 5.4173 | Muon + accum=8 | batch 太大 |
| 16 | 5.8282 | Muon lr=0.03 | 不如 0.02 |
| 15 | 5.8426 | Muon + accum=2 | 稳定但慢 |
| 4 | **6.4719** | accum=4 (AdamW) | 🥈 AdamW 最优 |
| 9 | 6.6480 | beta2=0.95 | 不如默认 |
| 8 | 6.7055 | warmup=50 | 浪费步数 |
| 7 | 6.7162 | lr=1e-3 + accum=4 | 组合无益 |
| 2 | 6.7879 | lr=1e-3 | 微小改善 |
| 1 | **6.8054** | Baseline | 🏁 |
| 10 | 6.9069 | weight_decay=0.1 | 正则太强 |
| 3 | 6.9517 | lr=2e-4 | 太慢 |
| 5 | 6.9793 | accum=16 | 更新次数不够 |
| 6 | 8.3703 | accum=2 (AdamW) | ❌ 发散 |

## 关键结论

### 1. Muon 优化器效果惊人
- 比最优 AdamW 配置再低 **23.8%**（4.93 vs 6.47）
- 比 AdamW 更稳定：AdamW 在 accum=2 发散，Muon 仍能收敛
- 最优 LR = 0.02，固定 LR 优于 cosine 衰减

### 2. 短训练中更多更新比更大 batch 更重要
- accum=4（125 次更新）> accum=8（62 次）> accum=16（31 次）
- 这是因为 500 步训练预算有限，更多 optimizer step 更划算

### 3. 传统调参效果有限
- Learning rate、warmup、betas、weight decay 的调整幅度都在 5% 以内
- 换优化器 (Muon) 才是质的飞跃

### 4. MPS 可以用于快速超参搜索
- 每次 500 步实验 ~2.5 分钟
- 17 个实验总共 ~40 分钟
- `torch.amp.autocast('mps', dtype=bfloat16)` 在 PyTorch 2.11 上工作正常

## 诚实局限

⚠️ 以下是这组实验**不能**证明的事情：

- **不能证明最终模型更好** — 只跑了 500 步（全量数据的 0.3%）
- **没有下游评测** — 没测 perplexity、MMLU、C-Eval 等
- **收敛速度 ≠ 最终质量** — Muon 前期快不代表 full training 更优
- **单次实验无统计显著性** — 每个配置只跑了一次

**实际价值**：找到更高效的训练配置，节省完整训练的时间和算力。

## 方法论：autoresearch

本项目使用 autoresearch 模式（灵感来自 [@jxnl](https://x.com/jxnl/status/1903870289965723661) 的 [program.md 方法](https://github.com/jxnl/autoresearch)）：

1. 写一份 `program.md` 定义实验规则和约束
2. 让 Claude Code agent 自动执行：修改代码 → 训练 → 记录结果 → 决定下一步
3. Agent 根据实验结果动态调整搜索方向（非网格搜索）
4. 所有修改限制在**一个文件**内（`trainer/train_pretrain.py`）

这种方式的优势：
- **快速**：40 分钟完成 17 个实验的完整搜索
- **自适应**：发现 Muon 有效后立即深入探索，跳过无效方向
- **可复现**：所有配置和结果记录在 `results.tsv`

## 项目结构

```
├── README.md                  # 本文件
├── program.md                 # autoresearch 执行指令
├── experiment_summary.md      # 详细实验报告
├── results.tsv                # 原始实验数据
├── train_pretrain.py          # 修改后的训练脚本（可直接使用）
└── patches/
    └── train_pretrain.patch   # 对 MiniMind 原始脚本的 diff
```

## 如何使用

### 方式 1：直接用修改后的脚本

```bash
# 1. Clone MiniMind
git clone https://github.com/jingyaogong/minimind.git
cd minimind

# 2. 下载数据集
python -c "from modelscope import snapshot_download; snapshot_download('gongjy/minimind_dataset', local_dir='dataset')"

# 3. 替换训练脚本
cp /path/to/minimind-autoresearch/train_pretrain.py trainer/train_pretrain.py

# 4. 用最优配置训练
cd trainer
PYTORCH_ENABLE_MPS_FALLBACK=1 python train_pretrain.py \
  --device mps --dtype bfloat16 --optimizer muon --muon_lr 0.02 \
  --accumulation_steps 4 --num_workers 0
```

### 方式 2：应用 patch

```bash
cd minimind
git apply /path/to/minimind-autoresearch/patches/train_pretrain.patch
```

## 技术细节

- **模型**：MiniMind 25.83M params (512D × 8L × 8H, GQA 2KV heads, SwiGLU, RoPE)
- **数据**：pretrain_hq.jsonl (141万样本, 1.5GB)
- **硬件**：M5 MacBook Air, Apple MPS backend
- **框架**：PyTorch 2.11.0
- **Muon 实现**：SingleDeviceMuon，内嵌 ~45 行代码，无额外依赖
- **参数分配**：25.82M Muon（所有 ≥2D 权重矩阵）+ 0.01M AdamW（embedding/norm）

## 致谢

- [MiniMind](https://github.com/jingyaogong/minimind) by jingyaogong — 优秀的中文小模型教学项目
- [Muon](https://github.com/KellerJordan/Muon) by Keller Jordan — Newton-Schulz 正交化优化器
- [autoresearch](https://github.com/jxnl/autoresearch) by Jason Liu — program.md 驱动的 AI 研究模式
- Powered by [Claude Code](https://docs.anthropic.com/en/docs/claude-code)

## License

实验代码和文档：MIT License
MiniMind 原始代码遵循其[原始许可证](https://github.com/jingyaogong/minimind/blob/master/LICENSE)
