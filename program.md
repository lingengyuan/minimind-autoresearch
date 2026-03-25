# AutoResearch Program: MiniMind 超参搜索

> 本文件是 Claude Code autoresearch 的执行指令。交给 AI agent 执行即可。

## 约束

- **只修改** `trainer/train_pretrain.py`，不改模型架构
- 每次实验固定 **500 steps**
- 用 **val_loss** 作为唯一指标
- 结果追加到 `results.tsv`
- 每轮实验判定：keep / discard / crash

## 实验顺序

### Phase 1: 基础超参
1. Baseline: 默认参数 (lr=5e-4, bs=8, accum=8, AdamW, bfloat16)
2. Learning rate: 1e-3, 2e-4
3. Effective batch size: accum=4, accum=16, accum=2

### Phase 2: 学习率调度
4. Warmup: warmup=50 steps
5. Cosine decay vs 默认 schedule

### Phase 3: 优化器
6. AdamW 参数调优: beta2, weight_decay
7. **Muon 优化器**: 嵌入 SingleDeviceMuon，搜索 lr

### Phase 4: MPS 特定
8. float32 vs bfloat16
9. num_workers 调优

### Phase 5: 组合验证
10. 最优配置组合 + 延长到 2000 steps 验证

## 执行规则

```
for each experiment:
  1. 修改参数（只改 train_pretrain.py）
  2. 运行 500 steps
  3. 记录 val_loss, 时间, commit
  4. 判定 keep/discard/crash
  5. 追加到 results.tsv
  6. 基于结果决定下一个实验
```

## 复现命令模板

```bash
cd minimind/trainer
PYTORCH_ENABLE_MPS_FALLBACK=1 python train_pretrain.py \
  --device mps \
  --dtype bfloat16 \
  --epochs 1 \
  --batch_size 8 \
  --accumulation_steps 4 \
  --learning_rate 5e-4 \
  --max_steps 500 \
  --num_workers 0 \
  --log_interval 100
```
