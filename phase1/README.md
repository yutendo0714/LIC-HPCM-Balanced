# Phase 1: Balanced R-D Optimization for HPCM

**目的**: HPCMにBalanced Rate-Distortion Optimizationを統合し、基本的な性能向上を検証

## 実装内容

### 1. コア実装

#### `src/optimizers/balanced.py`
- Balanced optimizer (CautiousAdam + FAMO)
- 2タスク（Distortion + BPP）のバランシング
- タスク重みの動的調整

#### `train.py`
- Balanced R-D対応の訓練スクリプト
- `--use_balanced`フラグで切り替え可能
- 標準Adam vs Balanced の比較実験をサポート

### 2. 主な変更点

#### 損失関数の拡張
```python
out["distortion"] = self.lmbda * 255 ** 2 * out["mse_loss"]
out["bpp_loss"] = ...
out["loss"] = out["distortion"] + out["bpp_loss"]
```
- Distortionを独立した項として分離
- Balanced optimizerが各タスク損失を動的に重み付け

#### 訓練ループの改変
```python
if use_balanced:
    task_losses = torch.stack([distortion, bpp_loss])
    weighted_loss = optimizer.backward_with_task_balancing(task_losses, model.parameters())
    optimizer.step(task_losses=task_losses)
    optimizer.update_task_weights(task_losses_new)
else:
    loss.backward()
    optimizer.step()
```

### 3. 使用方法

#### 標準訓練（ベースライン）
```bash
bash run_standard.sh
```

または

```bash
python train.py \
    --model_name HPCM_Base \
    --train_dataset /path/to/train \
    --test_dataset /path/to/test \
    --lambda 0.013 \
    --cuda
```

#### Balanced R-D訓練
```bash
bash run_balanced.sh
```

または

```bash
python train.py \
    --model_name HPCM_Base \
    --train_dataset /path/to/train \
    --test_dataset /path/to/test \
    --lambda 0.013 \
    --use_balanced \
    --gamma 0.003 \
    --w_lr 0.025 \
    --cuda
```

#### ハイパーパラメータ探索
```bash
bash run_hparam_search.sh
```

### 4. 主要パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `--use_balanced` | False | Balanced optimizerを使用 |
| `--gamma` | 0.003 | 正則化係数（タスク重みの発散防止） |
| `--w_lr` | 0.025 | タスク重みの学習率 |
| `--clip_max_norm` | 1.0 | 勾配クリッピング |

### 5. 期待される効果

- **BD-rate改善**: 1-2%の向上を期待
- **訓練安定性**: Rate/Distortionのバランス改善
- **λスイープ**: 各λ値での最適化品質向上

### 6. ディレクトリ構造

```
phase1/
├── src/
│   └── optimizers/
│       ├── __init__.py
│       └── balanced.py          # Balanced optimizer実装
├── train.py                     # メイン訓練スクリプト
├── run_standard.sh              # 標準訓練スクリプト
├── run_balanced.sh              # Balanced訓練スクリプト
├── run_hparam_search.sh         # ハイパラ探索スクリプト
└── README.md                    # このファイル
```

### 7. チェックポイント

訓練中、以下が保存されます：

- `epoch_best.pth.tar`: 最良モデル
- `epoch_{N}.pth.tar`: 500エポックごとのチェックポイント
- `balanced_state_best.pth`: Balanced optimizerの状態（use_balanced時）
- `balanced_state_{N}.pth`: 定期的なoptimizer状態

### 8. WandB ログ

以下のメトリックがログされます：

- `train/loss`, `test/loss`: 総合損失
- `train/distortion`, `test/distortion`: Distortion項
- `train/bpp_loss`, `test/bpp_loss`: BPP項
- `train/task_weight_distortion`: Distortionのタスク重み（Balanced時）
- `train/task_weight_bpp`: BPPのタスク重み（Balanced時）
- `train/psnr`, `test/psnr`: PSNR

### 9. トラブルシューティング

#### NaN損失が発生する場合
- `--gamma`を増やす（例: 0.005 → 0.01）
- `--clip_max_norm`を調整（例: 1.0 → 0.5）

#### 収束が遅い場合
- `--w_lr`を増やす（例: 0.025 → 0.05）
- `--gamma`を減らす（例: 0.003 → 0.001）

#### タスク重みが偏る場合
- 正常な挙動（一方が支配的になることがある）
- `min_losses`の初期値を調整（train.py内）

### 10. 次のステップ（Phase 2以降）

Phase 1で基本統合が完了したら：

1. **Phase 2**: ハイパーパラメータ最適化とFine-tuning
2. **Phase 3**: 階層的Balanced（s1/s2/s3スケール別）
3. **Phase 4**: 全λスイープでの評価とBD-rate計算

## 参考文献

- Balanced Rate-Distortion Optimization in Learned Image Compression (CVPR 2025)
- HPCM: Hierarchical Progressive Context Mining (オリジナル実装)
