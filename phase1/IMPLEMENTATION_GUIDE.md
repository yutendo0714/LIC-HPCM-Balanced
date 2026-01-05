# HPCM × Balanced R-D Optimization - 実装ガイド

## Phase 1: 基本統合 ✅ 完了

### 📁 ディレクトリ構成

```
phase1/
├── src/
│   ├── __init__.py
│   └── optimizers/
│       ├── __init__.py
│       └── balanced.py          # Balanced optimizer (CautiousAdam + FAMO)
├── train.py                     # メイン訓練スクリプト（Balanced対応）
├── run_standard.sh              # 標準Adam訓練スクリプト
├── run_balanced.sh              # Balanced R-D訓練スクリプト
├── run_hparam_search.sh         # ハイパーパラメータ探索スクリプト
├── test_phase1.py               # 実装テストスクリプト
└── README.md                    # 詳細ドキュメント
```

### 🎯 実装された機能

#### 1. Balanced Optimizer (`src/optimizers/balanced.py`)
- **CautiousAdam**: 勾配とモーメンタムの整合性チェック
- **FAMO**: 動的なタスク重み調整
- 2タスク（Distortion + BPP）のバランシング
- タスク重みの保存/読み込み機能

#### 2. 拡張損失関数 (`train.py`)
```python
out["distortion"] = self.lmbda * 255 ** 2 * out["mse_loss"]  # 独立項
out["bpp_loss"] = ...                                         # Rate項
out["loss"] = out["distortion"] + out["bpp_loss"]            # 総合損失
```

#### 3. 訓練ループ統合
- `--use_balanced`フラグで簡単切り替え
- タスク重みの自動調整
- WandBでのタスク重み可視化
- 標準Adamとの完全互換性

### 🚀 クイックスタート

#### 環境準備
```bash
cd /workspace/LIC-HPCM-Balanced/phase1

# データセットパスを設定
export TRAIN_DATA="/path/to/train/dataset"
export TEST_DATA="/path/to/test/dataset"
```

#### 1. 標準訓練（ベースライン）
```bash
# シェルスクリプトを編集してデータセットパスを設定
vim run_standard.sh  # パスを修正

# 実行
./run_standard.sh
```

#### 2. Balanced R-D訓練
```bash
# シェルスクリプトを編集してデータセットパスを設定
vim run_balanced.sh  # パスを修正

# 実行
./run_balanced.sh
```

#### 3. ハイパーパラメータ探索
```bash
# シェルスクリプトを編集
vim run_hparam_search.sh  # パスを修正

# 実行（12実験: gamma×4 × w_lr×3）
./run_hparam_search.sh
```

### 📊 主要パラメータ

| パラメータ | デフォルト | 推奨範囲 | 説明 |
|-----------|-----------|---------|------|
| `--gamma` | 0.003 | 0.001-0.01 | タスク重みの正則化係数 |
| `--w_lr` | 0.025 | 0.01-0.05 | タスク重み学習率 |
| `--clip_max_norm` | 1.0 | 0.5-2.0 | 勾配クリッピング |
| `--lambda` | 0.013 | - | Rate-Distortionトレードオフ |

### 🔍 動作確認

```bash
# Pythonテストスクリプト（PyTorch環境が必要）
python test_phase1.py

# 手動確認
python -c "from src.optimizers import Balanced; print('✓ Import OK')"
```

### 📈 期待される結果

#### 性能改善
- **BD-rate**: 1-2%の改善
- **訓練安定性**: Rate/Distortionバランス向上
- **λスイープ**: 各ポイントでの最適化品質向上

#### 観察されるパターン
1. **タスク重みの変動**
   - 初期: 均等（0.5, 0.5付近）
   - 中期: Rate優勢 or Distortion優勢
   - 後期: 安定化

2. **損失の推移**
   - Balanced: スムーズな収束
   - Standard: 振動が大きい可能性

### 🔧 トラブルシューティング

#### 問題1: NaN損失
**症状**: 訓練中にlossがNaNになる

**解決策**:
```bash
# gammaを増やす
--gamma 0.01

# gradient clippingを強化
--clip_max_norm 0.5

# 学習率を下げる
--learning-rate 1e-5
```

#### 問題2: タスク重みが極端に偏る
**症状**: task_weight_distortion が 0.99 や 0.01 など極端

**原因**: 正常な挙動（FAMOの特性）

**対処**:
- `min_losses`の初期値を調整（train.py L568付近）
- `gamma`を調整して正則化を強化

#### 問題3: 収束が遅い
**症状**: 標準Adamより学習が遅い

**解決策**:
```bash
# w_lrを増やす
--w_lr 0.05

# gammaを減らす
--gamma 0.001
```

### 📝 WandB モニタリング

ログされる主要メトリック：
- `train/loss`: 総合損失（Distortion + BPP）
- `train/distortion`: Distortion項のみ
- `train/bpp_loss`: BPP項のみ
- `train/task_weight_distortion`: Distortionのタスク重み
- `train/task_weight_bpp`: BPPのタスク重み
- `train/psnr`: 画質指標

**確認ポイント**:
1. タスク重みが0.2-0.8の範囲で推移するか
2. 損失が滑らかに減少するか
3. PSNRが標準Adamと同等以上か

### 🔄 チェックポイント管理

#### 保存されるファイル
```
outputs/
└── HPCM_Base_lmbda0.013_balanced/
    ├── epoch_best.pth.tar           # 最良モデル
    ├── epoch_500.pth.tar            # 定期チェックポイント
    ├── balanced_state_best.pth      # Balanced optimizer状態
    └── balanced_state_500.pth       # 定期optimizer状態
```

#### 再開方法
```bash
python train.py \
    --checkpoint ./outputs/.../epoch_500.pth.tar \
    --use_balanced \
    ...
```

### 📚 実装の詳細

#### Balanced Optimizerの動作フロー
```python
# 1. Forward pass
output = model(input)
loss = criterion(output, target)

# 2. タスク損失の構築
task_losses = torch.stack([distortion, bpp_loss])

# 3. 重み付き損失で逆伝播
weighted_loss = optimizer.backward_with_task_balancing(task_losses, model.parameters())

# 4. パラメータ更新（CautiousAdam）
optimizer.step(task_losses=task_losses)

# 5. タスク重み更新（FAMO）
optimizer.update_task_weights(new_task_losses)
```

#### タスク重みの計算（FAMO）
```python
# Softmax正規化
z = F.softmax(w, -1)

# 相対的な損失差
D = losses - min_losses + 1e-8

# 重み付き損失
weighted_loss = (D.log() * z / c).sum()
```

### 🎓 理論的背景

#### なぜBalanced R-Dが有効か？

1. **勾配衝突の回避**
   - HPCMは階層的構造（s1/s2/s3）
   - 各スケールでRate/Distortionの勾配が衝突
   - Balancedが動的にバランスを調整

2. **局所解の回避**
   - 固定λでは一方の項が支配的になりやすい
   - 動的な重み付けで探索範囲を拡大

3. **CautiousAdamの効果**
   - 勾配とモーメンタムの不一致を検出
   - 信頼できる方向のみ更新

### 🔬 実験プロトコル

#### ベースラインとの比較
```bash
# 1. 標準Adam訓練
./run_standard.sh

# 2. Balanced R-D訓練
./run_balanced.sh

# 3. 同じエポック数で比較
# - test/psnr
# - test/bpp_loss
# - BD-rate（後述）
```

#### ハイパーパラメータ最適化
```bash
# Grid search
./run_hparam_search.sh

# 結果の比較
# outputs/hparam_search/*/logs/ を確認
```

### 📊 次のステップ

#### Phase 2: ハイパーパラメータ最適化
- [ ] gammaの最適値を決定
- [ ] w_lrの最適値を決定
- [ ] Fine-tuning戦略の検討

#### Phase 3: 階層的Balanced（高度）
- [ ] s1/s2/s3スケール別のタスク分解
- [ ] 5タスクBalanced（Distortion + y_s1 + y_s2 + y_s3 + z）

#### Phase 4: 評価
- [ ] 全λスイープ（6点）での訓練
- [ ] BD-rate計算
- [ ] Kodak/CLIC/Tecnickでの評価

### 💡 Tips

1. **初回実験**
   - λ=0.013の1点で動作確認
   - epochs=100で早期検証

2. **ハイパラ探索**
   - epochs=1000で十分
   - WandBで損失曲線を比較

3. **本番訓練**
   - 最適パラメータでepochs=3000
   - 全λスイープ（時間がかかる）

4. **モニタリング**
   - タスク重みが極端に偏る場合は調整
   - PSNRが向上することを確認

### 🆘 サポート

問題が発生した場合：

1. **test_phase1.py**を実行して基本動作を確認
2. **WandB**のログを確認
3. **README.md**のトラブルシューティングを参照
4. **gamma/w_lr**を調整して再実験

---

**実装完了**: Phase 1の全ファイルが準備できました！
**次のアクション**: データセットパスを設定して訓練を開始してください。
