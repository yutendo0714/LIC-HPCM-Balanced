# HPCM × Balanced R-D Optimization プロジェクト

HPCMにBalanced Rate-Distortion Optimization（CVPR 2025）を統合し、学習ダイナミクス起因の性能損失を削減するプロジェクト。

## 🎯 プロジェクト目標

- **BD-rate改善**: 同じアーキテクチャで2-5%の性能向上
- **訓練安定性**: Rate/Distortionのバランス改善
- **λスイープ**: 各λ値での最適化品質向上

## 📁 プロジェクト構造

```
LIC-HPCM-Balanced/
├── phase1/                          # ✅ Phase 1: 基本統合（完了）
│   ├── src/optimizers/balanced.py   # Balanced optimizer実装
│   ├── train.py                     # Balanced対応訓練スクリプト
│   ├── run_standard.sh              # 標準Adam訓練
│   ├── run_balanced.sh              # Balanced R-D訓練
│   ├── run_hparam_search.sh         # ハイパラ探索
│   ├── test_phase1.py               # テストスクリプト
│   ├── README.md                    # 詳細ドキュメント
│   └── IMPLEMENTATION_GUIDE.md      # 実装ガイド
│
├── phase2/                          # 🔄 Phase 2: ハイパラ最適化（予定）
│   └── (未実装)
│
├── phase3/                          # 🔄 Phase 3: 階層的Balanced（予定）
│   └── (未実装)
│
├── src/                             # オリジナルHPCM実装
│   ├── models/
│   │   ├── HPCM_Base.py
│   │   └── HPCM_Large.py
│   ├── entropy_models/
│   └── ...
│
├── balanced-rd/                     # Balanced R-D参考実装
│   ├── train_balanced.py
│   └── ...
│
├── train.py                         # オリジナル訓練スクリプト
└── PHASES_OVERVIEW.md              # このファイル
```

## 🚀 Phase 1: 基本統合 ✅

**状態**: 実装完了

**実装内容**:
- Balanced optimizer（CautiousAdam + FAMO）
- 2タスクバランシング（Distortion + BPP）
- 標準Adamとの切り替え機能
- WandBでのタスク重み可視化

**使用方法**:
```bash
cd phase1

# 標準訓練（ベースライン）
./run_standard.sh

# Balanced R-D訓練
./run_balanced.sh

# ハイパーパラメータ探索
./run_hparam_search.sh
```

**期待される効果**:
- BD-rate: +1-2%
- 訓練の安定性向上
- λスイープでの品質向上

**詳細**: [phase1/README.md](phase1/README.md) と [phase1/IMPLEMENTATION_GUIDE.md](phase1/IMPLEMENTATION_GUIDE.md) を参照

---

## 🔄 Phase 2: ハイパーパラメータ最適化（予定）

**目的**: HPCMに最適なBalancedパラメータの探索

**実装予定**:
- [ ] グリッドサーチの自動化
- [ ] 動的gamma調整（エポックベース）
- [ ] Fine-tuning機能
- [ ] 結果分析スクリプト

**期間**: 1-2週間

**期待される効果**:
- BD-rate: +2-3%（累積）
- 最適パラメータの特定

---

## 🔄 Phase 3: 階層的Balanced（予定）

**目的**: HPCMの3段階構造（s1, s2, s3）に対応した階層的バランシング

**実装予定**:
- [ ] マルチスケールタスク分解（5タスク）
- [ ] HPCM_Base.py の forward 拡張
- [ ] 階層的損失関数
- [ ] スケール別タスク重み可視化

**期間**: 2-3週間

**期待される効果**:
- BD-rate: +3-5%（累積）
- よりきめ細かい最適化

**リスク**: 実装が複雑、Phase 1-2で十分な効果が出れば不要かも

---

## 📊 Phase 4: 評価とベンチマーク（予定）

**目的**: 全λスイープでの性能評価とBD-rate計算

**実装予定**:
- [ ] BD-rate計算スクリプト
- [ ] Kodak/CLIC/Tecnick評価
- [ ] 標準Adamとの詳細比較
- [ ] 結果可視化

**期間**: 1週間

---

## 📋 実装ロードマップ

| Phase | 内容 | 期間 | 難易度 | 期待効果 | 状態 |
|-------|------|------|--------|---------|------|
| **1** | 基本統合 | 1週間 | ★☆☆☆☆ | +1-2% | ✅ 完了 |
| **2** | ハイパラ最適化 | 1-2週間 | ★★☆☆☆ | +2-3% | 🔄 予定 |
| **4** | Fine-tuning | 1週間 | ★★☆☆☆ | +1-2% | 🔄 予定 |
| **3** | 階層的Balanced | 2-3週間 | ★★★★☆ | +3-5% | 🔄 後回し |
| **5** | 評価 | 1週間 | ★☆☆☆☆ | - | 🔄 予定 |

**推奨順序**: Phase 1 → Phase 2 → Phase 4 → Phase 5 → (必要なら Phase 3)

---

## 🔧 各Phaseの関係

```
Phase 1 (基本統合)
    ↓
    ├─→ Phase 2 (ハイパラ最適化)
    │       ↓
    │   Phase 4 (Fine-tuning)
    │       ↓
    └─→ Phase 5 (評価・BD-rate)
            ↓
        論文執筆・投稿

    (オプション)
    Phase 3 (階層的Balanced)
        ↓
    Phase 5 (追加評価)
```

---

## 🎓 理論的背景

### なぜBalanced R-Dが有効か？

1. **HPCMの特性**
   - 階層的エントロピーモデル（s1, s2, s3）
   - Rate項が支配的になりやすい
   - 局所解に陥りやすい

2. **Balanced R-Dの解決策**
   - 動的なタスク重み調整（FAMO）
   - 勾配とモーメンタムの整合性チェック（CautiousAdam）
   - 勾配衝突の回避

3. **期待される効果**
   - 同じアーキテクチャで性能向上
   - 訓練の安定性向上
   - Fine-tuningでも有効

### 参考文献

- **Balanced Rate-Distortion Optimization in Learned Image Compression** (CVPR 2025)
  - 著者: Zhang et al.
  - 論文: [balanced-rd/2502.20161v2.pdf](balanced-rd/2502.20161v2.pdf)
  - コード: [balanced-rd/](balanced-rd/)

- **HPCM: Hierarchical Progressive Context Mining**
  - オリジナル実装: [src/models/](src/models/)

---

## 🚀 クイックスタート

### Phase 1を始める

```bash
# 1. Phase 1ディレクトリへ移動
cd phase1

# 2. データセットパスを設定
vim run_balanced.sh  # パスを編集

# 3. 訓練開始
./run_balanced.sh

# 4. WandBで進捗確認
# https://wandb.ai/your-project/LIC_HPCM_Phase1
```

### テスト実行

```bash
cd phase1
python test_phase1.py  # PyTorch環境が必要
```

---

## 📝 開発メモ

### Phase 1 実装完了事項

- ✅ Balanced optimizer実装
- ✅ 損失関数の拡張（distortion分離）
- ✅ 訓練ループの改変
- ✅ タスク重み可視化
- ✅ チェックポイント保存/読み込み
- ✅ 実行スクリプト作成
- ✅ ドキュメント整備
- ✅ テストスクリプト作成

### 次のタスク（Phase 2）

- [ ] Phase 2ディレクトリ作成
- [ ] 自動ハイパラ探索スクリプト
- [ ] 動的gamma調整機能
- [ ] Fine-tuning専用スクリプト
- [ ] 結果分析ツール

---

## 🆘 トラブルシューティング

### Phase 1で問題が発生した場合

1. **[phase1/IMPLEMENTATION_GUIDE.md](phase1/IMPLEMENTATION_GUIDE.md)** のトラブルシューティングを確認
2. **[phase1/README.md](phase1/README.md)** の詳細を確認
3. **test_phase1.py** を実行して基本動作を確認

### 一般的な問題

- **NaN損失**: gamma を増やす、clip_max_norm を調整
- **収束が遅い**: w_lr を増やす
- **タスク重みが極端**: 正常な挙動、必要なら gamma で調整

---

## 📊 成果物（予定）

### 学術的成果
- [ ] 実験結果（BD-rate改善値）
- [ ] 訓練曲線の比較
- [ ] タスク重みの可視化
- [ ] 論文草稿

### 実装成果
- [x] Phase 1実装（完了）
- [ ] Phase 2実装
- [ ] Phase 4実装
- [ ] Phase 5実装
- [ ] （オプション）Phase 3実装

---

## 📞 連絡先

プロジェクト管理: yutendo (mayuyuto0714@gmail.com)

---

**最終更新**: 2026年1月5日
**現在のPhase**: Phase 1 ✅ 完了
**次のステップ**: Phase 1の訓練実行 → Phase 2の計画
