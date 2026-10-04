# Experiment Report — V2X CNN-BRNN + Attention

Generated from the completed local experiment on the paper-count-matched
synthetic proxy (official IEEE DataPort CSV was not available without login).

## 1. What the original paper did

Lazaga et al. (ICoAIT 2025) classified V2X `Message Content` into six
`Message Type` classes using BRNN, CNN-BRNN, BLSTM, and CNN-BLSTM after
Tokenizer + post-padding (`maxlen=12`) + Embedding. They reported CNN-BRNN
as best (Table V means: Acc 90%, F1 84.49%, Loss 0.45).

## 2. What we reproduced

- BRNN and CNN-BRNN with Flatten → Dense(sigmoid) → Softmax
- Categorical cross-entropy loss
- `maxlen=12`, post-padding, 10 epochs
- Same label order as the paper

## 3. What we changed

- Proposed **CNN-BRNN + Additive Attention** (Attention replaces Flatten)
- Train-only tokenizer fitting (no leakage)
- Fixed stratified 70/15/15 split with a held-out test set
- Explicit Attention visualization / inference tooling

## 4. Why Attention was added

To test whether learning token importance over bidirectional RNN states
improves classification versus flattening all timesteps, and to add an
interpretability view of important tokens.

## 5. Exact proposed architecture

```
Embedding(vocab, 64)
 → Conv1D(filters=64, kernel=3, padding=same, relu)
 → Bidirectional(SimpleRNN(64, return_sequences=True))
 → AdditiveAttention: score=vᵀtanh(Wh); α=softmax(score); context=Σ αh
 → Dense(64, sigmoid) → Dense(6, softmax)
```

## 6. Dataset details

| Item | Value |
|------|-------|
| Source used | Synthetic proxy (official CSV requires IEEE DataPort login) |
| Rows | 938 (paper counts) |
| Classes | Emergency 149, Hazard 131, Roadwork 125, Spam 236, Traffic 151, Weather 146 |
| Split | 656 / 141 / 141 (train/val/test), seed 42 |
| Formulation | **Multiclass** (not multilabel): Softmax + categorical CE |

## 7. Experimental setup

Identical settings for all three models except Attention vs Flatten:
embedding 64, CNN 64/3, RNN 64, dense 64, Adam 1e-3, batch 32, epochs 10.

## 8–10. Test-set results

| Model | Accuracy | Precision_macro | Recall_macro | F1_macro | Loss |
|-------|----------|-----------------|--------------|----------|------|
| BRNN | 0.9858 | 0.9840 | 0.9866 | 0.9852 | 0.0714 |
| CNN-BRNN | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.0468** |
| CNN-BRNN + Attention | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0562 |

Artifacts: `experiments/results/comparison_table.csv`, curves, confusion
matrices, attention plots.

## 11. Did Attention improve performance?

**No clear improvement over CNN-BRNN on this run.**

- Macro F1: tied at 1.0000 (ceiling on the proxy test set)
- Test loss: Attention **worse** (0.0562 vs 0.0468) → ΔLoss = +0.0094

## 12. F1-score change

Δ F1_macro (Attention − CNN-BRNN) = **0.0000** (no gain; both saturated).

## 13. Important observations

- CNN-BRNN still beats plain BRNN (F1 1.000 vs 0.985), consistent with the
  paper’s ranking that CNN-BRNN is strong.
- Attention provides useful token highlights for inference, even when F1
  does not improve (example: “Emergency vehicle approaching intersection”
  → predicted Emergency at 93.5% confidence).
- The synthetic proxy remains easier than the paper’s reported mean metrics;
  official CSV is required before claiming numerical reproduction of Table V.
- Paper terminology says “multi-label” but encoding + Softmax + categorical
  CE imply multiclass; we followed the math, not the label wording.

## 14. Limitations

- Official dataset unavailable without IEEE DataPort subscription
- Unstated paper hyperparameters filled with documented assumptions
- Ceiling effect limits Attention ablation sensitivity
- Attention weights ≠ causal explanations
- Single split / no significance testing

## 15. Possible future improvements

- Re-run on official `V2X_train - Sheet1.csv`
- Stronger regularization / fewer epochs / smaller models to avoid saturation
- Class-weighted loss, k-fold CV, statistical tests
- Multi-head attention / Transformer baselines
- True multilabel head only if labels co-occur in official data
