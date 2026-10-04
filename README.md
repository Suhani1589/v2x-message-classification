# V2X Message Classification with Attention

NLP research project for classifying Vehicle-to-Everything (V2X) messages into six categories, reproducing the strongest architecture from Lazaga et al. (ICoAIT 2025) and testing whether an **Attention** mechanism improves it.

## Research question

> Does incorporating an Attention mechanism into the CNN-BRNN architecture improve V2X message classification performance compared with the original CNN-BRNN model?

We do **not** assume Attention helps. The experiment measures that.

## Problem statement

V2X networks exchange short safety-critical and non-critical messages. Flooding the channel with spam harms **availability**. Automatic classification of message content into Emergency, Hazard, Roadwork, Spam, Traffic, and Weather supports prioritization and spam filtering.

## Motivation

The baseline paper shows CNN-BRNN outperforming BRNN, BLSTM, and CNN-BLSTM on the IEEE DataPort V2X message dataset. Flattening bidirectional RNN sequences treats all timesteps uniformly. Attention can learn token importance and also provides an interpretability view of which words the model focuses on.

## Original research paper

**Lazaga, C. F., Rosales, M. A., & Tan, G. P. (2025).** *V2X Message Multi-label Classification: A Deep Learning Approach.* 2025 1st International Conference on Artificial Intelligence Technology (ICoAIT). DOI: [10.1109/ICoAIT67446.2025.11308890](https://doi.org/10.1109/ICoAIT67446.2025.11308890)

Dataset DOI: [10.21227/yhfd-0d92](https://dx.doi.org/10.21227/yhfd-0d92)

See [`PAPER_NOTES.md`](PAPER_NOTES.md) for a strict paper vs. assumption breakdown.

### What the original paper did

- Preprocessed Message Content with Tokenizer + post-padding (`maxlen=12`) + Embedding
- Trained BRNN, CNN-BRNN, BLSTM, CNN-BLSTM for 10 epochs (Keras defaults)
- Reported CNN-BRNN as best (Table V mean Accuracy 90%, F1 84.49%)

### What we reproduce

- BRNN baseline
- CNN-BRNN (paper best model) with the documented output stack: Flatten → Dense(sigmoid) → Dense(softmax), Categorical Cross-Entropy

### What we change (proposed contribution)

- **CNN-BRNN + Attention**: replace Flatten with an explicit additive Attention pooling over bidirectional RNN outputs
- Add a held-out test split for fair model comparison
- Fit the tokenizer on the **training split only** (paper tokenized the full column)

## Classification formulation (important)

The paper calls the task “multi-label” but:

- assigns **one** integer class per message
- one-hot encodes that single class
- uses **Softmax** + **CategoricalCrossentropy**

That is **single-label multi-class** classification (6 exclusive classes). This project uses Softmax + categorical cross-entropy. If the official CSV ever shows multiple simultaneous Message Type values per row, revisit this decision.

## Dataset

| Item | Detail |
|------|--------|
| Official source | IEEE DataPort — `V2X_train - Sheet1.csv` (~1000 rows; paper used 938) |
| Text field | `Message Content` |
| Label field | `Message Type` |
| Classes | Emergency, Hazard, Roadwork, Spam, Traffic, Weather |
| Paper counts | Spam 236, Emergency 149, Traffic 151, Weather 146, Hazard 131, Roadwork 125 |

**Local status:** the official CSV requires IEEE DataPort login and was not present in this workspace. The pipeline therefore creates a **paper-count-matched synthetic proxy** so experiments can run. Place the official file in `data/raw/` to switch automatically.

```bash
# After downloading from IEEE DataPort:
cp "V2X_train - Sheet1.csv" data/raw/
python scripts/prepare_dataset.py
```

## NLP preprocessing

```
Raw V2X Message
 → light cleaning / lowercasing (Tokenizer defaults)
 → train-only Tokenizer vocabulary
 → integer sequences
 → post-padding / truncation to length 12 (configurable)
 → Embedding
 → deep model
```

## Architectures

### Baseline 1 — BRNN (paper)

`Embedding → Bidirectional(SimpleRNN, return_sequences=True) → Flatten → Dense(sigmoid) → Softmax`

### Baseline 2 — CNN-BRNN (paper best)

`Embedding → Conv1D → Bidirectional(SimpleRNN, return_sequences=True) → Flatten → Dense(sigmoid) → Softmax`

### Proposed — CNN-BRNN + Attention

```
Embedding
 → Conv1D
 → Bidirectional(SimpleRNN, return_sequences=True)   # h1..hT
 → Additive Attention
      score_i = vᵀ tanh(W h_i)
      α_i = softmax(score_i)
      context = Σ α_i h_i
 → Dense(sigmoid) → Softmax
```

Attention is implemented explicitly in `v2x_cls/models/attention.py` (not a black-box wrapper).

## Experimental setup

Controlled ablation between CNN-BRNN and CNN-BRNN + Attention:

- Same stratified 70/15/15 split (seed 42)
- Same tokenizer / vocabulary
- Same embedding size, CNN filters, RNN units, dense stack
- Same optimizer (Adam 1e-3), batch size 32, epochs 10 (paper)
- Only major architectural difference: Flatten vs Attention

Hyperparameters not stated in the paper are listed in `config.yaml` and `PAPER_NOTES.md`.

## Project structure

```
v2x-message-classification/
├── config.yaml
├── PAPER_NOTES.md
├── README.md
├── requirements.txt
├── data/
├── scripts/
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── run_experiment.py
├── v2x_cls/
│   ├── data.py
│   ├── preprocessing.py
│   ├── training.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── models/
│       ├── attention.py
│       └── __init__.py
├── checkpoints/
├── experiments/results/
└── notebooks/
```

## Install

```bash
cd v2x-message-classification
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to train

```bash
# Prepare data + tokenizer (train-only fit)
python scripts/prepare_dataset.py

# Train each model
python scripts/train.py --model brnn
python scripts/train.py --model cnn_brnn
python scripts/train.py --model cnn_brnn_attention

# Or full pipeline
python scripts/run_experiment.py
```

Smoke test (2 epochs):

```bash
python scripts/run_experiment.py --epochs 2
```

## How to evaluate

```bash
python scripts/evaluate.py --models brnn cnn_brnn cnn_brnn_attention
```

Outputs:

- `experiments/results/comparison_table.csv`
- per-model confusion matrices and metric JSON
- training curves under `experiments/results/<model>/curves/`
- attention examples for the proposed model

## How to run inference

```bash
python scripts/predict.py \
  --model cnn_brnn_attention \
  --text "Emergency vehicle approaching intersection"
```

Interactive mode:

```bash
python scripts/predict.py --model cnn_brnn_attention --interactive
```

Example output shape:

```
Predicted Class: Emergency
Confidence: XX%
Important tokens: Emergency, vehicle, intersection
```

Attention weights are an **interpretability aid**, not a perfect explanation of model reasoning.

## Results

Measured on the held-out test set after a full 10-epoch run
(`scripts/run_experiment.py --epochs 10 --regenerate-synthetic`).

**Data note:** official IEEE DataPort CSV was unavailable (login required), so
these numbers use the paper-count-matched synthetic proxy (938 rows). Replace
with the official CSV before claiming reproduction of the paper’s Table V.

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Loss |
|-------|----------|-------------------|----------------|------------|------|
| BRNN | 0.9858 | 0.9840 | 0.9866 | 0.9852 | 0.0714 |
| CNN-BRNN | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.0468** |
| CNN-BRNN + Attention | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0562 |

**Answer to the research question (this run):** Attention did **not** improve
over CNN-BRNN. Macro F1 tied at the ceiling (1.0); test loss was slightly
worse for Attention (0.0562 vs 0.0468). CNN-BRNN still outperformed plain BRNN,
which aligns with the paper’s finding that CNN-BRNN is the strongest family.

Full write-up: [`experiments/results/EXPERIMENT_REPORT.md`](experiments/results/EXPERIMENT_REPORT.md).

## Attention visualization

For sample messages, `scripts/evaluate.py` and `scripts/predict.py` write token→weight bar charts under `experiments/results/cnn_brnn_attention/attention_examples/`.

## Limitations

- Official IEEE DataPort CSV may be unavailable without login; synthetic proxy matches paper **counts**, not official wording.
- Paper omits embedding size, RNN units, Conv filters, batch size; we document assumptions.
- Paper metrics appear partly epoch-averaged / validation-oriented; our primary report uses a fixed test set.
- Softmax multi-class may differ from a true multi-label sigmoid setup if labels can co-occur (not evidenced by the paper’s encoding).
- Attention weights ≠ causal feature importance.

## Future work

- Re-run on the official CSV once available
- Compare sigmoid multi-label head if multi-label evidence appears
- Try multi-head / self-attention or Transformer encoders
- Class-weighted loss for imbalance
- Cross-validation and statistical significance tests

## Citation

If you use this repository, also cite the original paper and dataset:

```
J. M R, S. Shibu N B, and S. N Rao, “V2X Message Classification,
Prioritization, and Spam Detection Dataset,” IEEE DataPort.
https://dx.doi.org/10.21227/yhfd-0d92
```
