# Paper Notes — Lazaga, Rosales, Tan (ICoAIT 2025)

Source PDF: `V2X Message Multi-label Classification: A Deep Learning Approach`

This file separates **facts stated in the paper** from **implementation assumptions**.

## Facts from the paper

### Dataset
- Source: IEEE DataPort, Creative Commons, published 23 May 2023 [6]
  DOI: https://dx.doi.org/10.21227/yhfd-0d92
- Original fields: Message ID, Source Vehicle, Destination Vehicle, Message Type,
  Message Content, Priority, Spam
- Study uses only **Message Content** and **Message Type**
- Message types / classes:
  Emergency, Hazard, Roadwork, Spam, Traffic, Weather
- Reported counts (sum = 938):
  - Spam: 236 (25%)
  - Emergency: 149 (16%)
  - Traffic: 151 (16%)
  - Weather: 146 (16%)
  - Hazard: 131 (14%)
  - Roadwork: 125 (13%)
- IEEE DataPort abstract describes **1,000** messages; paper used **938**.
  The paper does not explain which 62 rows were excluded.

### Label encoding (paper)
- Emergency = 0
- Hazard = 1
- Roadwork = 2
- Spam = 3
- Traffic = 4
- Weather = 5
- Labels are converted to one-hot vectors

### Preprocessing (paper)
1. One-hot encoding of class labels
2. Keras `Tokenizer` on Message Content (punctuation removed by default)
3. Post-padding / truncation to **maximum length 12**
4. Embedding layer before the sequence models

### Models evaluated in the paper
1. BRNN (Bidirectional Simple RNN)
2. CNN-BRNN
3. BLSTM
4. CNN-BLSTM

### Output stack (paper)
- Flatten
- Dense with **sigmoid**
- Dense with **softmax**
- Loss: **CategoricalCrossentropy**
- Metrics reported: Accuracy, Precision, Recall, F1, Loss
- `validation_split = 0.3`
- Epochs = 10
- “Default Keras parameters”

### Reported summary (Table V means across epochs)
| Model      | Accuracy | Loss | Precision | Recall | F1    |
|------------|----------|------|-----------|--------|-------|
| BRNN       | 88.98    | 0.51 | 99.03     | 77.62  | 82.18 |
| CNN-BRNN   | 90.00    | 0.45 | 98.46     | 80.12  | 84.49 |
| BLSTM      | 74.44    | 0.82 | 78.86     | 54.77  | 61.27 |
| CNN-BLSTM  | 73.92    | 0.81 | 87.14     | 57.38  | 64.87 |

CNN-BRNN is the strongest model in the paper → our reproduction baseline.

## Critical formulation note (multi-label vs multi-class)

The paper **calls** the task “multi-label classification” and uses Softmax.

However, the paper also:
- assigns **one integer class id per message**
- one-hot encodes that single class
- optimizes **CategoricalCrossentropy**
- describes Message Type as *the* category of the message

That formulation is **single-label multi-class classification** (6 mutually exclusive classes),
not multi-label classification (where multiple labels can be active at once).

**Project decision:** treat the task as **multi-class**. Use Softmax + categorical
cross-entropy. If inspection of the official CSV ever shows multiple simultaneous
Message Type values for one row, revisit this decision.

## Ambiguities / assumptions used in this repository

| Item | Paper | Our assumption |
|------|-------|----------------|
| Embedding dimension | not stated | 64 |
| RNN units | “Keras defaults” | 64 |
| Conv1D filters / kernel | not stated | 64 / 3 |
| Dense hidden units | not stated | 64 |
| Batch size | not stated | 32 |
| Optimizer / LR | not stated | Adam, 1e-3 |
| Test split | only val_split=0.3 | fixed 70/15/15 stratified split |
| Tokenizer fit scope | “whole message content column” | **train split only** (avoid leakage) |
| Dropout | not mentioned | 0.0 for fair paper-style baseline |
| Why 938 of 1000 | unexplained | use all clean rows; synthetic proxy matches 938 counts |

## Dataset availability in this workspace

The official file `V2X_train - Sheet1.csv` requires an IEEE DataPort login/subscription
and was **not present locally**. This project therefore:

1. Provides a **paper-faithful synthetic proxy** matching the reported class counts and
   short V2X-style message content so the full pipeline can run.
2. Automatically prefers the official CSV if you place it under `data/raw/`.

Replace the synthetic file with the official dataset before claiming numerical
reproduction of Table V.
