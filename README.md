# Team 4 — Workshop #4: Autoencoders & Variational Autoencoders for Text Representation

Machine Learning (Semester 2026-III) — Systems Engineering,
**Universidad Distrital Francisco José de Caldas**. Instructor: Eng. Carlos Andrés Sierra, M.Sc.

Team: Xiomara Salome Arias Arias · Julian David Celis Giraldo · Alejandro Gomez Moyano

Unsupervised representation learning over 42,632 Disneyland reviews using autoencoders
and variational autoencoders. Input representations are CLS embeddings extracted from the
fine-tuned RoBERTa checkpoint of Workshop #2.

## Repository structure

```
src/
├── extract_embeddings.py   extract RoBERTa CLS vectors from the W2 checkpoint (run once)
├── autoencoder.py          AE and VAE architectures

checkpoints/
├── model_best.pt           RoBERTa fine-tuned checkpoint from Workshop #2 (via Git LFS)

data/
├── silver/reviews.parquet  cleaned corpus from Workshop #1 (42,632 rows, gitignored)
├── gold/splits.json        frozen 70/15/15 stratified split from Workshop #1 (gitignored)
├── gold/embeddings_bert.npy  RoBERTa CLS vectors (42,632 × 768) (gitignored)
└── gold/labels.npy         integer labels aligned to silver rows (gitignored)

```

## Setup

Python 3.12–3.14 and Poetry:

```bash
# CPU
poetry install

# AMD GPU (ROCm) — uncomment [tool.poetry.group.rocm.dependencies] first
poetry install --with rocm
```

## Data

Reuses silver and splits from Workshop #1, and the RoBERTa checkpoint from Workshop #2:

```bash
mkdir -p data/silver data/gold checkpoints
cp <workshop1>/data/silver/reviews.parquet data/silver/
cp <workshop1>/data/gold/splits.json       data/gold/
cp <workshop2>/checkpoints/model_best.pt   checkpoints/
```

## Step 1 — Extract embeddings (run once)

```bash
# CPU (≈ 45–70 min)
poetry run python src/extract_embeddings.py --checkpoint checkpoints/model_best.pt --batch_size 16

# AMD GPU
poetry run python src/extract_embeddings.py --checkpoint checkpoints/model_best.pt --batch_size 32
```

Saves `data/gold/embeddings_bert.npy` (42,632 × 768) and `data/gold/labels.npy`.

## Step 2 — Verify architectures

```bash
poetry run python src/autoencoder.py
```

Expected output:
```
AE   — input (8, 768) → z (8, 64) → recon (8, 768)
VAE  — input (8, 768) → µ (8, 64) → recon (8, 768)
AE parameters: 476,736
VAE parameters: 484,992
```

## Method

| | |
| --- | --- |
| Input | RoBERTa CLS token, 768-d float32 |
| Normalisation | `StandardScaler` fitted on train split only |
| Encoder | FC 768 → 256 → 128 → z (64-d), GELU + Dropout(0.2) |
| Decoder | FC 64 → 128 → 256 → 768, GELU + Dropout(0.2), linear output |
| AE loss | MSE |
| VAE loss | MSE + β · KL,  β ∈ {0.5, 1.0, 4.0} with KL annealing |
| Optimizer | Adam, 30–50 epochs, batch size 256 |
| Anomaly detection | Train on positive class only; MSE as anomaly score |

## Related

- Workshop #1: [JulianCelisCreator/ML-NLP-Sentimental-Analysis](https://github.com/JulianCelisCreator/ML-NLP-Sentimental-Analysis)
- Workshop #2: [JulianCelisCreator/JulianCelisCreator-ML-NLP-Sentimental-Analysis-W2](https://github.com/JulianCelisCreator/JulianCelisCreator-ML-NLP-Sentimental-Analysis-W2)
