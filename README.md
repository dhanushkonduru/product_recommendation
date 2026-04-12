# Multimodal Product Recommendation System

End-to-end recommender system that combines text and image understanding (CLIP) with two ranking backends:

- Option B baseline: lightweight time-decayed pooling + cosine-style scoring.
- Research transformer: sequential transformer with CLS token and BPR ranking loss.

This repository also includes a Streamlit app for interactive recommendations and model comparison.

This README is the single project documentation file and contains architecture + structure details.

## What This Project Is

This project solves a real recommendation problem: given a user's past fashion interactions, predict what they are likely to interact with next.

Core properties:

- Multimodal item representation: title text + product image.
- Sequential modeling: user history order is preserved (important for intent drift).
- Implicit-feedback ranking: trained/evaluated with retrieval metrics such as Recall@K and NDCG@K.
- Apple Silicon support: research model is designed to run well on MPS (Metal backend).

## What You Get In This Repo

- Data preparation pipeline from raw Amazon-style JSON.gz files.
- CLIP encoder wrapper for aligned text/image embeddings.
- Two recommendation models.
- Saved checkpoints and evaluation artifacts.
- Streamlit demo for qualitative and quantitative inspection.
- Consolidated architecture and project-structure documentation in this file.

## Final Results Achieved

Evaluation setup used in this project:

- 2000 sampled evaluation users.
- Top-K retrieval metrics: K = 5, 10, 20.
- Seen-item masking during ranking.

### Test Metrics

| Model | Recall@5 | Recall@10 | Recall@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|---|---:|---:|---:|---:|---:|---:|
| Research Transformer | 0.4339 | 0.5225 | 0.5564 | 0.3590 | 0.3908 | 0.4008 |
| Option B Baseline | 0.4786 | 0.5277 | 0.5601 | 0.4325 | 0.4523 | 0.4627 |

Result files available in `checkpoints/`:

- `checkpoints/research_mps_results.pkl`
- `checkpoints/option_b_results.pkl`

## High-Level Architecture

### 1) Item Embeddings (Shared by Both Models)

- Encoder: CLIP ViT-B/32.
- Text embedding and image embedding are fused by averaging.
- Final vector is L2-normalized.
- Embedding size: 512.

Fusion rule:

$$
e_{item} = \text{normalize}\left(\frac{e_{text} + e_{image}}{2}\right)
$$

### 2) Option B Baseline

- User vector = time-decayed weighted mean of the last 5 interacted items.
- Item scoring uses vectorized dot product on normalized vectors.
- Optional projection layer can be BPR-tuned.

### 3) Research Transformer

- Input tokens: `[CLS] + user_history + candidate_item`.
- Learnable position embeddings and modality/type embeddings.
- 4-layer PreNorm transformer encoder, 8 heads, GELU feed-forward block.
- Score head: CLS output -> MLP -> scalar relevance score.
- Loss: BPR pairwise ranking loss.

## Inference and Evaluation Design

Important engineering choice in this repo: vectorized scoring.

- For each user, candidate scoring is done in batch against all catalog items.
- Seen items are masked before top-K extraction.
- This avoids slow nested Python loops and keeps inference practical for demo/eval.

## Detailed Architecture

### Transformer Input and Token Layout

The research model builds a joint token sequence:

- token 0: learnable CLS token
- tokens 1..S: user history item embeddings
- final token: candidate item embedding

With `MAX_SEQ_LEN = 20`, the maximum token length is:

$$
	ext{MAX\_TOKENS} = 20 + 2 = 22
$$

### Learnable Embedding Components

- positional embedding over 22 positions
- modality/type embedding with 3 types:
	- type 0: CLS
	- type 1: user-context tokens
	- type 2: candidate token

### Encoder and Scoring Head

- 4-layer TransformerEncoder
- PreNorm (`norm_first=True`)
- 8 attention heads
- feed-forward block: 512 -> 2048 -> 512
- CLS output is layer-normalized and sent through MLP: 512 -> 256 -> 1

### Training Objective (BPR)

Pairwise ranking objective:

$$
\mathcal{L}_{BPR} = -\log\left(\sigma(s_{pos} - s_{neg})\right)
$$

where $s_{pos}$ is the score for a positive next item and $s_{neg}$ is for a sampled negative item.

### Data Flow (End-to-End)

1. Raw interactions and metadata are loaded and filtered.
2. Temporal splits are generated: train/val/test.
3. Item text/image embeddings are built with CLIP and fused.
4. Models are trained with ranking loss.
5. Metrics are computed on sampled users with top-K retrieval.
6. Streamlit app serves recommendations from saved checkpoints.

## Repository Map (Complete)

### Top Level

- `app.py`: Streamlit application used for interactive recommendation demo.
- `README.md`: Project overview and usage guide.
- `requirements.txt`: Python dependencies.

### Directory Tree

```text
product_recommendation/
├── README.md
├── requirements.txt
├── app.py
├── checkpoints/
│   ├── .gitkeep
│   ├── item_embedding_matrix.pt
│   ├── option_b_model.pt
│   ├── option_b_results.pkl
│   ├── research_mps_model.pt
│   └── research_mps_results.pkl
├── data/
│   ├── AMAZON_FASHION_5.json.gz
│   ├── meta_AMAZON_FASHION.json.gz
│   ├── image_encoder_data.pkl
│   └── prepared_data.pkl
├── models/
│   ├── clip_encoder.py
│   └── option_b_model.py
└── src/
	├── __init__.py
	├── data/
	│   ├── __init__.py
	│   └── prepare_dataset.py
	├── embeddings/
	│   ├── __init__.py
	│   └── clip_encoder.py
	├── evaluation/
	│   └── __init__.py
	├── models/
	│   ├── __init__.py
	│   ├── baseline.py
	│   └── transformer.py
	└── training/
		└── __init__.py
```

### Data and Artifacts

- `data/AMAZON_FASHION_5.json.gz`: raw interaction data.
- `data/meta_AMAZON_FASHION.json.gz`: item metadata.
- `data/prepared_data.pkl`: processed sequences/splits/metadata/index maps.
- `data/image_encoder_data.pkl`: precomputed image embeddings.
- `checkpoints/research_mps_model.pt`: transformer weights.
- `checkpoints/item_embedding_matrix.pt`: item embedding matrix used by transformer inference.
- `checkpoints/option_b_model.pt`: Option B model artifact.
- `checkpoints/research_mps_results.pkl`: transformer metrics.
- `checkpoints/option_b_results.pkl`: Option B metrics.

### Source Modules

- `src/data/prepare_dataset.py`: raw data loading, filtering, temporal split, metadata prep.
- `src/embeddings/clip_encoder.py`: CLIP wrapper (text/image encoders + compatibility wrappers).
- `src/models/baseline.py`: Option B baseline implementation and evaluation flow.
- `src/models/transformer.py`: research transformer training/evaluation implementation.
- `src/training/__init__.py`: placeholder package for future training module extraction.
- `src/evaluation/__init__.py`: placeholder package for future metrics module extraction.

### Legacy Compatibility Modules

- `models/clip_encoder.py`: compatibility CLIP encoder module.
- `models/option_b_model.py`: compatibility Option B module.

## How to Run

## 1) Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Run the Demo App (Fastest Way to Validate Project)

This repo already includes prepared data and checkpoints. You can run the app directly:

```bash
streamlit run app.py
```

App capabilities:

- Choose model backend (Option B or Research Transformer).
- Select/randomize users.
- View user recent history.
- Generate Top-K recommendations.
- View Recall@10 and NDCG@10 from saved evaluation outputs.
- Cold-start fallback via popularity ranking.

## 3) Data Preparation Script

Data preparation logic exists in `src/data/prepare_dataset.py` and includes:

- review parsing from gzipped JSON lines,
- user/item filtering,
- temporal split,
- metadata extraction,
- index mapping and serialization.

Note: it expects raw source file names consistent with the script (`reviews_Fashion.json.gz`, `meta_Fashion.json.gz`) unless adapted. The repository currently stores raw files as `data/AMAZON_FASHION_5.json.gz` and `data/meta_AMAZON_FASHION.json.gz`.

## 4) Transformer Training/Evaluation

Primary research training entrypoint:

```bash
python src/models/transformer.py --train_and_eval --epochs 5 --batch_size 128 --max_users 50000
```

Expected outputs in `checkpoints/`:

- `research_mps_model.pt`
- `item_embedding_matrix.pt`
- `research_mps_results.pkl`

## About Option B Training Script

The repository includes Option B training implementations under both `src/models/baseline.py` and `models/option_b_model.py`. The shipped checkpoints already let you run inference in the app immediately.

## Technical Stack

- PyTorch
- Transformers (Hugging Face)
- Streamlit
- NumPy
- Pandas
- Pillow
- tqdm

## Why This Project Matters

- Demonstrates that strong multimodal embeddings + efficient ranking can be highly competitive.
- Shows practical trade-off between lightweight baseline and expressive sequential transformer.
- Includes a full demo path from artifacts to UI, not only offline model scripts.
- Useful template for recommendation systems on Apple Silicon environments.

## Current Limitations and Next Improvements

- Training/evaluation utility code can be further modularized into `src/training/` and `src/evaluation/`.
- Path/config conventions are not yet fully unified across all scripts.
- Additional experiments (hard negatives, longer history, hyperparameter sweeps) can improve transformer quality.

## References

- Radford et al., CLIP (ICML 2021)
- Rendle et al., BPR (UAI 2009)
- McAuley et al., Amazon fashion recommendation datasets

---

If someone new reads this README, they should now understand:

- what problem is solved,
- how data flows through the system,
- how each major file/module contributes,
- which results were achieved,
- and how to run the project quickly.
