# Project Structure

```
product_recommendation/
│
├── README.md                        # Quick start guide, results, installation
├── ARCHITECTURE.md                  # Deep technical documentation
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore rules
│
├── app.py                           # Streamlit production demo (800 lines)
│
├── src/                             # Clean modular source code
│   ├── __init__.py
│   │
│   ├── data/                        # Data processing pipeline
│   │   ├── __init__.py
│   │   └── prepare_dataset.py      # Raw → prepared_data.pkl (450 lines)
│   │
│   ├── embeddings/                  # Multimodal embedding extraction
│   │   ├── __init__.py
│   │   └── clip_encoder.py         # Frozen CLIP ViT-B/32 wrapper (195 lines)
│   │
│   ├── models/                      # Model architectures
│   │   ├── __init__.py
│   │   ├── baseline.py             # Option B: Pooling + BPR (680 lines)
│   │   └── transformer.py          # Sequential Transformer (760 lines)
│   │
│   ├── training/                    # Training utilities (TODO: extract)
│   │   └── __init__.py
│   │
│   └── evaluation/                  # Evaluation metrics (TODO: extract)
│       └── __init__.py
│
├── data/                            # Datasets (gitignored)
│   ├── AMAZON_FASHION_5.json.gz    # Raw user-item interactions
│   ├── meta_AMAZON_FASHION.json.gz # Item metadata (title, image_url)
│   ├── prepared_data.pkl           # Processed: train/val/test splits
│   └── image_encoder_data.pkl      # CLIP image embeddings [798, 512]
│
├── checkpoints/                     # Model weights & results (gitignored)
│   ├── .gitkeep
│   ├── research_mps_model.pt       # Transformer weights (49.9 MB)
│   ├── item_embedding_matrix.pt    # Frozen CLIP embeddings [1000, 512] (2.0 MB)
│   ├── option_b_model.pt           # Baseline weights (2.1 MB)
│   ├── option_b_results.pkl        # Baseline metrics
│   └── research_mps_results.pkl    # Transformer metrics
│
└── models/                          # Legacy folder (kept for compatibility)
    ├── clip_encoder.py             # Same as src/embeddings/
    └── option_b_model.py           # Same as src/models/baseline.py

```

---

## File Descriptions

### Top-Level

- **README.md**: Production-grade documentation with badges, quick start, results table, citations
- **ARCHITECTURE.md**: 650-line technical deep-dive (model design, BPR loss, MPS optimization, vectorized eval)
- **app.py**: Streamlit app with model selector, user history viz, recommendation grid, caching
- **requirements.txt**: PyTorch, Transformers, Streamlit, Pillow, NumPy, tqdm

### src/ (Clean Modular Code)

**src/data/**
- `prepare_dataset.py`: Loads raw Amazon Fashion data, creates train/val/test splits, extracts metadata

**src/embeddings/**
- `clip_encoder.py`: Wraps HuggingFace CLIP (ViT-B/32), frozen weights, text + image encoding

**src/models/**
- `baseline.py`: Option B model (time-decayed pooling, BPR projection, dot product scoring)
- `transformer.py`: Research-grade 4-layer PreNorm Transformer with CLS token, BPR loss, MPS-optimized

**src/training/** (Planned Refactor)
- Extract `train()` function from transformer.py
- Shared training loop, optimizer setup, gradient clipping
- BPR loss implementation

**src/evaluation/** (Planned Refactor)
- Extract `evaluate()` function from transformer.py
- Recall@K, NDCG@K metrics
- Vectorized batch evaluation

### data/

- **AMAZON_FASHION_5.json.gz**: Raw 5-core filtered interactions (223k users, 1k items)
- **meta_AMAZON_FASHION.json.gz**: Item metadata (title, brand, image URLs)
- **prepared_data.pkl**: Preprocessed dict with `train_sequences`, `test_interactions`, `metadata`, etc.
- **image_encoder_data.pkl**: `{'item_image_embeddings': {item_id → Tensor[512]}}`

### checkpoints/

- **research_mps_model.pt**: Transformer state dict (12.76M params)
- **item_embedding_matrix.pt**: Frozen CLIP embeddings [1000, 512] float32
- **option_b_model.pt**: Baseline weights + projection layer + item_to_idx mapping
- **\*_results.pkl**: Evaluation metrics (Recall@K, NDCG@K for K=5,10,20)

### models/ (Legacy)

Kept for backward compatibility with existing scripts. Will be removed after full migration to `src/`.

---

## Key Design Principles

1. **Modular**: Clean separation of data/embeddings/models/training/eval
2. **Reproducible**: Fixed SEED=42, deterministic sampling
3. **Typed**: Type hints on all functions (str, List, Dict, Tensor)
4. **Documented**: Docstrings explain inputs/outputs/strategy
5. **Cached**: Streamlit `@st.cache_*` for data/models
6. **Vectorized**: Zero nested loops in scoring/evaluation
7. **MPS-first**: Optimized for Apple Silicon (not CUDA)

---

## Current vs Future Structure

### Current (After Refactor)

```
✅ src/data/prepare_dataset.py
✅ src/embeddings/clip_encoder.py
✅ src/models/baseline.py
✅ src/models/transformer.py (training + eval embedded)
⚠️  src/training/ (empty)
⚠️  src/evaluation/ (empty)
```

### Planned (Fully Modular)

```
src/data/prepare_dataset.py
src/embeddings/clip_encoder.py
src/models/baseline.py
src/models/transformer.py (architecture only)
src/training/trainer.py (BPR loss, AdamW, train loop)
src/evaluation/metrics.py (Recall@K, NDCG@K, vectorized scoring)
```

---

## Migration Guide

### Running Baseline

```bash
# New path
python src/models/baseline.py --train

# Old path (still works via legacy folder)
python models/option_b_model.py --train
```

### Running Transformer

```bash
# New path
python src/models/transformer.py --train_and_eval --epochs 5

# Old path (removed)
# python 09_research_multimodal_transformer_mps.py --train_and_eval
```

### Imports in Streamlit App

```python
# App loads models from checkpoints/ folder
load_option_b_model() → "checkpoints/option_b_model.pt"
load_research_transformer() → "checkpoints/research_mps_model.pt"
```

---

## Technical Debt & TODOs

- [ ] Extract training loop from transformer.py → src/training/trainer.py
- [ ] Extract evaluation metrics → src/evaluation/metrics.py
- [ ] Remove legacy models/ folder after testing
- [ ] Add unit tests for each module
- [ ] Add config.yaml for hyperparameters (replace hardcoded constants)
- [ ] Add logging (replace print statements)
- [ ] Add CLI with argparse in src/models/\*.py
- [ ] Add experiment tracking (Weights & Biases / MLflow)

---

*Structure Version: 1.0 | Date: February 2026*
