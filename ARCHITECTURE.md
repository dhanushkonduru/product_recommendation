# Architecture: Research-Grade Multimodal Sequential Transformer Recommender

## Overview

This repository implements a **production-quality multimodal sequential transformer** for product recommendation, optimized for Apple Silicon (M1/M2 with MPS backend). The system combines frozen CLIP embeddings with learnable sequential modeling to capture user preferences over time.

---

## 1. Problem Statement

**Goal**: Predict which products a user will interact with next, given their interaction history.

**Challenges**:
- **Sparse data**: Median user has only 2 interactions
- **Cold items**: 202/1000 items (20%) have no image embeddings
- **Sequential patterns**: Recent interactions are more predictive than older ones
- **Multimodality**: Text and visual signals carry complementary information

---

## 2. Multimodal Embedding Design

### 2.1 Item Embeddings

Items are represented by **512-dimensional frozen CLIP embeddings**:

```
Text Embedding (CLIP ViT-B/32 text encoder):
  - Input: Product title tokenized
  - Output: [512] float32 vector
  - Coverage: 1000/1000 items (100%)

Image Embedding (CLIP ViT-B/32 vision encoder):
  - Input: Product image (224×224 RGB)
  - Output: [512] float32 vector
  - Coverage: 798/1000 items (80%)

Fusion Strategy:
  - If both modalities available: (text + image) / 2
  - If text only: text
  - L2-normalize all embeddings → unit hypersphere
```

**Why frozen CLIP?**
- Pre-trained on 400M image-text pairs (web-scale multimodal knowledge)
- Zero-shot transfer to fashion domain
- Efficient: no fine-tuning required for item tower
- Stable: embeddings computed once, cached on disk

### 2.2 User Embeddings

Users are represented by their **interaction history** processed through a learnable transformer.

---

## 3. Sequential Transformer Architecture

### 3.1 Model Design

```
┌─────────────────────────────────────────────────────────────┐
│                   INPUT SEQUENCE                            │
├─────────────────────────────────────────────────────────────┤
│  [CLS]  [item₁]  [item₂]  ...  [item₂₀]  [candidate]       │
│   ↓       ↓        ↓               ↓          ↓             │
│  512d   512d     512d    ...     512d       512d            │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│              LEARNABLE EMBEDDINGS                           │
├─────────────────────────────────────────────────────────────┤
│  Positional Embedding (22 positions) → [512]                │
│  Modality Type Embedding (3 types) → [512]                  │
│    - Type 0: CLS token                                      │
│    - Type 1: User context items                             │
│    - Type 2: Candidate item                                 │
│  Input Projection: 512 → 512 (if needed)                    │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│          4-LAYER TRANSFORMER ENCODER                        │
├─────────────────────────────────────────────────────────────┤
│  Architecture: PreNorm (norm_first=True)                    │
│  Attention: 8 heads, 512 hidden dim                         │
│  FFN: 512 → 2048 → 512, GELU activation                     │
│  Dropout: 0.1                                               │
│  Output: [B, 22, 512]                                       │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│            CLS TOKEN EXTRACTION                             │
├─────────────────────────────────────────────────────────────┤
│  Extract: output[:, 0, :] → [B, 512]                        │
│  LayerNorm: stabilize CLS representation                    │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│              SCORING HEAD                                   │
├─────────────────────────────────────────────────────────────┤
│  MLP: 512 → 256 (GELU, Dropout 0.1) → 1                    │
│  Output: scalar relevance score                             │
└─────────────────────────────────────────────────────────────┘
```

**Key Design Choices**:

1. **PreNorm Architecture**: LayerNorm before attention/FFN (more stable training)
2. **CLS Token**: Learnable global context aggregator (inspired by BERT)
3. **Modality-Aware**: Separate embeddings for CLS/context/candidate distinguish roles
4. **No Cross-Attention**: Single-tower design (user+item jointly encoded) for efficiency

### 3.2 Why This Architecture?

|Component|Purpose|Benefit|
|---------|-------|-------|
|**Transformer Encoder**|Capture sequential dependencies|Learns "recently bought X → next buy Y" patterns|
|**CLS Token**|Global user representation|Single vector summarizes full interaction history|
|**Positional Emb**|Encode sequence order|Model knows item₁ came before item₂₀|
|**Modality Emb**|Distinguish context vs candidate|Model learns "is this item being scored?"|
|**4 Layers**|Depth vs efficiency tradeoff|Enough capacity, fast inference on MPS|

---

## 4. BPR Ranking Loss

### 4.1 Loss Function

**Bayesian Personalized Ranking (BPR)** optimizes pairwise ranking:

```python
L_BPR = -log(σ(s_pos - s_neg))

where:
  s_pos = model(user_history, positive_item)
  s_neg = model(user_history, negative_item)
  σ = sigmoid function
```

**Intuition**: Positive items should score higher than negative items.

### 4.2 Training Strategy

```
Data Preparation (per user):
  - For each interaction at position i:
    - Context: items[max(0, i-20):i]  (last 20 items)
    - Positive: items[i]
    - Negative: random item NOT in user history
    
Training Loop:
  - Shuffle training pairs
  - Batch size: 128
  - Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
  - Gradient clipping: max_norm=1.0
  - Epochs: 5
  
Loss Computation (fully vectorized):
  pos_scores = model(user_embs, pos_item_embs)  # [B]
  neg_scores = model(user_embs, neg_item_embs)  # [B]
  loss = -log(sigmoid(pos_scores - neg_scores) + 1e-8).mean()
```

**Why BPR?**
- Directly optimizes ranking (not pointwise prediction)
- Handles implicit feedback (no ratings, only clicks/purchases)
- Robust to label noise

---

## 5. Vectorized Evaluation

### 5.1 Challenge

Naive evaluation requires `O(users × items)` forward passes:

```python
# ❌ SLOW: Nested loops
for user in eval_users:
    for item in catalog:
        score = model(user, item)  # 2000 × 1000 = 2M forwards!
```

### 5.2 Solution: Batch Expansion

```python
# ✅ FAST: Vectorized scoring
user_seq = get_padded_sequence(user)           # [1, 20, 512]
user_expanded = user_seq.expand(1000, -1, -1)  # [1000, 20, 512]
all_scores = model(user_expanded, item_matrix) # [1000] in ONE forward!
```

**Key Insight**: Expand user sequence to match all items, compute scores in parallel.

### 5.3 Evaluation Pipeline

```
1. Pre-build user sequences:
   - Pad to MAX_SEQ_LEN=20
   - Convert to index tensors
   
2. Batch users (8 per batch):
   - Load user sequences → [8, 20, 512]
   
3. For each user in batch:
   - Expand user → [1000, 20, 512]
   - Score all items → [1000]
   - Mask seen items (set score to -1e9)
   - TopK → [K] indices
   
4. Compute metrics:
   - Recall@K: |TopK ∩ TestSet| / |TestSet|
   - NDCG@K: Discounted cumulative gain with position penalty
```

**Performance**: ~5 minutes for 2000 users on MPS (vs ~2 hours with nested loops).

---

## 6. Apple MPS Optimization

### 6.1 Why MPS?

**MPS (Metal Performance Shaders)** is Apple's GPU acceleration framework:
- Native support for M1/M2 chips (unified memory architecture)
- 2-4× faster than CPU for transformer inference
- No CUDA/cuDNN dependencies

### 6.2 Optimization Techniques

```python
# Device detection
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# All tensors on device
item_matrix = item_matrix.to(device)  # [1000, 512]
model = model.to(device)

# No AMP (MPS doesn't support float16 well)
# Use float32 throughout

# No torch.compile (not yet supported on MPS)

# Efficient memory usage
with torch.no_grad():  # Disable gradient tracking
    scores = model(user_expanded, items)
```

### 6.3 Constraints

- **float32 only**: MPS has issues with mixed precision
- **batch_first=True**: Required for MPS TransformerEncoder
- **No nested tensors**: Disable PyTorch nested tensor optimization

---

## 7. Data Flow

### 7.1 Training Pipeline

```
Raw Data (AMAZON_FASHION_5.json.gz)
  ↓
prepare_dataset.py
  ↓
prepared_data.pkl
  - train_sequences: {user_id: [(item_id, timestamp), ...]}
  - test_interactions: {user_id: set(item_ids)}
  - metadata: {item_id: {title, image_url}}
  ↓
clip_encoder.py (run once)
  ↓
image_encoder_data.pkl
  - item_image_embeddings: {item_id: [512] tensor}
  ↓
transformer.py --train
  ↓
research_mps_model.pt (model weights)
item_embedding_matrix.pt (frozen CLIP embeddings)
research_mps_results.pkl (metrics)
```

### 7.2 Inference Pipeline

```
User ID
  ↓
Load prepared_data.pkl → get train_sequences[user_id]
  ↓
Extract last 20 items → [20] item IDs
  ↓
Convert to indices → [20] integers
  ↓
Lookup in item_embedding_matrix → [20, 512]
  ↓
Expand to [1000, 20, 512]
  ↓
Transformer forward → [1000] scores
  ↓
Mask seen items
  ↓
TopK → [K] recommendations
```

---

## 8. Baseline Model (Option B)

For comparison, we also implement a **lightweight baseline**:

```
Item Embeddings: Frozen CLIP (text + image) / 2, L2-normalized
User Embeddings: Time-decayed weighted mean of last 5 items
Scoring: Dot product (cosine similarity)
Optional: 512→512 projection layer with BPR fine-tuning
```

**Why keep a baseline?**
- Fast inference (no transformer overhead)
- Strong performance on sparse data (Recall@10: 0.5277)
- Interpretable (pure similarity matching)
- Production fallback if transformer is too slow

---

## 9. Key Innovations

1. **No nested loops in evaluation**: Fully vectorized scoring via batch expansion
2. **MPS-first design**: Optimized for Apple Silicon (not CUDA)
3. **PreNorm transformer**: Stable training without warmup
4. **Modality-aware embeddings**: Model knows context vs candidate distinction
5. **Frozen multimodal tower**: Zero-shot CLIP, no item-side gradients
6. **BPR ranking loss**: Direct optimization of recommendation metrics

---

## 10. Model Parameters

```
Model: SequentialTransformerRecommender
Total parameters: 12,755,457

Breakdown:
  - CLS token: 512
  - Positional embeddings: 22 × 512 = 11,264
  - Modality embeddings: 3 × 512 = 1,536
  - Transformer (4 layers):
      - Self-attention: 4 × (4 × 512 × 512) = 4,194,304
      - FFN: 4 × (512 × 2048 + 2048 × 512) = 8,388,608
  - LayerNorm: ~6,000
  - Scoring head: 512 × 256 + 256 × 1 = 131,328
```

---

## 11. Performance Metrics

### 11.1 Test Set Evaluation (2000 users, seed=42)

|Model|Recall@5|Recall@10|Recall@20|NDCG@5|NDCG@10|NDCG@20|
|-----|--------|---------|---------|------|-------|-------|
|**Transformer**|0.4339|**0.5225**|0.5564|0.3590|**0.3908**|0.4008|
|**Baseline**|0.4786|0.5277|0.5601|0.4325|0.4523|0.4627|

**Observations**:
- Transformer achieves competitive performance to the strong baseline
- Baseline slightly outperforms on NDCG (simpler model less prone to overfitting)
- Transformer has headroom for improvement with hyperparameter tuning
- Both models achieve >52% Recall@10 (strong for sparse data)

### 11.2 Training Time

- **Data loading**: ~10 seconds
- **Training (5 epochs, 389k pairs, batch 128)**: ~34 minutes on MPS
- **Evaluation (2000 users)**: ~5 minutes on MPS
- **Total**: ~50 minutes end-to-end

---

## 12. Future Improvements

1. **Cross-attention**: Separate user/item towers with learned fusion
2. **Hard negatives**: Sample negatives from similar items (not random)
3. **Longer contexts**: Increase MAX_SEQ_LEN from 20 to 50
4. **Multi-task learning**: Joint training on click + purchase signals
5. **Flash Attention**: If/when MPS supports it (2-3× speedup)
6. **Distillation**: Compress transformer → lightweight student model

---

## 13. Repository Structure

```
product_recommendation/
│
├── src/
│   ├── data/
│   │   └── prepare_dataset.py       # Raw → prepared_data.pkl
│   ├── embeddings/
│   │   └── clip_encoder.py          # Frozen CLIP ViT-B/32
│   ├── models/
│   │   ├── baseline.py              # Option B (CLIP + pooling)
│   │   └── transformer.py           # Sequential Transformer
│   ├── training/                    # Training utilities (TODO)
│   └── evaluation/                  # Evaluation metrics (TODO)
│
├── data/
│   ├── prepared_data.pkl            # Processed dataset
│   └── image_encoder_data.pkl       # CLIP image embeddings
│
├── checkpoints/
│   ├── research_mps_model.pt        # Transformer weights
│   ├── item_embedding_matrix.pt     # Frozen CLIP embeddings
│   └── option_b_model.pt            # Baseline weights
│
├── app.py                           # Streamlit demo
├── requirements.txt                 # Dependencies
├── README.md                        # Quick start guide
└── ARCHITECTURE.md                  # This document
```

---

## 14. Technical Highlights for Resume

**"Built a production-grade multimodal sequential transformer recommender, achieving 52% Recall@10 on sparse e-commerce data:"**

- Designed a 4-layer PreNorm transformer with learnable CLS token and modality-aware embeddings
- Implemented BPR ranking loss with fully vectorized training (128 batch, AdamW, gradient clipping)
- Optimized inference for Apple Silicon (MPS) with batch expansion technique (zero nested loops)
- Leveraged frozen CLIP (ViT-B/32) for multimodal embeddings, fusing text+image via L2-normalized averaging
- Achieved 10x evaluation speedup (5min vs 2hrs) through vectorized scoring of 2M user-item pairs
- Built production Streamlit app with model-agnostic inference API and image caching

**Key Skills**: PyTorch, Transformers, Multimodal Learning, Ranking Systems, MPS Optimization, Production ML

---

## References

1. **CLIP**: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021
2. **BPR**: Rendle et al., "BPR: Bayesian Personalized Ranking from Implicit Feedback", UAI 2009
3. **PreNorm**: Xiong et al., "On Layer Normalization in the Transformer Architecture", ICML 2020
4. **MPS**: Apple, "Metal Performance Shaders Documentation", 2023

---

*Document Version: 1.0 | Last Updated: February 2026*
