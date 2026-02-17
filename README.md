# Multimodal Sequential Transformer Recommender

A research-grade multimodal product recommendation system using frozen CLIP embeddings and learnable transformer-based sequential modeling, optimized for Apple Silicon (M1/M2 with MPS backend).

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://python.org)
[![MPS](https://img.shields.io/badge/Accelerator-Apple_MPS-000000?logo=apple)](https://developer.apple.com/metal/)
[![Streamlit](https://img.shields.io/badge/Demo-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io)

---

## 🎯 Problem Statement

**Goal**: Recommend products to users based on their interaction history, leveraging both textual and visual signals.

**Challenges**:
- Sparse user interactions (median user has 2 interactions)
- Multimodal item representations (text + images)
- Sequential dependencies (recent items matter more)
- Real-time inference requirements

---

## 🏗️ Architecture

### Two Model Implementations

#### 1. **Research Transformer** (Primary Focus)
- 4-layer PreNorm Transformer with learnable CLS token
- Modality-aware positional embeddings (CLS, context, candidate)
- BPR ranking loss optimization
- **12.76M parameters**
- **Recall@10: 0.5225 | NDCG@10: 0.3908**

#### 2. **Baseline (Option B)**
- Time-decayed weighted pooling of last 5 items
- Optional BPR-tuned projection layer (512→512)
- Dot product scoring (cosine similarity)
- **Recall@10: 0.5277 | NDCG@10: 0.4523**

### Key Design Choices

```
Item Representation:
  - Frozen CLIP ViT-B/32 embeddings (text + image)
  - L2-normalized 512-dimensional vectors
  - 798/1000 items have images

User Representation:
  - Transformer: Learnable CLS token aggregating last 20 interactions
  - Baseline: Exponentially time-decayed mean of last 5 items

Training:
  - Loss: BPR (Bayesian Personalized Ranking)
  - Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
  - Batch size: 128, Epochs: 5
  - Device: Apple MPS (Metal Performance Shaders)

Evaluation:
  - Fully vectorized (zero nested loops)
  - Batch expansion technique for efficient scoring
  - 2000 users, 1000 items → 5 minutes on MPS
```

For detailed architecture documentation, see [**ARCHITECTURE.md**](ARCHITECTURE.md).

---

## 📊 Results

### Test Set Performance (2000 random users, seed=42)

|Model|Recall@5|Recall@10|Recall@20|NDCG@5|NDCG@10|NDCG@20|Training Time|
|-----|--------|---------|---------|------|-------|-------|-------------|
|**Transformer**|0.4339|0.5225|0.5564|0.3590|0.3908|0.4008|~34 min|
|**Baseline**|0.4786|0.5277|0.5601|0.4325|0.4523|0.4627|~25 sec|

**Key Insights**:
- Both models achieve >52% Recall@10 on highly sparse data
- Baseline is competitive due to strong CLIP embeddings + time decay
- Transformer has headroom for improvement (hyperparameter tuning, longer context)
- 10× faster evaluation vs naive nested loops (5min vs 2hr)

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/product_recommendation.git
cd product_recommendation

# Install dependencies
pip install -r requirements.txt
```

**Requirements**: Python 3.10+, PyTorch 2.0+, Apple M1/M2 Mac (for MPS acceleration)

### 2. Prepare Data

```bash
# Download Amazon Fashion dataset
# Place AMAZON_FASHION_5.json.gz and meta_AMAZON_FASHION.json.gz in data/

# Process raw data
python src/data/prepare_dataset.py
```

**Output**: `data/prepared_data.pkl`, `data/image_encoder_data.pkl`

### 3. Train Models

#### Option A: Train Baseline (Fast)

```bash
python src/models/baseline.py --train
```

**Time**: ~25 seconds on CPU  
**Output**: `option_b_model.pt`, `option_b_results.pkl`

#### Option B: Train Transformer (Research)

```bash
python src/models/transformer.py --train_and_eval --epochs 5 --batch_size 128 --max_users 50000
```

**Time**: ~50 minutes on MPS (M1 Max)  
**Output**: `research_mps_model.pt`, `item_embedding_matrix.pt`, `research_mps_results.pkl`

### 4. Launch Demo App

```bash
streamlit run app.py
```

Open browser at `http://localhost:8501` to interact with the recommendation system.

**Features**:
- Model selector (Baseline / Transformer)
- 2000 pre-sampled users with ≥5 interactions
- Random user button
- User history visualization (last 5 items with images)
- Top-K recommendations (3-21, adjustable)
- Evaluation metrics display
- Cold-start fallback (popularity-based)

---

## 📁 Project Structure

```
product_recommendation/
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── prepare_dataset.py       # Raw data → prepared_data.pkl
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── clip_encoder.py          # Frozen CLIP ViT-B/32 wrapper
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py              # Option B (CLIP + pooling + BPR)
│   │   └── transformer.py           # Sequential Transformer
│   ├── training/                    # Training utilities (modular)
│   │   └── __init__.py
│   └── evaluation/                  # Evaluation metrics (modular)
│       └── __init__.py
│
├── data/
│   ├── AMAZON_FASHION_5.json.gz     # Raw interactions
│   ├── meta_AMAZON_FASHION.json.gz  # Item metadata
│   ├── prepared_data.pkl            # Processed dataset
│   └── image_encoder_data.pkl       # CLIP image embeddings
│
├── checkpoints/
│   ├── research_mps_model.pt        # Transformer weights
│   ├── item_embedding_matrix.pt     # Frozen CLIP embeddings [1000, 512]
│   ├── option_b_model.pt            # Baseline weights
│   ├── option_b_results.pkl         # Baseline eval metrics
│   └── research_mps_results.pkl     # Transformer eval metrics
│
├── models/                          # Legacy folder (kept for compatibility)
│   ├── clip_encoder.py
│   └── option_b_model.py
│
├── app.py                           # Streamlit demo
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── ARCHITECTURE.md                  # Detailed technical documentation
└── .gitignore
```

---

## 🔬 Technical Highlights

### 1. **Fully Vectorized Evaluation**

No nested `user × item` loops. Batch expansion technique for efficient scoring:

```python
# ❌ Naive (slow): 2000 users × 1000 items = 2M forward passes
for user in users:
    for item in items:
        score = model(user, item)

# ✅ Optimized (fast): 2000 batched forwards
user_seq = pad_sequence(user)           # [1, 20, 512]
user_exp = user_seq.expand(1000, -1, -1) # [1000, 20, 512]
scores = model(user_exp, items)          # [1000] in ONE forward!
```

**Result**: 5 minutes vs 2 hours.

### 2. **Apple MPS Optimization**

```python
# Device detection
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Float32 only (no AMP on MPS)
model = model.to(device).float()

# Efficient memory
with torch.no_grad():
    scores = model(batch)
```

### 3. **Frozen CLIP Embeddings**

- **Text**: CLIP ViT-B/32 tokenizer + text encoder (frozen)
- **Image**: CLIP ViT-B/32 vision encoder (frozen) → 224×224 RGB
- **Fusion**: `L2_norm((text + image) / 2)`
- **Benefit**: Zero-shot multimodal transfer, no item-side gradients

### 4. **PreNorm Transformer**

```python
# LayerNorm BEFORE attention/FFN (not after)
norm_first=True   # More stable training
```

### 5. **BPR Ranking Loss**

```python
# Optimizes: positive items rank higher than negatives
loss = -log(sigmoid(score_pos - score_neg)).mean()
```

---

## 📈 Dataset

- **Source**: Amazon Fashion 5-core (McAuley et al., 2015)
- **Users**: 223,975
- **Items**: 1,000 (filtered for diversity)
- **Interactions**: Sparse sequences (median: 2 per user)
- **Splits**: 
  - Train: 70% (temporal split)
  - Validation: 10%
  - Test: 20%

---

## 🛠️ Technologies

- **PyTorch 2.0+**: Model implementation, MPS backend
- **Transformers (HuggingFace)**: CLIP model loading
- **Streamlit**: Interactive demo UI
- **NumPy, Pandas**: Data processing
- **Pillow**: Image loading
- **tqdm**: Progress bars

---

## 📝 Key Files

|File|Purpose|Lines|
|----|-------|-----|
|`src/models/transformer.py`|Sequential transformer model + training + eval|~760|
|`src/models/baseline.py`|Baseline model (pooling + BPR)|~680|
|`src/embeddings/clip_encoder.py`|Frozen CLIP wrapper|~195|
|`src/data/prepare_dataset.py`|Data preprocessing pipeline|~450|
|`app.py`|Streamlit production demo|~800|
|`ARCHITECTURE.md`|Technical deep-dive|~650|

---

## 🎓 Research Contributions

1. **MPS-first transformer recommender**: Optimized for Apple Silicon (not CUDA)
2. **Batch expansion evaluation**: Zero nested loops via tensor broadcasting
3. **Modality-aware embeddings**: Learnable type embeddings for context/candidate distinction
4. **Frozen multimodal tower**: CLIP embeddings with no fine-tuning (efficient)
5. **Production Streamlit app**: Model-agnostic inference API, user history viz

---

## 📚 References

1. Radford et al., **"Learning Transferable Visual Models From Natural Language Supervision"**, ICML 2021 (CLIP)
2. Rendle et al., **"BPR: Bayesian Personalized Ranking from Implicit Feedback"**, UAI 2009
3. Xiong et al., **"On Layer Normalization in the Transformer Architecture"**, ICML 2020 (PreNorm)
4. McAuley et al., **"Image-based Recommendations on Styles and Substitutes"**, SIGIR 2015 (Dataset)

---

## 🚧 Future Work

- [ ] Cross-attention between user/item towers
- [ ] Hard negative sampling (similar items)
- [ ] Longer context (MAX_SEQ_LEN: 20 → 50)
- [ ] Multi-task learning (click + purchase signals)
- [ ] Flash Attention (if MPS supports)
- [ ] Model distillation (transformer → lightweight)

---

## 📄 License

MIT License - See `LICENSE` file for details.

---

## 👤 Author

**Dhanu** — ML Research Engineer  
📧 Contact: [your.email@domain.com]  
🔗 LinkedIn: [linkedin.com/in/yourprofile]  
🐙 GitHub: [github.com/yourusername]

---

## 🌟 Citation

If you use this code in your research, please cite:

```bibtex
@misc{multimodal_transformer_recommender_2026,
  author = {Your Name},
  title = {Multimodal Sequential Transformer Recommender},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/yourusername/product_recommendation},
  note = {Research-grade implementation with Apple MPS optimization}
}
```

---

**Built with ❤️ using PyTorch and Apple Silicon**
