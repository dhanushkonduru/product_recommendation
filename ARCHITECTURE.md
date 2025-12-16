
# System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT LAYER                                   │
├─────────────────────────────────────────────────────────────────┤
│  User ID  │  Product Text (Title+Brand)  │  Product Image       │
└─────┬─────┴──────────────┬────────────────┴──────────┬──────────┘
      │                    │                            │
      │                    │                            │
┌─────▼────────────────────▼────────────────────────────▼──────────┐
│                    ENCODING LAYER                                │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  User Embedding      Text Encoder         Image Encoder          │
│  (Learned)           (DistilBERT)         (ResNet18)             │
│  [128-dim]           [768-dim]            [512-dim]              │
│      │                    │                    │                  │
│      │                    │                    │                  │
│      └────────────────────┼────────────────────┘                  │
│                           │                                      │
│                    Projection Layers                             │
│                    [128-dim each]                                │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│                    FUSION LAYER                                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Concatenate: [User(128) + Text(128) + Image(128)] = 384-dim    │
│                           │                                      │
│                    Fusion MLP                                    │
│      384 → 256 → 128 → 64 → 1 (Score)                           │
│                                                                   │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│                    OUTPUT LAYER                                  │
├──────────────────────────────────────────────────────────────────┤
│                    Ranking Score                                 │
│              (Higher = More Relevant)                            │
└──────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. User Embedding
- **Type:** Learned embedding layer
- **Dimension:** 128
- **Initialization:** Normal distribution (std=0.01)
- **Trainable:** Yes

### 2. Text Encoder
- **Model:** DistilBERT-base-uncased
- **Input:** Product title + brand (concatenated)
- **Output Dimension:** 768
- **Frozen:** Yes (pretrained weights)
- **Projection:** Linear layer (768 → 128)

### 3. Image Encoder
- **Model:** ResNet18 (pretrained on ImageNet)
- **Input:** Product image (224x224 RGB)
- **Output Dimension:** 512
- **Frozen:** Yes (pretrained weights)
- **Projection:** Linear layer (512 → 128)

### 4. Fusion Network
- **Strategy:** Concatenation + MLP
- **Input:** 384-dim (128+128+128)
- **Architecture:**
  - Linear(384 → 256) + ReLU + Dropout(0.2)
  - Linear(256 → 128) + ReLU + Dropout(0.2)
  - Linear(128 → 64) + ReLU + Dropout(0.2)
  - Linear(64 → 1)
- **Output:** Ranking score (logit)

### 5. Training
- **Loss:** Binary Cross-Entropy with Logits
- **Optimizer:** Adam (lr=0.001)
- **Negative Sampling:** 1 positive : 4 negatives
- **Epochs:** 10
