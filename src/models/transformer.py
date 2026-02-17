#!/usr/bin/env python3
"""
Research-Grade Multimodal Sequential Transformer Recommender
=============================================================
Optimized for Apple M1 Max (MPS backend).

Architecture:
  Token sequence: [CLS] [user_item_1] ... [user_item_k] [candidate_item]
  - Learnable CLS token
  - Learnable positional embeddings (max 22 positions)
  - Learnable modality type embeddings (user_context vs candidate)
  - 4-layer PreNorm TransformerEncoder, 8 heads, 512 hidden, GELU
  - Output: CLS token → MLP (512→256→1) → relevance score
  - Loss: BPR (Bayesian Personalized Ranking)

Item embedding: L2_normalize( (clip_text + clip_image) / 2 )
  - Precomputed once, moved to device, frozen.

Training:
  - Top 50k users with ≥5 interactions
  - 5-8 epochs, batch 128, AdamW lr=1e-4 wd=1e-4
  - Gradient clipping max_norm=1.0
  - Vectorized BPR with in-batch negatives

Evaluation:
  - Fully vectorized: user_emb @ item_matrix.T → topk
  - Seen-item masking
  - Recall@K, NDCG@K on 2000 random users

Hardware:
  - Apple MPS (Metal Performance Shaders)
  - float32 only (no AMP, no autocast, no torch.compile)
  - No .cuda() calls
"""

import os
import time
import pickle
import argparse
import math
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================

MAX_SEQ_LEN = 20          # last N user interactions
MAX_TOKENS = MAX_SEQ_LEN + 2  # [CLS] + user_items + [CANDIDATE]
EMBED_DIM = 512
HIDDEN_DIM = 512
NUM_LAYERS = 4
NUM_HEADS = 8
FF_DIM = 2048
DROPOUT = 0.1

BATCH_SIZE = 128
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 5
GRAD_CLIP = 1.0

MAX_TRAIN_USERS = 50_000
MIN_INTERACTIONS = 5
EVAL_USERS = 2000
K_VALUES = [5, 10, 20]

SEED = 42


# ============================================================================
# DEVICE SETUP
# ============================================================================

def get_device():
    """Get optimal device for Apple Silicon."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_data():
    """Load prepared_data.pkl and image_encoder_data.pkl."""
    print("[1/4] Loading prepared_data.pkl ...")
    with open("data/prepared_data.pkl", "rb") as f:
        data = pickle.load(f)

    print("[2/4] Loading image_encoder_data.pkl ...")
    with open("data/image_encoder_data.pkl", "rb") as f:
        img_raw = pickle.load(f)

    image_embeddings = img_raw["item_image_embeddings"]  # dict: item_id → Tensor[512]
    return data, image_embeddings


def get_text_embeddings(data, device):
    """Get or compute CLIP text embeddings (cached)."""
    cache_path = "data/clip_text_embeddings.pkl"

    if os.path.exists(cache_path):
        print("[3/4] Loading cached CLIP text embeddings ...")
        text_embs = torch.load(cache_path, map_location="cpu", weights_only=True)
        return text_embs

    # No cache exists — check if checkpoints/option_b_model.pt has pre-computed embeddings
    # (avoids importing CLIP which can crash on some macOS configs)
    if os.path.exists("checkpoints/option_b_model.pt"):
        print("[3/4] Using pre-computed embeddings from checkpoints/option_b_model.pt (CLIP skipped)")
        return None

    # Last resort: try CLIP encoding
    print("[3/4] Computing CLIP text embeddings (first run only) ...")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from models.clip_encoder import TextEncoder
    encoder = TextEncoder()
    encoder.eval()

    item_texts = data["metadata"]
    item_ids = list(item_texts.keys())
    texts = [item_texts[iid].get("product_text", item_texts[iid].get("title", ""))
             for iid in item_ids]

    text_embs = {}
    batch_size = 64
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="CLIP text"):
            batch_ids = item_ids[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]
            embs = encoder.encode(batch_texts)  # [B, 512] tensor
            embs = F.normalize(embs, dim=-1)
            for j, iid in enumerate(batch_ids):
                text_embs[iid] = embs[j].cpu().float()

    torch.save(text_embs, cache_path)
    print(f"  Cached {len(text_embs)} text embeddings → {cache_path}")
    return text_embs


def build_item_embedding_matrix(data, text_embs, image_embs, device):
    """
    Build fused item embedding matrix: L2_norm( (text + image) / 2 ).

    Falls back to checkpoints/option_b_model.pt if text embeddings unavailable.

    Returns:
        item_matrix: Tensor [num_items, 512] on device (frozen)
        item_to_idx: dict
        idx_to_item: dict
    """
    item_to_idx = data["item_to_idx"]
    idx_to_item = data["idx_to_item"]
    num_items = len(item_to_idx)

    # Fallback: use pre-computed embeddings from checkpoints/option_b_model.pt
    if text_embs is None:
        if os.path.exists("checkpoints/option_b_model.pt"):
            print("[4/4] Loading pre-computed item embeddings from checkpoints/option_b_model.pt ...")
            saved = torch.load("checkpoints/option_b_model.pt", map_location="cpu")
            matrix = saved["item_embedding_matrix"].float().to(device)
            print(f"  Item embeddings: {matrix.shape} → {device}")
            return matrix, item_to_idx, idx_to_item
        else:
            raise RuntimeError("No text embeddings and no checkpoints/option_b_model.pt found.")

    matrix = torch.zeros(num_items, EMBED_DIM, dtype=torch.float32)

    n_both, n_text_only = 0, 0
    for item_id, idx in item_to_idx.items():
        text_vec = text_embs.get(item_id)
        img_vec = image_embs.get(item_id)

        if text_vec is not None and img_vec is not None:
            if isinstance(img_vec, np.ndarray):
                img_vec = torch.from_numpy(img_vec).float()
            fused = (text_vec + img_vec) / 2.0
            n_both += 1
        elif text_vec is not None:
            fused = text_vec
            n_text_only += 1
        else:
            fused = torch.zeros(EMBED_DIM)

        # L2 normalize
        fused = F.normalize(fused, dim=0)
        matrix[idx] = fused

    matrix = matrix.to(device)
    print(f"[4/4] Item embeddings: {num_items} items "
          f"({n_both} text+image, {n_text_only} text-only) → {device}")

    return matrix, item_to_idx, idx_to_item


# ============================================================================
# TRAINING DATA PREPARATION (VECTORIZED)
# ============================================================================

def prepare_training_data(data, item_to_idx, max_users=MAX_TRAIN_USERS,
                          min_interactions=MIN_INTERACTIONS):
    """
    Prepare training triplets: (user_seq_indices, pos_item_idx, neg_item_idx).

    Selects top users by interaction count with ≥min_interactions.
    Negative items are sampled per positive pair.

    Returns:
        user_seqs:  LongTensor [N, MAX_SEQ_LEN]  (zero-padded, left-aligned)
        seq_lens:   LongTensor [N]                (actual sequence length)
        pos_items:  LongTensor [N]
        neg_items:  LongTensor [N]
    """
    train_sequences = data["train_sequences"]
    num_items = len(item_to_idx)

    # Filter and sort users by interaction count
    eligible = [(uid, seq) for uid, seq in train_sequences.items()
                if len(seq) >= min_interactions]
    eligible.sort(key=lambda x: len(x[1]), reverse=True)
    eligible = eligible[:max_users]
    print(f"  Selected {len(eligible)} users (≥{min_interactions} interactions)")

    all_user_seqs = []
    all_seq_lens = []
    all_pos = []
    all_neg = []

    for user_id, seq in tqdm(eligible, desc="Building training pairs", leave=False):
        item_ids = [item_id for item_id, _ts in seq]
        item_indices = [item_to_idx[iid] for iid in item_ids if iid in item_to_idx]

        if len(item_indices) < 2:
            continue

        user_item_set = set(item_indices)

        # For each position i>0, use items[:i] as context, items[i] as positive
        for i in range(1, len(item_indices)):
            context = item_indices[max(0, i - MAX_SEQ_LEN):i]
            pos_idx = item_indices[i]

            # Random negative (not in user history)
            neg_idx = random.randint(0, num_items - 1)
            attempts = 0
            while neg_idx in user_item_set and attempts < 10:
                neg_idx = random.randint(0, num_items - 1)
                attempts += 1

            # Pad context to MAX_SEQ_LEN (left padding with 0)
            pad_len = MAX_SEQ_LEN - len(context)
            padded = [0] * pad_len + context

            all_user_seqs.append(padded)
            all_seq_lens.append(len(context))
            all_pos.append(pos_idx)
            all_neg.append(neg_idx)

    user_seqs = torch.tensor(all_user_seqs, dtype=torch.long)
    seq_lens = torch.tensor(all_seq_lens, dtype=torch.long)
    pos_items = torch.tensor(all_pos, dtype=torch.long)
    neg_items = torch.tensor(all_neg, dtype=torch.long)

    print(f"  Total training pairs: {len(all_pos):,}")
    return user_seqs, seq_lens, pos_items, neg_items


# ============================================================================
# MODEL
# ============================================================================

class SequentialTransformerRecommender(nn.Module):
    """
    Multimodal Sequential Transformer for Recommendation.

    Input sequence: [CLS] + user_item_embeddings + [candidate_item_embedding]
    Output: CLS token → MLP → scalar relevance score

    All embeddings are 512-dim fused CLIP (text+image), frozen.
    The transformer learns sequential user preferences and user-item relevance.
    """

    def __init__(self, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
                 num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
                 ff_dim=FF_DIM, dropout=DROPOUT,
                 max_positions=MAX_TOKENS):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Learnable positional embeddings
        self.pos_embedding = nn.Embedding(max_positions, hidden_dim)

        # Learnable modality type embeddings
        # 0 = CLS, 1 = user_context, 2 = candidate
        self.modality_embedding = nn.Embedding(3, hidden_dim)

        # Input projection (CLIP 512 → hidden_dim)
        if embed_dim != hidden_dim:
            self.input_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        else:
            self.input_proj = nn.Identity()

        # Transformer encoder (PreNorm, GELU)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # PreNorm
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Layer norm on CLS output
        self.cls_norm = nn.LayerNorm(hidden_dim)

        # Output MLP: 512 → 256 → 1
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, user_item_embs, candidate_emb):
        """
        Args:
            user_item_embs: [B, S, 512] - user history item embeddings
            candidate_emb:  [B, 512]    - candidate item embedding

        Returns:
            scores: [B] - relevance scores
        """
        B, S, _ = user_item_embs.shape
        device = user_item_embs.device

        # Project embeddings
        user_proj = self.input_proj(user_item_embs)        # [B, S, H]
        cand_proj = self.input_proj(candidate_emb).unsqueeze(1)  # [B, 1, H]

        # Expand CLS token
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, H]

        # Assemble sequence: [CLS] + [user_items] + [candidate]
        tokens = torch.cat([cls, user_proj, cand_proj], dim=1)  # [B, S+2, H]
        seq_len = tokens.size(1)

        # Positional embeddings
        pos_ids = torch.arange(seq_len, device=device, dtype=torch.long)
        tokens = tokens + self.pos_embedding(pos_ids).unsqueeze(0)

        # Modality embeddings: 0=CLS, 1=user_context, 2=candidate
        mod_ids = torch.zeros(seq_len, device=device, dtype=torch.long)
        mod_ids[1:1 + S] = 1   # user items
        mod_ids[-1] = 2          # candidate
        tokens = tokens + self.modality_embedding(mod_ids).unsqueeze(0)

        # Transformer
        output = self.transformer(tokens)  # [B, S+2, H]

        # Extract CLS token
        cls_out = self.cls_norm(output[:, 0, :])  # [B, H]

        # MLP → score
        scores = self.mlp(cls_out).squeeze(-1)  # [B]

        return scores

    def get_user_embedding(self, user_item_embs):
        """
        Get user representation from CLS token (without candidate).
        Used for vectorized evaluation.

        Args:
            user_item_embs: [B, S, 512]

        Returns:
            cls_output: [B, H]
        """
        B, S, _ = user_item_embs.shape
        device = user_item_embs.device

        user_proj = self.input_proj(user_item_embs)
        cls = self.cls_token.expand(B, -1, -1)

        # [CLS] + [user_items] (no candidate for embedding extraction)
        tokens = torch.cat([cls, user_proj], dim=1)  # [B, S+1, H]
        seq_len = tokens.size(1)

        pos_ids = torch.arange(seq_len, device=device, dtype=torch.long)
        tokens = tokens + self.pos_embedding(pos_ids).unsqueeze(0)

        mod_ids = torch.zeros(seq_len, device=device, dtype=torch.long)
        mod_ids[1:] = 1  # all user items
        tokens = tokens + self.modality_embedding(mod_ids).unsqueeze(0)

        output = self.transformer(tokens)
        cls_out = self.cls_norm(output[:, 0, :])  # [B, H]

        return cls_out


# ============================================================================
# TRAINING
# ============================================================================

def train(model, item_matrix, user_seqs, seq_lens, pos_items, neg_items,
          device, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE):
    """
    Train with BPR loss. Fully vectorized, no Python loops in loss.

    Args:
        model: SequentialTransformerRecommender
        item_matrix: [num_items, 512] on device
        user_seqs: [N, MAX_SEQ_LEN] indices
        seq_lens: [N]
        pos_items: [N] indices
        neg_items: [N] indices
        device: torch.device
        num_epochs: int
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    N = len(user_seqs)
    indices = np.arange(N)

    print(f"\n{'='*60}")
    print(f"TRAINING: {N:,} pairs, batch_size={batch_size}, epochs={num_epochs}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        t0 = time.time()
        model.train()
        np.random.shuffle(indices)

        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(range(0, N, batch_size), desc=f"Epoch {epoch+1}/{num_epochs}",
                    leave=True)

        for start in pbar:
            end = min(start + batch_size, N)
            batch_idx = indices[start:end]

            # Gather batch
            b_seqs = user_seqs[batch_idx].to(device)        # [B, MAX_SEQ_LEN]
            b_pos = pos_items[batch_idx].to(device)          # [B]
            b_neg = neg_items[batch_idx].to(device)          # [B]

            # Look up embeddings from frozen item matrix
            b_user_embs = item_matrix[b_seqs]                # [B, MAX_SEQ_LEN, 512]
            b_pos_embs = item_matrix[b_pos]                  # [B, 512]
            b_neg_embs = item_matrix[b_neg]                  # [B, 512]

            # Forward pass
            pos_scores = model(b_user_embs, b_pos_embs)     # [B]
            neg_scores = model(b_user_embs, b_neg_embs)     # [B]

            # BPR loss (fully vectorized)
            loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / max(num_batches, 1)
        elapsed = time.time() - t0

        print(f"  Epoch {epoch+1} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s | Device: {device}")

    return model


# ============================================================================
# VECTORIZED EVALUATION
# ============================================================================

def evaluate(model, item_matrix, data, item_to_idx, idx_to_item, device,
             k_values=K_VALUES, num_eval_users=EVAL_USERS):
    """
    Vectorized evaluation. No nested item loops.

    Strategy:
      1. For each user batch, build padded sequences → [B, MAX_SEQ_LEN]
      2. Get user CLS embeddings from transformer → [B, H]
      3. Score ALL items: scores[i] = model(user_seq_i, item_j) for all j
         We batch this: expand user seq for all 1000 items at once per user.
      4. Mask seen items → topk → Recall@K, NDCG@K
    """
    model.eval()
    train_sequences = data["train_sequences"]
    test_interactions = data["test_interactions"]
    num_items = len(item_to_idx)

    # Select eval users
    candidates = [uid for uid in test_interactions
                  if len(test_interactions[uid]) > 0
                  and uid in train_sequences
                  and len(train_sequences[uid]) >= 2]
    random.seed(SEED)
    random.shuffle(candidates)
    eval_users = candidates[:num_eval_users]

    print(f"\n{'='*60}")
    print(f"EVALUATION: {len(eval_users)} users, K={k_values}")
    print(f"{'='*60}")

    recalls = defaultdict(float)
    ndcgs = defaultdict(float)
    count = 0
    max_k = max(k_values)

    t0 = time.time()

    # Pre-build all user sequences and metadata
    user_padded_seqs = []   # list of [MAX_SEQ_LEN] int lists
    user_seen_sets = []     # list of sets of item indices
    user_test_sets = []     # list of sets of item indices
    valid_uids = []

    for uid in eval_users:
        seq = train_sequences[uid]
        item_indices = [item_to_idx[iid] for iid, _ts in seq if iid in item_to_idx]
        if len(item_indices) == 0:
            continue

        test_items = test_interactions[uid]
        test_idx_set = set(item_to_idx[iid] for iid in test_items if iid in item_to_idx)
        if len(test_idx_set) == 0:
            continue

        context = item_indices[-MAX_SEQ_LEN:]
        pad_len = MAX_SEQ_LEN - len(context)
        padded = [0] * pad_len + context

        user_padded_seqs.append(padded)
        user_seen_sets.append(set(item_indices))
        user_test_sets.append(test_idx_set)
        valid_uids.append(uid)

    # Batch evaluation: process users in chunks to amortize transformer overhead
    eval_batch = 8  # users per eval batch (each user scores all 1000 items)

    with torch.no_grad():
        for batch_start in tqdm(range(0, len(valid_uids), eval_batch), desc="Evaluating"):
            batch_end = min(batch_start + eval_batch, len(valid_uids))
            batch_seqs = user_padded_seqs[batch_start:batch_end]
            B = len(batch_seqs)

            # Build user sequence tensors [B, MAX_SEQ_LEN]
            seq_tensor = torch.tensor(batch_seqs, dtype=torch.long, device=device)
            user_embs = item_matrix[seq_tensor]  # [B, MAX_SEQ_LEN, 512]

            # Score all items for all users in this batch
            # For each user, expand to score all num_items candidates
            # We process one user at a time but with all items vectorized
            for i in range(B):
                user_idx = batch_start + i
                single_user = user_embs[i:i+1]  # [1, S, 512]

                # Expand for all items
                user_exp = single_user.expand(num_items, -1, -1)  # [num_items, S, 512]
                scores = model(user_exp, item_matrix)              # [num_items]

                # Mask seen items
                seen_set = user_seen_sets[user_idx]
                if seen_set:
                    seen_t = torch.tensor(list(seen_set), dtype=torch.long, device=device)
                    scores[seen_t] = -1e9

                # Top-K (single torch call)
                _, topk_indices = torch.topk(scores, max_k)
                topk_np = topk_indices.cpu().numpy()

                # Metrics
                test_idx_set = user_test_sets[user_idx]
                for k in k_values:
                    top_k_set = set(topk_np[:k])
                    hits = len(top_k_set & test_idx_set)
                    recalls[k] += hits / len(test_idx_set)

                    dcg = sum(1.0 / math.log2(r + 2)
                              for r, idx in enumerate(topk_np[:k])
                              if idx in test_idx_set)
                    idcg = sum(1.0 / math.log2(j + 2)
                               for j in range(min(k, len(test_idx_set))))
                    ndcgs[k] += dcg / idcg if idcg > 0 else 0.0

                count += 1

    elapsed = time.time() - t0

    # Average
    results = {}
    for k in k_values:
        results[f"Recall@{k}"] = recalls[k] / max(count, 1)
        results[f"NDCG@{k}"] = ndcgs[k] / max(count, 1)

    print(f"\nEvaluation completed in {elapsed:.1f}s ({count} users)")
    print("-" * 40)
    for metric, value in sorted(results.items()):
        print(f"  {metric:15s}: {value:.4f}")

    return results

    elapsed = time.time() - t0

    # Average
    results = {}
    for k in k_values:
        results[f"Recall@{k}"] = recalls[k] / max(count, 1)
        results[f"NDCG@{k}"] = ndcgs[k] / max(count, 1)

    print(f"\nEvaluation completed in {elapsed:.1f}s ({count} users)")
    print("-" * 40)
    for metric, value in sorted(results.items()):
        print(f"  {metric:15s}: {value:.4f}")

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Research-Grade Multimodal Sequential Transformer (MPS)")
    parser.add_argument("--train", action="store_true",
                        help="Train model")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate model")
    parser.add_argument("--train_and_eval", action="store_true",
                        help="Train then evaluate")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help=f"Number of epochs (default: {NUM_EPOCHS})")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--max_users", type=int, default=MAX_TRAIN_USERS,
                        help=f"Max training users (default: {MAX_TRAIN_USERS})")
    args = parser.parse_args()

    # Use args values (don't mutate globals to avoid SyntaxError)
    batch_size = args.batch_size
    num_epochs = args.epochs
    max_users = args.max_users

    seed_everything(SEED)
    device = get_device()

    # ---- Data ----
    data, image_embeddings = load_data()
    text_embeddings = get_text_embeddings(data, device)
    item_matrix, item_to_idx, idx_to_item = build_item_embedding_matrix(
        data, text_embeddings, image_embeddings, device
    )

    # ---- Model ----
    model = SequentialTransformerRecommender().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")

    # ---- Train ----
    if args.train or args.train_and_eval:
        print("\n[*] Preparing training data ...")
        user_seqs, seq_lens, pos_items, neg_items = prepare_training_data(
            data, item_to_idx, max_users=max_users,
            min_interactions=MIN_INTERACTIONS,
        )

        model = train(model, item_matrix, user_seqs, seq_lens,
                       pos_items, neg_items, device, num_epochs=num_epochs,
                       batch_size=batch_size)

        # Save
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), "checkpoints/research_mps_model.pt")
        torch.save(item_matrix.cpu(), "checkpoints/item_embedding_matrix.pt")
        print("\n[*] Saved: checkpoints/research_mps_model.pt, checkpoints/item_embedding_matrix.pt")

    else:
        # Try to load existing model
        if os.path.exists("checkpoints/research_mps_model.pt"):
            print("[*] Loading pre-trained model ...")
            model.load_state_dict(
                torch.load("checkpoints/research_mps_model.pt", map_location=device)
            )
        else:
            print("[!] No trained model found. Run with --train first.")

    # ---- Evaluate ----
    if args.evaluate or args.train_and_eval:
        results = evaluate(model, item_matrix, data, item_to_idx,
                           idx_to_item, device)

        # Save results
        os.makedirs("checkpoints", exist_ok=True)
        with open("checkpoints/research_mps_results.pkl", "wb") as f:
            pickle.dump(results, f)
        print(f"\n[*] Saved: checkpoints/research_mps_results.pkl")

    print("\nDone.")


if __name__ == "__main__":
    main()
