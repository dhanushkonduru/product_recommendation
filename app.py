"""
Production-Grade Multimodal Product Recommendation Demo
========================================================
Streamlit app with two model backends:
  1. Option B  — Fast Baseline (CLIP + Time-Decayed Pooling + BPR)
  2. Research Transformer — Sequential Multimodal Transformer (MPS)

All inference is fully vectorized. No nested user×item loops.
"""

import math
import os
import pickle
import random
from collections import Counter
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageDraw


# ============================================================================
# CONSTANTS
# ============================================================================

MAX_SEQ_LEN = 20
EMBED_DIM = 512
HIDDEN_DIM = 512
NUM_LAYERS = 4
NUM_HEADS = 8
FF_DIM = 2048
DROPOUT = 0.1
MAX_TOKENS = MAX_SEQ_LEN + 2
NUM_SAMPLE_USERS = 2000
SEED = 42

MODEL_OPTIONS = {
    "Option B (Fast Baseline)": "option_b",
    "Research Transformer (MPS)": "transformer",
}


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================


class SequentialTransformerRecommender(nn.Module):
    """
    4-layer PreNorm Transformer for sequential recommendation.

    Input:  [CLS] + user_item_embeddings + [candidate_item]
    Output: CLS → LayerNorm → MLP (512→256→1) → scalar score
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        num_heads: int = NUM_HEADS,
        ff_dim: int = FF_DIM,
        dropout: float = DROPOUT,
        max_positions: int = MAX_TOKENS,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.pos_embedding = nn.Embedding(max_positions, hidden_dim)
        self.modality_embedding = nn.Embedding(3, hidden_dim)  # 0=CLS, 1=user, 2=candidate

        self.input_proj = (
            nn.Linear(embed_dim, hidden_dim, bias=False)
            if embed_dim != hidden_dim
            else nn.Identity()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self, user_item_embs: torch.Tensor, candidate_emb: torch.Tensor
    ) -> torch.Tensor:
        B, S, _ = user_item_embs.shape
        device = user_item_embs.device

        user_proj = self.input_proj(user_item_embs)
        cand_proj = self.input_proj(candidate_emb).unsqueeze(1)
        cls = self.cls_token.expand(B, -1, -1)

        tokens = torch.cat([cls, user_proj, cand_proj], dim=1)
        seq_len = tokens.size(1)

        pos_ids = torch.arange(seq_len, device=device, dtype=torch.long)
        tokens = tokens + self.pos_embedding(pos_ids).unsqueeze(0)

        mod_ids = torch.zeros(seq_len, device=device, dtype=torch.long)
        mod_ids[1 : 1 + S] = 1
        mod_ids[-1] = 2
        tokens = tokens + self.modality_embedding(mod_ids).unsqueeze(0)

        output = self.transformer(tokens)
        cls_out = self.cls_norm(output[:, 0, :])
        return self.mlp(cls_out).squeeze(-1)


class ProjectionLayer(nn.Module):
    """512→512 projection for Option B BPR tuning."""

    def __init__(self, dim: int = 512):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ============================================================================
# DEVICE DETECTION
# ============================================================================


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================================
# DATA LOADING (CACHED)
# ============================================================================


@st.cache_data(show_spinner="Loading dataset...")
def load_prepared_data() -> Dict[str, Any]:
    """Load prepared_data.pkl once. Precomputes popular items and sample users."""
    path = "data/prepared_data.pkl"
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Missing data/prepared_data.pkl — run 02_data_preparation.py first."
        )

    with open(path, "rb") as f:
        data = pickle.load(f)

    train_sequences = data["train_sequences"]

    # Popular items for cold-start fallback
    item_counts: Counter = Counter()
    for seq in train_sequences.values():
        for item_id, _ in seq:
            item_counts[item_id] += 1

    max_count = max(item_counts.values()) if item_counts else 1
    popular_items = [iid for iid, _ in item_counts.most_common(200)]
    popular_scores = {iid: item_counts[iid] / max_count for iid in popular_items}

    # Deterministic sample of 2000 users with >=5 interactions
    valid_users = [u for u, seq in train_sequences.items() if len(seq) >= 5]
    rng = random.Random(SEED)
    sample = rng.sample(valid_users, min(NUM_SAMPLE_USERS, len(valid_users)))
    sample.sort()

    data["popular_items"] = popular_items
    data["popular_scores"] = popular_scores
    data["sample_users"] = sample

    return data


@st.cache_data(show_spinner=False)
def load_all_metrics() -> Dict[str, Dict[str, float]]:
    """Load evaluation metrics for both models."""
    metrics: Dict[str, Dict[str, float]] = {}

    if os.path.exists("checkpoints/option_b_results.pkl"):
        with open("checkpoints/option_b_results.pkl", "rb") as f:
            ob = pickle.load(f)
        test = ob.get("results", {}).get("test", {})
        metrics["option_b"] = {
            "Recall@10": float(test.get("recall@10", 0)),
            "NDCG@10": float(test.get("ndcg@10", 0)),
        }

    if os.path.exists("checkpoints/research_mps_results.pkl"):
        with open("checkpoints/research_mps_results.pkl", "rb") as f:
            rt = pickle.load(f)
        metrics["transformer"] = {
            "Recall@10": float(rt.get("Recall@10", 0)),
            "NDCG@10": float(rt.get("NDCG@10", 0)),
        }

    return metrics


# ============================================================================
# MODEL LOADING (CACHED)
# ============================================================================


@st.cache_resource(show_spinner="Loading Option B model...")
def load_option_b_model() -> Optional[Dict[str, Any]]:
    """Load Option B: CLIP embeddings + optional BPR projection."""
    path = "checkpoints/option_b_model.pt"
    if not os.path.exists(path):
        return None

    saved = torch.load(path, map_location="cpu")
    item_embs: torch.Tensor = saved["item_embedding_matrix"]
    item_to_idx: Dict[str, int] = saved["item_to_idx"]
    proj_state = saved.get("projection_layer")
    has_proj = bool(saved.get("has_projection"))

    eval_embs = item_embs
    if has_proj and proj_state is not None:
        proj = ProjectionLayer(item_embs.shape[1])
        proj.load_state_dict(proj_state)
        proj.eval()
        with torch.no_grad():
            eval_embs = proj(eval_embs)
            eval_embs = eval_embs / (torch.norm(eval_embs, dim=1, keepdim=True) + 1e-8)

    idx_to_item = {v: k for k, v in item_to_idx.items()}

    return {
        "eval_embeddings": eval_embs,
        "item_to_idx": item_to_idx,
        "idx_to_item": idx_to_item,
    }


@st.cache_resource(show_spinner="Loading Research Transformer...")
def load_research_transformer(device_name: str) -> Optional[Tuple[nn.Module, torch.Tensor]]:
    """Load trained transformer + frozen item embedding matrix."""
    model_path = "checkpoints/research_mps_model.pt"
    matrix_path = "checkpoints/item_embedding_matrix.pt"

    if not os.path.exists(model_path) or not os.path.exists(matrix_path):
        return None

    device = torch.device(device_name)
    model = SequentialTransformerRecommender()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    item_matrix = torch.load(matrix_path, map_location=device)

    return model, item_matrix


# ============================================================================
# INFERENCE (FULLY VECTORIZED — NO NESTED LOOPS)
# ============================================================================


def recommend_option_b(
    user_id: str,
    top_k: int,
    data: Dict[str, Any],
    model_data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Option B: time-decayed weighted mean + dot product scoring."""
    train_sequences = data["train_sequences"]
    eval_embs = model_data["eval_embeddings"]
    item_to_idx = model_data["item_to_idx"]
    idx_to_item = model_data["idx_to_item"]

    user_seq = train_sequences.get(user_id, [])
    if not user_seq:
        return []

    # Time-decayed weighted mean of last 5 items
    recent = user_seq[-5:]
    indices: List[int] = []
    weights: List[float] = []
    for pos, (iid, _) in enumerate(recent):
        if iid in item_to_idx:
            indices.append(item_to_idx[iid])
            norm_pos = pos / max(len(recent) - 1, 1)
            weights.append(math.exp(2.0 * norm_pos))

    if not indices:
        return []

    w = torch.tensor(weights, dtype=torch.float32)
    w /= w.sum()
    idx_t = torch.tensor(indices, dtype=torch.long)

    user_emb = (eval_embs[idx_t] * w.unsqueeze(-1)).sum(dim=0)
    user_emb = user_emb / (torch.norm(user_emb) + 1e-8)

    # Vectorized scoring — single matmul
    with torch.no_grad():
        scores = torch.matmul(user_emb, eval_embs.T)

    # Mask seen items
    seen = {item_to_idx[iid] for iid, _ in user_seq if iid in item_to_idx}
    if seen:
        scores[torch.tensor(list(seen), dtype=torch.long)] = -float("inf")

    _, topk_idx = torch.topk(scores, min(top_k, scores.shape[0]))

    return [
        {"item_id": idx_to_item[i], "score": float(scores[i])}
        for i in topk_idx.tolist()
    ]


def recommend_transformer(
    user_id: str,
    top_k: int,
    data: Dict[str, Any],
    model: nn.Module,
    item_matrix: torch.Tensor,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Research Transformer: score all items via model.forward(). Fully vectorized."""
    train_sequences = data["train_sequences"]
    item_to_idx = data["item_to_idx"]
    idx_to_item = data["idx_to_item"]
    num_items = item_matrix.shape[0]

    user_seq = train_sequences.get(user_id, [])
    if not user_seq:
        return []

    item_indices = [item_to_idx[iid] for iid, _ in user_seq if iid in item_to_idx]
    if not item_indices:
        return []

    # Left-pad to MAX_SEQ_LEN
    context = item_indices[-MAX_SEQ_LEN:]
    padded = [0] * (MAX_SEQ_LEN - len(context)) + context

    seq_tensor = torch.tensor([padded], dtype=torch.long, device=device)
    user_embs = item_matrix[seq_tensor]  # [1, 20, 512]

    # Score ALL 1000 items in one forward pass (vectorized)
    user_expanded = user_embs.expand(num_items, -1, -1)  # [1000, 20, 512]

    with torch.no_grad():
        scores = model(user_expanded, item_matrix)  # [1000]

    # Mask seen items
    seen = {item_to_idx[iid] for iid, _ in user_seq if iid in item_to_idx}
    if seen:
        seen_t = torch.tensor(list(seen), dtype=torch.long, device=device)
        scores[seen_t] = -1e9

    _, topk_idx = torch.topk(scores, min(top_k, num_items))
    topk_cpu = topk_idx.cpu().tolist()
    scores_cpu = scores.cpu()

    return [
        {"item_id": idx_to_item[i], "score": float(scores_cpu[i])}
        for i in topk_cpu
    ]


def get_popular_recs(data: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
    """Popularity-based fallback for cold-start users."""
    popular = data.get("popular_items", [])
    scores = data.get("popular_scores", {})
    return [
        {"item_id": iid, "score": scores.get(iid, 0.0)}
        for iid in popular[:top_k]
    ]


# ============================================================================
# IMAGE HELPERS
# ============================================================================


@st.cache_data(show_spinner=False)
def _placeholder_bytes(size: int = 400) -> bytes:
    """Dark-theme placeholder image."""
    img = Image.new("RGB", (size, size), color=(35, 35, 40))
    draw = ImageDraw.Draw(img)
    text = "No Image"
    bbox = draw.textbbox((0, 0), text)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((size - w) / 2, (size - h) / 2), text, fill=(100, 100, 110))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@st.cache_data(show_spinner=False, max_entries=500)
def _fetch_image(url: str, timeout: int = 5) -> Optional[bytes]:
    """Fetch and cache a product image from URL. Returns None on failure."""
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.thumbnail((600, 600))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception:
        return None


def get_item_image(item_id: str, metadata: Dict[str, Any]) -> bytes:
    """Get image bytes for an item, with dark placeholder fallback."""
    url = metadata.get(item_id, {}).get("image_url")
    if url:
        img = _fetch_image(url)
        if img is not None:
            return img
    return _placeholder_bytes()


def item_has_image(item_id: str, metadata: Dict[str, Any]) -> bool:
    """Check if item has a real image URL."""
    return bool(metadata.get(item_id, {}).get("image_url"))


# ============================================================================
# STYLES
# ============================================================================


def inject_styles() -> None:
    st.markdown(
        """
<style>
/* Layout */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Model badge */
.model-badge {
    display: inline-block;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}

/* Metrics box in sidebar */
.metrics-box {
    background: rgba(99, 102, 241, 0.08);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 10px;
    padding: 14px 18px;
    margin-top: 8px;
}
.metrics-box h4 {
    margin: 0 0 8px 0;
    font-size: 0.85rem;
    opacity: 0.85;
}
.metrics-box .metric {
    font-size: 1.0rem;
    font-weight: 600;
    margin: 2px 0;
}

/* Card styling */
.rec-card {
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 8px;
    background: rgba(255, 255, 255, 0.02);
    transition: all 0.2s ease;
}
.rec-card:hover {
    border-color: rgba(99, 102, 241, 0.5);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
}
.rec-title {
    font-size: 0.92rem;
    font-weight: 600;
    margin-top: 8px;
    line-height: 1.35;
    min-height: 2.4rem;
}
.rec-meta {
    font-size: 0.78rem;
    opacity: 0.65;
    margin-top: 4px;
}

/* Section headers */
.section-header {
    font-size: 1.15rem;
    font-weight: 700;
    margin: 1.2rem 0 0.6rem 0;
    padding-bottom: 6px;
    border-bottom: 2px solid rgba(99, 102, 241, 0.3);
}

/* Sidebar device badge */
.sidebar-device {
    background: rgba(128, 128, 128, 0.1);
    border-radius: 8px;
    padding: 8px 12px;
    font-family: monospace;
    font-size: 0.85rem;
    margin-top: 4px;
}

/* History caption */
.history-caption {
    font-size: 0.8rem;
    text-align: center;
    opacity: 0.85;
    line-height: 1.25;
    margin-top: 4px;
}
</style>
""",
        unsafe_allow_html=True,
    )


# ============================================================================
# MAIN APP
# ============================================================================


def main() -> None:
    st.set_page_config(
        page_title="Multimodal Product Recommendations",
        page_icon="🛍️",
        layout="wide",
    )
    inject_styles()

    # ── Load data ──────────────────────────────────────────────────────────
    try:
        data = load_prepared_data()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    metadata: Dict[str, Any] = data["metadata"]
    train_sequences = data["train_sequences"]
    sample_users: List[str] = data["sample_users"]
    metrics = load_all_metrics()
    device = get_device()

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Controls")

        # Model selector
        model_label = st.selectbox(
            "Model",
            list(MODEL_OPTIONS.keys()),
            index=1,  # default: Research Transformer
        )
        model_key = MODEL_OPTIONS[model_label]

        # Top-K slider
        top_k = st.slider(
            "Top-K Recommendations",
            min_value=3,
            max_value=21,
            value=9,
            step=3,
        )

        # Hide no-image toggle
        hide_no_image = st.checkbox("Hide items without images", value=False)

        st.divider()

        # ── User selection ─────────────────────────────────────────────────
        st.subheader("👤 User Selection")

        # Session-state for random button
        if "selected_user" not in st.session_state:
            st.session_state.selected_user = sample_users[0]

        def _randomize_user():
            st.session_state.selected_user = random.choice(sample_users)

        st.button("🎲 Random User", on_click=_randomize_user)

        current = st.session_state.selected_user
        idx = sample_users.index(current) if current in sample_users else 0

        selected_user = st.selectbox(
            "Select User",
            sample_users,
            index=idx,
            format_func=lambda u: (
                f"{u}  ({len(train_sequences.get(u, []))} interactions)"
            ),
        )
        st.session_state.selected_user = selected_user

        # User info
        n_interactions = len(train_sequences.get(selected_user, []))

        if n_interactions == 0:
            st.warning("🆕 Cold-start user — will show popular items.")
        elif n_interactions < 5:
            st.warning(
                f"⚠️ Only {n_interactions} interaction"
                f"{'s' if n_interactions != 1 else ''}"
                " — recommendations may be less accurate."
            )
        else:
            st.success(f"✅ {n_interactions} interactions")

        st.divider()

        # Device info
        st.markdown(
            f'<div class="sidebar-device">'
            f"🖥️ Device: <strong>{device.type.upper()}</strong>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Model metrics
        if model_key in metrics:
            m = metrics[model_key]
            st.markdown(
                f"""<div class="metrics-box">
<h4>📊 Evaluation Metrics</h4>
<div class="metric">Recall@10: {m['Recall@10']:.4f}</div>
<div class="metric">NDCG@10: {m['NDCG@10']:.4f}</div>
</div>""",
                unsafe_allow_html=True,
            )

    # ── Header + Model Badge ──────────────────────────────────────────────
    col_title, col_badge = st.columns([3, 1])
    with col_title:
        st.title("🛍️ Multimodal Product Recommendations")
    with col_badge:
        badge_text = (
            "Research Multimodal Transformer (MPS)"
            if model_key == "transformer"
            else "Option B — Fast Baseline"
        )
        st.markdown(
            f'<div style="text-align:right; margin-top:28px;">'
            f'<span class="model-badge">{badge_text}</span></div>',
            unsafe_allow_html=True,
        )

    # ── User History ──────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">🕒 User History (Last 5 Interactions)</div>',
        unsafe_allow_html=True,
    )

    user_seq = train_sequences.get(selected_user, [])

    if not user_seq:
        st.info("No interaction history for this user.")
    else:
        history = user_seq[-5:]
        cols = st.columns(min(len(history), 5), gap="small")
        for i, (item_id, _ts) in enumerate(history):
            with cols[i]:
                img_bytes = get_item_image(item_id, metadata)
                st.image(img_bytes, width="stretch")
                title = metadata.get(item_id, {}).get("title", item_id)
                display_title = (title[:55] + "…") if len(title) > 55 else title
                st.markdown(
                    f'<div class="history-caption">{display_title}</div>',
                    unsafe_allow_html=True,
                )

    # ── Recommendations ───────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">🔥 Top Recommendations</div>',
        unsafe_allow_html=True,
    )

    recs: List[Dict[str, Any]] = []
    is_cold_start = len(user_seq) == 0

    if is_cold_start:
        recs = get_popular_recs(data, top_k * 3)
        st.info("🆕 Cold-start user — showing popular items as fallback.")
    else:
        # Over-fetch when filtering is on (80% of items have images)
        fetch_k = top_k * 3 if hide_no_image else top_k

        with st.spinner("Running inference..."):
            if model_key == "option_b":
                model_data = load_option_b_model()
                if model_data is None:
                    st.error(
                        "Option B model not found — "
                        "run `python models/option_b_model.py` first."
                    )
                    st.stop()
                recs = recommend_option_b(selected_user, fetch_k, data, model_data)
            else:
                result = load_research_transformer(device.type)
                if result is None:
                    st.error(
                        "Research Transformer not found — run "
                        "`python 09_research_multimodal_transformer_mps.py --train` first."
                    )
                    st.stop()
                transformer_model, item_matrix = result
                recs = recommend_transformer(
                    selected_user,
                    fetch_k,
                    data,
                    transformer_model,
                    item_matrix,
                    device,
                )

    # Apply no-image filter
    if hide_no_image:
        recs = [r for r in recs if item_has_image(r["item_id"], metadata)]

    recs = recs[:top_k]

    if not recs:
        st.warning("No recommendations produced for this user.")
    else:
        for row_start in range(0, len(recs), 3):
            row = recs[row_start : row_start + 3]
            cols = st.columns(3, gap="medium")

            for col, rec in zip(cols, row):
                item_id = rec["item_id"]
                title = metadata.get(item_id, {}).get("title", item_id)
                score = rec["score"]

                with col:
                    st.markdown('<div class="rec-card">', unsafe_allow_html=True)
                    img_bytes = get_item_image(item_id, metadata)
                    st.image(img_bytes, width="stretch")
                    st.markdown(
                        f'<div class="rec-title">{title}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="rec-meta">'
                        f"Score: {score:.4f} · ID: {item_id}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────
    st.divider()

    m_label = (
        "Research Transformer (MPS)"
        if model_key == "transformer"
        else "Option B (Baseline)"
    )
    m_data = metrics.get(model_key, {})
    recall = m_data.get("Recall@10", 0)
    ndcg = m_data.get("NDCG@10", 0)

    st.caption(
        f"**{m_label}** · Recall@10: {recall:.4f} · NDCG@10: {ndcg:.4f} · "
        f"Device: {device.type.upper()} · "
        f"{len(data['all_items'])} items · {len(data['all_users']):,} users"
    )


if __name__ == "__main__":
    main()
