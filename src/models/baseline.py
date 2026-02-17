"""
Simple but Strong Multimodal Recommender (Option B)
====================================================
Lightweight, efficient approach that outperforms heavy transformers on sparse data.

Key improvements:
- Time-decayed weighted pooling for user representation
- L2 normalization for stable scoring
- Optional lightweight BPR fine-tuning (optional via --train flag)
- Vectorized scoring with torch.matmul
- Efficient evaluation on 2000 random users

Architecture:
- Item Representation: L2-normalized CLIP (text + image averaged)
- User Representation: Time-decayed weighted mean of last N items
- Scoring: Normalized dot product
- Optional: 512→512 projection layer trained with BPR loss

NO transformers, NO cross-modal attention, NO deep fusion layers.
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Set, Tuple
from tqdm import tqdm
import math
import random
import argparse
import os
from .clip_encoder import TextEncoder


class ProjectionLayer(nn.Module):
    """Simple projection layer for optional fine-tuning."""
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, x):
        return self.linear(x)


def load_prepared_data():
    """Load prepared data from data/ directory."""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'prepared_data.pkl')
    with open(data_path, 'rb') as f:
        return pickle.load(f)


def load_image_encoder_data():
    """Load image encoder data from data/ directory."""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'image_encoder_data.pkl')
    with open(data_path, 'rb') as f:
        return pickle.load(f)


def recall_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    """Calculate Recall@K."""
    if len(relevant) == 0:
        return 0.0
    recommended_k = set(recommended[:k])
    return len(recommended_k & relevant) / len(relevant)


def ndcg_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    """Calculate NDCG@K."""
    if len(relevant) == 0:
        return 0.0
    
    recommended_k = recommended[:k]
    dcg = 0.0
    for i, item in enumerate(recommended_k):
        if item in relevant:
            dcg += 1.0 / math.log2(i + 2)
    
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def get_user_embedding(user_id: str, train_sequences: Dict, item_embedding_matrix: torch.Tensor,
                       item_to_idx: Dict, N: int = 5, use_time_decay: bool = True) -> torch.Tensor:
    """
    Get user embedding as time-decayed weighted mean of last N items.
    
    Args:
        user_id: User identifier
        train_sequences: Dict[user_id] = [(item_id, timestamp), ...]
        item_embedding_matrix: [num_items, embedding_dim] tensor (must be L2 normalized)
        item_to_idx: Dict mapping item_id to index
        N: Number of recent items to use for pooling (default: 5)
        use_time_decay: Whether to apply exponential time decay (default: True)
    
    Returns:
        L2-normalized user embedding [embedding_dim] tensor
    """
    user_seq = train_sequences.get(user_id, [])
    if len(user_seq) == 0:
        # Return zero embedding for cold-start users
        return torch.zeros(item_embedding_matrix.shape[1], device=item_embedding_matrix.device)
    
    # Get last N items
    recent_items = user_seq[-N:]  # [(item_id, timestamp), ...]
    
    # Map to indices
    item_indices = []
    weights = []
    
    for seq_pos, (item_id, timestamp) in enumerate(recent_items):
        if item_id in item_to_idx:
            item_indices.append(item_to_idx[item_id])
            
            # Time decay: newer items get higher weight
            # Weight = exp(alpha * normalized_position)
            # normalized_position in [0, 1], where 1 is the most recent
            if use_time_decay:
                norm_pos = seq_pos / max(len(recent_items) - 1, 1)
                weight = np.exp(2.0 * norm_pos)  # alpha=2.0 for exponential decay
            else:
                weight = 1.0
            weights.append(weight)
    
    if len(item_indices) == 0:
        return torch.zeros(item_embedding_matrix.shape[1], device=item_embedding_matrix.device)
    
    # Normalize weights to sum to 1
    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.sum()
    weights_tensor = torch.from_numpy(weights).to(item_embedding_matrix.device)
    
    # Get embeddings: [num_recent, embedding_dim]
    item_indices_tensor = torch.tensor(item_indices, device=item_embedding_matrix.device)
    item_embeddings = item_embedding_matrix[item_indices_tensor]
    
    # Weighted mean: [embedding_dim]
    user_emb = (item_embeddings * weights_tensor.unsqueeze(-1)).sum(dim=0)
    
    # L2 normalize
    user_emb = user_emb / (torch.norm(user_emb) + 1e-8)
    
    return user_emb


def build_item_embedding_matrix(metadata: Dict, item_to_idx: Dict, item_image_embeddings: Dict,
                                text_encoder: TextEncoder, device: torch.device) -> torch.Tensor:
    """
    Build final item embedding matrix by fusing text and image embeddings.
    
    For each item:
    - If image embedding exists: final_emb = (text_emb + image_emb) / 2
    - Else: final_emb = text_emb
    
    All embeddings are L2-normalized.
    
    Args:
        metadata: Dict containing item metadata with 'product_text'
        item_to_idx: Dict mapping item_id to index
        item_image_embeddings: Dict[item_id] = image_embedding (tensor)
        text_encoder: TextEncoder instance for encoding product texts
        device: torch device
    
    Returns:
        L2-normalized tensor of shape [num_items, embedding_dim]
    """
    num_items = len(item_to_idx)
    
    # Initialize embedding matrix
    item_embedding_matrix = None
    embedding_dim = None
    
    # Collect items with text
    items_with_text = []
    texts = []
    item_ids_list = []
    
    for item_id, idx in item_to_idx.items():
        if item_id in metadata and metadata[item_id].get('product_text'):
            items_with_text.append(idx)
            texts.append(metadata[item_id]['product_text'])
            item_ids_list.append(item_id)
    
    print(f"Encoding {len(texts)} items with text...")
    
    # Encode all texts at once
    text_embeddings = text_encoder.encode(texts)  # [num_items_with_text, embedding_dim]
    text_embeddings = text_embeddings.to(device)
    embedding_dim = text_embeddings.shape[1]
    
    # Initialize matrix
    item_embedding_matrix = torch.zeros(num_items, embedding_dim, device=device)
    
    # Fill in embeddings
    for idx, (item_idx, item_id) in enumerate(zip(items_with_text, item_ids_list)):
        text_emb = text_embeddings[idx]
        
        # Check if image embedding exists
        if item_id in item_image_embeddings:
            image_emb = item_image_embeddings[item_id]
            if isinstance(image_emb, np.ndarray):
                image_emb = torch.from_numpy(image_emb).float()
            image_emb = image_emb.to(device)
            
            # Fuse: mean of text and image
            final_emb = (text_emb + image_emb) / 2.0
        else:
            # Use text only
            final_emb = text_emb
        
        # L2 normalize
        final_emb = final_emb / (torch.norm(final_emb) + 1e-8)
        
        item_embedding_matrix[item_idx] = final_emb
    
    print(f"Built item embedding matrix: {item_embedding_matrix.shape}")
    print(f"  Items with both text+image: {sum(1 for iid in item_ids_list if iid in item_image_embeddings)}")
    print(f"  Items with text only: {len(item_ids_list) - sum(1 for iid in item_ids_list if iid in item_image_embeddings)}")
    print(f"  All embeddings are L2-normalized")
    
    return item_embedding_matrix


def evaluate_simple_multimodal(
    item_embedding_matrix: torch.Tensor,
    train_sequences: Dict,
    val_interactions: Dict,
    test_interactions: Dict,
    item_to_idx: Dict,
    user_to_idx: Dict,
    device: torch.device,
    projection_layer: nn.Module = None,
    k_values: List[int] = [5, 10, 20],
    num_eval_users: int = 2000
) -> Dict:
    """
    Evaluate simple multimodal recommender on subset of users.
    
    - All embeddings are L2-normalized
    - Vectorized scoring with torch.matmul
    - No nested loops over items
    - Removes seen items from recommendations
    
    Args:
        item_embedding_matrix: [num_items, embedding_dim] (L2-normalized)
        train_sequences: User training sequences
        val_interactions: Validation interactions
        test_interactions: Test interactions
        item_to_idx: Item to index mapping
        user_to_idx: User to index mapping
        device: torch device
        projection_layer: Optional projection layer (if fine-tuned)
        k_values: List of K values for metrics
        num_eval_users: Number of random users to evaluate (for speed)
    
    Returns:
        Dict with metrics for val and test sets
    """
    results = {}
    
    # Create inverse mapping: idx -> item_id (for safe lookup)
    idx_to_item = {idx: item_id for item_id, idx in item_to_idx.items()}
    
    # Sample random users for evaluation (for speed)
    all_users = list(test_interactions.keys())
    if len(all_users) > num_eval_users:
        random.seed(42)
        eval_users = random.sample(all_users, num_eval_users)
    else:
        eval_users = all_users
    
    print(f"Evaluating on {len(eval_users)} random users (seed=42)")
    
    # Optionally apply projection to item embeddings
    if projection_layer is not None:
        print("Applying learned projection layer to embeddings...")
        with torch.no_grad():
            projected_embeddings = projection_layer(item_embedding_matrix)
            projected_embeddings = projected_embeddings / (torch.norm(projected_embeddings, dim=1, keepdim=True) + 1e-8)
        eval_embeddings = projected_embeddings
    else:
        eval_embeddings = item_embedding_matrix
    
    for split_name, interactions in [('val', val_interactions), ('test', test_interactions)]:
        results[split_name] = {f'recall@{k}': [] for k in k_values}
        results[split_name].update({f'ndcg@{k}': [] for k in k_values})
        
        with torch.no_grad():
            for user_id in tqdm(eval_users, desc=f"Evaluating {split_name}"):
                if user_id not in interactions:
                    continue
                
                relevant_items = interactions[user_id]
                if len(relevant_items) == 0:
                    continue
                
                # Get user embedding (time-decayed, L2-normalized)
                user_emb = get_user_embedding(user_id, train_sequences, eval_embeddings,
                                            item_to_idx, N=5, use_time_decay=True)
                
                if user_emb.abs().sum() == 0:
                    # Cold-start user, skip
                    continue
                
                # Get seen items for this user
                seen_items = set([item_id for item_id, _ in train_sequences.get(user_id, [])])
                
                # Score all items: [embedding_dim] @ [embedding_dim, num_items] = [num_items]
                scores = torch.matmul(user_emb, eval_embeddings.T)
                
                # Zero out seen items
                for item_id in seen_items:
                    if item_id in item_to_idx:
                        item_idx = item_to_idx[item_id]
                        scores[item_idx] = -float('inf')
                
                # Get top-K items
                max_k = max(k_values)
                _, top_indices = torch.topk(scores, min(max_k + 1, eval_embeddings.shape[0]))
                
                # Filter out inf values and pad if needed
                top_indices = [idx for idx in top_indices if scores[idx] > -float('inf')][:max_k]
                ranked_items = [idx_to_item[idx.item()] for idx in top_indices]
                
                # Compute metrics for all K values
                for k in k_values:
                    results[split_name][f'recall@{k}'].append(
                        recall_at_k(ranked_items, set(relevant_items), k)
                    )
                    results[split_name][f'ndcg@{k}'].append(
                        ndcg_at_k(ranked_items, set(relevant_items), k)
                    )
        
        # Average metrics
        for k in k_values:
            if results[split_name][f'recall@{k}']:
                results[split_name][f'recall@{k}'] = np.mean(results[split_name][f'recall@{k}'])
            else:
                results[split_name][f'recall@{k}'] = 0.0
                
            if results[split_name][f'ndcg@{k}']:
                results[split_name][f'ndcg@{k}'] = np.mean(results[split_name][f'ndcg@{k}'])
            else:
                results[split_name][f'ndcg@{k}'] = 0.0
    
    return results


def train_projection_layer_bpr(item_embedding_matrix: torch.Tensor, train_sequences: Dict,
                               item_to_idx: Dict, device: torch.device, num_epochs: int = 5,
                               batch_size: int = 256, learning_rate: float = 0.001):
    """
    Light fine-tuning of projection layer using BPR loss.
    
    Args:
        item_embedding_matrix: [num_items, embedding_dim] (L2-normalized)
        train_sequences: User training sequences
        item_to_idx: Item to index mapping
        device: torch device
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
    
    Returns:
        Trained ProjectionLayer module
    """
    embedding_dim = item_embedding_matrix.shape[1]
    projection_layer = ProjectionLayer(embedding_dim).to(device)
    optimizer = torch.optim.Adam(projection_layer.parameters(), lr=learning_rate)
    
    print("\n" + "="*70)
    print("OPTIONAL BPR FINE-TUNING")
    print("="*70)
    print(f"Training for {num_epochs} epochs with BPR loss...")
    
    # Collect training samples
    training_samples = []
    for user_id, seq in train_sequences.items():
        if len(seq) < 2:
            continue
        for i in range(1, len(seq)):
            pos_item = seq[i][0]
            if pos_item in item_to_idx:
                training_samples.append((user_id, pos_item))
    
    print(f"Collected {len(training_samples)} training samples")
    
    total_items = len(item_to_idx)
    num_batches = max(1, len(training_samples) // batch_size)
    
    for epoch in range(num_epochs):
        random.shuffle(training_samples)
        epoch_loss = 0.0
        
        for batch_idx in range(0, len(training_samples), batch_size):
            batch_samples = training_samples[batch_idx:batch_idx + batch_size]
            
            user_ids, pos_items = zip(*batch_samples)
            
            # Random negative sampling
            neg_items = [random.choice(list(item_to_idx.keys())) for _ in user_ids]
            
            pos_indices = [item_to_idx[item] for item in pos_items]
            neg_indices = [item_to_idx[item] for item in neg_items]
            
            pos_indices = torch.tensor(pos_indices, device=device)
            neg_indices = torch.tensor(neg_indices, device=device)
            
            # Get user embeddings and project them
            user_embeddings = []
            for user_id in user_ids:
                user_emb = get_user_embedding(user_id, train_sequences, item_embedding_matrix,
                                            item_to_idx, N=5, use_time_decay=True)
                user_emb = projection_layer(user_emb.unsqueeze(0)).squeeze(0)
                user_emb = user_emb / (torch.norm(user_emb) + 1e-8)
                user_embeddings.append(user_emb)
            
            user_embeddings = torch.stack(user_embeddings)  # [batch_size, embedding_dim]
            
            # Project item embeddings
            projected_items = projection_layer(item_embedding_matrix)
            projected_items = projected_items / (torch.norm(projected_items, dim=1, keepdim=True) + 1e-8)
            
            pos_item_embs = projected_items[pos_indices]
            neg_item_embs = projected_items[neg_indices]
            
            # BPR loss: log_sigmoid(score_pos - score_neg)
            pos_scores = (user_embeddings * pos_item_embs).sum(dim=1)
            neg_scores = (user_embeddings * neg_item_embs).sum(dim=1)
            
            loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / max(1, len(training_samples) // batch_size)
        print(f"  Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
    
    projection_layer.eval()
    print("BPR fine-tuning completed ✓")
    
    return projection_layer


def main():
    """Main function for simple multimodal recommender."""
    parser = argparse.ArgumentParser(description="Option B: Simple Multimodal Recommender")
    parser.add_argument('--train', action='store_true', help='Enable optional BPR fine-tuning')
    args = parser.parse_args()
    
    print("="*70)
    print("SIMPLE BUT STRONG MULTIMODAL RECOMMENDER (OPTION B)")
    print("="*70)
    print("Architecture:")
    print("  - Item Representation: L2-normalized CLIP (text + image fused)")
    print("  - User Representation: Time-decayed mean of last 5 items")
    print("  - Scoring: Normalized dot product (cosine similarity)")
    if args.train:
        print("  - Tuning: Optional 512→512 projection with BPR loss")
    print("  - Evaluation: 2000 random users only (for speed)")
    print("  - NO transformers, NO cross-modal attention, NO deep fusion")
    
    # Load data
    print("\nLoading data...")
    data = load_prepared_data()
    train_sequences = data['train_sequences']
    val_interactions = data['val_interactions']
    test_interactions = data['test_interactions']
    metadata = data['metadata']
    item_to_idx = data['item_to_idx']
    user_to_idx = data['user_to_idx']
    
    print(f"  Users: {len(user_to_idx)}")
    print(f"  Items: {len(item_to_idx)}")
    print(f"  Val interactions: {sum(len(v) for v in val_interactions.values())}")
    print(f"  Test interactions: {sum(len(v) for v in test_interactions.values())}")
    
    # Load image embeddings
    print("\nLoading image embeddings...")
    image_data = load_image_encoder_data()
    item_image_embeddings = image_data['item_image_embeddings']
    print(f"  Loaded {len(item_image_embeddings)} image embeddings")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Initialize text encoder
    print("\nInitializing CLIP text encoder...")
    text_encoder = TextEncoder(model_name='openai/clip-vit-base-patch32', freeze=True)
    text_encoder = text_encoder.to(device)
    text_encoder.eval()
    
    # Build item embedding matrix
    print("\n" + "="*70)
    print("STEP 1: BUILD FINAL ITEM EMBEDDINGS")
    print("="*70)
    item_embedding_matrix = build_item_embedding_matrix(
        metadata, item_to_idx, item_image_embeddings, text_encoder, device
    )
    
    # Optional: Train projection layer with BPR
    projection_layer = None
    if args.train:
        projection_layer = train_projection_layer_bpr(
            item_embedding_matrix, train_sequences, item_to_idx, device,
            num_epochs=5, batch_size=256, learning_rate=0.001
        )
    
    # Evaluate
    print("\n" + "="*70)
    print("STEP 2: EVALUATE WITH OPTIMIZED SCORING")
    print("="*70)
    results = evaluate_simple_multimodal(
        item_embedding_matrix,
        train_sequences,
        val_interactions,
        test_interactions,
        item_to_idx,
        user_to_idx,
        device,
        projection_layer=projection_layer,
        k_values=[5, 10, 20],
        num_eval_users=2000
    )
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    for split in ['val', 'test']:
        print(f"\n{split.upper()} Set:")
        for metric in ['recall@5', 'recall@10', 'recall@20', 'ndcg@5', 'ndcg@10', 'ndcg@20']:
            print(f"  {metric}: {results[split][metric]:.4f}")
    
    # Save model and results
    print("\n" + "="*70)
    print("STEP 3: SAVE MODEL AND RESULTS")
    print("="*70)
    
    # Save model to root directory
    model_data = {
        'item_embedding_matrix': item_embedding_matrix.cpu(),
        'projection_layer': projection_layer.state_dict() if projection_layer else None,
        'item_to_idx': item_to_idx,
        'embedding_dim': item_embedding_matrix.shape[1],
        'has_projection': projection_layer is not None
    }
    
    # Save to checkpoints/ folder (project root)
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    checkpoints_dir = os.path.join(project_root, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    model_path = os.path.join(checkpoints_dir, 'option_b_model.pt')
    torch.save(model_data, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save results (compatible with Streamlit)
    option_b_results = {
        'results': results,
        'config': {
            'approach': 'simple_multimodal_with_time_decay_and_normalization',
            'user_representation': 'time_decayed_mean_of_last_5_items',
            'item_representation': 'L2_normalized_text_image_average',
            'scoring': 'cosine_similarity_dot_product',
            'fine_tuning': 'BPR_projection' if args.train else 'none',
            'num_eval_users': 2000,
            'encoder_type': 'CLIP',
            'embedding_dim': item_embedding_matrix.shape[1],
            'num_items_with_embeddings': item_embedding_matrix.shape[0],
            'num_items_with_images': len(item_image_embeddings)
        }
    }
    
    results_path = os.path.join(checkpoints_dir, 'option_b_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(option_b_results, f)
    
    print(f"Results saved to: {results_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("OPTION B: SIMPLE MULTIMODAL RECOMMENDER COMPLETE ✓")
    print("="*70)
    print("\nKey Metrics (Test Set):")
    print(f"  Recall@10: {results['test']['recall@10']:.4f}")
    print(f"  NDCG@10: {results['test']['ndcg@10']:.4f}")
    print(f"  Recall@20: {results['test']['recall@20']:.4f}")
    print(f"  NDCG@20: {results['test']['ndcg@20']:.4f}")
    
    if args.train:
        print("\nWith BPR Fine-Tuning: Enabled ✓")


def recommend(user_id: str, top_k: int = 10) -> List[Dict]:
    """
    Recommend items for a user (Streamlit-compatible API).
    
    Loads the checkpoints/option_b_model.pt and returns recommendations.
    Compatible with other model formats.
    
    Args:
        user_id: User identifier
        top_k: Number of items to recommend
    
    Returns:
        List of dicts: [{"item_id": ..., "score": ...}, ...]
    """
    # Load model from checkpoints/ folder
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    model_path = os.path.join(project_root, 'checkpoints', 'option_b_model.pt')
    model_data = torch.load(model_path, map_location='cpu')
    item_embedding_matrix = model_data['item_embedding_matrix']
    item_to_idx = model_data['item_to_idx']
    projection_layer = model_data['projection_layer']
    has_projection = model_data['has_projection']
    
    # Load data
    data = load_prepared_data()
    train_sequences = data['train_sequences']
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    item_embedding_matrix = item_embedding_matrix.to(device)
    
    # Apply projection if available
    if has_projection and projection_layer is not None:
        proj_layer = ProjectionLayer(item_embedding_matrix.shape[1])
        proj_layer.load_state_dict(projection_layer)
        proj_layer = proj_layer.to(device)
        proj_layer.eval()
        
        with torch.no_grad():
            eval_embeddings = proj_layer(item_embedding_matrix)
            eval_embeddings = eval_embeddings / (torch.norm(eval_embeddings, dim=1, keepdim=True) + 1e-8)
    else:
        eval_embeddings = item_embedding_matrix
    
    # Get user sequence
    user_train_seq = train_sequences.get(user_id, [])
    if len(user_train_seq) == 0:
        return []
    
    # Get seen items
    seen_items = set([item_id for item_id, _ in user_train_seq])
    
    # Compute user embedding
    user_emb = get_user_embedding(user_id, train_sequences, eval_embeddings,
                                 item_to_idx, N=5, use_time_decay=True)
    
    if user_emb.abs().sum() == 0:
        return []
    
    # Score all items
    with torch.no_grad():
        scores = torch.matmul(user_emb, eval_embeddings.T)
    
    # Get top-K unseen items
    scores_np = scores.cpu().numpy()
    
    # Create item list with scores, excluding seen items
    item_scores = []
    for item_id, idx in item_to_idx.items():
        if item_id not in seen_items:
            item_scores.append({
                'item_id': item_id,
                'score': float(scores_np[idx])
            })
    
    # Sort by score and take top-K
    item_scores.sort(key=lambda x: x['score'], reverse=True)
    recommendations = item_scores[:top_k]
    
    return recommendations


if __name__ == "__main__":
    main()
