"""
Multimodal Recommendation Model (IMPROVED)
==========================================
- Sequential user modeling (SASRec-style Transformer)
- CLIP-aligned text and image embeddings
- Late fusion (dot product)
- Ranking loss (BPR)
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Set, Optional, Tuple
from tqdm import tqdm
import math

from clip_encoder import TextEncoder
from sequential_user_model import SequentialUserModel, create_sequence_batch
from ranking_loss import BPRLoss


def load_prepared_data():
    """Load prepared data."""
    with open('prepared_data.pkl', 'rb') as f:
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


# TextEncoder is now imported from clip_encoder.py
# This uses CLIP for text encoding, ensuring alignment with image embeddings


class MultimodalRecommender(nn.Module):
    """
    Improved multimodal recommender with:
    - Sequential user modeling (SASRec-style Transformer)
    - CLIP-aligned text and image embeddings
    - Late fusion (dot product)
    """
    
    def __init__(
        self,
        n_items: int,
        text_embedding_dim: int,
        image_embedding_dim: int,
        user_model_dim: int = 128,
        max_seq_len: int = 50,
        dropout: float = 0.1
    ):
        """
        Args:
            n_items: Number of items
            text_embedding_dim: CLIP text embedding dimension (512)
            image_embedding_dim: CLIP image embedding dimension (512)
            user_model_dim: User embedding dimension from sequential model
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
        """
        super(MultimodalRecommender, self).__init__()
        
        # Sequential user model (replaces pointwise embeddings)
        self.user_model = SequentialUserModel(
            n_items=n_items,
            d_model=user_model_dim,
            nhead=4,
            num_layers=2,
            dim_feedforward=256,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        # Item embeddings: CLIP-aligned text + image
        # Since CLIP embeddings are already aligned, we concatenate and project
        item_embedding_dim = text_embedding_dim + image_embedding_dim
        self.item_mlp = nn.Sequential(
            nn.Linear(item_embedding_dim, user_model_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(user_model_dim * 2, user_model_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        item_sequences: torch.Tensor,
        sequence_mask: Optional[torch.Tensor],
        text_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            item_sequences: [batch_size, seq_len] - User interaction sequences
            sequence_mask: [batch_size, seq_len] - Mask for sequences
            text_embeddings: [batch_size, text_embedding_dim] - Item text embeddings
            image_embeddings: [batch_size, image_embedding_dim] - Item image embeddings
        
        Returns:
            scores: [batch_size] - Recommendation scores
        """
        # Get user embedding from sequence (sequential modeling)
        user_emb = self.user_model(item_sequences, sequence_mask)  # [batch_size, user_model_dim]
        
        # Fuse item embeddings (late fusion: concatenate CLIP-aligned embeddings)
        item_emb = torch.cat([text_embeddings, image_embeddings], dim=1)  # [batch_size, text_dim + image_dim]
        item_emb = self.item_mlp(item_emb)  # [batch_size, user_model_dim]
        
        # Late fusion: dot product (since embeddings are in same space)
        scores = torch.sum(user_emb * item_emb, dim=1)  # [batch_size]
        
        return scores


class MultimodalDataset(Dataset):
    """Dataset for multimodal training with sequences."""
    
    def __init__(
        self,
        train_sequences: Dict[str, List[Tuple[str, int]]],
        metadata: Dict,
        user_to_idx: Dict,
        item_to_idx: Dict,
        text_encoder: TextEncoder,
        item_image_embeddings: Dict,
        all_items: Set,
        max_seq_len: int = 50,
        negative_samples: int = 4
    ):
        self.train_sequences = train_sequences
        self.metadata = metadata
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.text_encoder = text_encoder
        self.item_image_embeddings = item_image_embeddings
        self.all_items = all_items
        self.max_seq_len = max_seq_len
        self.negative_samples = negative_samples
        
        # Pre-encode all item texts and images
        print("Pre-encoding item texts and images...")
        self.item_text_embeddings = {}
        self.item_multimodal_embeddings = {}
        
        items_with_both = []
        texts = []
        images = []
        item_ids = []
        
        for item_id in all_items:
            if (item_id in metadata and 
                metadata[item_id].get('product_text') and 
                item_id in item_image_embeddings):
                items_with_both.append(item_id)
                texts.append(metadata[item_id]['product_text'])
                images.append(item_image_embeddings[item_id])
                item_ids.append(item_id)
        
        # Encode texts in batches
        batch_size = 32
        all_text_emb = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
            batch_texts = texts[i:i+batch_size]
            batch_emb = text_encoder.encode(batch_texts)
            all_text_emb.append(batch_emb.cpu())
        
        if all_text_emb:
            text_embeddings = torch.cat(all_text_emb, dim=0)
            image_embeddings = torch.stack(images)
            
            for idx, item_id in enumerate(item_ids):
                self.item_text_embeddings[item_id] = text_embeddings[idx]
                self.item_multimodal_embeddings[item_id] = {
                    'text': text_embeddings[idx],
                    'image': image_embeddings[idx]
                }
        
        print(f"Encoded {len(self.item_multimodal_embeddings)} items")
        
        # Create training samples
        self.samples = []
        for user_id, sequence in train_sequences.items():
            if user_id not in user_to_idx:
                continue
            if len(sequence) == 0:
                continue
            
            # Use last item in sequence as positive
            pos_item_id = sequence[-1][0]  # (item_id, timestamp)
            
            if pos_item_id in self.item_multimodal_embeddings:
                # Use all previous items as context
                context_items = [item_id for item_id, _ in sequence[:-1]]
                self.samples.append((user_id, context_items, pos_item_id))
        
        print(f"Created {len(self.samples)} training samples")
    
    def __len__(self):
        return len(self.samples) * (1 + self.negative_samples)
    
    def __getitem__(self, idx):
        # Map to positive sample
        pos_idx = idx // (1 + self.negative_samples)
        is_positive = (idx % (1 + self.negative_samples)) == 0
        
        user_id, context_items, pos_item_id = self.samples[pos_idx]
        
        if is_positive:
            item_id = pos_item_id
            label = 1.0
        else:
            # Negative sampling
            user_items = set([item_id for item_id, _ in self.train_sequences.get(user_id, [])])
            negative_items = [item for item in self.all_items 
                            if item not in user_items 
                            and item in self.item_multimodal_embeddings]
            if len(negative_items) == 0:
                item_id = pos_item_id
            else:
                item_id = np.random.choice(negative_items)
            label = 0.0
        
        # Get embeddings
        text_emb = self.item_text_embeddings[item_id]
        image_emb = self.item_multimodal_embeddings[item_id]['image']
        
        # Create sequence (context + current item for positive, just context for negative)
        if is_positive:
            sequence = context_items + [item_id]
        else:
            sequence = context_items  # Don't include negative item in sequence
        
        return {
            'user_id': user_id,
            'sequence': sequence,
            'item_id': item_id,
            'text_embedding': text_emb,
            'image_embedding': image_emb,
            'label': label
        }


def collate_fn(batch):
    """Collate function for batching sequences."""
    user_ids = [item['user_id'] for item in batch]
    sequences = [item['sequence'] for item in batch]
    item_ids = [item['item_id'] for item in batch]
    text_embeddings = torch.stack([item['text_embedding'] for item in batch])
    image_embeddings = torch.stack([item['image_embedding'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.float32)
    
    return {
        'user_ids': user_ids,
        'sequences': sequences,
        'item_ids': item_ids,
        'text_embeddings': text_embeddings,
        'image_embeddings': image_embeddings,
        'labels': labels
    }


def train_multimodal_model(
    model: MultimodalRecommender,
    train_loader: DataLoader,
    item_to_idx: Dict,
    device: torch.device,
    n_epochs: int = 10,
    lr: float = 0.001
):
    """Train the multimodal recommendation model with BPR loss."""
    model = model.to(device)
    criterion = BPRLoss()  # Ranking loss instead of BCE
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
        for batch in pbar:
            # Prepare sequences
            sequences_tensor, sequence_mask = create_sequence_batch(
                batch['sequences'],
                item_to_idx,
                max_seq_len=model.user_model.max_seq_len
            )
            sequences_tensor = sequences_tensor.to(device)
            sequence_mask = sequence_mask.to(device)
            
            text_embeddings = batch['text_embeddings'].to(device)
            image_embeddings = batch['image_embeddings'].to(device)
            labels = batch['labels'].to(device)
            
            # Get scores for all items
            all_scores = model(
                sequences_tensor,
                sequence_mask,
                text_embeddings,
                image_embeddings
            )
            
            # Separate positive and negative samples
            pos_mask = labels == 1.0
            neg_mask = labels == 0.0
            
            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                pos_scores = all_scores[pos_mask]
                neg_scores = all_scores[neg_mask]
                
                # BPR: pair each positive with a negative
                n_pos = len(pos_scores)
                n_neg = len(neg_scores)
                if n_neg >= n_pos:
                    neg_indices = torch.randperm(n_neg)[:n_pos]
                    neg_scores_paired = neg_scores[neg_indices]
                else:
                    neg_scores_paired = neg_scores.repeat((n_pos // n_neg) + 1)[:n_pos]
                
                loss = criterion(pos_scores, neg_scores_paired)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
                pbar.set_postfix({'loss': loss.item()})
        
        if n_batches > 0:
            avg_loss = total_loss / n_batches
            print(f"Epoch {epoch+1}/{n_epochs}, Average Loss: {avg_loss:.4f}")
    
    return model


def evaluate_multimodal_model(
    model: MultimodalRecommender,
    text_encoder: TextEncoder,
    train_sequences: Dict,
    val_interactions: Dict,
    test_interactions: Dict,
    metadata: Dict,
    item_image_embeddings: Dict,
    user_to_idx: Dict,
    item_to_idx: Dict,
    all_items: Set,
    device: torch.device,
    k_values: List[int] = [5, 10, 20]
) -> Dict:
    """Evaluate multimodal model with sequences."""
    model.eval()
    text_encoder.eval()
    
    results = {}
    
    # Pre-compute item embeddings
    print("Pre-computing item embeddings for evaluation...")
    item_embeddings_cache = {}
    items_with_both = []
    texts = []
    images = []
    item_ids = []
    
    for item_id in all_items:
        if (item_id in metadata and 
            metadata[item_id].get('product_text') and 
            item_id in item_image_embeddings):
            items_with_both.append(item_id)
            texts.append(metadata[item_id]['product_text'])
            images.append(item_image_embeddings[item_id])
            item_ids.append(item_id)
    
    # Encode texts in batches
    batch_size = 32
    all_text_emb = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_emb = text_encoder.encode(batch_texts)
        all_text_emb.append(batch_emb.cpu())
    
    if all_text_emb:
        text_embeddings = torch.cat(all_text_emb, dim=0)
        image_embeddings = torch.stack(images)
        
        for idx, item_id in enumerate(item_ids):
            item_embeddings_cache[item_id] = {
                'text': text_embeddings[idx],
                'image': image_embeddings[idx]
            }
    
    print(f"Cached embeddings for {len(item_embeddings_cache)} items")
    
    for split_name, interactions in [('val', val_interactions), ('test', test_interactions)]:
        results[split_name] = {f'recall@{k}': [] for k in k_values}
        results[split_name].update({f'ndcg@{k}': [] for k in k_values})
        
        with torch.no_grad():
            for user_id, relevant_items in tqdm(interactions.items(), desc=f"Evaluating {split_name}"):
                if user_id not in user_to_idx:
                    continue
                
                # Get user's training sequence
                user_train_seq = train_sequences.get(user_id, [])
                if len(user_train_seq) == 0:
                    continue
                
                # Get context (all training items)
                context_items = [item_id for item_id, _ in user_train_seq]
                
                # Score all candidate items
                candidate_items = []
                candidate_text_emb = []
                candidate_image_emb = []
                
                for item_id in all_items:
                    if item_id in item_embeddings_cache:
                        candidate_items.append(item_id)
                        candidate_text_emb.append(item_embeddings_cache[item_id]['text'])
                        candidate_image_emb.append(item_embeddings_cache[item_id]['image'])
                
                if len(candidate_items) == 0:
                    continue
                
                # Prepare sequence
                sequences_tensor, sequence_mask = create_sequence_batch(
                    [context_items],
                    item_to_idx,
                    max_seq_len=model.user_model.max_seq_len
                )
                sequences_tensor = sequences_tensor.to(device)
                sequence_mask = sequence_mask.to(device)
                
                # Prepare item embeddings
                candidate_text_tensor = torch.stack(candidate_text_emb).to(device)
                candidate_image_tensor = torch.stack(candidate_image_emb).to(device)
                
                # Score all candidates
                sequences_tensor_expanded = sequences_tensor.repeat(len(candidate_items), 1)
                sequence_mask_expanded = sequence_mask.repeat(len(candidate_items), 1)
                
                scores = model(
                    sequences_tensor_expanded,
                    sequence_mask_expanded,
                    candidate_text_tensor,
                    candidate_image_tensor
                ).cpu().numpy()
                
                # Rank items
                ranked_items = [candidate_items[i] for i in np.argsort(scores)[::-1]]
                
                # Calculate metrics
                for k in k_values:
                    results[split_name][f'recall@{k}'].append(
                        recall_at_k(ranked_items, relevant_items, k)
                    )
                    results[split_name][f'ndcg@{k}'].append(
                        ndcg_at_k(ranked_items, relevant_items, k)
                    )
        
        # Average metrics
        for k in k_values:
            results[split_name][f'recall@{k}'] = np.mean(results[split_name][f'recall@{k}'])
            results[split_name][f'ndcg@{k}'] = np.mean(results[split_name][f'ndcg@{k}'])
    
    return results


def main():
    """Main function for multimodal fusion."""
    print("="*60)
    print("MULTIMODAL RECOMMENDATION MODEL")
    print("="*60)
    
    # Load data
    print("\nLoading prepared data...")
    data = load_prepared_data()
    
    train_sequences = data['train_sequences']  # Use sequences instead of interactions
    val_interactions = data['val_interactions']
    test_interactions = data['test_interactions']
    metadata = data['metadata']
    user_to_idx = data['user_to_idx']
    item_to_idx = data['item_to_idx']
    all_items = data['all_items']
    
    # Load image embeddings
    print("\nLoading image embeddings...")
    with open('image_encoder_data.pkl', 'rb') as f:
        image_data = pickle.load(f)
    item_image_embeddings = image_data['item_image_embeddings']
    image_embedding_dim = image_data['image_encoder_config']['embedding_dim']
    
    print(f"Loaded {len(item_image_embeddings)} image embeddings")
    
    # Initialize text encoder (CLIP-based)
    print("\n" + "="*60)
    print("Initializing Text Encoder (CLIP-based)")
    print("="*60)
    print("Using: CLIP ViT-B/32 (pretrained, frozen)")
    print("Note: CLIP embeddings are aligned - text and image in shared semantic space")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    text_encoder = TextEncoder(model_name='openai/clip-vit-base-patch32', freeze=True)
    text_encoder = text_encoder.to(device)
    text_embedding_dim = text_encoder.embedding_dim
    print(f"Text embedding dimension: {text_embedding_dim}")
    print(f"Image embedding dimension: {image_embedding_dim}")
    print(f"✓ Both embeddings are in the same {text_embedding_dim}-dim semantic space (CLIP-aligned)")
    
    # Create dataset
    print("\n" + "="*60)
    print("Creating Training Dataset")
    print("="*60)
    
    dataset = MultimodalDataset(
        train_sequences, metadata, user_to_idx, item_to_idx,
        text_encoder, item_image_embeddings, all_items,
        max_seq_len=50, negative_samples=4
    )
    
    train_loader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Initialize model
    print("\n" + "="*60)
    print("Initializing Improved Multimodal Recommender")
    print("="*60)
    print("Features:")
    print("  - Sequential user modeling (SASRec-style Transformer)")
    print("  - CLIP-aligned text and image embeddings")
    print("  - Late fusion (dot product)")
    print("  - Ranking loss (BPR)")
    
    model = MultimodalRecommender(
        n_items=len(item_to_idx),
        text_embedding_dim=text_embedding_dim,
        image_embedding_dim=image_embedding_dim,
        user_model_dim=128,
        max_seq_len=50,
        dropout=0.1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\n" + "="*60)
    print("Training Multimodal Recommender")
    print("="*60)
    
    model = train_multimodal_model(
        model, train_loader, item_to_idx, device, n_epochs=10, lr=0.001
    )
    
    # Evaluate
    print("\n" + "="*60)
    print("Evaluating Multimodal Recommender")
    print("="*60)
    
    results = evaluate_multimodal_model(
        model, text_encoder, train_sequences, val_interactions, test_interactions,
        metadata, item_image_embeddings, user_to_idx, item_to_idx, all_items, device
    )
    
    print("\nMultimodal Recommender Results:")
    for split in ['val', 'test']:
        print(f"\n{split.upper()} Set:")
        for metric in ['recall@5', 'recall@10', 'recall@20', 'ndcg@5', 'ndcg@10', 'ndcg@20']:
            print(f"  {metric}: {results[split][metric]:.4f}")
    
    # Save results
    multimodal_results = {
        'results': results,
        'model_state': model.state_dict(),
        'model_config': {
            'n_items': len(item_to_idx),
            'text_embedding_dim': text_embedding_dim,
            'image_embedding_dim': image_embedding_dim,
            'user_model_dim': 128,
            'max_seq_len': 50,
            'fusion_strategy': 'late_fusion_dot_product',
            'loss_type': 'bpr',
            'encoder_type': 'CLIP',
            'user_model_type': 'sequential_transformer',
            'embeddings_aligned': True
        }
    }
    
    with open('multimodal_results.pkl', 'wb') as f:
        pickle.dump(multimodal_results, f)
    
    print("\n" + "="*60)
    print("MULTIMODAL MODEL COMPLETE ✓")
    print("="*60)
    print("\nMultimodal Recommender Results (Test Set):")
    print(f"  Recall@10: {results['test']['recall@10']:.4f}")
    print(f"  NDCG@10: {results['test']['ndcg@10']:.4f}")
    print("\nResults saved to: multimodal_results.pkl")
    print("\nReady for: 07_evaluation.py")


if __name__ == "__main__":
    main()

