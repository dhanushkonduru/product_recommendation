"""
Text-Based Recommendation Model
================================
- Construct product text from title + brand
- Encode using pretrained DistilBERT (frozen)
- Fuse text embeddings with user embeddings
- Train ranking model with binary cross-entropy loss
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Set, Tuple
from tqdm import tqdm
import math
from clip_encoder import TextEncoder


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


class TextBasedRecommender(nn.Module):
    """
    Text-based recommender that fuses text embeddings with user embeddings.
    """
    
    def __init__(self, n_users: int, text_embedding_dim: int, hidden_dim: int = 128, 
                 output_dim: int = 64, dropout: float = 0.2):
        super(TextBasedRecommender, self).__init__()
        
        # User embedding
        self.user_embedding = nn.Embedding(n_users, hidden_dim)
        
        # Text projection (to match hidden_dim)
        self.text_projection = nn.Linear(text_embedding_dim, hidden_dim)
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, 1)  # Output score
        )
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
    
    def forward(self, user_ids: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        user_ids: [batch_size]
        text_embeddings: [batch_size, text_embedding_dim]
        """
        # Get user embeddings
        user_emb = self.user_embedding(user_ids)  # [batch_size, hidden_dim]
        
        # Project text embeddings
        text_emb = self.text_projection(text_embeddings)  # [batch_size, hidden_dim]
        
        # Concatenate and fuse
        fused = torch.cat([user_emb, text_emb], dim=1)  # [batch_size, hidden_dim * 2]
        score = self.fusion(fused)  # [batch_size, 1]
        
        return score.squeeze(1)  # [batch_size]


class RecommendationDataset(Dataset):
    """Dataset for training recommendation model."""
    
    def __init__(self, train_interactions: Dict, metadata: Dict, 
                 user_to_idx: Dict, item_to_idx: Dict, 
                 text_encoder: TextEncoder, all_items: Set, 
                 negative_samples: int = 4):
        self.train_interactions = train_interactions
        self.metadata = metadata
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.text_encoder = text_encoder
        self.all_items = all_items
        self.negative_samples = negative_samples
        
        # Pre-encode all item texts
        print("Pre-encoding item texts...")
        self.item_text_embeddings = {}
        items_with_text = []
        texts = []
        item_ids = []
        
        for item_id in all_items:
            if item_id in metadata and metadata[item_id].get('product_text'):
                items_with_text.append(item_id)
                texts.append(metadata[item_id]['product_text'])
                item_ids.append(item_id)
        
        # Encode in batches
        batch_size = 32
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = text_encoder.encode(batch_texts)
            all_embeddings.append(batch_embeddings.cpu())
        
        embeddings = torch.cat(all_embeddings, dim=0)
        
        for idx, item_id in enumerate(item_ids):
            self.item_text_embeddings[item_id] = embeddings[idx]
        
        print(f"Encoded {len(self.item_text_embeddings)} items")
        
        # Create training pairs
        self.samples = []
        for user_id, positive_items in train_interactions.items():
            if user_id not in user_to_idx:
                continue
            for item_id in positive_items:
                if item_id in self.item_text_embeddings:
                    self.samples.append((user_id, item_id, 1))  # Positive sample
        
        print(f"Created {len(self.samples)} positive samples")
    
    def __len__(self):
        return len(self.samples) * (1 + self.negative_samples)
    
    def __getitem__(self, idx):
        # Map to positive sample
        pos_idx = idx // (1 + self.negative_samples)
        is_positive = (idx % (1 + self.negative_samples)) == 0
        
        user_id, pos_item_id, label = self.samples[pos_idx]
        user_idx = self.user_to_idx[user_id]
        
        if is_positive:
            item_id = pos_item_id
            label = 1.0
        else:
            # Negative sampling: sample random item not in user's interactions
            user_items = self.train_interactions.get(user_id, set())
            negative_items = list(self.all_items - user_items)
            if len(negative_items) == 0:
                negative_items = list(self.all_items)
            
            # Only sample from items with text embeddings
            negative_items = [item for item in negative_items if item in self.item_text_embeddings]
            if len(negative_items) == 0:
                item_id = pos_item_id  # Fallback
            else:
                item_id = np.random.choice(negative_items)
            label = 0.0
        
        text_embedding = self.item_text_embeddings[item_id]
        
        return {
            'user_idx': torch.tensor(user_idx, dtype=torch.long),
            'text_embedding': text_embedding,
            'label': torch.tensor(label, dtype=torch.float32)
        }


def train_text_model(model: TextBasedRecommender, train_loader: DataLoader, 
                    device: torch.device, n_epochs: int = 10, lr: float = 0.001):
    """Train the text-based recommendation model."""
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
        for batch in pbar:
            user_indices = batch['user_idx'].to(device)
            text_embeddings = batch['text_embedding'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            scores = model(user_indices, text_embeddings)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch+1}/{n_epochs}, Average Loss: {avg_loss:.4f}")
    
    return model


def evaluate_text_model(model: TextBasedRecommender, text_encoder: TextEncoder,
                       train_interactions: Dict, val_interactions: Dict, 
                       test_interactions: Dict, metadata: Dict,
                       user_to_idx: Dict, item_to_idx: Dict, all_items: Set,
                       device: torch.device, k_values: List[int] = [5, 10, 20]) -> Dict:
    """Evaluate text-based model."""
    model.eval()
    text_encoder.eval()
    
    results = {}
    
    for split_name, interactions in [('val', val_interactions), ('test', test_interactions)]:
        results[split_name] = {f'recall@{k}': [] for k in k_values}
        results[split_name].update({f'ndcg@{k}': [] for k in k_values})
        
        with torch.no_grad():
            for user_id, relevant_items in tqdm(interactions.items(), desc=f"Evaluating {split_name}"):
                if user_id not in user_to_idx:
                    continue
                
                user_idx = user_to_idx[user_id]
                seen_items = train_interactions.get(user_id, set())
                
                # Score all items
                scores = []
                item_ids = []
                text_embeddings_list = []
                
                for item_id in all_items:
                    if item_id not in seen_items and item_id in metadata:
                        if metadata[item_id].get('product_text'):
                            item_ids.append(item_id)
                            text_embeddings_list.append(metadata[item_id]['product_text'])
                
                if len(item_ids) == 0:
                    continue
                
                # Encode texts in batches
                batch_size = 32
                all_text_emb = []
                for i in range(0, len(text_embeddings_list), batch_size):
                    batch_texts = text_embeddings_list[i:i+batch_size]
                    batch_emb = text_encoder.encode(batch_texts)
                    all_text_emb.append(batch_emb)
                
                text_embeddings = torch.cat(all_text_emb, dim=0).to(device)
                user_indices = torch.tensor([user_idx] * len(item_ids), dtype=torch.long).to(device)
                
                # Get scores
                item_scores = model(user_indices, text_embeddings).cpu().numpy()
                
                # Rank items
                ranked_items = [item_ids[i] for i in np.argsort(item_scores)[::-1]]
                
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
    """Main function for text encoder."""
    print("="*60)
    print("TEXT-BASED RECOMMENDATION MODEL")
    print("="*60)
    
    # Load data
    print("\nLoading prepared data...")
    data = load_prepared_data()
    
    train_interactions = data['train_interactions']
    val_interactions = data['val_interactions']
    test_interactions = data['test_interactions']
    metadata = data['metadata']
    user_to_idx = data['user_to_idx']
    item_to_idx = data['item_to_idx']
    all_items = data['all_items']
    
    print(f"Loaded data:")
    print(f"  Users: {len(user_to_idx)}")
    print(f"  Items: {len(all_items)}")
    
    # Initialize text encoder (CLIP-based, frozen)
    print("\n" + "="*60)
    print("Initializing Text Encoder (CLIP-based)")
    print("="*60)
    print("Using: CLIP ViT-B/32 (pretrained, frozen)")
    print("Note: CLIP embeddings are aligned with image embeddings in shared semantic space")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    text_encoder = TextEncoder(model_name='openai/clip-vit-base-patch32', freeze=True)
    text_encoder = text_encoder.to(device)
    text_embedding_dim = text_encoder.embedding_dim
    print(f"Text embedding dimension: {text_embedding_dim} (aligned with image embeddings)")
    
    # Create dataset
    print("\n" + "="*60)
    print("Creating Training Dataset")
    print("="*60)
    
    dataset = RecommendationDataset(
        train_interactions, metadata, user_to_idx, item_to_idx,
        text_encoder, all_items, negative_samples=4
    )
    
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    
    # Initialize model
    print("\n" + "="*60)
    print("Initializing Text-Based Recommender")
    print("="*60)
    
    model = TextBasedRecommender(
        n_users=len(user_to_idx),
        text_embedding_dim=text_embedding_dim,
        hidden_dim=128,
        output_dim=64,
        dropout=0.2
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\n" + "="*60)
    print("Training Text-Based Recommender")
    print("="*60)
    
    model = train_text_model(model, train_loader, device, n_epochs=10, lr=0.001)
    
    # Evaluate
    print("\n" + "="*60)
    print("Evaluating Text-Based Recommender")
    print("="*60)
    
    results = evaluate_text_model(
        model, text_encoder, train_interactions, val_interactions, test_interactions,
        metadata, user_to_idx, item_to_idx, all_items, device
    )
    
    print("\nText-Based Recommender Results:")
    for split in ['val', 'test']:
        print(f"\n{split.upper()} Set:")
        for metric in ['recall@5', 'recall@10', 'recall@20', 'ndcg@5', 'ndcg@10', 'ndcg@20']:
            print(f"  {metric}: {results[split][metric]:.4f}")
    
    # Save results
    text_results = {
        'results': results,
        'model_state': model.state_dict(),
        'text_encoder': text_encoder,
        'model_config': {
            'n_users': len(user_to_idx),
            'text_embedding_dim': text_embedding_dim,
            'hidden_dim': 128,
            'output_dim': 64
        }
    }
    
    with open('text_encoder_results.pkl', 'wb') as f:
        pickle.dump(text_results, f)
    
    print("\n" + "="*60)
    print("TEXT-BASED MODEL COMPLETE ✓")
    print("="*60)
    print("\nText-Based Recommender Results (Test Set):")
    print(f"  Recall@10: {results['test']['recall@10']:.4f}")
    print(f"  NDCG@10: {results['test']['ndcg@10']:.4f}")
    print("\nResults saved to: text_encoder_results.pkl")
    print("\nReady for: 05_image_encoder.py")


if __name__ == "__main__":
    main()

