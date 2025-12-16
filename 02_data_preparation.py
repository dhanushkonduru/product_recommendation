"""
Data Preparation and Preprocessing
==================================
- Filter users and items with minimum interactions
- Sort interactions by timestamp
- Build user interaction sequences
- Temporal train/validation/test split
- Prepare metadata for multimodal features
"""

import json
import gzip
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import pandas as pd
import numpy as np


def load_jsonl_gz(filepath: str) -> List[Dict]:
    """Load all data from a gzipped JSON Lines file."""
    data = []
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def prepare_reviews_data(reviews: List[Dict]) -> pd.DataFrame:
    """Convert reviews to DataFrame and prepare for processing."""
    # Extract essential fields
    data = []
    for r in reviews:
        data.append({
            'user_id': r.get('reviewerID'),
            'item_id': r.get('asin'),
            'rating': r.get('overall'),
            'timestamp': r.get('unixReviewTime')
        })
    
    df = pd.DataFrame(data)
    
    # Remove any nulls
    df = df.dropna()
    
    # Convert to implicit feedback (binary: 1 if reviewed, 0 otherwise)
    # For now, we'll mark all reviews as positive interactions
    df['interaction'] = 1
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    return df


def filter_users_items(df: pd.DataFrame, min_user_interactions: int = 5, 
                       min_item_interactions: int = 5) -> pd.DataFrame:
    """
    Filter users and items to ensure minimum interactions.
    Note: This is a 5-core dataset, so filtering may be minimal.
    """
    print(f"Initial data: {len(df)} interactions, {df['user_id'].nunique()} users, {df['item_id'].nunique()} items")
    
    # Filter users with at least min_user_interactions
    user_counts = df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= min_user_interactions].index
    df_filtered = df[df['user_id'].isin(valid_users)].copy()
    
    print(f"After user filtering: {len(df_filtered)} interactions, {df_filtered['user_id'].nunique()} users")
    
    # Filter items with at least min_item_interactions
    item_counts = df_filtered['item_id'].value_counts()
    valid_items = item_counts[item_counts >= min_item_interactions].index
    df_filtered = df_filtered[df_filtered['item_id'].isin(valid_items)].copy()
    
    print(f"After item filtering: {len(df_filtered)} interactions, {df_filtered['item_id'].nunique()} items")
    
    # Re-filter users in case some lost all their items
    user_counts = df_filtered['user_id'].value_counts()
    valid_users = user_counts[user_counts >= min_user_interactions].index
    df_filtered = df_filtered[df_filtered['user_id'].isin(valid_users)].copy()
    
    print(f"Final filtered data: {len(df_filtered)} interactions, {df_filtered['user_id'].nunique()} users, {df_filtered['item_id'].nunique()} items")
    
    return df_filtered


def build_user_sequences(df: pd.DataFrame) -> Dict[str, List[Tuple[str, int]]]:
    """
    Build interaction sequences per user, sorted by timestamp.
    Returns: {user_id: [(item_id, timestamp), ...]}
    """
    user_sequences = defaultdict(list)
    
    for _, row in df.iterrows():
        user_sequences[row['user_id']].append((row['item_id'], row['timestamp']))
    
    # Sort each user's sequence by timestamp
    for user_id in user_sequences:
        user_sequences[user_id].sort(key=lambda x: x[1])
    
    return dict(user_sequences)


def temporal_split(user_sequences: Dict[str, List[Tuple[str, int]]], 
                   train_ratio: float = 0.7, 
                   val_ratio: float = 0.15) -> Tuple[Dict, Dict, Dict]:
    """
    Split data temporally per user.
    For each user, use first train_ratio for training, next val_ratio for validation, rest for test.
    """
    train_sequences = {}
    val_sequences = {}
    test_sequences = {}
    
    for user_id, sequence in user_sequences.items():
        n = len(sequence)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_sequences[user_id] = sequence[:train_end]
        val_sequences[user_id] = sequence[train_end:val_end]
        test_sequences[user_id] = sequence[val_end:]
    
    # Statistics
    train_interactions = sum(len(seq) for seq in train_sequences.values())
    val_interactions = sum(len(seq) for seq in val_sequences.values())
    test_interactions = sum(len(seq) for seq in test_sequences.values())
    
    print(f"\nTemporal Split Statistics:")
    print(f"  Train: {train_interactions} interactions ({train_interactions/(train_interactions+val_interactions+test_interactions)*100:.1f}%)")
    print(f"  Validation: {val_interactions} interactions ({val_interactions/(train_interactions+val_interactions+test_interactions)*100:.1f}%)")
    print(f"  Test: {test_interactions} interactions ({test_interactions/(train_interactions+val_interactions+test_interactions)*100:.1f}%)")
    
    return train_sequences, val_sequences, test_sequences


def create_interaction_matrices(train_sequences: Dict, val_sequences: Dict, 
                                test_sequences: Dict) -> Tuple[Set, Set, Dict]:
    """
    Create sets of items and users, and prepare interaction dictionaries.
    Returns: (all_items, all_users, interaction_dicts)
    """
    # Collect all items and users
    all_items = set()
    all_users = set()
    
    for sequences in [train_sequences, val_sequences, test_sequences]:
        for user_id, sequence in sequences.items():
            all_users.add(user_id)
            for item_id, _ in sequence:
                all_items.add(item_id)
    
    print(f"\nUnique entities:")
    print(f"  Users: {len(all_users)}")
    print(f"  Items: {len(all_items)}")
    
    # Create interaction dictionaries for easy lookup
    # Format: {user_id: set(item_ids)}
    train_interactions = {user_id: set(item_id for item_id, _ in seq) 
                         for user_id, seq in train_sequences.items()}
    val_interactions = {user_id: set(item_id for item_id, _ in seq) 
                       for user_id, seq in val_sequences.items()}
    test_interactions = {user_id: set(item_id for item_id, _ in seq) 
                        for user_id, seq in test_sequences.items()}
    
    return all_items, all_users, {
        'train': train_interactions,
        'val': val_interactions,
        'test': test_interactions
    }


def prepare_metadata(metadata: List[Dict]) -> Dict[str, Dict]:
    """
    Prepare metadata dictionary indexed by asin.
    Extract: title, brand, imageURLHighRes
    """
    metadata_dict = {}
    
    for item in metadata:
        asin = item.get('asin')
        if not asin:
            continue
        
        # Extract text features
        title = item.get('title', '').strip()
        brand = item.get('brand', '').strip()
        
        # Construct product text: title + brand
        product_text = f"{title} {brand}".strip()
        
        # Extract image URL (use first high-res image if available)
        image_url = None
        if 'imageURLHighRes' in item and item['imageURLHighRes']:
            if isinstance(item['imageURLHighRes'], list) and len(item['imageURLHighRes']) > 0:
                image_url = item['imageURLHighRes'][0]
        elif 'imageURL' in item and item['imageURL']:
            if isinstance(item['imageURL'], list) and len(item['imageURL']) > 0:
                image_url = item['imageURL'][0]
        
        metadata_dict[asin] = {
            'title': title,
            'brand': brand,
            'product_text': product_text,
            'image_url': image_url
        }
    
    print(f"\nMetadata prepared: {len(metadata_dict)} items")
    print(f"  Items with text: {sum(1 for m in metadata_dict.values() if m['product_text'])}")
    print(f"  Items with images: {sum(1 for m in metadata_dict.values() if m['image_url'])}")
    
    return metadata_dict


def main():
    """Main function for data preparation."""
    print("="*60)
    print("DATA PREPARATION AND PREPROCESSING")
    print("="*60)
    
    # Load data
    print("\nLoading reviews data...")
    reviews = load_jsonl_gz("data/AMAZON_FASHION_5.json.gz")
    print(f"Loaded {len(reviews)} reviews")
    
    print("\nLoading metadata...")
    metadata = load_jsonl_gz("data/meta_AMAZON_FASHION.json.gz")
    print(f"Loaded {len(metadata)} metadata entries")
    
    # Prepare reviews
    print("\n" + "-"*60)
    print("Preparing reviews data...")
    df = prepare_reviews_data(reviews)
    
    # Filter users and items
    print("\n" + "-"*60)
    print("Filtering users and items...")
    df_filtered = filter_users_items(df, min_user_interactions=5, min_item_interactions=5)
    
    # Build user sequences
    print("\n" + "-"*60)
    print("Building user interaction sequences...")
    user_sequences = build_user_sequences(df_filtered)
    print(f"Built sequences for {len(user_sequences)} users")
    
    # Show sequence length statistics
    seq_lengths = [len(seq) for seq in user_sequences.values()]
    print(f"  Min sequence length: {min(seq_lengths)}")
    print(f"  Max sequence length: {max(seq_lengths)}")
    print(f"  Mean sequence length: {np.mean(seq_lengths):.2f}")
    print(f"  Median sequence length: {np.median(seq_lengths):.2f}")
    
    # Temporal split
    print("\n" + "-"*60)
    print("Creating temporal train/validation/test split...")
    train_sequences, val_sequences, test_sequences = temporal_split(
        user_sequences, train_ratio=0.7, val_ratio=0.15
    )
    
    # Create interaction matrices
    print("\n" + "-"*60)
    print("Creating interaction matrices...")
    all_items, all_users, interaction_dicts = create_interaction_matrices(
        train_sequences, val_sequences, test_sequences
    )
    
    # Prepare metadata
    print("\n" + "-"*60)
    print("Preparing metadata...")
    metadata_dict = prepare_metadata(metadata)
    
    # Save prepared data
    print("\n" + "-"*60)
    print("Saving prepared data...")
    
    prepared_data = {
        'train_sequences': train_sequences,
        'val_sequences': val_sequences,
        'test_sequences': test_sequences,
        'train_interactions': interaction_dicts['train'],
        'val_interactions': interaction_dicts['val'],
        'test_interactions': interaction_dicts['test'],
        'all_items': all_items,
        'all_users': all_users,
        'metadata': metadata_dict,
        'user_to_idx': {user_id: idx for idx, user_id in enumerate(sorted(all_users))},
        'item_to_idx': {item_id: idx for idx, item_id in enumerate(sorted(all_items))}
    }
    
    # Create reverse mappings
    prepared_data['idx_to_user'] = {idx: user_id for user_id, idx in prepared_data['user_to_idx'].items()}
    prepared_data['idx_to_item'] = {idx: item_id for item_id, idx in prepared_data['item_to_idx'].items()}
    
    with open('prepared_data.pkl', 'wb') as f:
        pickle.dump(prepared_data, f)
    
    print("Saved to: prepared_data.pkl")
    
    # Summary
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE ✓")
    print("="*60)
    print("\nPrepared data summary:")
    print(f"  Users: {len(all_users)}")
    print(f"  Items: {len(all_items)}")
    print(f"  Train interactions: {sum(len(seq) for seq in train_sequences.values())}")
    print(f"  Val interactions: {sum(len(seq) for seq in val_sequences.values())}")
    print(f"  Test interactions: {sum(len(seq) for seq in test_sequences.values())}")
    print(f"  Items with metadata: {len([i for i in all_items if i in metadata_dict])}")
    print("\nReady for: 03_baseline_models.py")


if __name__ == "__main__":
    main()

