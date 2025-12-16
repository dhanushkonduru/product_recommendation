"""
Baseline Recommendation Models
===============================
Implements and evaluates baseline recommendation methods:
1. Popularity-based recommender
2. Matrix Factorization

Evaluation metrics: Recall@K, NDCG@K
"""

import pickle
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import math


def load_prepared_data():
    """Load prepared data from step 2."""
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
            dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1) = 0
    
    # Ideal DCG
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_recommender(recommender_func, train_interactions: Dict, 
                        val_interactions: Dict, test_interactions: Dict,
                        all_items: Set, k_values: List[int] = [5, 10, 20]) -> Dict:
    """
    Evaluate a recommender function.
    recommender_func: function(user_id) -> List[item_id] (ranked recommendations)
    """
    results = {}
    
    for split_name, interactions in [('val', val_interactions), ('test', test_interactions)]:
        results[split_name] = {}
        for k in k_values:
            results[split_name][f'recall@{k}'] = []
            results[split_name][f'ndcg@{k}'] = []
        
        # Evaluate for each user
        for user_id, relevant_items in interactions.items():
            if len(relevant_items) == 0:
                continue
            
            # Get recommendations
            recommended = recommender_func(user_id)
            
            # Calculate metrics
            for k in k_values:
                results[split_name][f'recall@{k}'].append(
                    recall_at_k(recommended, relevant_items, k)
                )
                results[split_name][f'ndcg@{k}'].append(
                    ndcg_at_k(recommended, relevant_items, k)
                )
        
        # Average metrics
        for k in k_values:
            results[split_name][f'recall@{k}'] = np.mean(results[split_name][f'recall@{k}'])
            results[split_name][f'ndcg@{k}'] = np.mean(results[split_name][f'ndcg@{k}'])
    
    return results


class PopularityRecommender:
    """Popularity-based recommender: recommends most popular items."""
    
    def __init__(self, train_interactions: Dict, all_items: Set):
        # Count item popularity in training set
        item_counts = defaultdict(int)
        for user_items in train_interactions.values():
            for item in user_items:
                item_counts[item] += 1
        
        # Sort items by popularity (descending)
        sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
        self.popular_items = [item for item, count in sorted_items]
        
        # Add any items not in training set (with 0 popularity)
        for item in all_items:
            if item not in item_counts:
                self.popular_items.append(item)
        
        print(f"Popularity recommender initialized with {len(self.popular_items)} items")
        print(f"  Most popular item: {self.popular_items[0]} ({item_counts[self.popular_items[0]]} interactions)")
    
    def recommend(self, user_id: str, k: int = None) -> List[str]:
        """Recommend most popular items."""
        if k is None:
            return self.popular_items.copy()
        return self.popular_items[:k]


class SimpleMatrixFactorization:
    """
    Simple Matrix Factorization using SGD.
    Learns user and item embeddings.
    """
    
    def __init__(self, n_users: int, n_items: int, n_factors: int = 50, 
                 learning_rate: float = 0.01, reg: float = 0.01, n_epochs: int = 50):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.n_epochs = n_epochs
        
        # Initialize embeddings
        self.user_embeddings = np.random.normal(0, 0.1, (n_users, n_factors))
        self.item_embeddings = np.random.normal(0, 0.1, (n_items, n_factors))
        
        # Global bias
        self.global_bias = 0.0
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
    
    def fit(self, train_interactions: Dict, user_to_idx: Dict, item_to_idx: Dict):
        """Train the model using SGD."""
        print(f"\nTraining Matrix Factorization...")
        print(f"  Factors: {self.n_factors}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Regularization: {self.reg}")
        print(f"  Epochs: {self.n_epochs}")
        
        # Convert interactions to list of (user_idx, item_idx) pairs
        interactions = []
        for user_id, items in train_interactions.items():
            if user_id in user_to_idx:
                user_idx = user_to_idx[user_id]
                for item_id in items:
                    if item_id in item_to_idx:
                        item_idx = item_to_idx[item_id]
                        interactions.append((user_idx, item_idx))
        
        interactions = np.array(interactions)
        n_interactions = len(interactions)
        
        # Calculate global bias
        self.global_bias = 1.0  # All interactions are positive (implicit feedback)
        
        # Training loop
        for epoch in range(self.n_epochs):
            # Shuffle interactions
            np.random.shuffle(interactions)
            
            total_error = 0.0
            
            for user_idx, item_idx in interactions:
                # Predict
                pred = (self.global_bias + 
                       self.user_biases[user_idx] + 
                       self.item_biases[item_idx] +
                       np.dot(self.user_embeddings[user_idx], self.item_embeddings[item_idx]))
                
                # Error (target is 1 for implicit feedback)
                error = 1.0 - pred
                total_error += error ** 2
                
                # Update biases
                user_bias_update = self.learning_rate * (error - self.reg * self.user_biases[user_idx])
                item_bias_update = self.learning_rate * (error - self.reg * self.item_biases[item_idx])
                
                self.user_biases[user_idx] += user_bias_update
                self.item_biases[item_idx] += item_bias_update
                
                # Update embeddings
                user_emb = self.user_embeddings[user_idx].copy()
                item_emb = self.item_embeddings[item_idx].copy()
                
                self.user_embeddings[user_idx] += self.learning_rate * (
                    error * item_emb - self.reg * user_emb
                )
                self.item_embeddings[item_idx] += self.learning_rate * (
                    error * user_emb - self.reg * item_emb
                )
            
            if (epoch + 1) % 10 == 0:
                avg_error = total_error / n_interactions
                print(f"  Epoch {epoch + 1}/{self.n_epochs}, Avg Error: {avg_error:.4f}")
        
        print("Training complete!")
    
    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predict interaction score."""
        return (self.global_bias + 
               self.user_biases[user_idx] + 
               self.item_biases[item_idx] +
               np.dot(self.user_embeddings[user_idx], self.item_embeddings[item_idx]))
    
    def recommend(self, user_id: str, user_to_idx: Dict, item_to_idx: Dict, 
                 idx_to_item: Dict, train_interactions: Dict, k: int = None) -> List[str]:
        """Recommend items for a user."""
        if user_id not in user_to_idx:
            return []
        
        user_idx = user_to_idx[user_id]
        seen_items = train_interactions.get(user_id, set())
        
        # Score all items
        scores = []
        for item_id, item_idx in item_to_idx.items():
            if item_id not in seen_items:  # Don't recommend seen items
                score = self.predict(user_idx, item_idx)
                scores.append((item_id, score))
        
        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        recommended = [item_id for item_id, score in scores]
        
        if k is None:
            return recommended
        return recommended[:k]


def main():
    """Main function for baseline evaluation."""
    print("="*60)
    print("BASELINE RECOMMENDATION MODELS")
    print("="*60)
    
    # Load prepared data
    print("\nLoading prepared data...")
    data = load_prepared_data()
    
    train_interactions = data['train_interactions']
    val_interactions = data['val_interactions']
    test_interactions = data['test_interactions']
    all_items = data['all_items']
    user_to_idx = data['user_to_idx']
    item_to_idx = data['item_to_idx']
    idx_to_item = data['idx_to_item']
    
    print(f"Loaded data:")
    print(f"  Users: {len(user_to_idx)}")
    print(f"  Items: {len(item_to_idx)}")
    print(f"  Train interactions: {sum(len(items) for items in train_interactions.values())}")
    print(f"  Val interactions: {sum(len(items) for items in val_interactions.values())}")
    print(f"  Test interactions: {sum(len(items) for items in test_interactions.values())}")
    
    # Baseline 1: Popularity-based
    print("\n" + "="*60)
    print("BASELINE 1: Popularity-Based Recommender")
    print("="*60)
    
    pop_recommender = PopularityRecommender(train_interactions, all_items)
    
    def pop_recommend_func(user_id):
        return pop_recommender.recommend(user_id)
    
    pop_results = evaluate_recommender(
        pop_recommend_func, train_interactions, val_interactions, test_interactions, all_items
    )
    
    print("\nPopularity-Based Results:")
    for split in ['val', 'test']:
        print(f"\n{split.upper()} Set:")
        for metric in ['recall@5', 'recall@10', 'recall@20', 'ndcg@5', 'ndcg@10', 'ndcg@20']:
            print(f"  {metric}: {pop_results[split][metric]:.4f}")
    
    # Baseline 2: Matrix Factorization
    print("\n" + "="*60)
    print("BASELINE 2: Matrix Factorization")
    print("="*60)
    
    mf_model = SimpleMatrixFactorization(
        n_users=len(user_to_idx),
        n_items=len(item_to_idx),
        n_factors=32,  # Reduced for small dataset
        learning_rate=0.01,
        reg=0.01,
        n_epochs=50
    )
    
    mf_model.fit(train_interactions, user_to_idx, item_to_idx)
    
    def mf_recommend_func(user_id):
        return mf_model.recommend(user_id, user_to_idx, item_to_idx, idx_to_item, train_interactions)
    
    mf_results = evaluate_recommender(
        mf_recommend_func, train_interactions, val_interactions, test_interactions, all_items
    )
    
    print("\nMatrix Factorization Results:")
    for split in ['val', 'test']:
        print(f"\n{split.upper()} Set:")
        for metric in ['recall@5', 'recall@10', 'recall@20', 'ndcg@5', 'ndcg@10', 'ndcg@20']:
            print(f"  {metric}: {mf_results[split][metric]:.4f}")
    
    # Save results
    baseline_results = {
        'popularity': pop_results,
        'matrix_factorization': mf_results,
        'mf_model': mf_model  # Save model for later use
    }
    
    with open('baseline_results.pkl', 'wb') as f:
        pickle.dump(baseline_results, f)
    
    print("\n" + "="*60)
    print("BASELINE MODELS COMPLETE ✓")
    print("="*60)
    print("\nBaseline Results Summary:")
    print("\nPopularity-Based (Test Set):")
    print(f"  Recall@10: {pop_results['test']['recall@10']:.4f}")
    print(f"  NDCG@10: {pop_results['test']['ndcg@10']:.4f}")
    print("\nMatrix Factorization (Test Set):")
    print(f"  Recall@10: {mf_results['test']['recall@10']:.4f}")
    print(f"  NDCG@10: {mf_results['test']['ndcg@10']:.4f}")
    print("\nResults saved to: baseline_results.pkl")
    print("\nReady for: 04_text_based_model.py")


if __name__ == "__main__":
    main()

