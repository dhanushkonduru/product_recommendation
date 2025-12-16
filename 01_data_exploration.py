"""
Data Exploration and Analysis
==============================
Comprehensive analysis of the Amazon Fashion dataset.
Explores reviews data, metadata, and prepares for recommendation system.
"""

import json
import gzip
from collections import Counter, defaultdict
from typing import Dict, List, Any
import pandas as pd


def load_full_jsonl_gz(filepath: str) -> List[Dict[str, Any]]:
    """Load all data from a gzipped JSON Lines file."""
    data = []
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return []
    return data


def analyze_reviews_full(reviews: List[Dict]) -> None:
    """Comprehensive analysis of reviews data."""
    print(f"\n{'='*60}")
    print("FULL REVIEWS DATA ANALYSIS")
    print(f"{'='*60}\n")
    
    if not reviews:
        return
    
    total_reviews = len(reviews)
    print(f"Total Reviews: {total_reviews:,}")
    
    # Essential fields
    essential_fields = ['reviewerID', 'asin', 'overall', 'unixReviewTime']
    print("\nEssential Fields Coverage:")
    for field in essential_fields:
        present = sum(1 for r in reviews if field in r and r[field] is not None)
        print(f"  - {field}: {present:,}/{total_reviews} ({present/total_reviews*100:.2f}%)")
    
    # Rating distribution
    ratings = [r.get('overall') for r in reviews if r.get('overall') is not None]
    if ratings:
        rating_dist = Counter(ratings)
        print(f"\nRating Distribution:")
        for rating in sorted(rating_dist.keys()):
            count = rating_dist[rating]
            print(f"  {rating} stars: {count:,} ({count/len(ratings)*100:.2f}%)")
    
    # Timestamp analysis
    timestamps = [r.get('unixReviewTime') for r in reviews if r.get('unixReviewTime') is not None]
    if timestamps:
        print(f"\nTimestamp Analysis:")
        print(f"  Min: {min(timestamps)} ({pd.to_datetime(min(timestamps), unit='s')})")
        print(f"  Max: {max(timestamps)} ({pd.to_datetime(max(timestamps), unit='s')})")
        print(f"  Span: {(max(timestamps) - min(timestamps)) / (365.25 * 24 * 3600):.2f} years")
    
    # User and item statistics
    users = set(r.get('reviewerID') for r in reviews if r.get('reviewerID'))
    items = set(r.get('asin') for r in reviews if r.get('asin'))
    
    print(f"\nEntity Statistics:")
    print(f"  Unique Users: {len(users):,}")
    print(f"  Unique Items: {len(items):,}")
    print(f"  Total Interactions: {total_reviews:,}")
    print(f"  Avg interactions per user: {total_reviews/len(users):.2f}")
    print(f"  Avg interactions per item: {total_reviews/len(items):.2f}")
    
    # User interaction distribution
    user_interactions = defaultdict(int)
    for r in reviews:
        uid = r.get('reviewerID')
        if uid:
            user_interactions[uid] += 1
    
    interaction_counts = list(user_interactions.values())
    print(f"\nUser Interaction Distribution:")
    print(f"  Min interactions per user: {min(interaction_counts)}")
    print(f"  Max interactions per user: {max(interaction_counts)}")
    print(f"  Median interactions per user: {pd.Series(interaction_counts).median():.2f}")
    print(f"  Mean interactions per user: {pd.Series(interaction_counts).mean():.2f}")
    
    # Item interaction distribution
    item_interactions = defaultdict(int)
    for r in reviews:
        iid = r.get('asin')
        if iid:
            item_interactions[iid] += 1
    
    item_counts = list(item_interactions.values())
    print(f"\nItem Interaction Distribution:")
    print(f"  Min interactions per item: {min(item_counts)}")
    print(f"  Max interactions per item: {max(item_counts)}")
    print(f"  Median interactions per item: {pd.Series(item_counts).median():.2f}")
    print(f"  Mean interactions per item: {pd.Series(item_counts).mean():.2f}")
    
    # Check for text fields
    has_review_text = sum(1 for r in reviews if r.get('reviewText'))
    print(f"\nOptional Fields:")
    print(f"  reviewText: {has_review_text:,}/{total_reviews} ({has_review_text/total_reviews*100:.2f}%)")
    print(f"  verified: {sum(1 for r in reviews if r.get('verified')):,}/{total_reviews} ({sum(1 for r in reviews if r.get('verified'))/total_reviews*100:.2f}%)")


def analyze_metadata_full(metadata: List[Dict]) -> None:
    """Comprehensive analysis of metadata."""
    print(f"\n{'='*60}")
    print("FULL METADATA ANALYSIS")
    print(f"{'='*60}\n")
    
    if not metadata:
        return
    
    total_items = len(metadata)
    print(f"Total Items in Metadata: {total_items:,}")
    
    # Text fields
    text_fields = ['title', 'brand', 'description']
    print("\nText Fields Coverage:")
    for field in text_fields:
        if field == 'description':
            # Description can be list or string
            present = sum(1 for m in metadata 
                         if field in m and m[field] and 
                         (isinstance(m[field], str) and m[field].strip() or 
                          isinstance(m[field], list) and len(m[field]) > 0))
        else:
            present = sum(1 for m in metadata if field in m and m[field])
        print(f"  - {field}: {present:,}/{total_items} ({present/total_items*100:.2f}%)")
    
    # Check for category (might be in different format)
    has_category = sum(1 for m in metadata if 'category' in m and m['category'])
    if has_category > 0:
        print(f"  - category: {has_category:,}/{total_items} ({has_category/total_items*100:.2f}%)")
    else:
        # Check if category info is in rank or other fields
        print(f"  - category: Not found in standard field")
    
    # Image URLs
    image_fields = ['imageURL', 'imageURLHighRes', 'imUrl']
    print(f"\nImage Fields Coverage:")
    for field in image_fields:
        present = sum(1 for m in metadata 
                     if field in m and m[field] and
                     (isinstance(m[field], list) and len(m[field]) > 0 or
                      isinstance(m[field], str) and m[field].strip()))
        if present > 0:
            print(f"  - {field}: {present:,}/{total_items} ({present/total_items*100:.2f}%)")
            # Count items with multiple images
            multi_image = sum(1 for m in metadata 
                            if field in m and isinstance(m[field], list) and len(m[field]) > 1)
            if multi_image > 0:
                print(f"    └─ Items with multiple images: {multi_image:,}")
    
    # Sample image URL
    sample_image = next((m.get('imageURLHighRes') or m.get('imageURL') 
                        for m in metadata 
                        if (m.get('imageURLHighRes') or m.get('imageURL'))), None)
    if sample_image:
        if isinstance(sample_image, list):
            print(f"\nSample Image URL: {sample_image[0]}")
        else:
            print(f"\nSample Image URL: {sample_image}")
    
    # Check for price
    has_price = sum(1 for m in metadata if m.get('price'))
    if has_price > 0:
        print(f"\nPrice Field: {has_price:,}/{total_items} ({has_price/total_items*100:.2f}%)")


def check_data_overlap(reviews: List[Dict], metadata: List[Dict]) -> None:
    """Check overlap between reviews and metadata."""
    print(f"\n{'='*60}")
    print("DATA OVERLAP ANALYSIS")
    print(f"{'='*60}\n")
    
    review_items = set(r.get('asin') for r in reviews if r.get('asin'))
    metadata_items = set(m.get('asin') for m in metadata if m.get('asin'))
    
    overlap = review_items & metadata_items
    only_reviews = review_items - metadata_items
    only_metadata = metadata_items - review_items
    
    print(f"Items in Reviews: {len(review_items):,}")
    print(f"Items in Metadata: {len(metadata_items):,}")
    print(f"Items in Both: {len(overlap):,}")
    print(f"Items only in Reviews: {len(only_reviews):,}")
    print(f"Items only in Metadata: {len(only_metadata):,}")
    print(f"\nCoverage: {len(overlap)/len(review_items)*100:.2f}% of reviewed items have metadata")


def main():
    """Main function for full dataset analysis."""
    print("="*60)
    print("DATA EXPLORATION AND ANALYSIS")
    print("="*60)
    
    reviews_file = "data/AMAZON_FASHION_5.json.gz"
    metadata_file = "data/meta_AMAZON_FASHION.json.gz"
    
    print(f"\nLoading full reviews dataset...")
    reviews = load_full_jsonl_gz(reviews_file)
    
    print(f"Loading full metadata dataset...")
    metadata = load_full_jsonl_gz(metadata_file)
    
    # Analyze datasets
    analyze_reviews_full(reviews)
    analyze_metadata_full(metadata)
    check_data_overlap(reviews, metadata)
    
    # Final summary
    print(f"\n{'='*60}")
    print("ESSENTIAL COLUMNS FOR RECOMMENDATION SYSTEM")
    print(f"{'='*60}\n")
    
    print("REVIEWS DATA (User Behavior) - ESSENTIAL:")
    print("  ✓ reviewerID - User identifier (100% coverage)")
    print("  ✓ asin - Product identifier (100% coverage)")
    print("  ✓ overall - Rating 1-5 stars (for implicit feedback)")
    print("  ✓ unixReviewTime - Timestamp for temporal train/val/test split")
    print("\n  Optional (not used in baseline):")
    print("  ? reviewText - Review text (for future text features)")
    print("  ? verified - Verified purchase flag (for filtering)")
    
    print("\nMETADATA (Product Features) - ESSENTIAL:")
    print("  ✓ asin - Product identifier (join key with reviews)")
    print("  ✓ title - Product title (primary text feature)")
    print("  ✓ brand - Brand name (text feature, ~50% coverage)")
    print("  ✓ imageURLHighRes - High-res image URLs (image feature, ~80% coverage)")
    print("\n  Optional:")
    print("  ? description - Product description (if available)")
    print("  ? category - Product category (if available)")
    print("  ? price - Product price (for future features)")
    
    print(f"\n{'='*60}")
    print("IMPLICIT FEEDBACK CONVERSION")
    print(f"{'='*60}\n")
    print("""
Strategy: Binary Implicit Feedback
----------------------------------
- If a user reviewed an item → interaction = 1 (positive signal)
- If no review exists → interaction = 0 (negative/unknown)
- We ignore the actual rating value (1-5 stars) for simplicity
- This creates a binary user-item interaction matrix

Rationale:
- Even low ratings indicate user engagement with the product
- For fashion items, any review suggests the user considered the item
- Simpler for baseline models (can extend to weighted later)

Alternative (for future):
- Threshold: only ratings >= 4 are positive
- Weighted: rating value as interaction strength
    """)
    
    print(f"\n{'='*60}")
    print("DATA EXPLORATION COMPLETE ✓")
    print("="*60)
    print("\nReady to proceed to: 02_data_preparation.py")
    print("We now understand:")
    print("  - Data schema and coverage")
    print("  - Essential vs optional fields")
    print("  - How to convert reviews to implicit feedback")
    print("  - Data overlap between reviews and metadata")


if __name__ == "__main__":
    main()

