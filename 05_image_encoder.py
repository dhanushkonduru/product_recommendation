"""
Image Feature Extraction (CLIP-based)
======================================
- Extract image URLs from metadata
- Download product images (subset for memory efficiency)
- Encode images using CLIP (frozen) - aligned with text embeddings
- Prepare image embeddings for multimodal fusion
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import requests
from io import BytesIO
import os
from typing import Dict, List, Set
from tqdm import tqdm
import math
from clip_encoder import ImageEncoder


def load_prepared_data():
    """Load prepared data."""
    with open('prepared_data.pkl', 'rb') as f:
        return pickle.load(f)


def download_image(url: str, timeout: int = 5) -> Image.Image:
    """Download and return PIL Image from URL."""
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    except Exception as e:
        return None


# ImageEncoder is now imported from clip_encoder.py
# This maintains backward compatibility while using CLIP internally


def download_item_images(metadata: Dict, all_items: Set, max_items: int = None) -> Dict[str, Image.Image]:
    """
    Download images for items.
    Only download for items in all_items (our dataset).
    """
    print("Downloading item images...")
    item_images = {}
    
    items_to_download = list(all_items)
    if max_items:
        items_to_download = items_to_download[:max_items]
    
    for item_id in tqdm(items_to_download, desc="Downloading images"):
        if item_id in metadata:
            image_url = metadata[item_id].get('image_url')
            if image_url:
                img = download_image(image_url)
                if img:
                    item_images[item_id] = img
                else:
                    item_images[item_id] = None  # Mark as failed
            else:
                item_images[item_id] = None  # No URL
    
    successful = sum(1 for img in item_images.values() if img is not None)
    print(f"Successfully downloaded {successful}/{len(items_to_download)} images")
    
    return item_images


def main():
    """Main function for image encoder setup."""
    print("="*60)
    print("IMAGE FEATURE EXTRACTION")
    print("="*60)
    
    # Load data
    print("\nLoading prepared data...")
    data = load_prepared_data()
    
    metadata = data['metadata']
    all_items = data['all_items']
    
    print(f"Items in dataset: {len(all_items)}")
    
    # Download images (only for items in our dataset - small subset)
    print("\n" + "="*60)
    print("Downloading Images")
    print("="*60)
    print("Note: Only downloading images for items in our dataset (19 items)")
    
    item_images = download_item_images(metadata, all_items)
    
    # Initialize image encoder (CLIP-based)
    print("\n" + "="*60)
    print("Initializing Image Encoder (CLIP-based)")
    print("="*60)
    print("Using: CLIP ViT-B/32 (pretrained, frozen)")
    print("Note: CLIP embeddings are aligned with text embeddings in shared semantic space")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    image_encoder = ImageEncoder(model_name='openai/clip-vit-base-patch32', freeze=True)
    image_encoder = image_encoder.to(device)
    image_embedding_dim = image_encoder.embedding_dim
    print(f"Image embedding dimension: {image_embedding_dim} (aligned with text embeddings)")
    
    # Encode all item images
    print("\n" + "="*60)
    print("Encoding Item Images")
    print("="*60)
    
    item_image_embeddings = {}
    items_with_images = []
    images_list = []
    
    for item_id, img in item_images.items():
        if img is not None:
            items_with_images.append(item_id)
            images_list.append(img)
    
    # Encode in batches
    batch_size = 8
    all_embeddings = []
    for i in tqdm(range(0, len(images_list), batch_size), desc="Encoding images"):
        batch_images = images_list[i:i+batch_size]
        batch_embeddings = image_encoder.encode(batch_images)
        all_embeddings.append(batch_embeddings.cpu())
    
    if all_embeddings:
        embeddings = torch.cat(all_embeddings, dim=0)
        
        for idx, item_id in enumerate(items_with_images):
            item_image_embeddings[item_id] = embeddings[idx]
    
    print(f"Encoded {len(item_image_embeddings)} item images")
    
    # Save image embeddings
    image_data = {
        'item_image_embeddings': item_image_embeddings,
        'image_encoder_config': {
            'model_name': 'openai/clip-vit-base-patch32',
            'embedding_dim': image_embedding_dim,
            'frozen': True,
            'encoder_type': 'CLIP'
        }
    }
    
    with open('image_encoder_data.pkl', 'wb') as f:
        pickle.dump(image_data, f)
    
    print("\n" + "="*60)
    print("IMAGE ENCODER COMPLETE ✓")
    print("="*60)
    print(f"\nImage embeddings saved for {len(item_image_embeddings)} items")
    print("Results saved to: image_encoder_data.pkl")
    print("\nReady for: 06_multimodal_model.py")


if __name__ == "__main__":
    main()

