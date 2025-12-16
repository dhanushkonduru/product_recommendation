"""
CLIP-based Multimodal Encoder
==============================
Unified encoder for text and images using CLIP (Contrastive Language-Image Pre-training).
Both modalities produce embeddings in the same semantic space.

Key advantages:
- Text and image embeddings are aligned (same semantic space)
- No need for separate projections to align modalities
- Better multimodal understanding
- Same interface as previous encoders (backward compatible)
"""

import torch
import torch.nn as nn
from typing import List, Union
from PIL import Image
import torchvision.transforms as transforms


class CLIPEncoder(nn.Module):
    """
    CLIP-based encoder for text and images.
    Both modalities produce embeddings in the same semantic space.
    
    This replaces the separate DistilBERT (text) and ResNet18 (image) encoders
    with a unified CLIP model that was trained to align text and images.
    """
    
    def __init__(self, model_name: str = 'openai/clip-vit-base-patch32', freeze: bool = True):
        """
        Initialize CLIP encoder.
        
        Args:
            model_name: HuggingFace model name for CLIP
                - 'openai/clip-vit-base-patch32': 512-dim embeddings (default, faster)
                - 'openai/clip-vit-large-patch14': 768-dim embeddings (slower, better)
            freeze: Whether to freeze CLIP weights (default: True)
        """
        super(CLIPEncoder, self).__init__()
        
        try:
            from transformers import CLIPProcessor, CLIPModel
        except ImportError:
            raise ImportError(
                "CLIP requires transformers>=4.20.0. Install with: pip install transformers>=4.20.0"
            )
        
        self.model_name = model_name
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Get embedding dimension from model config
        # CLIP ViT-B/32: 512, CLIP ViT-L/14: 768
        self.embedding_dim = self.model.config.projection_dim
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Image preprocessing (CLIP uses its own preprocessing, but we keep this for compatibility)
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a batch of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Tensor of shape [batch_size, embedding_dim]
        """
        # Use CLIP processor for text
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad() if not any(p.requires_grad for p in self.model.parameters()) else torch.enable_grad():
            text_outputs = self.model.get_text_features(**inputs)
        
        # Normalize embeddings (CLIP uses normalized embeddings)
        text_embeddings = text_outputs / text_outputs.norm(dim=-1, keepdim=True)
        
        return text_embeddings
    
    def encode_image(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Encode a batch of PIL Images.
        
        Args:
            images: List of PIL Images (or None for missing images)
            
        Returns:
            Tensor of shape [batch_size, embedding_dim]
        """
        # Handle missing images
        processed_images = []
        for img in images:
            if img is None:
                # Create black image for missing images
                img = Image.new('RGB', (224, 224), color='black')
            processed_images.append(img)
        
        # Use CLIP processor for images
        inputs = self.processor(images=processed_images, return_tensors="pt")
        
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad() if not any(p.requires_grad for p in self.model.parameters()) else torch.enable_grad():
            image_outputs = self.model.get_image_features(**inputs)
        
        # Normalize embeddings (CLIP uses normalized embeddings)
        image_embeddings = image_outputs / image_outputs.norm(dim=-1, keepdim=True)
        
        return image_embeddings
    
    def encode(self, inputs: Union[List[str], List[Image.Image]]) -> torch.Tensor:
        """
        Unified encode method for backward compatibility.
        Automatically detects if input is text or images.
        
        Args:
            inputs: List of strings (text) or PIL Images
            
        Returns:
            Tensor of shape [batch_size, embedding_dim]
        """
        if len(inputs) == 0:
            device = next(self.model.parameters()).device
            return torch.empty(0, self.embedding_dim, device=device)
        
        # Detect input type
        if isinstance(inputs[0], str):
            return self.encode_text(inputs)
        elif isinstance(inputs[0], Image.Image) or inputs[0] is None:
            return self.encode_image(inputs)
        else:
            raise ValueError(f"Unsupported input type: {type(inputs[0])}")


# Backward compatibility: TextEncoder and ImageEncoder wrappers
class TextEncoder(nn.Module):
    """
    Wrapper around CLIPEncoder for text-only encoding.
    Maintains backward compatibility with existing code.
    """
    
    def __init__(self, model_name: str = 'openai/clip-vit-base-patch32', freeze: bool = True):
        super(TextEncoder, self).__init__()
        self.clip_encoder = CLIPEncoder(model_name=model_name, freeze=freeze)
        self.embedding_dim = self.clip_encoder.embedding_dim
    
    def encode(self, texts: List[str], max_length: int = 128) -> torch.Tensor:
        """
        Encode texts. max_length is ignored (CLIP handles it internally).
        """
        return self.clip_encoder.encode_text(texts)


class ImageEncoder(nn.Module):
    """
    Wrapper around CLIPEncoder for image-only encoding.
    Maintains backward compatibility with existing code.
    """
    
    def __init__(self, model_name: str = 'openai/clip-vit-base-patch32', freeze: bool = True):
        super(ImageEncoder, self).__init__()
        self.clip_encoder = CLIPEncoder(model_name=model_name, freeze=freeze)
        self.embedding_dim = self.clip_encoder.embedding_dim
        # Keep transform for compatibility (though CLIP uses its own)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    
    def encode(self, images: List[Image.Image]) -> torch.Tensor:
        """Encode images."""
        return self.clip_encoder.encode_image(images)

