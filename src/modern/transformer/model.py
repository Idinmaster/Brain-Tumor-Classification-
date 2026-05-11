"""
Vision Transformer (ViT) for Brain Tumor Classification
=========================================================
Supports:
  - Custom lightweight ViT (MiniViT)
  - Transfer-learning wrappers: ViT-B/16, Swin-T

Classes: glioma, meningioma, notumor, pituitary
"""

import math
import torch
import torch.nn as nn
from torchvision import models


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """
    Split image into non-overlapping patches and project to embedding dim.

    Args:
        img_size   : square image size (default 224)
        patch_size : square patch size (default 16) → 14×14 = 196 patches
        in_channels: number of input channels (3 for RGB)
        embed_dim  : transformer embedding dimension
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, embed_dim, H/P, W/P) → (B, N, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Standard scaled dot-product multi-head self-attention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)         # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """Pre-LN Transformer encoder block: Attention + MLP."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# MiniViT — lightweight custom Vision Transformer
# ---------------------------------------------------------------------------

class MiniViT(nn.Module):
    """
    Lightweight Vision Transformer for brain tumor classification.

    Default config (img_size=224, patch_size=16):
      • 196 patch tokens + 1 [CLS] token = 197 sequence length
      • embed_dim=384, num_heads=6, depth=8  (~25M params)

    Input  : (B, 3, 224, 224)
    Output : (B, num_classes)
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 4,
        embed_dim: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Learnable [CLS] token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)                               # (B, N, D)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)                # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                        # (B, N+1, D)
        x = self.pos_drop(x + self.pos_embed)

        # Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)

        # Classification from [CLS] token
        cls_out = x[:, 0]
        return self.head(cls_out)


# ---------------------------------------------------------------------------
# Transfer-learning wrappers (torchvision ViT / Swin)
# ---------------------------------------------------------------------------

def get_vit_b16(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """ViT-B/16 from torchvision with a fine-tuned classification head."""
    weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
    model = models.vit_b_16(weights=weights)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.GELU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
    )
    return model


def get_swin_t(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """Swin-Tiny from torchvision with a fine-tuned classification head."""
    weights = models.Swin_T_Weights.DEFAULT if pretrained else None
    model = models.swin_t(weights=weights)
    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.GELU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
    )
    return model


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

SUPPORTED_ARCHITECTURES = {
    "mini_vit":  lambda nc, pt: MiniViT(num_classes=nc),
    "vit_b16":   get_vit_b16,
    "swin_t":    get_swin_t,
}


def get_transformer_model(
    architecture: str = "swin_t",
    num_classes: int = 4,
    pretrained: bool = True,
) -> nn.Module:
    """
    Factory function for Transformer models.

    Args:
        architecture: One of 'mini_vit', 'vit_b16', 'swin_t'.
        num_classes:  Number of output classes (default 4).
        pretrained:   Use ImageNet pretrained weights (ignored for mini_vit).

    Returns:
        nn.Module ready for training.
    """
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Choose from: {list(SUPPORTED_ARCHITECTURES.keys())}"
        )
    return SUPPORTED_ARCHITECTURES[architecture](num_classes, pretrained)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy = torch.randn(2, 3, 224, 224).to(device)

    for arch in SUPPORTED_ARCHITECTURES:
        model = get_transformer_model(arch, num_classes=4, pretrained=False).to(device)
        out = model(dummy)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[{arch:10s}]  output={out.shape}  trainable params={params:,}")
