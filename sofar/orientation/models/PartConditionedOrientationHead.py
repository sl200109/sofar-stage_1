import hashlib
from typing import List

import torch
import torch.nn as nn


def _hash_instruction_tokens(text: str, vocab_size: int, max_tokens: int) -> List[int]:
    text = str(text or "").strip().lower()
    if not text:
        return [0]
    tokens = [tok for tok in text.replace(".", " ").replace(",", " ").split() if tok]
    if not tokens:
        return [0]
    hashed = []
    for token in tokens[:max_tokens]:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        hashed.append(int(digest[:8], 16) % vocab_size)
    return hashed or [0]


class PartConditionedOrientationHead(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.point_dim = int(getattr(config, "point_dim", 6))
        self.prior_dim = int(getattr(config, "prior_dim", 8))
        self.hidden_dim = int(getattr(config, "hidden_dim", 128))
        self.text_dim = int(getattr(config, "text_dim", 64))
        self.vocab_size = int(getattr(config, "vocab_size", 4096))
        self.max_tokens = int(getattr(config, "max_tokens", 32))

        self.point_encoder = nn.Sequential(
            nn.Linear(self.point_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.prior_encoder = nn.Sequential(
            nn.Linear(self.prior_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 2),
        )
        self.text_embedding = nn.Embedding(self.vocab_size, self.text_dim)
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 2),
        )
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim + self.hidden_dim // 2 + self.hidden_dim // 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 3),
        )

    def _encode_text(self, instructions):
        device = self.text_embedding.weight.device
        rows = []
        for instruction in instructions:
            token_ids = _hash_instruction_tokens(instruction, self.vocab_size, self.max_tokens)
            token_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
            rows.append(self.text_embedding(token_tensor).mean(dim=0))
        text_features = torch.stack(rows, dim=0)
        return self.text_proj(text_features)

    def forward(self, pts, instruction, priors=None):
        point_features = self.point_encoder(pts)
        point_features = point_features.max(dim=1).values

        if priors is None:
            priors = torch.zeros(
                point_features.size(0),
                self.prior_dim,
                device=point_features.device,
                dtype=point_features.dtype,
            )
        prior_features = self.prior_encoder(priors)
        text_features = self._encode_text(instruction)

        fused = torch.cat([point_features, prior_features, text_features], dim=-1)
        return self.head(fused)
