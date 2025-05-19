import torch
from typing import List


class DummyEmbedder:
    def __init__(self, embed_dim: int = 128):
        # Choose any embedding dimension you like
        self.embed_dim = embed_dim

    def __call__(
        self, code_texts: List[str], stmt_texts: List[str], lang: str
    ) -> torch.Tensor:
        """
        Args:
          code_texts: list of source‐code strings, length = batch_size
          stmt_texts: list of problem statements, length = batch_size
          lang: one of 'Cpp', 'Java', 'Python' (unused here)
        Returns:
          A float Tensor of shape [batch_size, embed_dim]
        """
        batch_size = len(code_texts)
        # Random tensor simulates real embeddings
        return torch.randn(batch_size, self.embed_dim)
