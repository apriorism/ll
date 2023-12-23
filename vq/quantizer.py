import torch
import torch.nn as nn


class Quantizer(nn.Module):
    num_embeddings: int
    embedding_dim: int
    commitment_cost: float

    def __init__(self,
                 num_embeddings: int = 512,
                 embedding_dim: int = 64,
                 commitment_cost: float = 0.25):
        super(Quantizer, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

    def forward(self, inputs: torch.Tensor):
        raise NotImplementedError()
