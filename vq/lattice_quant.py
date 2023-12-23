import torch
import torch.nn as nn
import torch.nn.functional as F
from vq.quantizer import Quantizer


class LatticeQuantizer(Quantizer):
    B: float
    sparcity_cost: float
    embedding: nn.Embedding

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 commitment_cost: float,
                 sparcity_cost: float,
                 initialize_embedding_b: bool = True):

        super(LatticeQuantizer, self).__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost
        )

        self.sparcity_cost = sparcity_cost
        self.B = 1.0 / ((self.num_embeddings ** (1.0 / self.embedding_dim)) - 1.0) # noqa

        self.embedding = nn.Embedding(1, self.embedding_dim)
        if initialize_embedding_b:
            self.embedding.weight.data.uniform_(-self.B, self.B)
        else:
            self.embedding.weight.data.uniform_(-1, 1)

    def forward(self, latents: torch.Tensor):
        latents = latents.permute(
            0, 2, 3, 1
        ).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        flat_latents = latents.view(-1, self.embedding_dim)  # [BHW x D]

        # Babai estimate
        babai_estimate = torch.round(
            torch.mul(flat_latents, 1 / self.embedding.weight)
        )

        # Quantize the latents
        quantized_latents_flat = torch.mul(
            self.embedding.weight, babai_estimate
        )
        quantized_latents = quantized_latents_flat.view(latents.shape)

        # Compute the LQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())
        size_loss = -torch.sum(torch.abs(self.embedding.weight))

        lq_loss = (
            embedding_loss
            + self.commitment_cost * commitment_loss
            + self.sparsity_cost * size_loss
        )

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        # convert quantized from BHWC -> BCHW
        return {
            "vq_loss": lq_loss,
            "quantized": quantized_latents.permute(0, 3, 1, 2).contiguous(),
            "quantized_flat": quantized_latents_flat,
        }
