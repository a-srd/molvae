import torch
import torch.nn as nn
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from config import DEVICE as device
from typing import Tuple

# Hyperparameters
from config import ENCODER_EMBEDDING_SIZE as encoder_embedding_size
from config import LATENT_EMBEDDING_SIZE as latent_embedding_size
from config import DECODER_SIZE as decoder_size
from config import EDGE_DIM as edge_dim
from config import NO_HEADS as no_heads
from config import NUM_ENCODER_LAYERS
from config import NUM_DECODER_LAYERS


class MOLVAE(nn.Module):
    """
    Graph Variational Autoencoder (GVAE) for molecular graph generation.
    """
    def __init__(self, feature_size: int):
        super(MOLVAE, self).__init__()

        self.encoder_emb_size = encoder_embedding_size
        self.latent_embedding_size = latent_embedding_size
        self.dec_size = decoder_size
        self.edge_dim = edge_dim
        self.no_heads = no_heads

        # Encoder layers
        encoder_layers = []
        for _ in range(NUM_ENCODER_LAYERS):
            encoder_layers.extend([
                TransformerConv(self.encoder_emb_size, self.encoder_emb_size, heads=self.no_heads, concat=False, beta=True, edge_dim=self.edge_dim),
                nn.BatchNorm1d(self.encoder_emb_size),
                nn.ReLU()
            ])

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent transform layers (mean and log variance)
        self.mu_transform = TransformerConv(self.encoder_emb_size, self.latent_embedding_size, heads=self.no_heads, concat=False, beta=True, edge_dim=self.edge_dim)
        self.logvar_transform = TransformerConv(self.encoder_emb_size, self.latent_embedding_size, heads=self.no_heads, concat=False, beta=True, edge_dim=self.edge_dim)

        # Decoder layers
        decoder_layers = []
        for _ in range(NUM_DECODER_LAYERS - 1):  # One less loop for the final layer
            decoder_layers.extend([
                nn.Linear(self.latent_embedding_size * 2, self.dec_size),
                nn.BatchNorm1d(self.dec_size),
                nn.ReLU()
            ])
        decoder_layers.append(nn.Linear(self.dec_size, 1))  # Final layer for logits

        self.decoder = nn.Sequential(*decoder_layers) 

    def encode(self, x: torch.Tensor, edge_attr: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes input graph features into mean and log variance of the latent space."""
        x = self.encoder(x, edge_index, edge_attr)
        mu = self.mu_transform(x, edge_index, edge_attr)
        logvar = self.logvar_transform(x, edge_index, edge_attr)
        return mu, logvar


    def decode(self, z: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        """Decodes latent representation into upper triangular logits of the adjacency matrix."""
        inputs = []

        # Process each graph in the batch separately
        for graph_id in torch.unique(batch_index):
            graph_mask = batch_index == graph_id
            graph_z = z[graph_mask]
            edge_indices = torch.triu_indices(graph_z.shape[0], graph_z.shape[0], offset=1)  # Upper triangular indices
            
            # Prepare input for the decoder
            source_indices = edge_indices[0].repeat_interleave(self.latent_embedding_size).reshape(-1, self.latent_embedding_size).to(device)
            target_indices = edge_indices[1].repeat_interleave(self.latent_embedding_size).reshape(-1, self.latent_embedding_size).to(device)
            sources_feats = torch.gather(graph_z, 0, source_indices)
            target_feats = torch.gather(graph_z, 0, target_indices)
            graph_inputs = torch.cat([sources_feats, target_feats], dim=1)
            inputs.append(graph_inputs)

        inputs = torch.cat(inputs)  # Concatenate for the entire batch
        edge_logits = self.decoder(inputs)
        return edge_logits

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from the latent distribution."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu  

    def forward(self, x, edge_attr, edge_index, batch_index):
        mu, logvar = self.encode(x, edge_attr, edge_index)
        z = self.reparameterize(mu, logvar)
        triu_logits = self.decode(z, batch_index)
        return triu_logits, mu, logvar
