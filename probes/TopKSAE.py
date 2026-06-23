import torch
import torch.nn as nn
import torch.nn.functional as F

# Standard top-K sparse autoencoder.
# Sparsity is enforced structurally: only the top-K features by magnitude survive.
# No sparsity loss, no gates, no adaptive coefficient.

class TopKSAE(nn.Module):
    def __init__(self, embed_dim, expansion_factor):
        super().__init__()
        self.embed_dim = embed_dim
        self.feature_dim = embed_dim * expansion_factor

        # dictionary weights
        self.encoder_weight = nn.Parameter(torch.randn(self.feature_dim, self.embed_dim) / self.embed_dim ** 0.5)
        self.encoder_bias = nn.Parameter(torch.randn(self.feature_dim) / self.embed_dim ** 0.5)
        self.decoder_weight = nn.Parameter(torch.randn(self.embed_dim, self.feature_dim) / self.embed_dim ** 0.5)
        self.decoder_bias = nn.Parameter(torch.randn(self.embed_dim) / self.embed_dim ** 0.5)

    def encode(self, input, k):
        input = input.to(self.encoder_weight.dtype)
        # center the input, project to feature space, ReLU
        pre = (input - self.decoder_bias) @ self.encoder_weight.T + self.encoder_bias
        pre = F.relu(pre)
        # keep only top-K features per token
        topk = torch.topk(pre, k=k, dim=-1)
        features = torch.zeros_like(pre)
        features.scatter_(-1, topk.indices, topk.values)
        active_count = (features > 0).float().sum(-1).mean()
        return features, active_count

    def decode(self, features):
        return features @ self.decoder_weight.T


# trains TopKSAE from streaming activations.
# returns: (loss, recon_loss, recon_pct, active_count)
def train_TopKSAE(
    sae: TopKSAE,
    activation,
    optimizer,
    k,
    clip_grad_norm=None,
    clip_grad_value=None,
):
    features, active_count = sae.encode(activation, k=k)

    pred = sae.decode(features)
    # average squared error over both tokens and embedding dimensions (per-coordinate MSE)
    recon_loss = (pred - activation).pow(2).mean()
    # percentage of activation energy not reconstructed
    activation_energy = activation.pow(2).mean()
    recon_pct = (recon_loss / activation_energy) * 100.0
    loss = recon_loss

    optimizer.zero_grad()
    loss.backward()
    if clip_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=clip_grad_norm)
    if clip_grad_value is not None:
        torch.nn.utils.clip_grad_value_(sae.parameters(), clip_value=clip_grad_value)
    optimizer.step()

    # keep decoder feature columns uniform
    with torch.no_grad():
        sae.decoder_weight /= sae.decoder_weight.norm(dim=0, keepdim=True)
    return loss.item(), recon_loss.item(), recon_pct.item(), active_count.item()
