import torch
import torch.nn as nn
import torch.nn.functional as F

# Advanced top-K SAE:
#   - no encoder bias (threshold fixed at zero)
#   - auxiliary K loss on dead features (AuxK)
#   - feature resampling
#   - intended to be used with Adam beta1 = 0

class AdvTopKSAE(nn.Module):
    def __init__(self, embed_dim, expansion_factor, auxk_coeff=1.0/32.0, dead_threshold=1e-5):
        super().__init__()
        self.embed_dim = embed_dim
        self.feature_dim = embed_dim * expansion_factor
        self.auxk_coeff = auxk_coeff
        self.dead_threshold = dead_threshold

        self.encoder_weight = nn.Parameter(torch.randn(self.feature_dim, self.embed_dim) / self.embed_dim ** 0.5)
        self.decoder_weight = nn.Parameter(torch.randn(self.embed_dim, self.feature_dim) / self.embed_dim ** 0.5)
        self.decoder_bias = nn.Parameter(torch.randn(self.embed_dim) / self.embed_dim ** 0.5)

        # activation frequency EMA per feature
        self.register_buffer("activation_ema", torch.zeros(self.feature_dim))
        # small ring buffer of recent activations for resampling
        self.resample_buffer = []
        self.resample_buffer_max_entries = 128
        self.resample_tokens_per_entry = 16

    def encode(self, input, k):
        input = input.to(self.encoder_weight.dtype)
        # no encoder bias: threshold fixed at zero
        pre = (input - self.decoder_bias) @ self.encoder_weight.T
        pre_relu = F.relu(pre)
        topk = torch.topk(pre_relu, k=k, dim=-1)
        features = torch.zeros_like(pre_relu)
        features.scatter_(-1, topk.indices, topk.values)
        active_count = (features > 0).float().sum(-1).mean()
        return features, active_count, pre

    def decode(self, features):
        return features @ self.decoder_weight.T

    @torch.no_grad()
    def update_dead_features(self, features):
        # features: (tokens, feature_dim)
        freq = (features > 0).float().mean(dim=0)
        self.activation_ema = 0.99 * self.activation_ema + 0.01 * freq

    def get_dead_mask(self):
        return self.activation_ema < self.dead_threshold

    @torch.no_grad()
    def add_to_resample_buffer(self, activations):
        # activations: (tokens, embed_dim)
        activations = activations.detach().cpu()
        n = activations.shape[0]
        if n == 0:
            return
        k = min(self.resample_tokens_per_entry, n)
        idx = torch.randperm(n)[:k]
        self.resample_buffer.append(activations[idx])
        if len(self.resample_buffer) > self.resample_buffer_max_entries:
            self.resample_buffer.pop(0)

    @torch.no_grad()
    def resample_dead_features(self, optimizer):
        dead_mask = self.get_dead_mask()
        dead_indices = torch.where(dead_mask)[0]
        n_dead = dead_indices.numel()
        if n_dead == 0 or len(self.resample_buffer) == 0:
            return

        device = self.encoder_weight.device
        dtype = self.encoder_weight.dtype

        all_acts = torch.cat(self.resample_buffer, dim=0).to(device).to(dtype)
        idx = torch.randint(0, all_acts.shape[0], (n_dead,))
        samples = all_acts[idx]
        samples = samples / (samples.norm(dim=1, keepdim=True) + 1e-8)

        alive_mask = ~dead_mask
        if alive_mask.any():
            avg_alive_norm = self.encoder_weight[alive_mask].norm(dim=1).mean()
        else:
            avg_alive_norm = 1.0

        self.encoder_weight[dead_indices] = samples * avg_alive_norm
        self.decoder_weight[:, dead_indices] = samples.T
        self.activation_ema[dead_indices] = 1.0

        # reset optimizer state for dead features (best-effort)
        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state.get(p)
                if not state:
                    continue
                for key in ("exp_avg", "exp_avg_sq"):
                    buf = state.get(key)
                    if buf is None:
                        continue
                    try:
                        buf[dead_indices] = 0.0
                    except Exception:
                        pass


# trains AdvTopKSAE from streaming activations.
# returns: (loss, recon_loss, recon_pct, active_count)
def train_AdvTopKSAE(
    sae: AdvTopKSAE,
    activation,
    optimizer,
    k,
    step,
    resample_every=1000,
    auxk_coeff=None,
    clip_grad_norm=None,
    clip_grad_value=None,
):
    if auxk_coeff is None:
        auxk_coeff = sae.auxk_coeff

    features, active_count, pre = sae.encode(activation, k)
    pred = sae.decode(features)

    # average squared error over both tokens and embedding dimensions (per-coordinate MSE)
    recon_loss = (pred - activation).pow(2).mean()
    activation_energy = activation.pow(2).mean()
    recon_pct = (recon_loss / activation_energy) * 100.0

    # AuxK: use top-k dead features to reconstruct the residual
    residual = activation - pred
    dead_mask = sae.get_dead_mask()
    n_dead = dead_mask.sum().item()

    auxk_loss = activation.new_tensor(0.0)
    if n_dead > 0:
        dead_mask_t = dead_mask.to(pre.dtype)
        dead_pre = pre * dead_mask_t
        if n_dead >= k:
            topk_dead = torch.topk(dead_pre, k=k, dim=-1)
            aux_features = torch.zeros_like(pre)
            aux_features.scatter_(-1, topk_dead.indices, F.relu(topk_dead.values))
        else:
            aux_features = F.relu(pre) * dead_mask_t
        aux_pred = sae.decode(aux_features)
        auxk_loss = (aux_pred - residual).pow(2).mean() * auxk_coeff

    loss = recon_loss + auxk_loss

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

    sae.update_dead_features(features)
    sae.add_to_resample_buffer(activation)
    if step % resample_every == 0 and step > 0:
        sae.resample_dead_features(optimizer)

    return loss.item(), recon_loss.item(), recon_pct.item(), active_count.item()
