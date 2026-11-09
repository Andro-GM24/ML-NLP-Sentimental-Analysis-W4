"""AE and VAE architectures for Workshop #4.

Both models operate on 768-d RoBERTa CLS vectors (already normalised with
StandardScaler before being passed in).  The encoder compresses to a small
latent code; the decoder reconstructs the original vector.

Architectures
-------------
AE  : deterministic bottleneck, trained with MSE reconstruction loss.
VAE : stochastic bottleneck (µ, log σ²) + reparameterisation trick,
      trained with ELBO = MSE + β · KL.  Supports KL annealing.

Usage (shape check)
-------------------
    poetry run python src/autoencoder.py
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ── shared building block ─────────────────────────────────────────────────────

def _block(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    """FC (x2)→ GELU → Dropout."""
    #test GELU
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.GELU(),
        nn.Dropout(dropout),
    )


# ── Autoencoder ───────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """768 → 256 → 128 → z_dim."""

    def __init__(self, input_dim: int = 768, z_dim: int = 64, dropout: float = 0.2,input_dropout: float = 0.2):
        super().__init__()
        self.noise = nn.Dropout(input_dropout)
        self.net = nn.Sequential(
            _block(input_dim, 256, dropout),
            _block(256, 128, dropout),
            nn.Linear(128, z_dim),   # no activation on bottleneck
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.noise(x)  # Corrupción de entrada para Denoising AE
        return self.net(x)


class Decoder(nn.Module):
    """z_dim → 128 → 256 → 768  (no output activation — MSE over real-valued vectors)."""

    def __init__(self, z_dim: int = 64, output_dim: int = 768, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            _block(z_dim, 128, dropout),
            _block(128, 256, dropout),
            nn.Linear(256, output_dim),  # linear output for sequential targets
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class AE(nn.Module):
    """Standard autoencoder."""

    def __init__(self, input_dim: int = 768, z_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.encoder = Encoder(input_dim, z_dim, dropout)
        self.decoder = Decoder(z_dim, input_dim, dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (reconstruction, latent code)."""
        z    = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ── Variational Autoencoder ───────────────────────────────────────────────────

class VAEEncoder(nn.Module):
    """768 ->(Dropout)-> 256 -> 128 -> (µ, log σ²)."""

    def __init__(self, input_dim: int = 768, z_dim: int = 64, dropout: float = 0.2, input_dropout: float = 0.2):
        super().__init__()
        self.noise = nn.Dropout(input_dropout)  
        self.shared = nn.Sequential(
            _block(input_dim, 256, dropout),
            _block(256, 128, dropout),
        )
        self.fc_mu      = nn.Linear(128, z_dim)
        self.fc_log_var = nn.Linear(128, z_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x       = self.noise(x)  
        h       = self.shared(x)
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var


class VAE(nn.Module):
    """Variational autoencoder with reparameterisation trick."""

    def __init__(self, input_dim: int = 768, z_dim: int = 64, dropout: float = 0.2,input_dropout: float = 0.2):
        super().__init__()
        self.encoder = VAEEncoder(input_dim, z_dim, dropout, input_dropout)
        self.decoder = Decoder(z_dim, input_dim, dropout)

    def reparameterise(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """z = µ + σ ⊙ ε,  ε ~ N(0, I).  Differentiable via µ and log σ²."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + std * eps
        # At eval time use µ directly (deterministic, as the workshop requests)
        return mu

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (reconstruction, µ, log σ²)."""
        mu, log_var = self.encoder(x)
        z           = self.reparameterise(mu, log_var)
        x_hat       = self.decoder(z)
        return x_hat, mu, log_var

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return µ (deterministic latent code for downstream analysis)."""
        mu, _ = self.encoder(x)
        return mu

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        """Sample n points from the prior N(0, I) and decode them."""
        z = torch.randn(n, self.encoder.fc_mu.out_features, device=device)
        return self.decoder(z)


# ── losses ────────────────────────────────────────────────────────────────────

def ae_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """MSE reconstruction loss."""
    return nn.functional.mse_loss(x_hat, x)


def vae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1.0,
    kl_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ELBO = MSE + β · kl_weight · KL.

    kl_weight is used for KL annealing: start at 0.0 and ramp to 1.0
    over the first N epochs before β takes over.

    Returns (total_loss, recon_loss, kl_loss) for logging.
    """
    recon = nn.functional.mse_loss(x_hat, x)
    # KL divergence: -0.5 * sum(1 + log σ² - µ² - σ²)
    kl    = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    total = recon + beta * kl_weight * kl
    return total, recon, kl


# ── shape check ───────────────────────────────────────────────────────────────

def _shape_check() -> None:
    batch = torch.randn(8, 768)

    ae = AE()
    x_hat, z = ae(batch)
    assert x_hat.shape == (8, 768), f"AE recon shape {x_hat.shape}"
    assert z.shape      == (8, 64),  f"AE latent shape {z.shape}"
    print(f"AE   — input {tuple(batch.shape)} → z {tuple(z.shape)} → recon {tuple(x_hat.shape)}")

    vae = VAE()
    x_hat, mu, log_var = vae(batch)
    assert x_hat.shape  == (8, 768), f"VAE recon shape {x_hat.shape}"
    assert mu.shape     == (8, 64),  f"VAE µ shape {mu.shape}"
    assert log_var.shape == (8, 64), f"VAE log σ² shape {log_var.shape}"
    print(f"VAE  — input {tuple(batch.shape)} → µ {tuple(mu.shape)} → recon {tuple(x_hat.shape)}")

    loss_ae = ae_loss(batch, x_hat)
    total, recon, kl = vae_loss(batch, x_hat, mu, log_var, beta=1.0, kl_weight=0.5)
    print(f"AE loss  : {loss_ae.item():.4f}")
    print(f"VAE loss : total {total.item():.4f} | recon {recon.item():.4f} | kl {kl.item():.4f}")

    # Parameter counts
    for name, model in [("AE", ae), ("VAE", vae)]:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{name} parameters: {total_params:,}")


if __name__ == "__main__":
    _shape_check()