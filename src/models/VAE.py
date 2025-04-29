import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.logit = nn.Conv2d(128, 1, 8)

    def forward(self, x):
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        out = self.logit(f3).view(-1)
        return out, [f1, f2, f3]

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) with a convolutional encoder and decoder.
    Args:
        in_channels (int): Number of channels in the input image.
        latent_dim (int): Dimensionality of the latent space.
        image_size (int): Height/width of the (square) input image.
    """
    def __init__(self, in_channels=3, latent_dim=128, use_adv: bool = False, beta: float = 1.0, T: int = 2500):
        super(VAE, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.beta = beta
        self.use_adv = use_adv
        self.T = T
        self.iter = 0

        # -----------------------
        # ENCODER
        # -----------------------
        self.encoder = nn.Sequential(
            # Input: (in_channels, 64, 64)
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # -> (32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),           # -> (64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),          # -> (128, 8, 8)
            nn.ReLU(),
            nn.Flatten()  # -> (128*8*8)
        )

        # Fully-connected layers to produce the latent Gaussian parameters
        self.fc_mu = nn.Linear(128*8*8, latent_dim)
        self.fc_logvar = nn.Linear(128*8*8, latent_dim)

        # -----------------------
        # DECODER
        # -----------------------
        # Map latent vector back to flattened feature map
        self.decoder_input = nn.Linear(latent_dim, 128*8*8)
        self.decoder = nn.Sequential(
            # Reshape to (128, 8, 8)
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> (64, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # -> (32, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),  # -> (in_channels, 64, 64)
            nn.Sigmoid()  # Normalizes the output to [0, 1]
        )

        if use_adv:
            self.discriminator = Discriminator(in_channels)
            self.register_buffer('real_label', torch.tensor(1.))
            self.register_buffer('fake_label', torch.tensor(0.))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar) # Standard deviation from log variance
        eps = torch.randn_like(std) # Sample epsilon from standard normal with matching dim as std
        return mu + eps * std # Return reparameterized sample

    def forward(self, x):
        x_enc = self.encoder(x)
        mu, logvar = self.fc_mu(x_enc), self.fc_logvar(x_enc)
        z = self.reparameterize(mu, logvar)
        x_rec = self.decoder(self.decoder_input(z))
        return x_rec, mu, logvar

    def _gamma(self, layer_idx: int):
        t = self.iter / self.T
        return min(max(t - layer_idx, 0.0), 1.0)

    def loss(self, x, mu, logvar, sigma=1.0):
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(self.decoder_input(z))
        log_px_given_z = dist.Normal(recon_x, sigma).log_prob(x)
        recon_loss = -torch.sum(log_px_given_z)

        # VEA+ terms
        adv_loss = torch.tensor(0.0, device=x.device)
        if self.use_adv:
            x_rec = recon_x.detach()
            real_logits, feats_real = self.discriminator(x)
            fake_logits, feats_fake = self.discriminator(x_rec)
            bce = F.binary_cross_entropy_with_logits
            d_loss = bce(real_logits, self.real_label.expand_as(real_logits)) + \
                     bce(fake_logits, self.fake_label.expand_as(fake_logits))

            fm_losses = []
            for i, (fr, ff) in enumerate(zip(feats_real, feats_fake)):
                w = self._gamma(i)
                fm_losses.append(w * F.mse_loss(ff, fr.detach()))
            adv_loss = sum(fm_losses)
            self.iter += 1
        return recon_loss, kl_loss, adv_loss, (d_loss if self.use_adv else None)

    
    def decode(self, z): # used in sample_generation.py when we have to decode a random z
        return self.decoder(self.decoder_input(z))
    
    def encode(self, x, return_stats=False):
        enc = self.encoder(x)
        mu, logvar = self.fc_mu(enc), self.fc_logvar(enc)
        z = self.reparameterize(mu, logvar)
        return (z, mu, logvar) if return_stats else z
