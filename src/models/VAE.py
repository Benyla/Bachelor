import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        # 4 conv layers with 5×5 kernels, stride=2, padding=2
        # BatchNorm + LeakyReLU(0.01) as per Lafarge et al.
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
                nn.LeakyReLU(0.01)
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
                nn.LeakyReLU(0.01)
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
                nn.LeakyReLU(0.01)
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
                nn.LeakyReLU(0.01)
            )
        ])
        # classifier head: from 256×4×4 → 1 logit
        self.classifier = nn.Conv2d(256, 1, kernel_size=4)

    def forward(self, x):
        feats = []
        out = x
        for layer in self.layers:
            out = layer(out)
            feats.append(out)
        logits = self.classifier(out).view(-1)
        return logits, feats

class VAE(nn.Module):
    def __init__(self,
                 in_channels=3,
                 latent_dim=256,
                 beta=2.0,
                 T=2500,
                 use_adv=True,
                 overfit=False):
        super().__init__()
        self.beta = beta
        self.T = T
        self.iter = 0
        self.use_adv = use_adv
        self.overfit = overfit
        print(f"[Model Init] use_adv={self.use_adv}, overfit={self.overfit}")

        # Encoder: 4 conv blocks → flatten
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2), nn.LeakyReLU(0.01),
            nn.Conv2d(32,           64, kernel_size=5, stride=2, padding=2), nn.LeakyReLU(0.01),
            nn.Conv2d(64,          128, kernel_size=5, stride=2, padding=2), nn.LeakyReLU(0.01),
            nn.Conv2d(128,         256, kernel_size=5, stride=2, padding=2), nn.LeakyReLU(0.01),
            nn.Flatten()
        )
        self.fc_mu     = nn.Linear(256*4*4, latent_dim)
        self.fc_logvar = nn.Linear(256*4*4, latent_dim)

        # Decoder: mirror of encoder
        self.decoder_input = nn.Linear(latent_dim, 256*4*4)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1), nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(128,  64, kernel_size=5, stride=2, padding=2, output_padding=1), nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64,   32, kernel_size=5, stride=2, padding=2, output_padding=1), nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, in_channels, kernel_size=5, stride=2, padding=2, output_padding=1), nn.Sigmoid()
        )

        if use_adv:
            self.discriminator = Discriminator(in_channels)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.register_buffer('real_label', torch.tensor(1., device=device))
            self.register_buffer('fake_label', torch.tensor(0., device=device))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def overfit_reparameterize(self, mu, logvar):
        return mu

    def encode(self, x):
        enc = self.encoder(x)
        mu, logvar = self.fc_mu(enc), self.fc_logvar(enc)
        if self.overfit:
            z = self.overfit_reparameterize(mu, logvar)
        else:
            z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        return self.decoder(self.decoder_input(z))

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, mu, logvar, z

    def _gamma(self, layer_idx: int):
        t = self.iter / self.T
        return min(max(t - layer_idx, 0.0), 1.0)
    
    def get_beta(self):
        t = min(self.iter / self.T, 1.0)
        return t * self.beta

    def loss(self, x, x_rec, mu, logvar, sigma=1.0):
        # --- ELBO terms ---
        # 1) Recon loss
        recon_loss = F.mse_loss(x_rec, x, reduction='sum')
        # 2) KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # --- VAE+ adversarial terms ---
        adv_fm_loss = torch.tensor(0., device=x.device)
        if self.use_adv:
            # (a) Discriminator classification update
            real_logits, feats_real = self.discriminator(x)
            fake_logits_detach, _   = self.discriminator(x_rec.detach())
            bce = F.binary_cross_entropy_with_logits
            # (b) Feature-matching loss for generator
            # Temporarily freeze D
            for p in self.discriminator.parameters(): p.requires_grad = False
            _, feats_fake = self.discriminator(x_rec)
            # Unfreeze D
            for p in self.discriminator.parameters(): p.requires_grad = True

            fm_losses = []
            for i, (fr, ff) in enumerate(zip(feats_real, feats_fake)):
                w = self._gamma(i)
                fm_losses.append(w * F.mse_loss(ff, fr.detach(), reduction='sum'))
            adv_fm_loss = sum(fm_losses)
            self.iter += 1

        return recon_loss, kl_loss, adv_fm_loss

    def loss_discriminator(self, x, x_rec):
        """
        Discriminator loss: binary cross-entropy on real vs. reconstructed images.
        """
        bce = F.binary_cross_entropy_with_logits
        real_logits, _ = self.discriminator(x)
        fake_logits, _ = self.discriminator(x_rec.detach())
        d_loss = (
            bce(real_logits, self.real_label.expand_as(real_logits)) +
            bce(fake_logits, self.fake_label.expand_as(fake_logits))
        )
        return d_loss
