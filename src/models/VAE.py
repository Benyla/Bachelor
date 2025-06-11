# Build with inspiration from https://proceedings.mlr.press/v102/lafarge19a.html
# And https://github.com/AntixK/PyTorch-VAE?

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        # 4 convolutional layers → classifier
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
        # out will be [N, 256, 4, 4] when passed here → [N, 1, 1, 1] - this is just N numbers 
        self.classifier = nn.Conv2d(256, 1, kernel_size=4)

    def forward(self, x):
        feats = []
        out = x
        for layer in self.layers:
            out = layer(out)
            feats.append(out) # save intermediate features for each layer
        logits = self.classifier(out).view(-1) # [N, 1, 1, 1] → [N] collabsing tensor to list of N numbers
        return logits, feats # returning classifier logits and intermediate features

class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256, beta=2.0, T=16687, use_adv=True, overfit=False):
        super().__init__()
        self.beta = beta
        self.T = T
        self.iter = 0
        self.use_adv = use_adv
        self.overfit = overfit
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        print(f"[Model Init] use_adv={self.use_adv}, overfit={self.overfit}")

        # Encoder: 4 convolutional layers  → flatten
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels,  32, kernel_size=5, stride=2, padding=2), nn.LeakyReLU(0.01), # [N,  3, 64, 64] → [N,  32, 32, 32]
            nn.Conv2d(32,           64, kernel_size=5, stride=2, padding=2), nn.LeakyReLU(0.01), # [N,  32, 32, 32] → [N,  64, 16, 16]
            nn.Conv2d(64,          128, kernel_size=5, stride=2, padding=2), nn.LeakyReLU(0.01), # [N,  64, 16, 16] → [N, 128, 8, 8]
            nn.Conv2d(128,         256, kernel_size=5, stride=2, padding=2), nn.LeakyReLU(0.01), # [N, 128, 8, 8]   → [N, 256, 4, 4]
            nn.Flatten() # [N, 256, 4, 4]   → [N, 4096]
        )
        self.fc_mu     = nn.Linear(256*4*4, latent_dim) # spatial dims: 4x4
        self.fc_logvar = nn.Linear(256*4*4, latent_dim) # spatial dims: 4x4

        # Decoder: opposite  of encoder, output_padding to maintain spatial dims (ConvTranspose problems)
        self.decoder_input = nn.Linear(latent_dim, 256*4*4)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 4, 4)), # [N, 256*4*4] → [N, 256, 4, 4]
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1), nn.LeakyReLU(0.01), # [N, 256, 4, 4] → [N, 128, 8, 8]
            nn.ConvTranspose2d(128,  64, kernel_size=5, stride=2, padding=2, output_padding=1), nn.LeakyReLU(0.01), # [N, 128, 8, 8] → [N, 64, 16, 16]
            nn.ConvTranspose2d(64,   32, kernel_size=5, stride=2, padding=2, output_padding=1), nn.LeakyReLU(0.01), # [N, 64, 16, 16] → [N, 32, 32, 32]
            nn.ConvTranspose2d(32, in_channels, kernel_size=5, stride=2, padding=2, output_padding=1), nn.Sigmoid() # [N, 32, 32, 32] → [N, 3, 64, 64]
        )

        if use_adv:
            self.discriminator = Discriminator(in_channels) # init discriminator
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.register_buffer('real_label', torch.tensor(1., device=device)) # real_label = 1
            self.register_buffer('fake_label', torch.tensor(0., device=device)) # fake_label = 0

    def reparameterize(self, mu, logvar): # reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # [N, latent_dim] - normally distributed with mean=0, std=1
        return mu + eps * std

    def overfit_reparameterize(self, mu, logvar): # we dont need uncertainty in overfitting
        return mu

    def encode(self, x):
        enc = self.encoder(x)
        mu, logvar = self.fc_mu(enc), torch.clamp(self.fc_logvar(enc), min=-10.0, max=10.0) # to prevent eplosion in reparameterization
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
    
    def _gamma(self, layer_idx: int): # gamma function to follow lafarge2019capturing
        return min(max((self.iter/self.T) - layer_idx, 0.0), 1.0)

    def loss(self, x, x_rec, mu, logvar):

        # Reconstruction loss - this implementation will add a constant to the loss
        log_px = dist.Normal(x_rec, 1).log_prob(x)
        recon_loss = -torch.sum(log_px) # returning the negative log likelihood makes the loss positive, as we want to minimize it 
        # logpx is a density function, so loss can be bigger than 1.

        # KL loss
        kl_loss = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar)

        #cAdversarial loss
        adv_fm_loss = torch.tensor(0., device=x.device)
        if self.use_adv:
            # Freeze discriminator parameters as we dont want to update them here
            for p in self.discriminator.parameters():
                p.requires_grad = False

            _, feats_real = self.discriminator(x)       # real features
            _, feats_fake = self.discriminator(x_rec)   # fake features

            # Unfreeze discriminator parameters again
            for p in self.discriminator.parameters():
                p.requires_grad = True

            # feature matching loss
            # Forces the VAE to generate images that are similar to the real ones in the feature space of the discriminator
            # This does not influence the discriminator 
            # We simply uses the fact that the discriminator pushes real and fake images apart in the feature space
            # Then if the VAE can generate images that are close to the real ones in the  discriminator feature space
            # It must be hard to distinguish between real and fake images
            fm_losses = [
                self._gamma(i) * F.mse_loss(ff, fr.detach(), reduction='sum')
                for i, (fr, ff) in enumerate(zip(feats_real, feats_fake))
            ]
            adv_fm_loss = sum(fm_losses)

            # iteration counter
            self.iter += 1

        return recon_loss, kl_loss, adv_fm_loss

    def loss_discriminator(self, x, x_rec):
        # Discriminator loss - gets lower when D is good at classifying real vs fake
        # this mean assigning 1 to real images and 0 to fake images within the logits
        bce = F.binary_cross_entropy_with_logits # applies sigmoid internally, to get numbers between 0 and 1
        real_logits, _ = self.discriminator(x)
        fake_logits, _ = self.discriminator(x_rec.detach())
        d_loss = (
            bce(real_logits, self.real_label.expand_as(real_logits)) + 
            bce(fake_logits, self.fake_label.expand_as(fake_logits))
        )
        return d_loss

